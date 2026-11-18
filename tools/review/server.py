#!/usr/bin/env python3
"""Local review UI for ringgrid detection results.

Serves the review app (built into ``app/dist``; the single-file ``index.html``
until it is built) plus a small JSON API on 127.0.0.1. Everything stays on
the local machine: the server refuses to bind a non-loopback address, the page
is served with a CSP that forbids any external request, and the output
directory (feedback, tile cache, detection runs) must be git-ignored.

Standard library only (Pillow is needed solely for ``--split`` or non-PNG
inputs). See ``tools/review/README.md``.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import ipaddress
import json
import math
import os
import re
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from strip_tiles import SKIP_STRIPS, save_tile, tile_name  # noqa: E402

SCHEMA = "ringgrid.review.v1"
VERDICTS = ("ok", "has_issues", "unusable_image")
FLAG_KINDS = ("wrong_id", "wrong_center", "false_positive", "ok_confirmed")
PIN_TYPES = ("missed_marker", "note")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BROWSER_NATIVE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAX_BODY = 8 * 1024 * 1024
DEFAULT_MARKER_RADIUS_PX = 16.0
# Match radius for pairing the same marker across runs, and the shift beyond
# which a paired marker counts as "position differs" (pixels).
DEFAULT_MATCH_PX = 10.0
DEFAULT_POS_TOL_PX = 0.5
# The page is the React app built into app/dist (`bun install && bun run build`
# in tools/review/app). Every file it loads is in that bundle, so scripts,
# styles and fonts come from 'self' only. Inline styles are allowed because the
# components position layers with style attributes; no inline script is.
APP_DIST = HERE / "app" / "dist"
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; "
    "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
# The single-file page (index.html) the bundle replaces: served only while the
# bundle has not been built. Its script and styles are inline.
LEGACY_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'"
)
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}
BUILD_HINT = "the review app is not built: run `bun install && bun run build` in tools/review/app (serving the legacy page meanwhile)"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Defensive detection-JSON normalisation (single place; shapes vary by version)
# --------------------------------------------------------------------------


def _num(v):
    """Finite float or None (bools and strings are not numbers here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        return int(v)
    return None


def _pair(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        a, b = _num(v[0]), _num(v[1])
        if a is not None and b is not None:
            return [a, b]
    return None


def _int_pair(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        a, b = _int(v[0]), _int(v[1])
        if a is not None and b is not None:
            return [a, b]
    return None


def _ellipse(v):
    """Ellipse as {cx, cy, a, b, angle(rad)} or None.

    Accepts the object form (``cx, cy, a, b, angle``) and a 5-element array.
    """
    vals = None
    if isinstance(v, dict):
        vals = [_num(v.get(k)) for k in ("cx", "cy", "a", "b", "angle")]
        if vals[4] is None and "angle" not in v:
            vals[4] = 0.0
    elif isinstance(v, (list, tuple)) and len(v) == 5:
        vals = [_num(x) for x in v]
    if vals is None or any(x is None for x in vals) or vals[2] <= 0 or vals[3] <= 0:
        return None
    return dict(zip(("cx", "cy", "a", "b", "angle"), vals))


def _scalar_details(diag) -> dict:
    """Flat ``group.field`` map of scalar leaves of a per-marker diagnostics
    record (at most one nesting level; bulky point lists are dropped)."""
    out = {}
    if not isinstance(diag, dict):
        return out
    for k, v in diag.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (bool, str)) or _num(v2) is not None:
                    out[f"{k}.{k2}"] = v2
        elif isinstance(v, (bool, str)) or _num(v) is not None:
            out[k] = v
    return out


def normalise_detection(data, tile: str = "", run: str = "") -> dict:
    """Return one stable shape for any detector-version JSON.

    Output: ``{tile, run, image_size, frame, markers[], n_markers, n_decoded,
    radius_hint, warnings[]}`` where each marker has ``index, id, confidence,
    center, grid_coord, board_xy_mm, ellipse_outer, ellipse_inner,
    center_source, details`` (missing fields are ``None``). Never raises.
    """
    warnings: list[str] = []
    if isinstance(data, list):  # a bare marker array
        data = {"detected_markers": data}
    if not isinstance(data, dict):
        return _finish_norm(tile, run, None, None, [], [f"top level is {type(data).__name__}, not an object"])
    raw = data.get("detected_markers", data.get("markers"))
    if raw is None:
        warnings.append("no detected_markers array")
        raw = []
    elif not isinstance(raw, list):
        warnings.append("detected_markers is not an array")
        raw = []
    diag_markers = None
    diag = data.get("diagnostics")
    if isinstance(diag, dict) and isinstance(diag.get("markers"), list):
        # Diagnostics are positionally aligned 1:1 with detected_markers.
        if len(diag["markers"]) == len(raw):
            diag_markers = diag["markers"]
        else:
            warnings.append("diagnostics.markers length differs from detected_markers; ignored")

    markers = []
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            warnings.append(f"marker {i}: not an object, skipped")
            continue
        d = diag_markers[i] if diag_markers is not None and isinstance(diag_markers[i], dict) else {}
        outer = _ellipse(m.get("ellipse_outer")) or _ellipse(d.get("ellipse_outer"))
        inner = _ellipse(m.get("ellipse_inner")) or _ellipse(d.get("ellipse_inner"))
        center = _pair(m.get("center"))
        if center is None and outer is not None:
            center = [outer["cx"], outer["cy"]]
            warnings.append(f"marker {i}: center missing, using ellipse centre")
        if center is None:
            warnings.append(f"marker {i}: no usable centre, skipped")
            continue
        source = m.get("center_source", d.get("center_source"))
        details = _scalar_details(d)
        if "source" in d and isinstance(d["source"], str):
            details["source"] = d["source"]
        markers.append(
            {
                "index": i,
                "id": _int(m.get("id")),
                "confidence": _num(m.get("confidence")),
                "center": center,
                "grid_coord": _int_pair(m.get("grid_coord")),
                "board_xy_mm": _pair(m.get("board_xy_mm")),
                "ellipse_outer": outer,
                "ellipse_inner": inner,
                "center_source": source if isinstance(source, str) else None,
                "details": details,
            }
        )
    size = data.get("image_size")
    size = [_int(size[0]), _int(size[1])] if isinstance(size, list) and len(size) == 2 else None
    if size is not None and None in size:
        size = None
    frame = data.get("center_frame")
    return _finish_norm(tile, run, size, frame if isinstance(frame, str) else None, markers, warnings)


def _finish_norm(tile, run, size, frame, markers, warnings) -> dict:
    return {
        "tile": tile,
        "run": run,
        "image_size": size,
        "frame": frame,
        "markers": markers,
        "n_markers": len(markers),
        "n_decoded": sum(1 for m in markers if m["id"] is not None),
        "radius_hint": _radius_hint(markers),
        "warnings": warnings[:50],
    }


def _radius_hint(markers) -> float:
    """Circle radius for markers without ellipses: median outer semi-major
    axis if any marker has one, else 0.35 x median nearest-neighbour distance,
    else a fixed fallback."""
    radii = [m["ellipse_outer"]["a"] for m in markers if m["ellipse_outer"]]
    if radii:
        return float(statistics.median(radii))
    pts = [m["center"] for m in markers]
    if len(pts) >= 2:
        nn = []
        for i, p in enumerate(pts):
            nn.append(min(math.hypot(p[0] - q[0], p[1] - q[1]) for j, q in enumerate(pts) if j != i))
        med = statistics.median(nn)
        if med > 0:
            return 0.35 * med
    return DEFAULT_MARKER_RADIUS_PX


# --------------------------------------------------------------------------
# Run comparison
# --------------------------------------------------------------------------


def compare_markers(a: list[dict], b: list[dict], match_px=DEFAULT_MATCH_PX, pos_px=DEFAULT_POS_TOL_PX) -> dict:
    """Greedy nearest-pair matching of two normalised marker lists.

    ``pairs`` carry ``cls`` in ``same | id_differs | position_differs``;
    unmatched markers are listed in ``only_a`` / ``only_b`` (marker indices).
    """
    cand = []
    for i, ma in enumerate(a):
        for j, mb in enumerate(b):
            d = math.hypot(ma["center"][0] - mb["center"][0], ma["center"][1] - mb["center"][1])
            if d <= match_px:
                cand.append((d, i, j))
    cand.sort()
    used_a, used_b, pairs = set(), set(), []
    for d, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        if a[i]["id"] != b[j]["id"]:
            cls = "id_differs"
        elif d > pos_px:
            cls = "position_differs"
        else:
            cls = "same"
        pairs.append({"a": a[i]["index"], "b": b[j]["index"], "dist": d, "cls": cls})
    only_a = [m["index"] for i, m in enumerate(a) if i not in used_a]
    only_b = [m["index"] for j, m in enumerate(b) if j not in used_b]
    counts = {"same": 0, "id_differs": 0, "position_differs": 0, "only_a": len(only_a), "only_b": len(only_b)}
    for p in pairs:
        counts[p["cls"]] += 1
    return {"pairs": pairs, "only_a": only_a, "only_b": only_b, "counts": counts}


def differs(counts: dict) -> bool:
    return any(v for k, v in counts.items() if k != "same")


# --------------------------------------------------------------------------
# Feedback schema + atomic store
# --------------------------------------------------------------------------


class FeedbackError(ValueError):
    pass


def empty_feedback(dataset_name: str) -> dict:
    return {"schema": SCHEMA, "dataset": dataset_name, "updated_at": now_iso(), "general_remarks": [], "tiles": {}}


def _expect(cond, msg):
    if not cond:
        raise FeedbackError(msg)


def _only_keys(obj, allowed, where):
    _expect(isinstance(obj, dict), f"{where}: expected object")
    extra = set(obj) - set(allowed)
    _expect(not extra, f"{where}: unknown keys {sorted(extra)}")


def _text(v, where, limit=20000):
    _expect(isinstance(v, str) and len(v) <= limit, f"{where}: expected string (<= {limit} chars)")
    return v


def _opt_text(obj, key, where):
    if obj.get(key) is not None:
        _text(obj[key], f"{where}.{key}")


def _opt_id(obj, key, where):
    v = obj.get(key)
    _expect(v is None or (_int(v) is not None and _int(v) >= 0), f"{where}.{key}: expected non-negative integer or null")


def validate_tile_record(rec, where="tile") -> dict:
    _only_keys(rec, ("verdict", "remark", "reviewed_at", "runs_seen", "marker_flags", "pins"), where)
    _expect(rec.get("verdict") in (None, *VERDICTS), f"{where}.verdict: must be one of {VERDICTS} or null")
    _opt_text(rec, "remark", where)
    _opt_text(rec, "reviewed_at", where)
    seen = rec.get("runs_seen", [])
    _expect(isinstance(seen, list) and all(isinstance(r, str) and NAME_RE.match(r) for r in seen), f"{where}.runs_seen: list of run names")
    flags = rec.get("marker_flags", [])
    _expect(isinstance(flags, list) and len(flags) <= 5000, f"{where}.marker_flags: list")
    for n, f in enumerate(flags):
        w = f"{where}.marker_flags[{n}]"
        _only_keys(f, ("run", "kind", "marker", "correct_id", "remark", "at"), w)
        _expect(isinstance(f.get("run"), str) and NAME_RE.match(f["run"]), f"{w}.run: run name")
        _expect(f.get("kind") in FLAG_KINDS, f"{w}.kind: must be one of {FLAG_KINDS}")
        mk = f.get("marker")
        _only_keys(mk, ("index", "id", "center"), f"{w}.marker")
        _expect(_int(mk.get("index")) is not None, f"{w}.marker.index: integer")
        _opt_id(mk, "id", f"{w}.marker")
        _expect(_pair(mk.get("center")) is not None, f"{w}.marker.center: [x, y]")
        _opt_id(f, "correct_id", w)
        _opt_text(f, "remark", w)
        _opt_text(f, "at", w)
    pins = rec.get("pins", [])
    _expect(isinstance(pins, list) and len(pins) <= 5000, f"{where}.pins: list")
    uids = set()
    for n, p in enumerate(pins):
        w = f"{where}.pins[{n}]"
        _only_keys(p, ("uid", "type", "x", "y", "true_id", "remark", "text", "at"), w)
        _expect(isinstance(p.get("uid"), str) and 0 < len(p["uid"]) <= 64 and p["uid"] not in uids, f"{w}.uid: unique string")
        uids.add(p["uid"])
        _expect(p.get("type") in PIN_TYPES, f"{w}.type: must be one of {PIN_TYPES}")
        _expect(_num(p.get("x")) is not None and _num(p.get("y")) is not None, f"{w}: x/y numbers")
        _opt_id(p, "true_id", w)
        _opt_text(p, "remark", w)
        _opt_text(p, "text", w)
        _opt_text(p, "at", w)
        if p["type"] == "note":
            _expect(p.get("true_id") is None, f"{w}: notes carry text, not true_id")
        else:
            _expect(p.get("text") is None, f"{w}: missed_marker carries remark, not text")
    return rec


def validate_remarks(rem, where="general_remarks") -> list:
    _expect(isinstance(rem, list) and len(rem) <= 5000, f"{where}: list")
    for n, r in enumerate(rem):
        _only_keys(r, ("at", "text"), f"{where}[{n}]")
        _text(r.get("at"), f"{where}[{n}].at", 64)
        _text(r.get("text"), f"{where}[{n}].text")
    return rem


def validate_feedback(fb, tile_names=None) -> dict:
    _only_keys(fb, ("schema", "dataset", "updated_at", "general_remarks", "tiles"), "feedback")
    _expect(fb.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    _expect(isinstance(fb.get("dataset"), str), "dataset: string")
    validate_remarks(fb.get("general_remarks", []))
    tiles = fb.get("tiles", {})
    _expect(isinstance(tiles, dict), "tiles: object")
    for name, rec in tiles.items():
        _expect(tile_names is None or name in tile_names, f"unknown tile {name!r}")
        validate_tile_record(rec, f"tiles[{name}]")
    return fb


class FeedbackStore:
    """feedback.json with atomic replace (temp file + fsync + rename)."""

    def __init__(self, path: Path, dataset_name: str, tile_names):
        self.path = path
        self.tile_names = set(tile_names)
        self.lock = threading.Lock()
        if path.exists():
            try:
                self.data = validate_feedback(json.loads(path.read_text()), None)
            except (ValueError, OSError) as exc:
                raise SystemExit(f"{path} exists but is not a valid {SCHEMA} file ({exc}); refusing to overwrite it")
        else:
            self.data = empty_feedback(dataset_name)

    def _write(self, data: dict) -> None:
        data["updated_at"] = now_iso()
        fd, tmp = tempfile.mkstemp(prefix=".feedback.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def replace(self, fb: dict) -> dict:
        validate_feedback(fb, self.tile_names)
        with self.lock:
            self._write(fb)
            self.data = fb
            return fb

    def patch(self, tiles: dict | None, general_remarks) -> dict:
        _expect(tiles is None or isinstance(tiles, dict), "tiles: object")
        for name, rec in (tiles or {}).items():
            _expect(name in self.tile_names, f"unknown tile {name!r}")
            validate_tile_record(rec, f"tiles[{name}]")
        if general_remarks is not None:
            validate_remarks(general_remarks)
        with self.lock:
            new = json.loads(json.dumps(self.data))
            for name, rec in (tiles or {}).items():
                if rec.get("verdict") is None and not any(rec.get(k) for k in ("remark", "marker_flags", "pins")):
                    new["tiles"].pop(name, None)  # fully cleared tile: drop the record
                else:
                    new["tiles"][name] = rec
            if general_remarks is not None:
                new["general_remarks"] = general_remarks
            self._write(new)
            self.data = new
            return new


# --------------------------------------------------------------------------
# Output-directory privacy check
# --------------------------------------------------------------------------


def check_output_ignored(out_dir: Path) -> tuple[bool, str]:
    """(ok, message). ok when ``out_dir`` is outside any git work tree or is
    covered by a gitignore rule, so outputs can never be committed by accident."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        top = subprocess.run(
            ["git", "-C", str(out_dir), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return True, "git unavailable; could not verify that the output directory is ignored"
    if top.returncode != 0:
        return True, "output directory is outside any git repository"
    probe = out_dir.resolve() / "feedback.json"
    res = subprocess.run(
        ["git", "-C", top.stdout.strip(), "check-ignore", "-q", "--no-index", str(probe)], capture_output=True, timeout=20
    )
    if res.returncode == 0:
        return True, "output directory is git-ignored"
    return False, f"output directory {out_dir} is inside git repo {top.stdout.strip()} and NOT ignored"


# --------------------------------------------------------------------------
# Dataset, runs
# --------------------------------------------------------------------------


def png_size(path: Path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    return None


def image_size(path: Path):
    size = png_size(path) if path.suffix.lower() == ".png" else None
    if size:
        return size
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit(f"Pillow is required to read {path.name}; use PNG inputs or the repo .venv")
    with Image.open(path) as im:
        return im.size


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class Dataset:
    def __init__(self, data_dir: Path, split: int = 1, out_dir: Path | None = None):
        self.dir = Path(data_dir).resolve()
        self.name = self.dir.name
        self.split = split
        self.tiles_dir = (out_dir or self.dir) / "tiles"
        files = sorted((p for p in self.dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: natural_key(p.name))
        if not files:
            raise SystemExit(f"no images in {self.dir}")
        self.tiles: dict[str, dict] = {}
        for f in files:
            w, h = image_size(f)
            if split > 1:
                if w % split:
                    print(f"warning: {f.name}: width {w} not divisible by {split}, skipped", file=sys.stderr)
                    continue
                for i in range(split):
                    self._add(tile_name(f.stem, i), f, i, w // split, h, f.stem in SKIP_STRIPS)
            else:
                self._add(f.stem, f, None, w, h, False)
        self._lock = threading.Lock()

    def _add(self, name, src, index, w, h, eval_skip):
        if not NAME_RE.match(name):
            print(f"warning: tile name {name!r} unsupported, skipped", file=sys.stderr)
            return
        self.tiles[name] = {"name": name, "source": src.name, "index": index, "width": w, "height": h, "eval_skip": eval_skip}

    def image_path(self, name: str) -> Path:
        """Path of a file the browser (and the detector) can read for ``name``."""
        t = self.tiles[name]
        src = self.dir / t["source"]
        if t["index"] is None and src.suffix.lower() in BROWSER_NATIVE:
            return src
        cached = self.tiles_dir / f"{name}.png"
        with self._lock:
            if not cached.exists():
                self.tiles_dir.mkdir(parents=True, exist_ok=True)
                tmp = cached.with_suffix(".tmp.png")
                if t["index"] is None:
                    from PIL import Image

                    with Image.open(src) as im:
                        im.convert("L").save(tmp)
                else:
                    save_tile(src, t["index"], self.split, tmp)
                os.replace(tmp, cached)
        return cached


class Run:
    def __init__(self, name: str, directory: Path, manifest: dict, state: str = "done"):
        self.name = name
        self.dir = directory
        self.manifest = manifest  # shared with the writer thread, which updates `complete`
        self.status = {"state": state, "done": 0, "total": 0, "errors": []}
        self._cache: dict[str, tuple[float, dict]] = {}

    def tile_json_path(self, tile: str) -> Path | None:
        exact = self.dir / f"{tile}.json"
        if exact.is_file():
            return exact
        matches = [p for p in self.dir.glob(f"{tile}_*.json") if p.is_file()]
        return matches[0] if len(matches) == 1 else None

    def detection(self, tile: str) -> dict | None:
        """Normalised detection for a tile, or None when the run has no output for it."""
        path = self.tile_json_path(tile)
        if path is None:
            return None
        mtime = path.stat().st_mtime
        hit = self._cache.get(tile)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            norm = normalise_detection(None, tile, self.name)
            norm["warnings"] = [f"unreadable detection file: {exc}"]
        else:
            norm = normalise_detection(data, tile, self.name)
        self._cache[tile] = (mtime, norm)
        return norm


def detector_version(binary: Path) -> dict:
    info = {}
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=20)
        info["version"] = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["version"] = None
    try:
        g = subprocess.run(
            ["git", "-C", str(binary.resolve().parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20
        )
        info["git_commit"] = g.stdout.strip() if g.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


def split_user_args(raw) -> list[str]:
    args = shlex.split(raw) if isinstance(raw, str) else raw
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError("args must be a string or a list of strings")
    for a in args:
        if a.split("=", 1)[0] in ("--image", "--out"):
            raise ValueError(f"{a.split('=', 1)[0]} is set by the review tool")
    return args


class RunManager:
    def __init__(self, out_dir: Path, dataset: Dataset, allowed_binaries: dict[str, Path]):
        self.runs_dir = out_dir / "runs"
        self.dataset = dataset
        self.binaries = allowed_binaries  # label -> resolved path
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2))
        if self.runs_dir.is_dir():
            for d in sorted(self.runs_dir.iterdir()):
                mf = d / "manifest.json"
                if d.is_dir() and NAME_RE.match(d.name) and mf.is_file():
                    try:
                        manifest = json.loads(mf.read_text())
                    except ValueError:
                        continue
                    state = "done" if manifest.get("complete") else "incomplete"
                    self.runs[d.name] = Run(d.name, d, manifest, state)

    def import_dir(self, name: str, directory: Path) -> None:
        if not NAME_RE.match(name):
            raise SystemExit(f"invalid run name {name!r}")
        if not directory.is_dir():
            raise SystemExit(f"--run {name}: {directory} is not a directory")
        if name in self.runs:
            raise SystemExit(f"run name {name!r} already exists")
        mtime = datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc).astimezone().isoformat(timespec="seconds")
        self.runs[name] = Run(name, directory.resolve(), {"imported": True, "path": str(directory.resolve()), "timestamp": mtime})

    def start(self, name: str, binary: Path, args: list[str], wait: bool = False) -> Run:
        """Create (or resume) a run by invoking an allow-listed detector binary per tile."""
        if not NAME_RE.match(name):
            raise ValueError("invalid run name (letters, digits, . _ - only)")
        if binary.resolve() not in {p.resolve() for p in self.binaries.values()}:
            raise PermissionError("binary is not in the allow-list")
        with self.lock:
            existing = self.runs.get(name)
            directory = self.runs_dir / name
            if existing:
                same = existing.manifest.get("binary") == str(binary) and existing.manifest.get("args") == args
                if not same or existing.status["state"] == "running" or existing.manifest.get("imported"):
                    raise FileExistsError(f"run {name!r} already exists with a different configuration")
                if existing.status["state"] == "done":
                    return existing
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {
                "binary": str(binary),
                "args": args,
                "timestamp": now_iso(),
                "complete": False,
                "dataset": self.dataset.name,
                "split": self.dataset.split,
                **detector_version(binary),
            }
            self._write_manifest(directory, manifest)
            run = Run(name, directory, manifest, "running")
            run.status["total"] = len(self.dataset.tiles)
            self.runs[name] = run
        thread = threading.Thread(target=self._execute, args=(run, binary, args, manifest), daemon=True)
        thread.start()
        if wait:
            thread.join()
        return run

    @staticmethod
    def _write_manifest(directory: Path, manifest: dict) -> None:
        tmp = directory / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp, directory / "manifest.json")

    def _execute(self, run: Run, binary: Path, args: list[str], manifest: dict) -> None:
        def one(tile: str) -> None:
            out = run.dir / f"{tile}.json"
            if out.is_file() and out.stat().st_size > 0:
                return
            image = self.dataset.image_path(tile)
            tmp = run.dir / f"{tile}.json.tmp"
            cmd = [str(binary), "detect", "--image", str(image), "--out", str(tmp), *args]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if res.returncode != 0 or not tmp.is_file():
                raise RuntimeError(f"{tile}: exit {res.returncode}: {res.stderr.strip()[-300:]}")
            os.replace(tmp, out)

        futures = {self.pool.submit(one, t): t for t in self.dataset.tiles}
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - reported per tile
                run.status["errors"].append(str(exc)[:400])
            run.status["done"] += 1
        failed = bool(run.status["errors"])
        manifest["complete"] = not failed
        self._write_manifest(run.dir, manifest)
        run.status["state"] = "failed" if failed else "done"
        if failed:
            (run.dir / "errors.log").write_text("\n".join(run.status["errors"]) + "\n")


# --------------------------------------------------------------------------
# Application + HTTP layer
# --------------------------------------------------------------------------


class App:
    def __init__(self, dataset: Dataset, out_dir: Path, binaries: dict[str, Path]):
        self.dataset = dataset
        self.out_dir = out_dir
        self.store = FeedbackStore(out_dir / "feedback.json", dataset.name, dataset.tiles)
        self.runs = RunManager(out_dir, dataset, binaries)
        self._cmp_cache: dict = {}

    def compare(self, tile: str, a: str, b: str, match_px: float, pos_px: float) -> dict | None:
        da, db = self.runs.runs[a].detection(tile), self.runs.runs[b].detection(tile)
        if da is None or db is None:
            return None
        key = (tile, a, b, match_px, pos_px, id(da), id(db))
        if key not in self._cmp_cache:
            if len(self._cmp_cache) > 4000:
                self._cmp_cache.clear()
            self._cmp_cache[key] = compare_markers(da["markers"], db["markers"], match_px, pos_px)
        return self._cmp_cache[key]

    def state(self) -> dict:
        runs = []
        names = list(self.runs.runs)
        for name in names:
            run = self.runs.runs[name]
            counts = {}
            for t in self.dataset.tiles:
                det = run.detection(t)
                counts[t] = None if det is None else [det["n_markers"], det["n_decoded"]]
            runs.append({"name": name, "manifest": run.manifest, "status": run.status, "counts": counts})
        differing = []
        for t in self.dataset.tiles:
            flag = False
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    cmp = self.compare(t, a, b, DEFAULT_MATCH_PX, DEFAULT_POS_TOL_PX)
                    flag = flag or (cmp is not None and differs(cmp["counts"]))
            if flag:
                differing.append(t)
        return {
            "dataset": {"name": self.dataset.name, "split": self.dataset.split, "tiles": list(self.dataset.tiles.values())},
            "runs": runs,
            "differing_tiles": differing,
            "binaries": [{"label": k, "path": str(v)} for k, v in self.runs.binaries.items()],
            "defaults": {"match_px": DEFAULT_MATCH_PX, "pos_px": DEFAULT_POS_TOL_PX},
            "feedback": self.store.data,
        }


def app_built(app_dir: Path | None) -> bool:
    return app_dir is not None and (app_dir / "index.html").is_file()


def static_file(app_dir: Path, parts: list[str]) -> Path | None:
    """The built app's file for URL path ``parts``, or None. Never a path
    outside ``app_dir``, a directory, or a type the page does not load."""
    if not parts or any(p in ("", ".", "..") or "/" in p or "\\" in p or "\x00" in p for p in parts):
        return None
    root = app_dir.resolve()
    path = (root / Path(*parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in STATIC_TYPES:
        return None
    return path


def make_handler(app: App, port_holder: list, app_dir: Path | None = APP_DIST):
    """Request handler. ``app_dir`` is the built front end; while it has no
    index.html (not built yet) the legacy single-file page is served instead.
    Checked per request, so building the app needs no server restart."""
    legacy_html = (HERE / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ringgrid-review"

        def log_message(self, fmt, *args):  # quiet; errors are printed explicitly
            pass

        # -- helpers ---------------------------------------------------
        def _send(self, status, body: bytes, ctype: str, extra=None, csp=CSP):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", csp)
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200):
            self._send(status, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def _err(self, status, msg):
            self._json({"error": msg}, status)

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            allowed = {f"127.0.0.1:{port_holder[0]}", f"localhost:{port_holder[0]}", f"[::1]:{port_holder[0]}"}
            return host in allowed

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return origin.lower() in {f"http://127.0.0.1:{port_holder[0]}", f"http://localhost:{port_holder[0]}", f"http://[::1]:{port_holder[0]}"}

        def _read_json(self):
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise ValueError("Content-Length required")
            if n < 0 or n > MAX_BODY:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"null")

        def _tile(self, seg):
            return seg if seg in app.dataset.tiles else None

        def _run(self, seg):
            return app.runs.runs.get(seg)

        # -- routes ----------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return self._err(403, "bad Host header")
            url = urlsplit(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            try:
                if not parts:
                    if app_built(app_dir):
                        return self._send(200, (app_dir / "index.html").read_bytes(), STATIC_TYPES[".html"])
                    return self._send(200, legacy_html, STATIC_TYPES[".html"], csp=LEGACY_CSP)
                if parts[0] != "api":
                    path = static_file(app_dir, parts) if app_built(app_dir) else None
                    if path is None:
                        return self._err(404, "not found")
                    return self._send(200, path.read_bytes(), STATIC_TYPES[path.suffix.lower()])
                if parts == ["api", "state"]:
                    return self._json(app.state())
                if len(parts) == 4 and parts[:2] == ["api", "tile"] and parts[3] == "image":
                    tile = self._tile(parts[2])
                    if tile is None:
                        return self._err(404, "unknown tile")
                    path = app.dataset.image_path(tile)
                    ctype = BROWSER_NATIVE.get(path.suffix.lower(), "image/png")
                    return self._send(200, path.read_bytes(), ctype)
                if len(parts) == 5 and parts[:2] == ["api", "run"] and parts[3] == "tile":
                    run, tile = self._run(parts[2]), self._tile(parts[4])
                    if run is None or tile is None:
                        return self._err(404, "unknown run or tile")
                    det = run.detection(tile)
                    if det is None:
                        return self._err(404, "no detection for this tile in this run")
                    return self._json(det)
                if len(parts) == 3 and parts[:2] == ["api", "compare"]:
                    tile = self._tile(parts[2])
                    a, b = query.get("a", [""])[0], query.get("b", [""])[0]
                    if tile is None or a not in app.runs.runs or b not in app.runs.runs:
                        return self._err(404, "unknown tile or run")
                    try:
                        match = float(query.get("match", [DEFAULT_MATCH_PX])[0])
                        pos = float(query.get("pos", [DEFAULT_POS_TOL_PX])[0])
                    except ValueError:
                        return self._err(400, "match/pos must be numbers")
                    if not (math.isfinite(match) and math.isfinite(pos) and 0 < match < 1e4 and 0 <= pos < 1e4):
                        return self._err(400, "match/pos out of range")
                    cmp = app.compare(tile, a, b, match, pos)
                    return self._err(404, "missing detections") if cmp is None else self._json(cmp)
                return self._err(404, "not found")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                print(f"GET {self.path}: {exc!r}", file=sys.stderr)
                return self._err(500, "internal error")

        def do_POST(self):
            if not self._host_ok() or not self._origin_ok():
                return self._err(403, "bad Host/Origin")
            path = urlsplit(self.path).path
            try:
                body = self._read_json()
                if path == "/api/screen-target":
                    try:
                        return self._json(generate_screen_target(body))
                    except RuntimeError as exc:
                        return self._err(503, str(exc))
                if path == "/api/feedback":
                    if not isinstance(body, dict):
                        raise FeedbackError("body must be an object")
                    if "replace" in body:
                        _only_keys(body, ("replace",), "body")
                        fb = app.store.replace(body["replace"])
                    else:
                        _only_keys(body, ("tiles", "general_remarks"), "body")
                        fb = app.store.patch(body.get("tiles"), body.get("general_remarks"))
                    return self._json({"ok": True, "updated_at": fb["updated_at"]})
                if path == "/api/runs":
                    if not isinstance(body, dict):
                        raise ValueError("body must be an object")
                    binary = body.get("binary")
                    allowed = {str(p): p for p in app.runs.binaries.values()}
                    if binary not in allowed:
                        return self._err(403, "binary is not one of the server's allowed detectors")
                    args = split_user_args(body.get("args", ""))
                    run = app.runs.start(str(body.get("name", "")), allowed[binary], args)
                    return self._json({"ok": True, "run": run.name}, 202)
                return self._err(404, "not found")
            except FeedbackError as exc:
                return self._err(400, str(exc))
            except FileExistsError as exc:
                return self._err(409, str(exc))
            except PermissionError as exc:
                return self._err(403, str(exc))
            except (ValueError, UnicodeDecodeError) as exc:
                return self._err(400, str(exc))
            except Exception as exc:  # noqa: BLE001
                print(f"POST {self.path}: {exc!r}", file=sys.stderr)
                return self._err(500, "internal error")

    return Handler


def generate_screen_target(body):
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    _only_keys(body, ("template", "screen"), "body")
    if body.get("template") not in ("coded_hex", "plain_rect"):
        raise ValueError("unknown screen target template")
    options = body.get("screen")
    if not isinstance(options, dict):
        raise ValueError("screen must be an object")
    try:
        from ringgrid import TargetLayout
        render = TargetLayout.render_fitted_screen_target
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Build the RingGrid Python bindings in the Review server environment to generate targets.") from exc
    target = TargetLayout.default_hex() if body["template"] == "coded_hex" else TargetLayout.rect_24x24()
    bundle = render(target, options)
    return {
        "png_base64": base64.b64encode(bundle.png_bytes).decode("ascii"),
        "json_text": bundle.json_text,
        "metadata": bundle.metadata,
    }


def require_loopback(host: str) -> str:
    """Return ``host`` if it is a loopback literal; otherwise raise ValueError."""
    if host == "localhost":
        return "127.0.0.1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"refusing to bind {host!r}: only loopback addresses are allowed")
    if not ip.is_loopback:
        raise ValueError(f"refusing to bind {host!r}: only loopback addresses are allowed")
    return host


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765, app_dir: Path | None = APP_DIST) -> ThreadingHTTPServer:
    host = require_loopback(host)
    holder = [port]

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = __import__("socket").AF_INET6 if ":" in host else __import__("socket").AF_INET

    srv = Server((host, port), make_handler(app, holder, app_dir))
    holder[0] = srv.server_address[1]
    return srv


def default_binaries() -> dict[str, Path]:
    found = {}
    for profile in ("release", "debug"):
        p = REPO_ROOT / "target" / profile / "ringgrid-dev"
        if p.is_file() and os.access(p, os.X_OK):
            found[f"repo-{profile}"] = p.resolve()
    return found


def parse_args(argv: list[str]):
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1 :]
    else:
        tail = []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="directory of images")
    ap.add_argument("--split", type=int, default=1, help="split each image into N equal-width tiles (rtv3d strips: 6)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default tools/out/review/<dataset>)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=DIR", help="import a directory of detection JSONs")
    ap.add_argument(
        "--detect", action="append", default=[], metavar="NAME=BINARY", help="create a run with this detector binary; args after `--`"
    )
    ap.add_argument("--host", default="127.0.0.1", help="loopback address only")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--allow-unignored-out", action="store_true", help="do not refuse when --out is not git-ignored")
    args = ap.parse_args(argv)
    return args, tail


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args, tail = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.split < 1:
        raise SystemExit("--split must be >= 1")
    try:
        require_loopback(args.host)
    except ValueError as exc:
        raise SystemExit(str(exc))
    data = args.data.resolve()
    if not data.is_dir():
        raise SystemExit(f"--data {data} is not a directory")
    out = (args.out or REPO_ROOT / "tools" / "out" / "review" / data.name).resolve()
    ok, msg = check_output_ignored(out)
    if not ok:
        banner = "!" * 72
        print(f"{banner}\nPRIVACY WARNING: {msg}\nFeedback and tiles of private images could be committed.\n{banner}", file=sys.stderr)
        if not args.allow_unignored_out:
            raise SystemExit("refusing to start; use an ignored --out (e.g. under tools/out/ or data/) or --allow-unignored-out")
    else:
        print(f"output: {out} ({msg})")

    dataset = Dataset(data, args.split, out)
    binaries = default_binaries()
    detect_jobs = []
    for spec in args.detect:
        name, _, path = spec.partition("=")
        p = Path(path).resolve()
        if not path or not p.is_file() or not os.access(p, os.X_OK):
            raise SystemExit(f"--detect {spec}: binary not found or not executable")
        binaries.pop(next((k for k, v in binaries.items() if v == p and k.startswith("repo-")), None), None)
        binaries[name] = p
        detect_jobs.append((name, p))
    app = App(dataset, out, binaries)
    for spec in args.run:
        name, _, path = spec.partition("=")
        app.runs.import_dir(name, Path(path))
    for name, p in detect_jobs:
        app.runs.start(name, p, tail)

    srv = make_server(app, args.host, args.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    if not app_built(APP_DIST):
        print(f"note: {BUILD_HINT}", file=sys.stderr)
    print(f"{len(dataset.tiles)} tiles, {len(app.runs.runs)} runs; serving {url} (Ctrl-C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
