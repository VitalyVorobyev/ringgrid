#!/usr/bin/env python3
"""Generate the documentation performance-page data (``data.json``).

This is the *performance test* behind the GitHub Pages performance dashboard
(``.github/pages/performance/index.html``). For a small set of deliberately
**varied** scenes — a real capture, a sparse and a dense coded hex board, and a
plain rect board — it measures:

  * **Detection result + overlay** — one ``ringgrid detect`` run per scene
    supplies the marker count and a rendered detection overlay (so the page
    shows *what was actually detected*, not just the input).
  * **Timing** — the ``ringgrid bench`` subcommand: single-pass detection with
    warmup + repeats, median per stage (proposal / fit+decode / finalize) and
    end-to-end wall-clock.
  * **Accuracy** — the synthetic reference benchmark
    (``tools/run_reference_benchmark.sh``): precision, recall and subpixel
    centre error, for the ``none`` and ``projective_center`` refinement modes.

The result is a stable, committed snapshot (``--out``). CI only deploys it; it
is regenerated locally on a quiet machine when the algorithm changes:

    .venv/bin/python tools/gen_pages_perf.py

Run from the repository root.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / ".github/pages/performance/data.json"
DEFAULT_IMG_DIR = REPO_ROOT / ".github/pages/performance/img"
RELEASE_BIN = REPO_ROOT / "target/release/ringgrid"
REFERENCE_OUT_DIR = REPO_ROOT / "tools/out/reference_benchmark_post_pipeline"
SCENES_DIR = REPO_ROOT / "tools/out/perf_scenes"
PYTHON = sys.executable


# ── scene catalogue ──────────────────────────────────────────────────────────
# Each scene is either a fixed real capture or a synthetic board generated with
# explicit geometry, chosen so the set varies in resolution, lattice size, and
# target type (coded hex vs plain rect). Marker counts and overlays come from a
# full `detect` run; per-stage timing comes from `bench`.
SCENES = [
    {
        "key": "real_hex",
        "label": "Coded hex — real capture",
        "kind": "real",
        "image": "testdata/target_3_split_00.png",
        "target": None,
        "marker_diameter": None,
        "gen": None,
        "note": "Dense hex lattice photographed in the lab; decoded to absolute IDs.",
    },
    {
        "key": "hex_sparse",
        "label": "Coded hex — sparse board",
        "kind": "synthetic",
        "marker_diameter": None,
        "gen": ("hex", dict(img_w=900, img_h=700, board_mm=120.0, pitch_mm=10.0,
                            seed=3, blur=0.8)),
        "note": "45-cell hex at 0.63 MP — sparse lattice, larger markers.",
    },
    {
        "key": "hex_dense",
        "label": "Coded hex — dense board",
        "kind": "synthetic",
        "marker_diameter": 32.0,
        "gen": ("hex", dict(img_w=1280, img_h=960, board_mm=200.0, pitch_mm=8.0,
                            seed=42, blur=0.8)),
        "note": "203-cell hex at 1.23 MP — the standard reference density.",
    },
    {
        "key": "rect_plain",
        "label": "Plain rect — origin-anchored",
        "kind": "synthetic",
        "marker_diameter": None,
        "gen": ("rect", dict(img_w=1000, img_h=1000, rows=12, cols=12,
                             seed=5, blur=0.8)),
        "note": "144-cell plain rect with origin dots — grid-labeled, anchored to "
                "absolute mm (no per-marker IDs).",
    },
]


# ── shell helpers ────────────────────────────────────────────────────────────
def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kwargs)


def capture(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                             capture_output=True, text=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_sha() -> str:
    return capture(["git", "rev-parse", "--short", "HEAD"]) or "unknown"


def rustc_version() -> str:
    raw = capture(["rustc", "--version"])  # "rustc 1.88.0 (abc 2025-...)"
    return raw.split()[1] if raw else "unknown"


def cpu_name() -> str:
    if sys.platform == "darwin":
        name = capture(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown CPU"


def today() -> str:
    return datetime.date.today().isoformat()


# ── scene generation ─────────────────────────────────────────────────────────
def generate_scene(scene: dict, reuse: bool) -> tuple[Path, Path | None]:
    """Return (image_path, target_spec_path) for a scene, generating if needed."""
    if scene["gen"] is None:
        return REPO_ROOT / scene["image"], None

    kind, params = scene["gen"]
    out_dir = SCENES_DIR / scene["key"]
    image = out_dir / "img_0000.png"
    target = out_dir / ("board_spec.json" if kind == "hex" else "target_spec.json")
    if reuse and image.exists() and target.exists():
        return image, target

    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "hex":
        cmd = [PYTHON, "tools/gen_synth.py", "--out_dir", str(out_dir),
               "--n_images", "1",
               "--img_w", str(params["img_w"]), "--img_h", str(params["img_h"]),
               "--board_mm", str(params["board_mm"]),
               "--pitch_mm", str(params["pitch_mm"]),
               "--seed", str(params["seed"]), "--blur_px", str(params["blur"])]
    else:  # rect
        cmd = [PYTHON, "tools/gen_synth_rect.py", "--out_dir", str(out_dir),
               "--n_images", "1",
               "--img_w", str(params["img_w"]), "--img_h", str(params["img_h"]),
               "--rows", str(params["rows"]), "--cols", str(params["cols"]),
               "--seed", str(params["seed"]), "--blur_px", str(params["blur"])]
    run(cmd, env={"MPLBACKEND": "Agg", **_env()})
    return image, target


def detect_scene(binary: Path, image: Path, target: Path | None,
                 marker_diameter: float | None, out_json: Path) -> dict:
    cmd = [str(binary), "detect", "--image", str(image), "--out", str(out_json)]
    if target is not None:
        cmd += ["--target", str(target)]
    if marker_diameter is not None:
        cmd += ["--marker-diameter", str(marker_diameter)]
    run(cmd, env={"RUST_LOG": "warn", **_env()})
    return json.loads(out_json.read_text())


def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Green (confident) → amber (weak) ramp."""
    conf = max(0.0, min(1.0, conf))
    if conf >= 0.5:
        return (34, 197, 94)   # emerald
    return (245, 158, 11)      # amber


WEB_MAX_DIM = 900  # keep the page light: cap preview/overlay long edge (px)


def _save_web(img: Image.Image, out_path: Path) -> None:
    """Downscale to a web-friendly size and save a compact JPEG.

    JPEG (not PNG): these dense synthetic board renders are photo-like and
    compress poorly as PNG; JPEG keeps the whole page-image set to ~1 MB.
    """
    w, h = img.size
    scale = WEB_MAX_DIM / max(w, h)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, quality=85, optimize=True, progressive=True)


def draw_overlay(image_path: Path, det: dict, out_path: Path) -> None:
    """Render fitted outer ellipses + centers over the input image."""
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    for m in det.get("detected_markers", []):
        conf = float(m.get("confidence") or 1.0)
        color = _confidence_color(conf)
        e = m.get("ellipse_outer")
        if e:
            cx, cy, a, b, ang = e["cx"], e["cy"], e["a"], e["b"], e["angle"]
            pts = []
            for i in range(48):
                t = 2.0 * math.pi * i / 48
                ct, st = math.cos(t), math.sin(t)
                x = cx + a * ct * math.cos(ang) - b * st * math.sin(ang)
                y = cy + a * ct * math.sin(ang) + b * st * math.cos(ang)
                pts.append((x, y))
            draw.line(pts + [pts[0]], fill=color, width=2)
        c = m.get("center")
        if c:
            draw.ellipse([c[0] - 2, c[1] - 2, c[0] + 2, c[1] + 2],
                         fill=(239, 68, 68))
    _save_web(base, out_path)


def _env() -> dict:
    import os
    return dict(os.environ)


# ── timing (ringgrid bench) ──────────────────────────────────────────────────
def bench_scene(binary: Path, image: Path, target: Path | None,
                marker_diameter: float | None, repeats: int) -> dict:
    out_path = REPO_ROOT / "tools/out/_pages_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "bench", "--out", str(out_path), "--repeats", str(repeats),
           "--image", str(image)]
    if target is not None:
        cmd += ["--target", str(target)]
    if marker_diameter is not None:
        cmd += ["--marker-diameter", str(marker_diameter)]
    run(cmd, env={"RUST_LOG": "warn", **_env()})
    return json.loads(out_path.read_text())["images"][0]


# ── accuracy (reference benchmark) ───────────────────────────────────────────
def run_reference_benchmark(reuse: bool) -> dict | None:
    summary_path = REFERENCE_OUT_DIR / "summary.json"
    if not reuse:
        script = REPO_ROOT / "tools/run_reference_benchmark.sh"
        try:
            run(["bash", str(script)])
        except subprocess.CalledProcessError as exc:
            print(f"  ! reference benchmark failed ({exc}); accuracy omitted")
            return None
    if not summary_path.exists():
        print(f"  ! {summary_path} not found; accuracy omitted")
        return None
    return json.loads(summary_path.read_text())


def accuracy_sets(summary: dict) -> list[dict]:
    modes = summary.get("modes", {})
    labels = [
        ("none__none", "No refinement"),
        ("projective_center__none", "Projective centre"),
    ]
    sets = []
    for key, label in labels:
        m = modes.get(key)
        if not m:
            continue
        sets.append({
            "label": label,
            "precision": m.get("avg_precision"),
            "recall": m.get("avg_recall"),
            "center_mean_px": m.get("avg_center_mean_px"),
            "h_vs_gt_mean_px": m.get("avg_h_vs_gt_mean_px"),
            "h_vs_gt_p95_px": m.get("avg_h_vs_gt_p95_px"),
        })
    return sets


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--img-out-dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--repeats", type=int, default=15,
                    help="Timed repeats per image (median reported).")
    ap.add_argument("--reuse-scenes", action="store_true",
                    help="Reuse already-generated synthetic scene images.")
    ap.add_argument("--reuse-summary", action="store_true",
                    help="Reuse an existing reference-benchmark summary.json.")
    ap.add_argument("--skip-accuracy", action="store_true",
                    help="Skip the accuracy panel entirely (timing only).")
    args = ap.parse_args()

    print("[1/4] Building release CLI")
    run(["cargo", "build", "--release", "-p", "ringgrid-cli"])
    if not RELEASE_BIN.exists():
        print(f"error: {RELEASE_BIN} not found after build", file=sys.stderr)
        return 1

    print("[2/4] Accuracy — synthetic reference benchmark")
    summary = None if args.skip_accuracy else run_reference_benchmark(args.reuse_summary)

    print("[3/4] Scenes — generate, detect, overlay, time")
    args.img_out_dir.mkdir(parents=True, exist_ok=True)
    det_dir = SCENES_DIR / "_detect"
    det_dir.mkdir(parents=True, exist_ok=True)

    images, frames = [], []
    for scene in SCENES:
        print(f"  · {scene['key']}")
        image, target = generate_scene(scene, args.reuse_scenes)
        if not image.exists():
            print(f"  ! {image} missing; skipping {scene['key']}")
            continue
        det = detect_scene(RELEASE_BIN, image, target, scene["marker_diameter"],
                           det_dir / f"{scene['key']}.json")
        bench = bench_scene(RELEASE_BIN, image, target, scene["marker_diameter"],
                            args.repeats)

        # Input preview + detection overlay, both downscaled into the img dir.
        preview = f"{scene['key']}_input.jpg"
        overlay = f"{scene['key']}_overlay.jpg"
        _save_web(Image.open(image).convert("RGB"), args.img_out_dir / preview)
        draw_overlay(image, det, args.img_out_dir / overlay)

        w, h = det.get("image_size", [bench["width"], bench["height"]])
        markers = det.get("detected_markers", [])
        n_id = sum(1 for m in markers if m.get("id") is not None)
        entry = {
            "label": scene["label"],
            "kind": scene["kind"],
            "file": Path(scene["image"]).name if scene["gen"] is None
                    else f"{scene['key']}/img_0000.png",
            "width": w,
            "height": h,
            "markers": len(markers),
            "decoded": n_id,
            "board_frame": det.get("board_frame"),
            "proposal_ms": bench["proposal_ms"],
            "fit_decode_ms": bench["fit_decode_ms"],
            "finalize_ms": bench["finalize_ms"],
            "img": f"./img/{preview}",
            "overlay": f"./img/{overlay}",
            "note": scene["note"],
        }
        images.append(entry)
        frames.append({"file": entry["file"], "note": scene["note"],
                       "ms": bench["total_ms"]})

    print("[4/4] Assembling data.json")
    data = {
        "meta": {
            "cpu": cpu_name(),
            "rustc": rustc_version(),
            "git_sha": git_sha(),
            "generated": today(),
            "repeats": args.repeats,
            "source": "tools/gen_pages_perf.py",
        },
        "images": images,
        "end_to_end": {
            "desc": "End-to-end single-pass latency (median of "
                    f"{args.repeats} repeats) across the scene set.",
            "frames": frames,
        },
    }

    if summary is not None:
        sets = accuracy_sets(summary)
        if sets:
            data["accuracy"] = {
                "desc": "Synthetic reference set (3 hex boards, <1 px blur) scored "
                        "against generated ground truth. Centre error is the mean "
                        "per-marker distance to the true centre; lower is sharper. "
                        "Projective-centre refinement removes the perspective bias "
                        "of the raw ellipse centre.",
                "sets": sets,
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    try:
        shown = args.out.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.out
    print(f"\nWrote {shown} "
          f"({len(images)} scenes, accuracy={'yes' if 'accuracy' in data else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
