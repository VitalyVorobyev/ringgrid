#!/usr/bin/env python3
"""Coverage evaluation on the private IPGS precision-plate dataset.

The plate is a 58x58 plain rect lattice (pitch 2 mm, rings 0.8 / 0.4 mm outer
/ inner radius) with an origin-dot triad; ``Cam1_*.png`` / ``Cam2_*.png`` are
2048x1536 views from frontal to strongly oblique. The data is private and
lives outside the repo; when the directory is absent the script says so and
exits 0 (like ``run_rtv3d_eval.py``).

Per image it

1. runs the configuration under test with the published CLI
   (``ringgrid detect``: adaptive multi-scale unless ``--marker-diameter`` is
   in the detect args), timing it;
2. builds a *reference* — the expected set must not depend on the
   configuration under test, and on the oblique views that configuration
   labels too few rings to fit anything:

   - a relaxed single-pass probe (``ringgrid-dev detect``: wide scale prior,
     ``min_confidence = 0``; cached in ``<out>/reference``) supplies labelled
     ring centres for a board-to-image fit; the reference is whichever of probe
     and run-under-test has more labelled markers;
   - its lattice basis is checked against the rings' shapes: a labelling in a
     sheared basis of the same point lattice fits a homography just as tightly
     but predicts the wrong ellipses, so the unimodular relabelling whose
     orientation-preserving model best predicts the detected outer ellipses
     is applied first;
   - its labels are re-anchored by the *image*, not by the detector: every
     placement of the target's origin-dot triad on the lattice (lattice
     rotation x translation) is projected through the fit, and the placement
     whose three dot positions are all dark is taken when it is unambiguous.
     A detector anchor that disagrees shows up as wrong-label suspects;

3. scores the run with ``score_coverage.py`` against that reference and
   writes a per-image table plus per-camera totals for the images processed
   (``--overlays`` also draws expected vs detected rings).

The generated target is ringgrid's ``58x58`` plain rect layout with automatic
origin dots. Its triad sits midway in both axes, while the plate's sits 28/30
columns from the edges on one axis: absolute labels can differ from the
plate's reference file by one column, and one plate column lies off the
layout. Every view here shows the plate's central ~16x12 cells, so neither
affects the scores.

Usage (from the repo root):
    cargo build --release -p ringgrid-cli --bin ringgrid-dev
    cargo build --release -p ringgrid --features cli --bin ringgrid
    .venv/bin/python tools/run_ipgs_eval.py \\
        --data ../calibration-rs/privatedata/ipgs-ring --label baseline \\
        [--detect-args "--refine photometric"] [--jobs 1] [--overlays]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from gen_synth_target import origin_dots_mm  # noqa: E402
from score_coverage import (  # noqa: E402
    SHAPE_ERR_MAX,
    Lattice,
    Model,
    fit_models,
    labelled_markers,
    ring_shape_errors,
    score_coverage,
    summary_line,
)
from score_detect_target import ROTATIONS  # noqa: E402
from target_json import load_target_spec  # noqa: E402

DEFAULT_DATA = REPO.parent / "calibration-rs" / "privatedata" / "ipgs-ring"
DEFAULT_OUT = REPO / "tools" / "out" / "ipgs_eval"

# Plate geometry (Ref127x127_2mm_Code0x000000.ref): 58x58 rings at 2 mm pitch,
# outer / inner radius 0.8 / 0.4 mm, three Ø0.4 mm origin dots in the lattice
# gaps around the centre.
PLATE = {
    "pitch_mm": 2.0,
    "rows": 58,
    "cols": 58,
    "outer_radius_mm": 0.8,
    "inner_radius_mm": 0.4,
}
# Relaxed reference probe: one wide scale tier, no confidence floor.
REFERENCE_ARGS = [
    "--marker-diameter-min", "50",
    "--marker-diameter-max", "170",
    "--refine", "photometric",
]
MIN_FIT = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="IPGS image directory")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--label", default="baseline", help="name of the run under test")
    p.add_argument("--detect-args", default="--refine photometric",
                   help="extra arguments for `ringgrid detect` (quoted string)")
    p.add_argument("--target", type=Path, help="target spec (default: generated for the plate)")
    p.add_argument("--images", nargs="+", default=["Cam*.png"], help="image globs inside --data")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel detections (>1 inflates the recorded runtimes)")
    p.add_argument("--reuse", action="store_true",
                   help="reuse existing detection / reference JSONs instead of re-running")
    p.add_argument("--overlays", action="store_true", help="draw expected vs detected rings")
    p.add_argument("--bin-dir", type=Path, default=REPO / "target" / "release")
    return p.parse_args()


# ── Detection runs ───────────────────────────────────────────────────────


def ensure_target(args, out: Path) -> Path:
    if args.target:
        return args.target
    target_dir = out / "target"
    spec = target_dir / "target_spec.json"
    if not spec.exists():
        cmd = [
            str(args.bin_dir / "ringgrid-dev"), "gen-target", "rect",
            "--pitch_mm", str(PLATE["pitch_mm"]),
            "--rows", str(PLATE["rows"]), "--cols", str(PLATE["cols"]),
            "--marker_outer_radius_mm", str(PLATE["outer_radius_mm"]),
            "--marker_inner_radius_mm", str(PLATE["inner_radius_mm"]),
            "--dots", "--name", "ipgs_58x58_2mm",
            "--out_dir", str(target_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return spec


def run_cmd(cmd: list[str], out_json: Path, reuse: bool) -> float | None:
    """Run a detection command; returns wall time (s), None when reused."""
    if reuse and out_json.exists():
        return None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr[-400:]}")
    return dt


def detect_under_test(args, image: Path, target: Path, out_json: Path) -> float | None:
    cmd = [
        str(args.bin_dir / "ringgrid"), "detect",
        "--image", str(image), "--target", str(target), "--out", str(out_json),
        *shlex.split(args.detect_args),
    ]
    return run_cmd(cmd, out_json, args.reuse)


def detect_reference_probe(args, image: Path, target: Path, out_json: Path, cfg: Path) -> None:
    cmd = [
        str(args.bin_dir / "ringgrid-dev"), "detect",
        "--image", str(image), "--target", str(target), "--out", str(out_json),
        "--config", str(cfg), *REFERENCE_ARGS,
    ]
    run_cmd(cmd, out_json, True)


# ── Image-based anchoring of the reference ───────────────────────────────


def load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def disk_mean(img: np.ndarray, centre, radius: float) -> float:
    """Mean intensity over a small disk (nearest-pixel sampling)."""
    h, w = img.shape
    r = max(radius, 1.0)
    offs = np.arange(-int(np.ceil(r)), int(np.ceil(r)) + 1)
    dx, dy = np.meshgrid(offs, offs)
    keep = dx * dx + dy * dy <= r * r
    xs = np.rint(centre[0] + dx[keep]).astype(int)
    ys = np.rint(centre[1] + dy[keep]).astype(int)
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= w or ys.max() >= h:
        return float("nan")
    return float(img[ys, xs].mean())


def anchor_by_dots(img, model, lattice: Lattice, spec: dict):
    """Find the triad placement whose three dot positions are dark.

    Returns ``(rotation, translation, info)`` mapping absolute board
    coordinates to the reference frame (``ref = R·abs + t``), or
    ``(None, None, info)`` when no placement is unambiguous.
    """
    fid = spec.get("fiducials")
    if not fid:
        return None, None, {"reason": "target has no origin dots"}
    dots_abs = np.asarray(origin_dots_mm(spec["lattice"]), dtype=float)
    # Absolute board mm -> fractional lattice coordinates.
    uvw = np.hstack([dots_abs, np.ones((3, 1))]) @ lattice.affine_inv.T
    dots_uv = uvw[:, :2]
    pitch_mm = float(np.linalg.norm(lattice.xy_mm([[1, 0]]) - lattice.xy_mm([[0, 0]])))
    h, w = img.shape
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
    corner_uv, _ = lattice.nearest_coord(model.unproject(corners))
    lo, hi = corner_uv.min(axis=0) - 2, corner_uv.max(axis=0) + 2

    hyps = []
    for rot in ROTATIONS[lattice.kind]:
        r = np.asarray(rot, dtype=float)
        rotated = dots_uv @ r.T
        for tu in range(lo[0], hi[0] + 1):
            for tv in range(lo[1], hi[1] + 1):
                uv = rotated + np.array([tu, tv])
                xy = lattice.xy_mm(uv)
                px = model.project(xy)
                if not np.all(np.isfinite(px)):
                    continue
                # Sample radius: a third of the dot's projected radius along
                # its most foreshortened direction (local scale), so the disk
                # stays on the dot even when the printed dot is undersized.
                scale = np.minimum(
                    np.linalg.norm(model.project(xy + [pitch_mm, 0.0]) - px, axis=1),
                    np.linalg.norm(model.project(xy + [0.0, pitch_mm]) - px, axis=1),
                ) / pitch_mm
                rad = fid["dot_radius_mm"] * scale / 3.0
                vals = [disk_mean(img, p, rr) for p, rr in zip(px, rad)]
                if not np.all(np.isfinite(vals)):
                    continue
                hyps.append((max(vals), rot, (tu, tv), px))
    if not hyps:
        return None, None, {"reason": "no triad placement inside the image"}
    paper = float(np.median([hh[0] for hh in hyps]))
    hyps.sort(key=lambda hh: hh[0])
    best, second = hyps[0], (hyps[1] if len(hyps) > 1 else None)
    info = {
        "paper_level": paper,
        "best_darkest_dot_max": best[0],
        "second_darkest_dot_max": second[0] if second else None,
        "dots_px": best[3].tolist(),
    }
    # All three dots dark, and the runner-up placement clearly not.
    if best[0] > 0.5 * paper or (second and second[0] < 0.65 * paper):
        info["reason"] = "ambiguous or invisible origin dots"
        return None, None, info
    return best[1], best[2], info


def relabel(det: dict, matrix, translation=(0, 0), lattice: Lattice | None = None,
            frame: str | None = None) -> dict:
    """Copy of a detection with every label mapped to ``M·c + t``."""
    m_int = np.asarray(matrix, dtype=int)
    out = dict(det)
    markers = []
    for m in det.get("detected_markers", []):
        m = dict(m)
        gc = m.get("grid_coord")
        if gc is not None:
            c = m_int @ np.asarray(gc) + np.asarray(translation)
            m["grid_coord"] = [int(c[0]), int(c[1])]
            if lattice is not None:
                m["board_xy_mm"] = lattice.xy_mm([c]).ravel().tolist()
        markers.append(m)
    out["detected_markers"] = markers
    if frame is not None:
        out["board_frame"] = frame
    return out


def unimodular_candidates(bound: int = 2):
    """Integer 2x2 matrices with entries in [-bound, bound] and det = ±1."""
    rng = range(-bound, bound + 1)
    for a in rng:
        for b in rng:
            for c in rng:
                for d in rng:
                    if abs(a * d - b * c) == 1:
                        yield ((a, b), (c, d))


def orientation_preserving(model: Model, xy_mm) -> bool:
    """Whether the model's Jacobian at a board point has a positive determinant."""
    return bool(np.linalg.det(model.jacobian(xy_mm)) > 0)


def fix_lattice_basis(det: dict, fit: dict, lattice: Lattice, image_size):
    """Undo a sheared labelling: the lattice basis that predicts the rings' shapes.

    A labelling in another basis of the same point lattice (``c' = A c``,
    ``A`` unimodular) is fitted by a homography exactly as tightly as the
    board's own, but projects the wrong ring shapes. Relabelling by ``A``
    needs no refit: ``H' = H · T · A^-1 · T^-1`` with ``T`` the coords -> mm
    map. Ring shapes cannot tell a basis from its mirror image, so only bases
    under which the board images orientation-preserved (board frame y-down
    like the image, seen from the front) are considered; the origin dots then
    pick the rotation. Returns ``(A, model, median shape error)``.
    """
    labelled = labelled_markers(det)
    t = np.vstack([lattice.affine.T, [0.0, 0.0, 1.0]])
    best = None
    for a in unimodular_candidates():
        a_h = np.eye(3)
        a_h[:2, :2] = np.linalg.inv(np.asarray(a, dtype=float))
        h_new = fit["h"].h @ t @ a_h @ np.linalg.inv(t)
        model = Model(h_new, 0.0, image_size)
        moved = [(i, tuple(np.asarray(a) @ np.asarray(c)), p) for i, c, p in labelled]
        centre_mm = lattice.xy_mm([c for _, c, _ in moved]).mean(axis=0)
        if not orientation_preserving(model, centre_mm):
            continue
        err = float(np.median(ring_shape_errors(model, lattice, det, moved)))
        if best is None or err < best[2]:
            best = (a, model, err)
    return best


def build_reference(image: Path, candidates: list[dict], lattice: Lattice, spec: dict):
    """Reference labelling for one image (see the module docstring).

    Takes the run with the most labelled markers, fixes its lattice basis by
    the ring shapes, and anchors it by the origin dots in the image.
    """
    det = max(candidates, key=lambda d: len(labelled_markers(d)))
    image_size = tuple(det["image_size"])
    info = {"source": det.get("_source"), "n_labelled": len(labelled_markers(det))}
    fit, reason = fit_models(det, lattice, image_size, 3.0, MIN_FIT)
    if fit is None:
        return None, {**info, "reason": reason}
    basis, model, shape_err = fix_lattice_basis(det, fit, lattice, image_size)
    info["basis"], info["ring_shape_err_median"] = basis, shape_err
    if shape_err > SHAPE_ERR_MAX:
        return None, {**info, "reason": "no lattice basis predicts the ring shapes"}
    det = relabel(det, basis)
    rot, t, dots = anchor_by_dots(load_gray(image), model, lattice, spec)
    info.update(dots)
    if rot is None:
        info["anchored"] = False
        return det, info
    info["anchored"] = True
    info["rotation"], info["translation"] = rot, list(t)
    # ref = R·abs + t  =>  abs = R^-1·ref - R^-1·t
    r_inv = np.rint(np.linalg.inv(np.asarray(rot, dtype=float))).astype(int)
    return relabel(det, r_inv, -(r_inv @ np.asarray(t)), lattice, "absolute"), info


# ── Overlays ─────────────────────────────────────────────────────────────


def draw_overlay(image: Path, det: dict, cov: dict, out_png: Path) -> None:
    """Expected rings (green: detected, orange: missing), detections (red),
    wrong-label suspects (magenta boxes)."""
    from PIL import Image, ImageDraw

    im = Image.open(image).convert("RGB")
    dr = ImageDraw.Draw(im)
    missing = {tuple(c) for c in cov.get("missing", [])}
    for e in cov.get("expected", []):
        x, y = e["center"]
        col = (255, 150, 0) if tuple(e["cell"]) in missing else (0, 200, 0)
        dr.ellipse([x - 16, y - 16, x + 16, y + 16], outline=col, width=6)
    for m in det.get("detected_markers", []):
        x, y = m["center"]
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 0, 0))
    for s in cov.get("wrong_label_suspects", []):
        x, y = s["center"]
        dr.rectangle([x - 28, y - 28, x + 28, y + 28], outline=(255, 0, 255), width=6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.resize((im.width // 2, im.height // 2)).save(out_png)


# ── Main ─────────────────────────────────────────────────────────────────


def fmt(v, spec="{:.3f}"):
    return "-" if v is None else spec.format(v)


def main() -> int:
    args = parse_args()
    if not args.data.is_dir():
        print(f"IPGS data not found at {args.data}; skipping (private dataset).")
        return 0
    for b in ("ringgrid", "ringgrid-dev"):
        if not (args.bin_dir / b).exists():
            raise SystemExit(
                f"{args.bin_dir / b} missing; build with `cargo build --release -p ringgrid-cli "
                "--bin ringgrid-dev` and `cargo build --release -p ringgrid --features cli --bin ringgrid`"
            )
    images = sorted({p for pattern in args.images for p in args.data.glob(pattern)})
    if not images:
        raise SystemExit(f"no images matching {args.images} in {args.data}")

    out = args.out
    run_dir = out / args.label
    ref_dir = out / "reference"
    run_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    target = ensure_target(args, out)
    spec = load_target_spec(target)
    lattice = Lattice(spec)
    ref_cfg = ref_dir / "probe_config.json"
    ref_cfg.write_text(json.dumps({"min_confidence": 0.0}))

    def work(image: Path):
        stem = image.stem
        runtime = detect_under_test(args, image, target, run_dir / f"{stem}.json")
        detect_reference_probe(args, image, target, ref_dir / f"{stem}.probe.json", ref_cfg)
        return stem, runtime

    timings_path = run_dir / "runtimes.json"
    runtimes = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for stem, rt in pool.map(work, images):
            if rt is not None:
                runtimes[stem] = rt
            print(f"  detected {stem}" + (f" in {rt:.2f}s" if rt is not None else " (reused)"))
    timings_path.write_text(json.dumps(runtimes, indent=2))

    rows = []
    for image in images:
        stem = image.stem
        det = json.loads((run_dir / f"{stem}.json").read_text())
        probe = json.loads((ref_dir / f"{stem}.probe.json").read_text())
        probe["_source"], det_c = "probe", dict(det, _source="run")
        ref, ref_info = build_reference(image, [probe, det_c], lattice, spec)
        (ref_dir / f"{stem}.reference_info.json").write_text(json.dumps(ref_info, indent=2, default=str))
        if ref is None:
            cov = score_coverage(det, spec, None, min_fit=MIN_FIT)
        else:
            ref.pop("_source", None)
            (ref_dir / f"{stem}.reference.json").write_text(json.dumps(ref))
            cov = score_coverage(det, spec, ref, min_fit=MIN_FIT)
            own = score_coverage(det, spec, None, min_fit=MIN_FIT)
            for k in ("h_rms_px", "h_max_px", "hk1_rms_px", "hk1_max_px"):
                cov[k] = own.get(k)
            cov["pred_fit_scorable"] = own["scorable"]
            if not own["scorable"]:
                cov["pred_unscorable_reason"] = own["unscorable_reason"]
        cov["reference"] = ref_info
        cov["runtime_s"] = runtimes.get(stem)
        (run_dir / f"{stem}.coverage.json").write_text(json.dumps(cov, indent=2))
        if args.overlays and cov.get("scorable"):
            draw_overlay(image, det, cov, run_dir / "overlays" / f"{stem}.png")
        rows.append((stem, cov))
        print(f"{stem}: {summary_line(cov)}")

    write_report(rows, run_dir, args)
    return 0


def write_report(rows, run_dir: Path, args) -> None:
    lines = [
        f"# IPGS coverage — {args.label}",
        "",
        f"detect args: `{args.detect_args}`",
        "",
        "| image | frame | ref anchored | expected | detected | missing | wrong-label | centre-out "
        "| H rms/max px | H+k1 rms/max px | runtime s |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    totals: dict[str, dict] = {}
    for stem, c in rows:
        cam = stem.split("_")[0]
        t = totals.setdefault(cam, {"images": 0, "expected": 0, "detected": 0, "wrong": 0,
                                    "unscorable": 0, "complete": 0, "runtime": 0.0})
        t["images"] += 1
        t["runtime"] += c.get("runtime_s") or 0.0
        if not c.get("scorable"):
            t["unscorable"] += 1
            lines.append(f"| {stem} | {c.get('board_frame')} | - | unscorable: {c.get('unscorable_reason')} "
                         f"| {c.get('n_pred')} | | | | | | {fmt(c.get('runtime_s'), '{:.1f}')} |")
            continue
        if not c.get("pred_fit_scorable", True):
            t["unscorable"] += 1
        t["expected"] += c["n_expected"]
        t["detected"] += c["n_detected_expected"]
        t["wrong"] += c["n_wrong_label_suspects"]
        t["complete"] += int(c["n_detected_expected"] == c["n_expected"] and not c["n_wrong_label_suspects"])
        ref = c.get("reference") or {}
        lines.append(
            f"| {stem} | {c.get('board_frame')} | {ref.get('anchored')} | {c['n_expected']} "
            f"| {c['n_detected_expected']} | {len(c['missing'])} | {c['n_wrong_label_suspects']} "
            f"| {c['n_centre_outliers']} "
            f"| {fmt(c.get('h_rms_px'))}/{fmt(c.get('h_max_px'))} "
            f"| {fmt(c.get('hk1_rms_px'))}/{fmt(c.get('hk1_max_px'))} "
            f"| {fmt(c.get('runtime_s'), '{:.1f}')} |"
        )
    lines += ["", "| camera | images | expected | detected | recall | wrong-label | complete | "
              "unscorable (own fit) | mean runtime s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cam, t in sorted(totals.items()):
        recall = t["detected"] / t["expected"] if t["expected"] else 0.0
        lines.append(
            f"| {cam} | {t['images']} | {t['expected']} | {t['detected']} | {recall:.3f} | {t['wrong']} "
            f"| {t['complete']} | {t['unscorable']} | {t['runtime'] / max(1, t['images']):.1f} |"
        )
    report = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(report)
    (run_dir / "summary.json").write_text(json.dumps(totals, indent=2))
    print(report)


if __name__ == "__main__":
    raise SystemExit(main())
