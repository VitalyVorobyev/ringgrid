#!/usr/bin/env python3
"""Target-matrix synthetic benchmark: generate -> detect -> score -> summarize.

One case per target combination the library supports beyond the hex-coded
reference/distortion/blur2 benchmarks:

- rect_plain_dots / rect_plain_nodots: the 24x24 plain rect board with and
  without its origin-dot triad (absolute vs relative canonical frame).
- rect_coded: coded rings on a rect lattice (IDs anchor the board).
- hex_plain_dots / hex_plain_nodots: plain rings on the hex lattice.
- hex_coded_extended: a coded hex board with more cells than the 893-word base
  codebook, so the Extended profile's extension IDs are decoded end to end.

Every case shares the scene parameters (blur, noise, seed, image count); a case
fixes only its target and the image size that puts its markers at a sensible
pixel diameter. Writes per-image detections/scores plus a summary.json consumed
by the regression gate (``target_matrix_benchmark`` in
``tools/ci/regression_baseline.json``).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from gen_synth_target import (
    DEFAULT_COLS,
    DEFAULT_DOT_RADIUS_MM,
    DEFAULT_INNER_RADIUS_MM,
    DEFAULT_OUTER_RADIUS_MM,
    DEFAULT_PITCH_MM,
    DEFAULT_ROWS,
    TARGET_SCHEMA,
    plain_rect_spec,
)

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


def hex_spec(name, rows, long_row_cols, coding, dot_radius_mm=None) -> dict:
    """A v6 hex spec with the classic ringgrid marker (pitch 8, rings 4.8/3.2 mm)."""
    spec = {
        "schema": TARGET_SCHEMA,
        "name": name,
        "lattice": {
            "kind": "hex",
            "rows": rows,
            "long_row_cols": long_row_cols,
            "pitch_mm": 8.0,
        },
        "marker": {"outer_radius_mm": 4.8, "inner_radius_mm": 3.2},
        "coding": coding,
    }
    if dot_radius_mm is not None:
        spec["fiducials"] = {"dot_radius_mm": dot_radius_mm}
    return spec


def rect_plain(dots: bool) -> dict:
    return plain_rect_spec(
        DEFAULT_ROWS,
        DEFAULT_COLS,
        DEFAULT_PITCH_MM,
        DEFAULT_OUTER_RADIUS_MM,
        DEFAULT_INNER_RADIUS_MM,
        DEFAULT_DOT_RADIUS_MM if dots else None,
    )


CODED = {"kind": "coded16", "ring_width_mm": 1.152}

# img: square image side (px). marker_diameter: nominal outer-ring diameter in
# px for the detector's scale prior — the generator fills ~70% of the image
# with the board span, so diameter ≈ 0.7 * img / span_mm * 2 * outer_radius_mm.
CASES = {
    # Original rect benchmark (formerly modes dots / no_dots) — parameters frozen.
    "rect_plain_dots": {"spec": rect_plain(True), "img": 1100, "marker_diameter": 26.0},
    "rect_plain_nodots": {"spec": rect_plain(False), "img": 1100, "marker_diameter": 26.0},
    # examples/targets/rect_coded.toml geometry: 20x20, pitch 14, span 276.8 mm.
    "rect_coded": {
        "spec": {
            "schema": TARGET_SCHEMA,
            "name": "synth_rect_coded",
            "lattice": {"kind": "rect", "rows": 20, "cols": 20, "pitch_mm": 14.0},
            "marker": {"outer_radius_mm": 4.8, "inner_radius_mm": 3.2},
            "coding": CODED,
        },
        "img": 1280,
        "marker_diameter": 31.0,
    },
    # examples/targets/hex_plain_*.toml geometry: 15x14, span 189.7 mm. The dot
    # radius is what the recipe's `fiducials = "auto"` derives (0.1 * pitch).
    "hex_plain_dots": {
        "spec": hex_spec("synth_hex_plain_dots", 15, 14, {"kind": "plain"}, 0.8),
        "img": 900,
        "marker_diameter": 32.0,
    },
    "hex_plain_nodots": {
        "spec": hex_spec("synth_hex_plain_nodots", 15, 14, {"kind": "plain"}),
        "img": 900,
        "marker_diameter": 32.0,
    },
    # 37 rows x 34 long-row cols = 1239 cells: IDs 893..1238 (346 markers) come
    # from the Extended profile's extension. Span 467.8 mm.
    "hex_coded_extended": {
        "spec": hex_spec(
            "synth_hex_coded_extended",
            37,
            34,
            {**CODED, "codebook_profile": "extended"},
        ),
        "img": 2200,
        "marker_diameter": 32.0,
    },
}


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"command failed: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(scores, n_images) -> dict:
    summary = {
        "n_images": n_images,
        "avg_precision": mean([s["precision"] for s in scores]),
        "avg_recall": mean([s["recall"] for s in scores]),
        "avg_center_mean_px": mean([s["center_err_mean_px"] for s in scores]),
        "avg_coord_accuracy": mean([s["coord_accuracy"] for s in scores]),
        "origin_resolution_rate": mean(
            [1.0 if s["origin_resolved"] else 0.0 for s in scores]
        ),
        "origin_correct_rate": mean(
            [1.0 if s["origin_correct"] else 0.0 for s in scores]
        ),
        "avg_h_vs_gt_mean_px": mean([s["h_vs_gt_mean_px"] for s in scores]),
    }
    # Mirror metrics exist only in the relative frame, identity metrics only
    # for coded targets (and ext_* only for the Extended profile).
    if "labels_mirrored" in scores[0]:
        summary["avg_coord_accuracy_up_to_mirror"] = mean(
            [s["coord_accuracy_up_to_mirror"] for s in scores]
        )
        summary["mirrored_rate"] = mean(
            [1.0 if s["labels_mirrored"] else 0.0 for s in scores]
        )
    for key in ("id_accuracy", "decoded_fraction", "ext_recall", "ext_decoded_rate"):
        if key in scores[0]:
            summary[f"avg_{key}"] = mean([s[key] for s in scores])
    if "n_id_wrong" in scores[0]:
        summary["total_id_wrong"] = sum(s["n_id_wrong"] for s in scores)
    return summary


def run_case(name, case, args, out_root: Path, python: str) -> dict:
    case_dir = out_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    # The generator re-emits the spec it rendered as target_spec.json.
    spec_path = case_dir / "target_spec.json"
    with open(spec_path, "w") as f:
        json.dump(case["spec"], f, indent=2)

    run(
        [
            python,
            str(TOOLS / "gen_synth_target.py"),
            "--out_dir",
            str(case_dir),
            "--target",
            str(spec_path),
            "--n_images",
            str(args.n_images),
            "--img_w",
            str(case["img"]),
            "--img_h",
            str(case["img"]),
            "--blur_px",
            str(args.blur_px),
            "--noise_sigma",
            str(args.noise_sigma),
            "--seed",
            str(args.seed),
        ],
        cwd=str(TOOLS),
    )

    scores = []
    for idx in range(args.n_images):
        img = case_dir / f"img_{idx:04d}.png"
        det = case_dir / f"det_{idx:04d}.json"
        score = case_dir / f"score_{idx:04d}.json"
        run(
            [
                "cargo",
                "run",
                "--bin",
                "ringgrid-dev",
                "--release",
                "-q",
                "--",
                "detect",
                "--target",
                str(spec_path),
                "--image",
                str(img),
                "--out",
                str(det),
                "--marker-diameter",
                str(case["marker_diameter"]),
            ],
            cwd=str(ROOT),
        )
        run(
            [
                python,
                str(TOOLS / "score_detect_target.py"),
                "--gt",
                str(case_dir / f"gt_{idx:04d}.json"),
                "--pred",
                str(det),
                "--gate",
                str(args.gate),
                "--out",
                str(score),
            ]
        )
        with open(score) as f:
            scores.append(json.load(f))
    return summarize(scores, args.n_images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="tools/out/target_matrix_benchmark")
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        default=list(CASES),
        help="Subset of cases to run (default: all).",
    )
    ap.add_argument("--n_images", type=int, default=3)
    ap.add_argument("--blur_px", type=float, default=0.8)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--gate", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    summary = {"modes": {}}

    for name in args.cases:
        t0 = time.monotonic()
        mode_summary = run_case(name, CASES[name], args, out_root, python)
        summary["modes"][name] = mode_summary
        print(f"[{name}] ({time.monotonic() - t0:.1f}s) {json.dumps(mode_summary, indent=2)}")

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
