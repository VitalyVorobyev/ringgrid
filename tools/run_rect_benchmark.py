#!/usr/bin/env python3
"""Rect-plain synthetic benchmark: generate -> detect -> score -> summarize.

Runs two modes over the same random draws:
- dots:    ISRA-style board with origin dots — expects absolute board frame,
           correct anchoring, and homography quality vs ground truth.
- no_dots: same board without fiducials — expects the relative canonical
           frame with symmetry-consistent coordinates and no mm positions.

Writes per-image detections/scores plus a summary.json consumed by the
regression gate.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="tools/out/rect_benchmark")
    ap.add_argument("--n_images", type=int, default=3)
    ap.add_argument("--blur_px", type=float, default=0.8)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--marker_diameter", type=float, default=26.0)
    ap.add_argument("--gate", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    python = sys.executable
    summary = {"modes": {}}

    for mode in ("dots", "no_dots"):
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        gen_cmd = [
            python,
            str(TOOLS / "gen_synth_rect.py"),
            "--out_dir",
            str(mode_dir),
            "--n_images",
            str(args.n_images),
            "--blur_px",
            str(args.blur_px),
            "--noise_sigma",
            str(args.noise_sigma),
            "--seed",
            str(args.seed),
        ]
        if mode == "no_dots":
            gen_cmd.append("--no_dots")
        run(gen_cmd, cwd=str(TOOLS))

        scores = []
        for idx in range(args.n_images):
            img = mode_dir / f"img_{idx:04d}.png"
            det = mode_dir / f"det_{idx:04d}.json"
            score = mode_dir / f"score_{idx:04d}.json"
            run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "-q",
                    "--",
                    "detect",
                    "--target",
                    str(mode_dir / "target_spec.json"),
                    "--image",
                    str(img),
                    "--out",
                    str(det),
                    "--marker-diameter",
                    str(args.marker_diameter),
                ],
                cwd=str(ROOT),
            )
            run(
                [
                    python,
                    str(TOOLS / "score_detect_rect.py"),
                    "--gt",
                    str(mode_dir / f"gt_{idx:04d}.json"),
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

        mode_summary = {
            "n_images": args.n_images,
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
        summary["modes"][mode] = mode_summary
        print(f"[{mode}] {json.dumps(mode_summary, indent=2)}")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
