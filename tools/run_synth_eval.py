#!/usr/bin/env python3
"""End-to-end synthetic evaluation: generate → detect → score.

Usage:
    python tools/run_synth_eval.py --n 3 [--blur_px 5.0] [--marker_diameter 14.0]

Prerequisites:
    - tools/gen_synth.py must be runnable
    - ringgrid binary must be built (cargo build)
    - tools/score_detect.py must exist
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_ringgrid_binary() -> str:
    """Find the ringgrid binary."""
    candidates = [
        "target/release/ringgrid",
        "target/debug/ringgrid",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try cargo run
    return None


def main():
    parser = argparse.ArgumentParser(description="Run synthetic eval pipeline")
    parser.add_argument("--n", type=int, default=3, help="Number of images")
    parser.add_argument("--blur_px", type=float, default=1.0, help="Blur sigma in pixels")
    parser.add_argument("--marker_diameter", type=float, default=32.0, help="Expected marker diameter in pixels")
    parser.add_argument("--gate", type=float, default=8.0, help="Matching gate in pixels")
    parser.add_argument("--out_dir", type=str, default="tools/out/eval_run", help="Output directory")
    parser.add_argument("--codebook", type=str, default="tools/codebook.json", help="Codebook JSON path")
    parser.add_argument("--skip_gen", action="store_true", help="Skip image generation (reuse existing)")
    parser.add_argument(
        "--stress_inner_confusion",
        action="store_true",
        help="Enable gen_synth stress mode that increases inner-edge confusion.",
    )
    parser.add_argument(
        "--tilt_strength",
        type=float,
        default=0.3,
        help="Projective tilt strength passed through to tools/gen_synth.py (larger => stronger perspective).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    synth_dir = out_dir / "synth"
    det_dir = out_dir / "det"
    synth_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate synthetic images
    if not args.skip_gen:
        print(f"[1/3] Generating {args.n} synthetic images (blur={args.blur_px}px)...")
        gen_cmd = [
            sys.executable, "tools/gen_synth.py",
            "--out_dir", str(synth_dir),
            "--n_images", str(args.n),
            "--blur_px", str(args.blur_px),
            "--codebook", args.codebook,
            "--tilt_strength", str(args.tilt_strength),
        ]
        if args.stress_inner_confusion:
            gen_cmd.append("--stress-inner-confusion")
        result = subprocess.run(gen_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"gen_synth.py failed:\n{result.stderr}")
            sys.exit(1)
        print(result.stdout.strip())
    else:
        print(f"[1/3] Skipping generation, using existing images in {synth_dir}")

    # Step 2: Run detection on each image
    binary = find_ringgrid_binary()
    use_cargo_run = binary is None

    print(f"\n[2/3] Running detection on {args.n} images...")
    for i in range(args.n):
        img_path = synth_dir / f"img_{i:04d}.png"
        det_path = det_dir / f"det_{i:04d}.json"
        debug_path = det_dir / f"debug_{i:04d}.json"

        if not img_path.exists():
            print(f"  WARNING: {img_path} not found, skipping")
            continue

        if use_cargo_run:
            cmd = [
                "cargo", "run", "--quiet", "--",
                "detect",
                "--image", str(img_path),
                "--out", str(det_path),
                "--debug-json", str(debug_path),
                "--marker-diameter", str(args.marker_diameter),
            ]
        else:
            cmd = [
                binary,
                "detect",
                "--image", str(img_path),
                "--out", str(det_path),
                "--debug-json", str(debug_path),
                "--marker-diameter", str(args.marker_diameter),
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Detection failed for {img_path}:\n{result.stderr}")
            continue

        # Count detections
        with open(det_path) as f:
            det_data = json.load(f)
        n_det = len(det_data["detected_markers"])
        n_with_id = sum(1 for m in det_data["detected_markers"] if m.get("id") is not None)
        print(f"  [{i+1}/{args.n}] {img_path.name}: {n_det} detections ({n_with_id} with ID)")

    # Step 3: Score each image and aggregate
    print(f"\n[3/3] Scoring results...")
    all_results = []
    for i in range(args.n):
        gt_path = synth_dir / f"gt_{i:04d}.json"
        det_path = det_dir / f"det_{i:04d}.json"

        if not gt_path.exists() or not det_path.exists():
            continue

        cmd = [
            sys.executable, "tools/score_detect.py",
            "--gt", str(gt_path),
            "--pred", str(det_path),
            "--gate", str(args.gate),
            "--out", str(det_dir / f"score_{i:04d}.json"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout.strip())

        score_path = det_dir / f"score_{i:04d}.json"
        if score_path.exists():
            with open(score_path) as f:
                all_results.append(json.load(f))

    # Aggregate
    if all_results:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        n = len(all_results)
        avg_precision = sum(r["precision"] for r in all_results) / n
        avg_recall = sum(r["recall"] for r in all_results) / n
        avg_tp = sum(r["n_tp"] for r in all_results) / n
        avg_fp = sum(r["n_fp"] for r in all_results) / n
        avg_miss = sum(r["n_miss"] for r in all_results) / n
        total_gt = sum(r["n_gt"] for r in all_results)
        total_tp = sum(r["n_tp"] for r in all_results)

        print(f"Images scored:       {n}")
        print(f"Total GT markers:    {total_gt}")
        print(f"Total TP:            {total_tp}")
        print(f"Avg precision:       {avg_precision:.3f}")
        print(f"Avg recall:          {avg_recall:.3f}")
        print(f"Avg TP per image:    {avg_tp:.0f}")
        print(f"Avg FP per image:    {avg_fp:.0f}")
        print(f"Avg missed:          {avg_miss:.0f}")

        # Aggregate center errors
        all_ce = []
        for r in all_results:
            ce = r.get("center_error", {})
            if ce:
                all_ce.append(ce["mean"])
        if all_ce:
            print(f"Avg center error:    {sum(all_ce)/len(all_ce):.2f} px")

        # Aggregate RANSAC stats
        all_ransac = [r["ransac_stats"] for r in all_results if r.get("ransac_stats")]
        avg_reproj = None
        if all_ransac:
            avg_reproj = sum(rs["mean_err_px"] for rs in all_ransac) / len(all_ransac)
            avg_p95 = sum(rs["p95_err_px"] for rs in all_ransac) / len(all_ransac)
            avg_inliers = sum(rs["n_inliers"] for rs in all_ransac) / len(all_ransac)
            print(f"Avg RANSAC inliers:  {avg_inliers:.0f}")
            print(f"Avg reproj error:    {avg_reproj:.2f} px")
            print(f"Avg reproj p95:      {avg_p95:.2f} px")

        # Write aggregate
        agg = {
            "n_images": n,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_tp": avg_tp,
            "avg_fp": avg_fp,
            "avg_miss": avg_miss,
            "avg_center_error": sum(all_ce) / len(all_ce) if all_ce else None,
            "avg_reproj_error": avg_reproj,
            "per_image": all_results,
        }
        agg_path = det_dir / "aggregate.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nAggregate scores written to {agg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
