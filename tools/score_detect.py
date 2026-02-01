#!/usr/bin/env python3
"""Score detection results against ground truth.

Computes:
  - Detection precision and recall (by ID match within distance gate)
  - Center error statistics for true positives
  - Decode accuracy statistics
  - Lists of missed IDs and false positives

Usage:
    python tools/score_detect.py \
        --gt tools/out/synth_001/gt_0000.json \
        --pred /tmp/det_0000.json \
        [--gate 8.0] [--out scores.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_gt(path: str) -> list[dict]:
    """Load ground truth markers (visible only)."""
    with open(path) as f:
        data = json.load(f)
    return [m for m in data["markers"] if m.get("visible", True)]


def load_pred(path: str) -> tuple[list[dict], dict]:
    """Load predicted (detected) markers and metadata."""
    with open(path) as f:
        data = json.load(f)
    return data["detected_markers"], data


def dist2(a: list, b: list) -> float:
    """Squared Euclidean distance between two 2D points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def score(
    gt_markers: list[dict],
    pred_markers: list[dict],
    gate: float = 8.0,
) -> dict:
    """Score predictions against ground truth.

    Matching: for each GT marker, find the closest prediction with matching ID
    within `gate` pixels. Greedy matching by increasing distance.

    Returns a dict with precision, recall, center error stats, etc.
    """
    gate_sq = gate * gate

    # Build lookup: gt_id -> list of (gt_marker, index)
    gt_by_id: dict[int, list[tuple[dict, int]]] = {}
    for i, m in enumerate(gt_markers):
        mid = m["id"]
        gt_by_id.setdefault(mid, []).append((m, i))

    # Build lookup: pred_id -> list of (pred_marker, index)
    pred_with_id = [(m, i) for i, m in enumerate(pred_markers) if m.get("id") is not None]

    # Match: for each pred with ID, try to find a GT with same ID within gate
    gt_matched = set()
    pred_matched = set()
    matches = []  # (gt_idx, pred_idx, center_error)

    # Sort predictions by confidence descending for greedy matching
    pred_with_id.sort(key=lambda x: -x[0].get("confidence", 0.0))

    for pred_m, pred_idx in pred_with_id:
        pred_id = pred_m["id"]
        pred_center = pred_m["center"]

        if pred_id not in gt_by_id:
            continue

        best_gt_idx = None
        best_dist_sq = gate_sq

        for gt_m, gt_idx in gt_by_id[pred_id]:
            if gt_idx in gt_matched:
                continue
            gt_center = gt_m["true_image_center"]
            d2 = dist2(pred_center, gt_center)
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            gt_matched.add(best_gt_idx)
            pred_matched.add(pred_idx)
            matches.append((best_gt_idx, pred_idx, math.sqrt(best_dist_sq)))

    # Compute metrics
    n_gt = len(gt_markers)
    n_pred = len(pred_markers)
    n_pred_with_id = len(pred_with_id)
    n_tp = len(matches)
    n_fp = n_pred - n_tp  # all unmatched predictions are FP
    n_miss = n_gt - n_tp

    precision = n_tp / max(n_pred, 1)
    recall = n_tp / max(n_gt, 1)

    # Center error statistics
    center_errors = [e for _, _, e in matches]
    center_errors.sort()

    center_stats = {}
    if center_errors:
        center_stats = {
            "mean": sum(center_errors) / len(center_errors),
            "median": center_errors[len(center_errors) // 2],
            "p95": center_errors[int(0.95 * len(center_errors))],
            "max": center_errors[-1],
        }

    # Decode distance distribution (for matched predictions)
    decode_dists = []
    for _, pred_idx, _ in matches:
        m = pred_markers[pred_idx]
        if m.get("decode") and m["decode"].get("best_dist") is not None:
            decode_dists.append(m["decode"]["best_dist"])

    decode_dist_hist = {}
    for d in decode_dists:
        decode_dist_hist[str(d)] = decode_dist_hist.get(str(d), 0) + 1

    # ID accuracy among decoded predictions
    n_id_correct = n_tp
    n_id_total = n_pred_with_id

    # Missed GT IDs
    missed_ids = []
    for i, m in enumerate(gt_markers):
        if i not in gt_matched:
            missed_ids.append({
                "id": m["id"],
                "center": m["true_image_center"],
                "q": m.get("q"),
                "r": m.get("r"),
            })

    # False positive predictions (unmatched with ID)
    false_positives = []
    for pred_m, pred_idx in pred_with_id:
        if pred_idx not in pred_matched:
            false_positives.append({
                "pred_id": pred_m["id"],
                "center": pred_m["center"],
                "confidence": pred_m.get("confidence", 0),
            })
    # Also count predictions without ID as FP
    n_pred_no_id = sum(1 for m in pred_markers if m.get("id") is None)

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_pred_with_id": n_pred_with_id,
        "n_pred_no_id": n_pred_no_id,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_miss": n_miss,
        "precision": precision,
        "recall": recall,
        "center_error": center_stats,
        "decode_dist_histogram": decode_dist_hist,
        "missed_ids_top20": missed_ids[:20],
        "false_positives_top20": sorted(
            false_positives, key=lambda x: -x["confidence"]
        )[:20],
        "gate_px": gate,
    }


def extract_ransac_stats(pred_data: dict) -> dict | None:
    """Extract RANSAC/homography stats from prediction metadata."""
    return pred_data.get("ransac")


def print_report(result: dict) -> None:
    """Print a human-readable report."""
    print("=" * 60)
    print("ringgrid detection scoring")
    print("=" * 60)
    print(f"GT markers (visible):    {result['n_gt']}")
    print(f"Predictions total:       {result['n_pred']}")
    print(f"  with decoded ID:       {result['n_pred_with_id']}")
    print(f"  without ID:            {result['n_pred_no_id']}")
    print(f"True positives:          {result['n_tp']}")
    print(f"False positives:         {result['n_fp']}")
    print(f"Missed:                  {result['n_miss']}")
    print(f"Precision:               {result['precision']:.3f}")
    print(f"Recall:                  {result['recall']:.3f}")
    print(f"Gate:                    {result['gate_px']:.1f} px")

    ce = result.get("center_error", {})
    if ce:
        print(f"\nCenter error (TP):")
        print(f"  mean:   {ce['mean']:.2f} px")
        print(f"  median: {ce['median']:.2f} px")
        print(f"  p95:    {ce['p95']:.2f} px")
        print(f"  max:    {ce['max']:.2f} px")

    dh = result.get("decode_dist_histogram", {})
    if dh:
        print(f"\nDecode distance histogram:")
        for d in sorted(dh.keys(), key=int):
            print(f"  dist={d}: {dh[d]}")

    missed = result.get("missed_ids_top20", [])
    if missed:
        print(f"\nTop missed IDs ({len(missed)} shown):")
        for m in missed[:10]:
            print(f"  id={m['id']:4d} at ({m['center'][0]:.0f}, {m['center'][1]:.0f})")

    fps = result.get("false_positives_top20", [])
    if fps:
        print(f"\nTop false positives ({len(fps)} shown):")
        for fp in fps[:10]:
            print(f"  pred_id={fp['pred_id']:4d} conf={fp['confidence']:.3f} at ({fp['center'][0]:.0f}, {fp['center'][1]:.0f})")

    rs = result.get("ransac_stats")
    if rs:
        print(f"\nHomography RANSAC:")
        print(f"  candidates:    {rs['n_candidates']}")
        print(f"  inliers:       {rs['n_inliers']}")
        print(f"  threshold:     {rs['threshold_px']:.1f} px")
        print(f"  mean reproj:   {rs['mean_err_px']:.2f} px")
        print(f"  p95 reproj:    {rs['p95_err_px']:.2f} px")

    print()


def main():
    parser = argparse.ArgumentParser(description="Score ringgrid detection results")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON")
    parser.add_argument("--pred", required=True, help="Path to detection result JSON")
    parser.add_argument("--gate", type=float, default=8.0, help="Max center distance for matching (px)")
    parser.add_argument("--out", type=str, default=None, help="Optional: write scores JSON to this path")
    args = parser.parse_args()

    gt_markers = load_gt(args.gt)
    pred_markers, pred_data = load_pred(args.pred)

    result = score(gt_markers, pred_markers, gate=args.gate)

    # Attach RANSAC stats from prediction file if present
    ransac_stats = extract_ransac_stats(pred_data)
    if ransac_stats:
        result["ransac_stats"] = ransac_stats

    print_report(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Scores written to {args.out}")


if __name__ == "__main__":
    main()
