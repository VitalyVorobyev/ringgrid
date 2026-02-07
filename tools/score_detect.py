#!/usr/bin/env python3
"""Score detection results against ground truth.

Computes:
  - Detection precision and recall (by ID match within distance gate)
  - Center error statistics for true positives (`marker.center` vs GT center)
  - Homography-vs-GT center error (if prediction JSON contains `homography`)
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
from typing import Any, Optional


def load_gt_data(path: str) -> dict[str, Any]:
    """Load ground truth JSON."""
    with open(path) as f:
        return json.load(f)


def visible_gt_markers(gt_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Load ground truth markers (visible only)."""
    return [m for m in gt_data["markers"] if m.get("visible", True)]


def load_pred(path: str) -> tuple[list[dict], dict]:
    """Load predicted (detected) markers and metadata."""
    with open(path) as f:
        data = json.load(f)
    return data["detected_markers"], data


def dist2(a: list, b: list) -> float:
    """Squared Euclidean distance between two 2D points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def pred_inner_outer_ratio(pred_marker: dict[str, Any]) -> Optional[float]:
    eo = pred_marker.get("ellipse_outer")
    ei = pred_marker.get("ellipse_inner")
    if not eo or not ei:
        return None
    oa, ob = (eo.get("semi_axes") or [None, None])[:2]
    ia, ib = (ei.get("semi_axes") or [None, None])[:2]
    if oa is None or ob is None or ia is None or ib is None:
        return None
    oa = float(oa)
    ob = float(ob)
    ia = float(ia)
    ib = float(ib)
    if oa <= 0.0 or ob <= 0.0:
        return None
    return 0.5 * ((ia / oa) + (ib / ob))


def stats_1d(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    return {
        "mean": sum(xs) / len(xs),
        "median": xs[len(xs) // 2],
        "p95": xs[int(0.95 * len(xs))],
        "max": xs[-1],
    }


def gt_center_for_mode(gt_marker: dict[str, Any], mode: str) -> Optional[list[float]]:
    """Return GT center for frame mode: 'image' or 'working'."""
    if mode not in ("image", "working"):
        raise ValueError(f"unsupported GT center mode: {mode}")
    key = "true_working_center" if mode == "working" else "true_image_center"
    center = gt_marker.get(key)
    if not isinstance(center, list) or len(center) < 2:
        # Backward compatibility with older GT JSONs.
        fallback_key = "true_image_center" if mode == "working" else "true_working_center"
        center = gt_marker.get(fallback_key)
    if not isinstance(center, list) or len(center) < 2:
        return None
    x = float(center[0])
    y = float(center[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return [x, y]


def resolve_gt_mode(mode: str, pred_data: dict[str, Any]) -> str:
    """Resolve 'auto' GT mode from prediction metadata."""
    if mode != "auto":
        return mode
    return "working" if pred_data.get("camera") is not None else "image"


def score(
    gt_markers: list[dict],
    pred_markers: list[dict],
    gate: float = 8.0,
    *,
    center_gt_mode: str = "image",
    expected_inner_ratio: Optional[float] = None,
    inner_ratio_tol: float = 0.08,
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
    matches = []  # (gt_idx, pred_idx, center_error_primary)

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
            gt_center = gt_center_for_mode(gt_m, center_gt_mode)
            if gt_center is None:
                continue
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

    # Center error statistics (primary center field only).
    center_errors_primary = [e for _, _, e in matches]
    center_stats = stats_1d(center_errors_primary)

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
            center = gt_center_for_mode(m, center_gt_mode)
            if center is None:
                center = m.get("true_image_center") or m.get("true_working_center")
            if not isinstance(center, list) or len(center) < 2:
                center = [float("nan"), float("nan")]
            missed_ids.append({
                "id": m["id"],
                "center": center,
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

    result = {
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
        "center_gt_frame": center_gt_mode,
    }

    if expected_inner_ratio is not None:
        pred_with_both = sum(
            1
            for m in pred_markers
            if m.get("ellipse_outer") is not None and m.get("ellipse_inner") is not None
        )

        rows = []
        ratios = []
        abs_errs = []
        for _, pred_idx, center_err in matches:
            m = pred_markers[pred_idx]
            ratio = pred_inner_outer_ratio(m)
            if ratio is None:
                continue
            ae = abs(ratio - expected_inner_ratio)
            ratios.append(ratio)
            abs_errs.append(ae)
            rows.append(
                {
                    "id": m.get("id"),
                    "pred_ratio": ratio,
                    "abs_err": ae,
                    "center_err_px": center_err,
                    "confidence": m.get("confidence", 0.0),
                }
            )

        rows.sort(key=lambda r: -r["abs_err"])

        result["inner_ratio"] = {
            "expected": expected_inner_ratio,
            "tol": inner_ratio_tol,
            "n_pred_with_both_ellipses": pred_with_both,
            "n_tp_with_both_ellipses": len(ratios),
            "ratio_stats": stats_1d(ratios),
            "abs_err_stats": stats_1d(abs_errs),
            "worst5": rows[:5],
        }

    return result


def extract_ransac_stats(pred_data: dict) -> dict | None:
    """Extract RANSAC/homography stats from prediction metadata."""
    return pred_data.get("ransac")


def project_h(h: list[list[float]], x: float, y: float) -> Optional[list[float]]:
    """Project 2D point through a 3x3 homography matrix."""
    q0 = h[0][0] * x + h[0][1] * y + h[0][2]
    q1 = h[1][0] * x + h[1][1] * y + h[1][2]
    q2 = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(q2) < 1e-12:
        return None
    return [q0 / q2, q1 / q2]


def homography_error_vs_gt(
    gt_markers: list[dict],
    pred_data: dict,
    gt_mode: str,
) -> Optional[dict]:
    """Compute absolute geometric error between predicted H and GT projected centers.

    For each visible GT marker with valid board/image coordinates:
      err = || project(H_pred, board_xy_mm) - true_<frame>_center ||
    """
    h = pred_data.get("homography")
    if not isinstance(h, list) or len(h) != 3:
        return None
    if any(not isinstance(row, list) or len(row) != 3 for row in h):
        return None

    errors = []
    for m in gt_markers:
        board_xy = m.get("board_xy_mm")
        true_center = gt_center_for_mode(m, gt_mode)
        if (
            not isinstance(board_xy, list)
            or len(board_xy) < 2
            or true_center is None
        ):
            continue
        proj = project_h(h, float(board_xy[0]), float(board_xy[1]))
        if proj is None:
            continue
        errors.append(math.sqrt(dist2(proj, [float(true_center[0]), float(true_center[1])])))

    if not errors:
        return None

    out = stats_1d(errors)
    out["n_markers_used"] = len(errors)
    out["gt_frame"] = gt_mode
    return out


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
    print(f"Center GT frame:         {result.get('center_gt_frame', 'image')}")

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

    hg = result.get("homography_error_vs_gt")
    if hg:
        print("Homography vs GT centers:")
        print(f"  GT frame:      {hg.get('gt_frame', 'image')}")
        print(f"  markers used:  {hg['n_markers_used']}")
        print(f"  mean:          {hg['mean']:.2f} px")
        print(f"  median:        {hg['median']:.2f} px")
        print(f"  p95:           {hg['p95']:.2f} px")
        print(f"  max:           {hg['max']:.2f} px")

    ir = result.get("inner_ratio")
    if ir:
        print("\nInner/outer ellipse ratio (TP only):")
        print(f"  expected:      {ir['expected']:.4f}")
        print(f"  tol:           {ir['tol']:.4f}")
        print(f"  pred w/ both:  {ir['n_pred_with_both_ellipses']}")
        print(f"  TP w/ both:    {ir['n_tp_with_both_ellipses']}")
        ars = ir.get("abs_err_stats", {}) or {}
        if ars:
            print(f"  abs_err mean:  {ars['mean']:.4f}")
            print(f"  abs_err p95:   {ars['p95']:.4f}")
            print(f"  abs_err max:   {ars['max']:.4f}")
        worst = ir.get("worst5") or []
        if worst:
            print("  worst5:")
            for r in worst:
                tag = " !!" if r["abs_err"] > ir["tol"] else ""
                print(
                    f"    id={r['id']:4d} ratio={r['pred_ratio']:.4f} abs_err={r['abs_err']:.4f}{tag}"
                )

    print()


def main():
    parser = argparse.ArgumentParser(description="Score ringgrid detection results")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON")
    parser.add_argument("--pred", required=True, help="Path to detection result JSON")
    parser.add_argument("--gate", type=float, default=8.0, help="Max center distance for matching (px)")
    parser.add_argument("--out", type=str, default=None, help="Optional: write scores JSON to this path")
    parser.add_argument(
        "--center-gt-key",
        choices=["auto", "image", "working"],
        default="auto",
        help=(
            "GT frame for center_error matching: "
            "'image' -> true_image_center, 'working' -> true_working_center, "
            "'auto' selects working when prediction JSON contains camera metadata."
        ),
    )
    parser.add_argument(
        "--homography-gt-key",
        choices=["auto", "image", "working"],
        default="auto",
        help=(
            "GT frame for homography_error_vs_gt: "
            "'image' -> true_image_center, 'working' -> true_working_center, "
            "'auto' selects working when prediction JSON contains camera metadata."
        ),
    )
    parser.add_argument(
        "--check-inner-ratio",
        action="store_true",
        help="Compute inner/outer ellipse ratio stats (requires ellipse_outer+ellipse_inner in predictions).",
    )
    parser.add_argument(
        "--inner-ratio-tol",
        type=float,
        default=0.08,
        help="Tolerance for flagging |ratio-expected| outliers (used with --check-inner-ratio).",
    )
    args = parser.parse_args()

    gt_data = load_gt_data(args.gt)
    gt_markers = visible_gt_markers(gt_data)
    pred_markers, pred_data = load_pred(args.pred)
    center_gt_mode = resolve_gt_mode(args.center_gt_key, pred_data)
    homography_gt_mode = resolve_gt_mode(args.homography_gt_key, pred_data)

    expected_ratio = None
    if args.check_inner_ratio:
        inner_mm = gt_data.get("inner_radius_mm")
        outer_mm = gt_data.get("outer_radius_mm")
        stress = bool(gt_data.get("stress_inner_confusion", False))
        if inner_mm is not None and outer_mm is not None and float(outer_mm) > 0.0:
            # Keep in sync with the detector's marker spec and the synthetic renderer:
            # - gen_synth draws rings with width `outer_radius * 0.12` (or 0.16 in stress mode)
            # - the detector's outer/inner ellipses correspond to the *outer/inner boundary*
            #   of the merged dark band, i.e. (inner_radius - ring_width) / (outer_radius + ring_width).
            ring_width = float(outer_mm) * (0.16 if stress else 0.12)
            denom = float(outer_mm) + ring_width
            if denom > 0.0:
                expected_ratio = (float(inner_mm) - ring_width) / denom
        else:
            print(
                "WARNING: GT missing inner_radius_mm/outer_radius_mm; skipping inner ratio stats.",
                file=sys.stderr,
            )

    result = score(
        gt_markers,
        pred_markers,
        gate=args.gate,
        center_gt_mode=center_gt_mode,
        expected_inner_ratio=expected_ratio,
        inner_ratio_tol=args.inner_ratio_tol,
    )

    # Attach RANSAC stats from prediction file if present
    ransac_stats = extract_ransac_stats(pred_data)
    if ransac_stats:
        result["ransac_stats"] = ransac_stats

    h_gt_stats = homography_error_vs_gt(gt_markers, pred_data, homography_gt_mode)
    if h_gt_stats:
        result["homography_error_vs_gt"] = h_gt_stats

    print_report(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Scores written to {args.out}")


if __name__ == "__main__":
    main()
