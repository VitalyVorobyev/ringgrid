#!/usr/bin/env python3
"""Visualize ringgrid detection debug dumps (ringgrid.debug.v1).

Overlays detection candidates / final detections on top of the input image.

Examples:
  python3 tools/viz_detect_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --debug_json tools/out/synth_001/debug_0000.json \
    --out tools/out/synth_001/det_overlay_0000.png

  # Focus on a single marker id
  python3 tools/viz_detect_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --debug_json tools/out/synth_001/debug_0000.json \
    --id 42 --zoom 4.0 \
    --out tools/out/synth_001/det_overlay_id42.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def maybe_use_agg(out_path: str | None) -> None:
    # Use Agg for file output (headless-friendly).
    if out_path:
        import matplotlib

        matplotlib.use("Agg")


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def sample_ellipse_xy(ell: dict[str, Any], n: int = 128) -> tuple[list[float], list[float]]:
    """Sample ellipse polyline from EllipseParams-like dict.

    Expects:
      center_xy: [cx, cy]
      semi_axes: [a, b]
      angle: radians
    """
    cx, cy = ell["center_xy"]
    a, b = ell["semi_axes"]
    ang = ell["angle"]

    xs: list[float] = []
    ys: list[float] = []
    ca = math.cos(ang)
    sa = math.sin(ang)
    for i in range(n + 1):
        t = (i / n) * 2.0 * math.pi
        x = a * math.cos(t)
        y = b * math.sin(t)
        xr = ca * x - sa * y + cx
        yr = sa * x + ca * y + cy
        xs.append(float(xr))
        ys.append(float(yr))
    return xs, ys


def iter_stage_candidates(debug: dict[str, Any], stage_name: str) -> Iterable[dict[str, Any]]:
    stages = debug.get("stages", {})
    stage = stages.get(stage_name, {})
    for c in stage.get("candidates", []) or []:
        yield c


def find_final_marker(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    final_ = debug.get("stages", {}).get("final", {})
    for m in final_.get("detections", []) or []:
        if m.get("id") == marker_id:
            return m
    return None


def refine_entry_for_id(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    refine = debug.get("stages", {}).get("stage4_refine")
    if not refine:
        return None
    for m in refine.get("refined_markers", []) or []:
        if m.get("id") == marker_id:
            return m
    return None


def compute_zoom_window(center_xy: list[float], img_w: int, img_h: int, zoom: float) -> tuple[float, float, float, float]:
    span = min(img_w, img_h) / max(zoom, 1e-6)
    half = span / 2.0
    cx, cy = center_xy
    x0 = max(0.0, cx - half)
    x1 = min(float(img_w), cx + half)
    y0 = max(0.0, cy - half)
    y1 = min(float(img_h), cy + half)
    return x0, x1, y0, y1


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ringgrid.debug.v1 overlays")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--debug_json", required=True, type=str)
    parser.add_argument("--out", default=None, type=str, help="Write PNG to this path (otherwise interactive window)")
    parser.add_argument(
        "--stage",
        type=str,
        default="final",
        choices=["final", "stage0_proposals", "stage1_fit_decode", "stage3_ransac", "stage4_refine"],
    )
    parser.add_argument("--only-inliers", action="store_true")
    parser.add_argument("--id", type=int, default=None, help="Focus on a single decoded id")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom factor when --id is provided (default 4.0)")
    parser.add_argument("--show-ellipses", dest="show_ellipses", action="store_true", default=True)
    parser.add_argument("--no-ellipses", dest="show_ellipses", action="store_false")
    parser.add_argument("--show-candidates", dest="show_candidates", action="store_true", default=True)
    parser.add_argument("--no-candidates", dest="show_candidates", action="store_false")
    parser.add_argument("--show-edge-points", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    if args.id is not None and args.zoom is None:
        args.zoom = 4.0

    maybe_use_agg(args.out)
    import matplotlib.pyplot as plt

    debug = load_json(args.debug_json)
    schema_version = debug.get("schema_version")
    if schema_version != "ringgrid.debug.v1":
        if "detected_markers" in debug and "image_size" in debug:
            raise SystemExit(
                "Input looks like DetectionResult (normal --out JSON), not a debug dump.\n"
                "Re-run detection with `ringgrid detect --debug-json <path>` and pass that file here."
            )
        raise SystemExit(f"Unsupported debug schema_version: {schema_version!r}")

    img = plt.imread(args.image)
    # image dims from debug (authoritative for plotting extents)
    img_w = int(debug.get("image", {}).get("width", img.shape[1]))
    img_h = int(debug.get("image", {}).get("height", img.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, cmap="gray")

    title = f"ringgrid debug overlay ({args.stage})"
    if args.id is not None:
        title += f" id={args.id}"
    ax.set_title(title)

    # Helpers for ransac inlier/outlier sets (ids)
    ransac = debug.get("stages", {}).get("stage3_ransac", {})
    inlier_ids = set(ransac.get("inlier_ids", []) or [])
    outlier_ids = set(ransac.get("outlier_ids", []) or [])

    # Candidate scatter
    if args.show_candidates and args.stage in ("stage0_proposals", "stage1_fit_decode", "stage3_ransac"):
        stage_name = "stage0_proposals" if args.stage == "stage0_proposals" else "stage1_fit_decode"
        for c in iter_stage_candidates(debug, stage_name):
            prop = c.get("proposal", {})
            x, y = prop.get("center_xy", [None, None])
            if x is None or y is None:
                continue

            status = (c.get("decision", {}) or {}).get("status", "rejected")
            derived_id = (c.get("derived", {}) or {}).get("id")

            if args.id is not None and derived_id is not None and int(derived_id) != int(args.id):
                continue
            if args.only_inliers and derived_id is not None and int(derived_id) not in inlier_ids:
                continue

            color = "green" if status == "accepted" else "red"
            if args.stage == "stage3_ransac" and derived_id is not None:
                if int(derived_id) in outlier_ids:
                    color = "magenta"
                elif int(derived_id) in inlier_ids:
                    color = "green"

            ax.plot(float(x), float(y), "o", color=color, markersize=3, alpha=args.alpha)

    # Refine stage overlay
    if args.stage == "stage4_refine":
        refine = debug.get("stages", {}).get("stage4_refine")
        if refine:
            for m in refine.get("refined_markers", []) or []:
                mid = m.get("id")
                if args.id is not None and mid != args.id:
                    continue
                if args.only_inliers and mid is not None and int(mid) not in inlier_ids:
                    continue
                px, py = m.get("prior_center_xy", [None, None])
                rx, ry = m.get("refined_center_xy", [None, None])
                if px is not None and py is not None:
                    ax.plot(float(px), float(py), "x", color="yellow", markersize=6, alpha=args.alpha)
                if rx is not None and ry is not None:
                    ax.plot(float(rx), float(ry), "o", color="green", markersize=4, alpha=args.alpha)
                    ax.text(float(rx) + 4, float(ry) - 4, str(mid), fontsize=6, color="yellow")

                if args.show_ellipses:
                    for key, color in (("ellipse_outer", "lime"), ("ellipse_inner", "cyan")):
                        ell = m.get(key)
                        if ell:
                            xs, ys = sample_ellipse_xy(ell)
                            ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=args.alpha)

    # Final detections overlay
    if args.stage == "final":
        final_ = debug.get("stages", {}).get("final", {})
        for m in final_.get("detections", []) or []:
            mid = m.get("id")
            if args.id is not None and mid != args.id:
                continue
            if args.only_inliers and mid is not None and int(mid) not in inlier_ids:
                continue

            cx, cy = m.get("center", [None, None])
            if cx is None or cy is None:
                continue

            ax.plot(float(cx), float(cy), "o", color="green", markersize=4, alpha=args.alpha)
            if mid is not None:
                ax.text(float(cx) + 4, float(cy) - 4, str(mid), fontsize=6, color="yellow")

            if args.show_ellipses:
                for key, color in (("ellipse_outer", "lime"), ("ellipse_inner", "cyan")):
                    ell = m.get(key)
                    if ell:
                        xs, ys = sample_ellipse_xy(ell)
                        ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=args.alpha)

    # If focusing on an id and not in final/refine stage, draw stage1 ellipses if available.
    if args.id is not None and args.show_ellipses and args.stage in ("stage0_proposals", "stage1_fit_decode", "stage3_ransac"):
        for c in iter_stage_candidates(debug, "stage1_fit_decode"):
            derived_id = (c.get("derived", {}) or {}).get("id")
            if derived_id is None or int(derived_id) != int(args.id):
                continue
            rf = c.get("ring_fit")
            if not rf:
                break
            for key, color in (("ellipse_outer", "lime"), ("ellipse_inner", "cyan")):
                ell = rf.get(key)
                if ell:
                    xs, ys = sample_ellipse_xy(ell)
                    ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=args.alpha)
            break

    # Optional edge points when focusing on an id
    if args.show_edge_points and args.id is not None:
        # Prefer refine stage points if present, otherwise stage1 candidate ring_fit points
        # (Stage1 data is keyed by candidate index; we match by derived id.)
        points_drawn = False
        for c in iter_stage_candidates(debug, "stage1_fit_decode"):
            derived_id = (c.get("derived", {}) or {}).get("id")
            if derived_id is None or int(derived_id) != int(args.id):
                continue
            rf = c.get("ring_fit")
            if not rf:
                continue
            for key, color in (("points_outer", "lime"), ("points_inner", "cyan")):
                pts = rf.get(key)
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ax.plot(xs, ys, ".", color=color, markersize=1.5, alpha=args.alpha)
                    points_drawn = True
            break
        if not points_drawn:
            ax.text(10, 20, "edge points not present (run with --debug-store-points)", color="white", fontsize=10)

    # Default axes extents in image coordinates
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    # Zoom if requested
    if args.id is not None and args.zoom is not None:
        # Choose zoom center: refine refined center, else final center, else stage1 derived center.
        center = None
        refine_m = refine_entry_for_id(debug, args.id)
        if refine_m:
            center = refine_m.get("refined_center_xy")
        if center is None:
            fm = find_final_marker(debug, args.id)
            if fm:
                center = fm.get("center")
        if center is None:
            for c in iter_stage_candidates(debug, "stage1_fit_decode"):
                if (c.get("derived", {}) or {}).get("id") == args.id:
                    center = (c.get("derived", {}) or {}).get("center_xy")
                    break

        if center:
            x0, x1, y0, y1 = compute_zoom_window(center, img_w, img_h, args.zoom)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # invert y for image coords
    ax.set_aspect("equal")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote overlay to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
