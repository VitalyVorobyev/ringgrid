#!/usr/bin/env python3
"""Visualize ringgrid DetectionResult overlays: markers, edge points, and proposals.

Extends viz_detect.py with edge-point scatter and proposal-center rendering.
All overlay components are independently togglable.

Examples
--------
# Show markers + both ellipses (same as viz_detect.py defaults):
  python3 viz_detect_edges.py --image img.png --det_json det.json

# Add outer + inner edge scatter:
  python3 viz_detect_edges.py --image img.png --det_json det.json --show-edge-points

# Show proposals only (no marker labels):
  python3 viz_detect_edges.py --image img.png --det_json det.json \\
      --show-proposals --no-ellipses --no-confidence

# Zoom to one marker with edges:
  python3 viz_detect_edges.py --image img.png --det_json det.json \\
      --id 19 --zoom 8 --show-edge-points

# Save to file:
  python3 viz_detect_edges.py --image img.png --det_json det.json \\
      --show-edge-points --show-proposals --out out/overlay.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def maybe_use_agg(out_path: str | None) -> None:
    if out_path:
        import matplotlib

        matplotlib.use("Agg")


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def sample_ellipse_xy(ellipse: dict[str, Any], n: int = 128) -> tuple[list[float], list[float]]:
    if "center_xy" in ellipse:
        cx, cy = ellipse["center_xy"]
    else:
        cx, cy = ellipse["cx"], ellipse["cy"]
    if "semi_axes" in ellipse:
        a, b = ellipse["semi_axes"]
    else:
        a, b = ellipse["a"], ellipse["b"]
    angle = ellipse["angle"]
    ca = math.cos(angle)
    sa = math.sin(angle)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n + 1):
        t = (i / n) * 2.0 * math.pi
        ex = a * math.cos(t)
        ey = b * math.sin(t)
        xs.append(ca * ex - sa * ey + cx)
        ys.append(sa * ex + ca * ey + cy)
    return xs, ys


def compute_zoom_window(
    center_xy: list[float], img_w: int, img_h: int, zoom: float
) -> tuple[float, float, float, float]:
    span = min(img_w, img_h) / max(zoom, 1e-6)
    half = span / 2.0
    cx, cy = center_xy
    return (
        max(0.0, cx - half),
        min(float(img_w), cx + half),
        max(0.0, cy - half),
        min(float(img_h), cy + half),
    )


def marker_confidence(marker: dict[str, Any]) -> float | None:
    confidence = marker.get("confidence")
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def confidence_color(confidence: float | None) -> tuple[float, float, float]:
    if confidence is None:
        return (0.0, 1.0, 0.0)
    r = 1.0 - 0.9 * confidence
    g = 0.1 + 0.9 * confidence
    return (r, g, 0.1)


def plot_detection(
    ax,
    marker: dict[str, Any],
    *,
    alpha: float,
    show_ellipses: bool,
    show_confidence: bool,
) -> None:
    import matplotlib.patheffects as pe

    cx, cy = marker.get("center", [None, None])
    if cx is None or cy is None:
        return

    confidence = marker_confidence(marker) if show_confidence else None
    point_color = confidence_color(confidence)
    ax.plot(float(cx), float(cy), "o", color=point_color, markersize=4, alpha=alpha)

    marker_id = marker.get("id")
    label = None
    if marker_id is not None and confidence is not None:
        label = f"{marker_id} ({confidence:.2f})"
    elif marker_id is not None:
        label = str(marker_id)
    elif confidence is not None:
        label = f"{confidence:.2f}"

    if label is not None:
        ax.text(
            float(cx) + 4.0,
            float(cy) - 4.0,
            label,
            fontsize=6,
            color="white",
            ha="left",
            va="bottom",
            path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()],
        )

    if not show_ellipses:
        return
    for key, color in (("ellipse_outer", "lime"), ("ellipse_inner", "cyan")):
        ellipse = marker.get(key)
        if not ellipse:
            continue
        xs, ys = sample_ellipse_xy(ellipse)
        ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=alpha)


def plot_edge_points(
    ax,
    marker: dict[str, Any],
    *,
    show_outer: bool,
    show_inner: bool,
    alpha: float,
) -> None:
    """Scatter raw sub-pixel edge sample points for a single marker.

    Outer points are rendered in orange, inner points in yellow, so they
    are visually distinct from the fitted ellipses (lime/cyan).
    """
    if show_outer:
        pts = marker.get("edge_points_outer")
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=6, c="orange", linewidths=0, alpha=alpha, zorder=3)

    if show_inner:
        pts = marker.get("edge_points_inner")
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=6, c="yellow", linewidths=0, alpha=alpha, zorder=3)


def plot_proposals(
    ax,
    proposals: list[dict[str, Any]],
    *,
    alpha: float,
) -> None:
    """Scatter proposal centers sized by their gradient-vote score.

    Proposals are rendered in magenta behind detected markers so they
    do not occlude detection results. Marker size scales from 2 to 10 pt
    proportional to score relative to the strongest proposal.
    """
    if not proposals:
        return
    scores = [float(p.get("score", 0.0)) for p in proposals]
    max_score = max(scores) if scores else 1.0
    if max_score <= 0.0:
        max_score = 1.0
    xs = []
    ys = []
    sizes = []
    for p, score in zip(proposals, scores):
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        sizes.append(2.0 + (score / max_score) * 10.0)
    if xs:
        ax.scatter(xs, ys, s=sizes, c="magenta", linewidths=0, alpha=alpha, zorder=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ringgrid DetectionResult overlays (markers, edges, proposals)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- I/O ---
    parser.add_argument("--image", required=True, type=str, help="Input image file")
    parser.add_argument("--det_json", required=True, type=str, help="DetectionResult JSON file")
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Write PNG to this path (omit for interactive window)",
    )
    parser.add_argument("--id", type=int, default=None, help="Focus on a single marker ID")
    parser.add_argument(
        "--zoom", type=float, default=None, help="Zoom factor when --id is set (default: 4)"
    )
    parser.add_argument("--alpha", type=float, default=0.85, help="Global overlay alpha (default: 0.85)")

    # --- marker / ellipse ---
    parser.add_argument(
        "--show-ellipses",
        dest="show_ellipses",
        action="store_true",
        default=True,
        help="Overlay fitted ellipses (default: on)",
    )
    parser.add_argument(
        "--no-ellipses", dest="show_ellipses", action="store_false", help="Hide fitted ellipses"
    )
    parser.add_argument(
        "--show-confidence",
        dest="show_confidence",
        action="store_true",
        default=True,
        help="Color centers by confidence and include it in labels (default: on)",
    )
    parser.add_argument(
        "--no-confidence",
        dest="show_confidence",
        action="store_false",
        help="Disable confidence coloring and labels",
    )

    # --- edge points ---
    parser.add_argument(
        "--show-edge-points",
        dest="show_edge_points",
        action="store_true",
        default=False,
        help="Shorthand: enable both --show-outer-edge and --show-inner-edge",
    )
    parser.add_argument(
        "--show-outer-edge",
        dest="show_outer_edge",
        action="store_true",
        default=False,
        help="Overlay outer ellipse edge sample points (orange, default: off)",
    )
    parser.add_argument("--no-outer-edge", dest="show_outer_edge", action="store_false")
    parser.add_argument(
        "--show-inner-edge",
        dest="show_inner_edge",
        action="store_true",
        default=False,
        help="Overlay inner ellipse edge sample points (yellow, default: off)",
    )
    parser.add_argument("--no-inner-edge", dest="show_inner_edge", action="store_false")

    # --- proposals ---
    parser.add_argument(
        "--show-proposals",
        dest="show_proposals",
        action="store_true",
        default=False,
        help="Overlay proposal centers sized by gradient-vote score (magenta, default: off)",
    )
    parser.add_argument("--no-proposals", dest="show_proposals", action="store_false")

    args = parser.parse_args()

    # --show-edge-points is a shorthand for both outer and inner
    if args.show_edge_points:
        args.show_outer_edge = True
        args.show_inner_edge = True

    if args.id is not None and args.zoom is None:
        args.zoom = 4.0

    maybe_use_agg(args.out)
    import matplotlib.pyplot as plt

    image = plt.imread(args.image)
    detection = load_json(args.det_json)
    markers = detection.get("detected_markers")
    if not isinstance(markers, list):
        raise SystemExit("Input JSON does not look like DetectionResult (missing detected_markers)")

    image_size = detection.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2:
        img_w, img_h = int(image_size[0]), int(image_size[1])
    else:
        img_h, img_w = image.shape[:2]

    render_dpi = 100
    fig = plt.figure(
        figsize=(img_w / render_dpi, img_h / render_dpi),
        dpi=render_dpi,
        frameon=False,
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()

    # Proposals first — rendered behind everything else
    if args.show_proposals:
        proposals = detection.get("proposals", [])
        if not proposals:
            print("Warning: --show-proposals set but no 'proposals' key found in JSON")
        plot_proposals(ax, proposals, alpha=args.alpha * 0.6)

    # Per-marker overlays
    show_any_edge = args.show_outer_edge or args.show_inner_edge
    for marker in markers:
        marker_id = marker.get("id")
        if args.id is not None and marker_id != args.id:
            continue
        plot_detection(
            ax,
            marker,
            alpha=args.alpha,
            show_ellipses=args.show_ellipses,
            show_confidence=args.show_confidence,
        )
        if show_any_edge:
            plot_edge_points(
                ax,
                marker,
                show_outer=args.show_outer_edge,
                show_inner=args.show_inner_edge,
                alpha=args.alpha * 0.75,
            )

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")

    if args.id is not None and args.zoom is not None:
        center = None
        for marker in markers:
            if marker.get("id") == args.id:
                center = marker.get("center")
                break
        if center is not None:
            x0, x1, y0, y1 = compute_zoom_window(center, img_w, img_h, args.zoom)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=render_dpi, pad_inches=0)
        plt.close(fig)
        print(f"Wrote overlay to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
