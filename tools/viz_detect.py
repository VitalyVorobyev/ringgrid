#!/usr/bin/env python3
"""Visualize ringgrid DetectionResult JSON overlays."""

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
        x = a * math.cos(t)
        y = b * math.sin(t)
        xr = ca * x - sa * y + cx
        yr = sa * x + ca * y + cy
        xs.append(float(xr))
        ys.append(float(yr))
    return xs, ys


def compute_zoom_window(
    center_xy: list[float], img_w: int, img_h: int, zoom: float
) -> tuple[float, float, float, float]:
    span = min(img_w, img_h) / max(zoom, 1e-6)
    half = span / 2.0
    cx, cy = center_xy
    x0 = max(0.0, cx - half)
    x1 = min(float(img_w), cx + half)
    y0 = max(0.0, cy - half)
    y1 = min(float(img_h), cy + half)
    return x0, x1, y0, y1


def plot_detection(
    ax,
    marker: dict[str, Any],
    *,
    alpha: float,
    show_ellipses: bool,
) -> None:
    import matplotlib.patheffects as pe

    cx, cy = marker.get("center", [None, None])
    if cx is None or cy is None:
        return

    ax.plot(float(cx), float(cy), "o", color="lime", markersize=4, alpha=alpha)

    marker_id = marker.get("id")
    if marker_id is not None:
        ax.text(
            float(cx) + 4.0,
            float(cy) - 4.0,
            str(marker_id),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ringgrid DetectionResult overlays")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--det_json", required=True, type=str)
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Write PNG to this path (otherwise interactive window)",
    )
    parser.add_argument("--id", type=int, default=None, help="Focus on a single marker id")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom factor when --id is set")
    parser.add_argument("--show-ellipses", dest="show_ellipses", action="store_true", default=True)
    parser.add_argument("--no-ellipses", dest="show_ellipses", action="store_false")
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

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
        img_h, img_w = image.shape[0], image.shape[1]

    render_dpi = 100
    fig = plt.figure(
        figsize=(img_w / render_dpi, img_h / render_dpi),
        dpi=render_dpi,
        frameon=False,
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()

    for marker in markers:
        marker_id = marker.get("id")
        if args.id is not None and marker_id != args.id:
            continue
        plot_detection(ax, marker, alpha=args.alpha, show_ellipses=args.show_ellipses)

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
