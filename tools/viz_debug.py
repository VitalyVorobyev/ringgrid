#!/usr/bin/env python3
"""Visualize synthetic ground truth overlaid on generated images.

Given an image and its gt.json, overlay:
  - True marker centers (crosses)
  - Marker IDs as text labels
  - Outer/inner ellipses (optional)

Usage:
    python tools/viz_debug.py --image tools/out/synth_001/img_0000.png \\
                              --gt tools/out/synth_001/gt_0000.json \\
                              --out tools/out/synth_001/viz_0000.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_ellipse(ax, ell: dict, color: str = "lime", lw: float = 0.8):
    """Draw an ellipse from parameter dict."""
    cx = ell["cx"]
    cy = ell["cy"]
    a = ell["a"]
    b = ell["b"]
    angle_deg = math.degrees(ell["angle"])

    e = patches.Ellipse(
        (cx, cy), 2 * a, 2 * b,
        angle=angle_deg,
        fill=False, edgecolor=color, linewidth=lw,
    )
    ax.add_patch(e)


def main():
    parser = argparse.ArgumentParser(description="Visualize ringgrid GT overlay")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--show_ellipses", action="store_true", default=True)
    parser.add_argument("--no_ellipses", dest="show_ellipses", action="store_false")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # Load image
    img = plt.imread(args.image)

    # Load GT
    with open(args.gt) as f:
        gt = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1 if img.dtype == np.float64 else 255)
    ax.set_title(f"ringgrid viz: {gt['image_file']} ({gt['n_markers']} markers)")

    for m in gt["markers"]:
        cx, cy = m["true_image_center"]
        if not m["visible"]:
            continue

        # Draw cross at true center
        ax.plot(cx, cy, "+", color="red", markersize=6, markeredgewidth=0.8)

        # Label with ID
        ax.text(
            cx + 4, cy - 4, str(m["id"]),
            fontsize=4, color="yellow",
            ha="left", va="bottom",
        )

        # Draw ellipses
        if args.show_ellipses:
            draw_ellipse(ax, m["outer_ellipse"], color="lime", lw=0.5)
            draw_ellipse(ax, m["inner_ellipse"], color="cyan", lw=0.5)

    ax.set_xlim(0, gt["image_size"][0])
    ax.set_ylim(gt["image_size"][1], 0)
    ax.set_aspect("equal")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {args.out}")


if __name__ == "__main__":
    main()
