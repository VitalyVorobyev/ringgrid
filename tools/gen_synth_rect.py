#!/usr/bin/env python3
"""Generate synthetic rect-lattice plain-ring target images with ground truth.

Rect counterpart of gen_synth.py for the ISRA-style target: a rows x cols grid
of plain (uncoded) filled annuli at uniform pitch, with optional origin dots at
cell-gap centers. Reuses gen_synth's homography/blur/illumination/noise
pipeline; only the lattice, the annulus rendering, and the coordinate-keyed
ground truth are specific to this generator.

Ground truth is keyed by lattice coordinate (u, v) — plain rings carry no IDs.
The board frame matches ringgrid's TargetLayout: cell (0, 0) at (0, 0) mm,
+u toward +x, +v toward +y.

Usage:
    python tools/gen_synth_rect.py --out_dir tools/out/synth_rect --n_images 3
    python tools/gen_synth_rect.py --out_dir ... --no_dots   # unresolvable origin
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from gen_synth import (
    apply_anisotropic_blur,
    apply_illumination_gradient,
    apply_noise,
    make_random_homography,
    project_point,
    write_png_gray,
)

# ISRA drawing 5256-57-102 defaults (24x24, pitch 14 mm, ring outer/inner
# diameter 11.2/5.6 mm, three Ø2.8 mm dots at cell-gap centers near center).
DEFAULT_ROWS = 24
DEFAULT_COLS = 24
DEFAULT_PITCH_MM = 14.0
DEFAULT_OUTER_RADIUS_MM = 5.6
DEFAULT_INNER_RADIUS_MM = 2.8
DEFAULT_DOT_RADIUS_MM = 1.4
DEFAULT_DOTS_MM = [[161.0, 161.0], [147.0, 161.0], [161.0, 175.0]]


def generate_rect_lattice(rows: int, cols: int, pitch_mm: float):
    """Cells as (u, v, x_mm, y_mm), row-major, cell (0, 0) at the origin."""
    return [
        (u, v, u * pitch_mm, v * pitch_mm) for v in range(rows) for u in range(cols)
    ]


def render_rect_board(
    img_w: int,
    img_h: int,
    cells,
    H: np.ndarray,
    outer_radius_mm: float,
    inner_radius_mm: float,
    dots_mm,
    dot_radius_mm: float,
) -> np.ndarray:
    """Backward-map every pixel through H^-1 and ink annuli + dots."""
    img = np.ones((img_h, img_w), dtype=np.float64) * 0.85

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return img

    ys, xs = np.mgrid[0:img_h, 0:img_w]
    xs_flat = xs.ravel().astype(np.float64)
    ys_flat = ys.ravel().astype(np.float64)
    pts_board = H_inv @ np.vstack([xs_flat, ys_flat, np.ones_like(xs_flat)])
    bx = pts_board[0] / pts_board[2]
    by = pts_board[1] / pts_board[2]

    img_flat = img.ravel()
    for _u, _v, mx, my in cells:
        ix, iy = project_point(H, mx, my)
        if ix < -100 or ix > img_w + 100 or iy < -100 or iy > img_h + 100:
            continue
        dist_sq = (bx - mx) ** 2 + (by - my) ** 2
        mask = (dist_sq >= inner_radius_mm**2) & (dist_sq <= outer_radius_mm**2)
        img_flat[mask] = 0.1

    for dx_mm, dy_mm in dots_mm:
        dist_sq = (bx - dx_mm) ** 2 + (by - dy_mm) ** 2
        img_flat[dist_sq <= dot_radius_mm**2] = 0.1

    return img_flat.reshape(img_h, img_w)


def target_spec_v5(args, dots_mm) -> dict:
    """Emit the ringgrid.target.v5 spec matching this synthetic board."""
    spec = {
        "schema": "ringgrid.target.v5",
        "name": "synth_rect_plain",
        "lattice": {
            "kind": "rect",
            "rows": args.rows,
            "cols": args.cols,
            "pitch_mm": args.pitch_mm,
        },
        "marker": {
            "outer_radius_mm": args.outer_radius_mm,
            "inner_radius_mm": args.inner_radius_mm,
        },
        "coding": {"kind": "plain"},
    }
    if dots_mm:
        spec["fiducials"] = {
            "dot_radius_mm": args.dot_radius_mm,
            "dots_mm": dots_mm,
        }
    return spec


def generate_one_sample(
    idx: int,
    out_dir: Path,
    args,
    dots_mm,
    rng: np.random.RandomState,
) -> dict:
    cells = generate_rect_lattice(args.rows, args.cols, args.pitch_mm)
    board_w = (args.cols - 1) * args.pitch_mm
    board_h = (args.rows - 1) * args.pitch_mm
    board_span = max(board_w, board_h) + 2.0 * args.outer_radius_mm

    # gen_synth's homography centers board coords around (0, 0); compose a
    # shift of the board center plus a full in-plane rotation for origin-
    # resolution coverage.
    H_base = make_random_homography(
        rng, args.img_w, args.img_h, board_span, tilt_strength=args.tilt_strength
    )
    theta = math.radians(rng.uniform(-args.rot_max_deg, args.rot_max_deg))
    ca, sa = math.cos(theta), math.sin(theta)
    cx_mm, cy_mm = board_w / 2.0, board_h / 2.0
    # Rotate about the board center, then re-center at the board frame origin.
    H_rot = np.array(
        [
            [ca, -sa, cx_mm - ca * cx_mm + sa * cy_mm],
            [sa, ca, cy_mm - sa * cx_mm - ca * cy_mm],
            [0.0, 0.0, 1.0],
        ]
    )
    H_center = np.array(
        [[1.0, 0.0, -cx_mm], [0.0, 1.0, -cy_mm], [0.0, 0.0, 1.0]]
    )
    # Board frame (cell (0,0) at origin) -> image pixels.
    H = H_base @ H_center @ H_rot

    img = render_rect_board(
        args.img_w,
        args.img_h,
        cells,
        H,
        args.outer_radius_mm,
        args.inner_radius_mm,
        dots_mm,
        args.dot_radius_mm,
    )
    img = apply_anisotropic_blur(img, args.blur_px, rng)
    img = apply_illumination_gradient(img, rng, strength=args.illum_strength)
    img = apply_noise(img, rng, sigma=args.noise_sigma)

    img_name = f"img_{idx:04d}.png"
    write_png_gray(str(out_dir / img_name), img)

    gt_cells = []
    for u, v, mx, my in cells:
        ix, iy = project_point(H, mx, my)
        visible = (0 <= ix < args.img_w) and (0 <= iy < args.img_h)
        gt_cells.append(
            {
                "u": u,
                "v": v,
                "board_xy_mm": [mx, my],
                "true_image_center": [float(ix), float(iy)],
                "visible": visible,
            }
        )

    gt_dots = []
    for dx_mm, dy_mm in dots_mm:
        ix, iy = project_point(H, dx_mm, dy_mm)
        gt_dots.append(
            {
                "board_xy_mm": [dx_mm, dy_mm],
                "true_image_center": [float(ix), float(iy)],
                "visible": (0 <= ix < args.img_w) and (0 <= iy < args.img_h),
            }
        )

    gt = {
        "image_file": img_name,
        "image_size": [args.img_w, args.img_h],
        "lattice": {"kind": "rect", "rows": args.rows, "cols": args.cols},
        "pitch_mm": args.pitch_mm,
        "outer_radius_mm": args.outer_radius_mm,
        "inner_radius_mm": args.inner_radius_mm,
        "dot_radius_mm": args.dot_radius_mm if dots_mm else None,
        "homography": H.tolist(),
        "rotation_deg": math.degrees(theta),
        "blur_px": args.blur_px,
        "noise_sigma": args.noise_sigma,
        "seed": args.seed + idx,
        "n_cells": len(cells),
        "cells": gt_cells,
        "dots": gt_dots,
    }
    with open(out_dir / f"gt_{idx:04d}.json", "w") as f:
        json.dump(gt, f, indent=2)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_images", type=int, default=3)
    ap.add_argument("--img_w", type=int, default=1100)
    ap.add_argument("--img_h", type=int, default=1100)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS)
    ap.add_argument("--pitch_mm", type=float, default=DEFAULT_PITCH_MM)
    ap.add_argument("--outer_radius_mm", type=float, default=DEFAULT_OUTER_RADIUS_MM)
    ap.add_argument("--inner_radius_mm", type=float, default=DEFAULT_INNER_RADIUS_MM)
    ap.add_argument("--dot_radius_mm", type=float, default=DEFAULT_DOT_RADIUS_MM)
    ap.add_argument(
        "--no_dots",
        action="store_true",
        help="Omit origin dots (origin stays unresolvable).",
    )
    ap.add_argument("--blur_px", type=float, default=0.8)
    ap.add_argument("--illum_strength", type=float, default=0.15)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--tilt_strength", type=float, default=0.3)
    ap.add_argument(
        "--rot_max_deg",
        type=float,
        default=180.0,
        help="Uniform in-plane rotation range (± degrees).",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dots_mm = [] if args.no_dots else [list(d) for d in DEFAULT_DOTS_MM]
    if dots_mm and (args.rows, args.cols, args.pitch_mm) != (24, 24, 14.0):
        # Keep the L-shape at the board center for non-default dimensions.
        cu = (args.cols - 1) // 2
        cv = (args.rows - 1) // 2
        gx = (cu + 0.5) * args.pitch_mm
        gy = (cv + 0.5) * args.pitch_mm
        dots_mm = [
            [gx, gy],
            [gx - args.pitch_mm, gy],
            [gx, gy + args.pitch_mm],
        ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "target_spec.json", "w") as f:
        json.dump(target_spec_v5(args, dots_mm), f, indent=2)

    rng = np.random.RandomState(args.seed)
    for idx in range(args.n_images):
        gt = generate_one_sample(idx, out_dir, args, dots_mm, rng)
        n_vis = sum(1 for c in gt["cells"] if c["visible"])
        print(
            f"[{idx + 1}/{args.n_images}] {gt['image_file']}: "
            f"{n_vis}/{gt['n_cells']} cells visible, rot {gt['rotation_deg']:.1f} deg"
        )


if __name__ == "__main__":
    main()
