#!/usr/bin/env python3
"""Generate synthetic target images with ground truth for any ringgrid target.

Renders every combination the target model supports — lattice (hex | rect) x
coding (coded16 | plain) x origin dots — from a ``ringgrid.target.v6`` spec,
seen through a random homography with a full in-plane rotation. Reuses
gen_synth's homography/blur/illumination/noise pipeline; only the lattice, the
marker rendering and the ground truth are specific to this generator.

The rendering mirrors the library's printed target (``target_generation.rs``):

- plain markers are a filled annulus between the inner and outer radii;
- coded markers are two stroked rings (``ring_width_mm`` wide, centred on the
  radii) with a 16-sector code band between them, bright where the codeword bit
  is 1, IDs assigned in generation order (or from ``id_assignment``) and words
  read from the target's codebook profile in ``tools/codebook.json``;
- origin dots are filled disks at the positions the library derives.

Ground truth is keyed by lattice coordinate (u, v) — axial (q, r) on hex — and,
for coded targets, also carries the marker ID. The board frame matches
ringgrid's TargetLayout: cells in generation order (top row first), the first
cell at (0, 0) mm, +x right, +y down.

Without ``--target`` the geometry flags describe a plain rect board (the
original rect benchmark's shorthand).

Usage:
    python tools/gen_synth_target.py --out_dir out/rect --n_images 3
    python tools/gen_synth_target.py --out_dir out/rect --no_dots
    python tools/gen_synth_target.py --out_dir out/hex --target spec.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gen_synth import (
    apply_anisotropic_blur,
    apply_illumination_gradient,
    apply_noise,
    make_random_homography,
    project_point,
    write_png_gray,
)

TOOLS = Path(__file__).resolve().parent
CODEBOOK_JSON = TOOLS / "codebook.json"
TARGET_SCHEMA = "ringgrid.target.v6"
CODE_SECTORS = 16
PAPER = 0.85
INK = 0.1

# 24x24 plain rect defaults for the flag shorthand (pitch 14 mm, ring outer/inner
# diameter 11.2/5.6 mm, three Ø2.8 mm dots at cell-gap centers near center).
DEFAULT_ROWS = 24
DEFAULT_COLS = 24
DEFAULT_PITCH_MM = 14.0
DEFAULT_OUTER_RADIUS_MM = 5.6
DEFAULT_INNER_RADIUS_MM = 2.8
DEFAULT_DOT_RADIUS_MM = 1.4


# ── Lattice (mirrors ringgrid::target::lattice) ─────────────────────────


def center_index(n: int) -> int:
    """Index of the centre cell along an axis of ``n`` cells (mirrors Rust)."""
    return (n - 1) // 2


def generate_rect_lattice(rows: int, cols: int, pitch_mm: float):
    """Rect cells as (u, v, x_mm, y_mm), row-major.

    Coordinates are centred to match ringgrid's rect lattice: cell (0, 0) is the
    central cell, so a 24-wide board runs -11..=12.
    """
    cu, cv = center_index(cols), center_index(rows)
    return [
        (u - cu, v - cv, u * pitch_mm, v * pitch_mm)
        for v in range(rows)
        for u in range(cols)
    ]


def generate_hex_lattice(rows: int, long_row_cols: int, pitch_mm: float):
    """Hex cells as (q, r, x_mm, y_mm) in generation order (top row first).

    Mirrors ``generate_hex_cells``: pointy-top axial coordinates centred on the
    middle row, rows alternating between ``long_row_cols`` and one fewer, and
    positions normalized so the first generated cell sits at (0, 0) mm.
    """
    row_mid = rows // 2
    sqrt3 = math.sqrt(3.0)
    cells = []
    for row_idx in range(rows):
        r = row_idx - row_mid
        long_row = rows == 1 or ((r + long_row_cols - 1) & 1) == 0
        n_cols = long_row_cols if long_row else long_row_cols - 1
        # Rust integer division truncates toward zero.
        q_start = -int((r + n_cols - 1) / 2)
        for col_idx in range(n_cols):
            q = q_start + col_idx
            x = pitch_mm * (sqrt3 * q + 0.5 * sqrt3 * r)
            y = pitch_mm * (1.5 * r)
            cells.append((q, r, x, y))
    ax, ay = cells[0][2], cells[0][3]
    return [(q, r, x - ax, y - ay) for q, r, x, y in cells]


def generate_cells(lattice: dict):
    """Cells of a v6 ``lattice`` object, in the library's generation order."""
    if lattice["kind"] == "rect":
        return generate_rect_lattice(
            lattice["rows"], lattice["cols"], lattice["pitch_mm"]
        )
    if lattice["kind"] == "hex":
        return generate_hex_lattice(
            lattice["rows"], lattice["long_row_cols"], lattice["pitch_mm"]
        )
    raise ValueError(f"unknown lattice kind {lattice['kind']!r}")


# ── Origin dots (mirrors ringgrid::target::fiducials) ───────────────────


def rect_origin_dots_mm(rows: int, cols: int, pitch_mm: float):
    """Rect origin-dot triad: an L of three lattice gaps around cell (0, 0)."""
    ox = pitch_mm * center_index(cols)
    oy = pitch_mm * center_index(rows)
    h = 0.5 * pitch_mm
    ax, ay = ox + h, oy + h
    return [[ax, ay], [ax - pitch_mm, ay], [ax, ay + pitch_mm]]


def hex_origin_dots_mm(rows: int, long_row_cols: int, pitch_mm: float):
    """Hex origin-dot triad: the triangle holes at 30°, 90° and 150° around cell (0, 0)."""
    cells = generate_hex_lattice(rows, long_row_cols, pitch_mm)
    ox, oy = next((x, y) for q, r, x, y in cells if (q, r) == (0, 0))
    return [
        [ox + pitch_mm * math.cos(math.radians(deg)),
         oy + pitch_mm * math.sin(math.radians(deg))]
        for deg in (30.0, 90.0, 150.0)
    ]


def origin_dots_mm(lattice: dict):
    """Origin-dot positions of a v6 ``lattice`` object.

    Mirrors ``ringgrid::target::fiducials::origin_dot_positions_mm``. Kept in
    step by ``tools/tests/test_gen_synth_target.py``.
    """
    if lattice["kind"] == "rect":
        return rect_origin_dots_mm(
            lattice["rows"], lattice["cols"], lattice["pitch_mm"]
        )
    return hex_origin_dots_mm(
        lattice["rows"], lattice["long_row_cols"], lattice["pitch_mm"]
    )


# ── Coding ──────────────────────────────────────────────────────────────


def load_codebook(profile: str):
    """Codewords and base-table size for a codebook profile (base | extended)."""
    with open(CODEBOOK_JSON) as f:
        data = json.load(f)
    words = [int(w, 16) for w in data["profiles"][profile]["codewords"]]
    return words, data["profiles"]["base"]["n"]


def marker_ids(coding: dict, n_cells: int):
    """Codebook ID per cell: ``id_assignment`` when present, else sequential."""
    ids = coding.get("id_assignment") or list(range(n_cells))
    if len(ids) != n_cells:
        raise ValueError(f"id_assignment has {len(ids)} entries for {n_cells} cells")
    return ids


# ── Rendering ───────────────────────────────────────────────────────────


def projected_window(H, mx, my, radius_mm, img_w, img_h):
    """Pixel window containing the image of the disk (mx, my, radius_mm).

    A homography maps the disk's bounding square onto a convex quad, so the
    pixel bounds of its four projected corners contain every pixel whose
    pre-image lies in the disk. Returns None when the window is empty.
    """
    xs, ys = [], []
    for sx in (-radius_mm, radius_mm):
        for sy in (-radius_mm, radius_mm):
            px, py = project_point(H, mx + sx, my + sy)
            xs.append(px)
            ys.append(py)
    x0 = max(0, int(math.floor(min(xs))) - 1)
    x1 = min(img_w, int(math.ceil(max(xs))) + 2)
    y0 = max(0, int(math.floor(min(ys))) - 1)
    y1 = min(img_h, int(math.ceil(max(ys))) + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    return slice(y0, y1), slice(x0, x1)


def ink_marker(window, dist_sq, dx, dy, marker, codeword, ring_half_width_mm):
    """Ink one marker into ``window`` (a view of the image)."""
    ro, ri = marker["outer_radius_mm"], marker["inner_radius_mm"]
    if codeword is None:
        # Plain marker: filled annulus between the radii.
        window[(dist_sq >= ri**2) & (dist_sq <= ro**2)] = INK
        return

    h = ring_half_width_mm
    window[(dist_sq >= (ro - h) ** 2) & (dist_sq <= (ro + h) ** 2)] = INK
    window[(dist_sq >= (ri - h) ** 2) & (dist_sq <= (ri + h) ** 2)] = INK
    band = (dist_sq >= (ri + h) ** 2) & (dist_sq <= (ro - h) ** 2)
    if np.any(band):
        angle = np.arctan2(dy[band], dx[band])
        sector = np.floor((angle / (2.0 * math.pi) + 0.5) * CODE_SECTORS).astype(int)
        sector %= CODE_SECTORS
        bits = np.array([(codeword >> s) & 1 for s in range(CODE_SECTORS)])
        window[band] = np.where(bits[sector] == 1, PAPER, INK)


def render_board(img_w, img_h, cells, codewords, H, marker, ring_half_width_mm,
                 dots_mm, dot_radius_mm):
    """Backward-map every pixel through H^-1 and ink markers + dots."""
    img = np.ones((img_h, img_w), dtype=np.float64) * PAPER

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return img

    ys, xs = np.mgrid[0:img_h, 0:img_w]
    xs_flat = xs.ravel().astype(np.float64)
    ys_flat = ys.ravel().astype(np.float64)
    pts_board = H_inv @ np.vstack([xs_flat, ys_flat, np.ones_like(xs_flat)])
    bx = (pts_board[0] / pts_board[2]).reshape(img_h, img_w)
    by = (pts_board[1] / pts_board[2]).reshape(img_h, img_w)

    draw_radius_mm = marker["outer_radius_mm"] + ring_half_width_mm
    for (_u, _v, mx, my), codeword in zip(cells, codewords):
        win = projected_window(H, mx, my, draw_radius_mm, img_w, img_h)
        if win is None:
            continue
        dx = bx[win] - mx
        dy = by[win] - my
        ink_marker(img[win], dx**2 + dy**2, dx, dy, marker, codeword,
                   ring_half_width_mm)

    for dx_mm, dy_mm in dots_mm:
        dist_sq = (bx - dx_mm) ** 2 + (by - dy_mm) ** 2
        img[dist_sq <= dot_radius_mm**2] = INK

    return img


# ── Target spec ─────────────────────────────────────────────────────────


def plain_rect_spec(rows, cols, pitch_mm, outer_radius_mm, inner_radius_mm,
                    dot_radius_mm=None) -> dict:
    """A ringgrid.target.v6 plain rect spec (the flag shorthand's target)."""
    spec = {
        "schema": TARGET_SCHEMA,
        "name": "synth_rect_plain",
        "lattice": {
            "kind": "rect",
            "rows": rows,
            "cols": cols,
            "pitch_mm": pitch_mm,
        },
        "marker": {
            "outer_radius_mm": outer_radius_mm,
            "inner_radius_mm": inner_radius_mm,
        },
        "coding": {"kind": "plain"},
    }
    if dot_radius_mm is not None:
        # v6 stores only the size; positions derive from the lattice.
        spec["fiducials"] = {"dot_radius_mm": dot_radius_mm}
    return spec


class Board:
    """Everything the renderer and ground truth need, derived from a v6 spec."""

    def __init__(self, spec: dict):
        if spec.get("schema") != TARGET_SCHEMA:
            raise ValueError(f"expected a {TARGET_SCHEMA} spec, got {spec.get('schema')!r}")
        self.spec = spec
        self.lattice = spec["lattice"]
        self.marker = spec["marker"]
        self.coding = spec["coding"]
        self.cells = generate_cells(self.lattice)

        fiducials = spec.get("fiducials")
        self.dot_radius_mm = fiducials["dot_radius_mm"] if fiducials else None
        self.dots_mm = origin_dots_mm(self.lattice) if fiducials else []

        if self.coding["kind"] == "coded16":
            self.ring_half_width_mm = 0.5 * self.coding["ring_width_mm"]
            self.profile = self.coding.get("codebook_profile", "base")
            words, self.base_n = load_codebook(self.profile)
            self.ids = marker_ids(self.coding, len(self.cells))
            self.codewords = [words[i] for i in self.ids]
        elif self.coding["kind"] == "plain":
            self.ring_half_width_mm = 0.0
            self.ids = None
            self.codewords = [None] * len(self.cells)
        else:
            raise ValueError(f"unknown coding kind {self.coding['kind']!r}")

    @property
    def is_coded(self) -> bool:
        return self.ids is not None

    def bounds_mm(self):
        xs = [c[2] for c in self.cells]
        ys = [c[3] for c in self.cells]
        return min(xs), min(ys), max(xs), max(ys)


# ── Sample generation ───────────────────────────────────────────────────


def generate_one_sample(idx, out_dir: Path, args, board: Board,
                        rng: np.random.RandomState) -> dict:
    x0, y0, x1, y1 = board.bounds_mm()
    draw_radius_mm = board.marker["outer_radius_mm"] + board.ring_half_width_mm
    board_span = max(x1 - x0, y1 - y0) + 2.0 * draw_radius_mm

    # gen_synth's homography centers board coords around (0, 0); compose a
    # shift of the board center plus a full in-plane rotation for origin-
    # resolution coverage.
    H_base = make_random_homography(
        rng, args.img_w, args.img_h, board_span, tilt_strength=args.tilt_strength
    )
    theta = math.radians(rng.uniform(-args.rot_max_deg, args.rot_max_deg))
    ca, sa = math.cos(theta), math.sin(theta)
    cx_mm, cy_mm = (x0 + x1) / 2.0, (y0 + y1) / 2.0
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
    # Board frame (first cell at origin) -> image pixels.
    H = H_base @ H_center @ H_rot

    img = render_board(
        args.img_w,
        args.img_h,
        board.cells,
        board.codewords,
        H,
        board.marker,
        board.ring_half_width_mm,
        board.dots_mm,
        board.dot_radius_mm,
    )
    img = apply_anisotropic_blur(img, args.blur_px, rng)
    img = apply_illumination_gradient(img, rng, strength=args.illum_strength)
    img = apply_noise(img, rng, sigma=args.noise_sigma)

    img_name = f"img_{idx:04d}.png"
    write_png_gray(str(out_dir / img_name), img)

    gt_cells = []
    for i, (u, v, mx, my) in enumerate(board.cells):
        ix, iy = project_point(H, mx, my)
        cell = {
            "u": u,
            "v": v,
            "board_xy_mm": [mx, my],
            "true_image_center": [float(ix), float(iy)],
            "visible": (0 <= ix < args.img_w) and (0 <= iy < args.img_h),
        }
        if board.is_coded:
            cell["id"] = board.ids[i]
        gt_cells.append(cell)

    gt_dots = []
    for dx_mm, dy_mm in board.dots_mm:
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
        "lattice": {k: v for k, v in board.lattice.items() if k != "pitch_mm"},
        "pitch_mm": board.lattice["pitch_mm"],
        "outer_radius_mm": board.marker["outer_radius_mm"],
        "inner_radius_mm": board.marker["inner_radius_mm"],
        "dot_radius_mm": board.dot_radius_mm,
        "homography": H.tolist(),
        "rotation_deg": math.degrees(theta),
        "blur_px": args.blur_px,
        "noise_sigma": args.noise_sigma,
        "seed": args.seed + idx,
        "n_cells": len(board.cells),
        "cells": gt_cells,
        "dots": gt_dots,
    }
    if board.is_coded:
        gt["codebook_profile"] = board.profile
        gt["codebook_base_n"] = board.base_n
    with open(out_dir / f"gt_{idx:04d}.json", "w") as f:
        json.dump(gt, f, indent=2)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--target",
        help="ringgrid.target.v6 spec JSON to render (any lattice/coding/dots). "
        "When given, the plain-rect geometry flags below are not used.",
    )
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

    if args.target:
        if args.no_dots:
            ap.error("--no_dots does not apply to --target; edit the spec's fiducials")
        with open(args.target) as f:
            spec = json.load(f)
    else:
        spec = plain_rect_spec(
            args.rows,
            args.cols,
            args.pitch_mm,
            args.outer_radius_mm,
            args.inner_radius_mm,
            None if args.no_dots else args.dot_radius_mm,
        )
    board = Board(spec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "target_spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    rng = np.random.RandomState(args.seed)
    for idx in range(args.n_images):
        gt = generate_one_sample(idx, out_dir, args, board, rng)
        n_vis = sum(1 for c in gt["cells"] if c["visible"])
        print(
            f"[{idx + 1}/{args.n_images}] {gt['image_file']}: "
            f"{n_vis}/{gt['n_cells']} cells visible, rot {gt['rotation_deg']:.1f} deg"
        )


if __name__ == "__main__":
    main()
