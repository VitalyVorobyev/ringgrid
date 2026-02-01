#!/usr/bin/env python3
"""Generate synthetic ringgrid calibration target images with ground truth.

Renders a hex (triangular) lattice board with two-edge ring markers, each
carrying a 16-sector binary code band. Applies projective warp, anisotropic
Gaussian blur, illumination gradient, and noise to simulate Scheimpflug-like
imaging conditions.

Usage:
    python tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 10

Dependencies: numpy, matplotlib (for viz_debug.py), json (stdlib).
"""

import argparse
import json
import math
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Optional

import numpy as np


# ── Hex lattice utilities ───────────────────────────────────────────────

def hex_axial_to_xy(q: int, r: int, pitch: float) -> tuple[float, float]:
    """Convert axial hex coordinates (q, r) to Cartesian (x, y) in mm.

    Uses "pointy-top" orientation:
      x = pitch * (sqrt(3) * q + sqrt(3)/2 * r)
      y = pitch * (3/2 * r)
    """
    x = pitch * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = pitch * (1.5 * r)
    return x, y


def generate_hex_lattice(
    board_mm: float,
    pitch_mm: float,
    n_markers: Optional[int] = None,
) -> list[tuple[int, int, float, float]]:
    """Generate hex lattice points within a board_mm x board_mm area.

    Returns list of (q, r, x_mm, y_mm) tuples, centered around (0, 0).
    If n_markers is set, trim to roughly that many.
    """
    half = board_mm / 2.0
    # Generous range for q, r
    max_coord = int(half / pitch_mm) + 2
    points = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            x, y = hex_axial_to_xy(q, r, pitch_mm)
            margin = pitch_mm * 0.6  # keep markers away from edge
            if abs(x) < half - margin and abs(y) < half - margin:
                points.append((q, r, x, y))

    # Sort by (r, q) for stable ordering
    points.sort(key=lambda p: (p[1], p[0]))

    if n_markers is not None and len(points) > n_markers:
        # Keep the n_markers closest to center
        points.sort(key=lambda p: p[2] ** 2 + p[3] ** 2)
        points = points[:n_markers]
        points.sort(key=lambda p: (p[1], p[0]))

    return points


# ── Codebook loading ────────────────────────────────────────────────────

def load_codebook(path: str) -> list[int]:
    """Load codebook from JSON, return list of integer codewords."""
    with open(path) as f:
        data = json.load(f)
    return [int(s, 16) for s in data["codewords"]]


# ── Rendering ───────────────────────────────────────────────────────────

def make_random_homography(
    rng: np.random.RandomState,
    img_w: int,
    img_h: int,
    board_mm: float,
    tilt_strength: float = 0.3,
) -> np.ndarray:
    """Create a projective homography mapping board coords (mm) to image pixels.

    Composes: centering + scale + mild perspective tilt.
    """
    # Scale so board fills ~70% of image
    scale = 0.7 * min(img_w, img_h) / board_mm

    # Base affine: center board in image
    H_base = np.array([
        [scale, 0, img_w / 2.0],
        [0, scale, img_h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)

    # Small projective perturbation
    p1 = rng.uniform(-tilt_strength, tilt_strength) * 1e-3
    p2 = rng.uniform(-tilt_strength, tilt_strength) * 1e-3
    H_proj = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [p1, p2, 1],
    ], dtype=np.float64)

    # Small rotation
    angle = rng.uniform(-0.15, 0.15)
    ca, sa = math.cos(angle), math.sin(angle)
    H_rot = np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return H_base @ H_proj @ H_rot


def project_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply homography H to point (x, y)."""
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def project_ellipse_params(
    H: np.ndarray,
    cx: float, cy: float,
    radius: float,
) -> dict:
    """Project a circle (center, radius) through homography H.

    Returns approximate ellipse parameters in image coords.
    Uses sample-based approach: project points on the circle, fit ellipse.
    """
    n = 64
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_board = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.ones(n),
    ])
    pts_img = (H @ pts_board.T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]

    # Fit ellipse via direct conic fit
    return fit_ellipse_from_points(pts_img)


def fit_ellipse_from_points(pts: np.ndarray) -> dict:
    """Direct least-squares ellipse fit from 2D points.

    Returns dict with cx, cy, a, b, angle.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])

    S = D.T @ D
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=np.float64)

    S11 = S[:3, :3]
    S12 = S[:3, 3:]
    S22 = S[3:, 3:]

    try:
        S22_inv = np.linalg.inv(S22)
    except np.linalg.LinAlgError:
        cx = float(np.mean(x))
        cy = float(np.mean(y))
        return {"cx": cx, "cy": cy, "a": 1.0, "b": 1.0, "angle": 0.0}

    M = S11 - S12 @ S22_inv @ S12.T
    C1_inv = np.linalg.inv(C1)
    system = C1_inv @ M

    eigvals, eigvecs = np.linalg.eig(system)
    # Find eigenvector satisfying ellipse constraint
    best_idx = None
    for i in range(3):
        v = eigvecs[:, i].real
        constraint = 4 * v[0] * v[2] - v[1] ** 2
        if constraint > 0:
            if best_idx is None or abs(eigvals[i].real) < abs(eigvals[best_idx].real):
                best_idx = i

    if best_idx is None:
        cx = float(np.mean(x))
        cy = float(np.mean(y))
        return {"cx": cx, "cy": cy, "a": 1.0, "b": 1.0, "angle": 0.0}

    a1 = eigvecs[:, best_idx].real
    a2 = -S22_inv @ S12.T @ a1
    coeffs = np.concatenate([a1, a2])

    A, B, C, D_c, E, F = coeffs
    denom = B**2 - 4*A*C
    if abs(denom) < 1e-15:
        return {"cx": float(np.mean(x)), "cy": float(np.mean(y)), "a": 1.0, "b": 1.0, "angle": 0.0}

    cx_e = (2*C*D_c - B*E) / (-denom)
    cy_e = (2*A*E - B*D_c) / (-denom)

    # Rotation angle
    if abs(A - C) < 1e-15:
        angle = math.pi / 4 if B > 0 else (-math.pi / 4 if B < 0 else 0)
    else:
        angle = 0.5 * math.atan2(B, A - C)

    # Semi-axes
    sum_ac = A + C
    diff = math.sqrt((A - C)**2 + B**2)
    lam1 = (sum_ac + diff) / 2
    lam2 = (sum_ac - diff) / 2
    f_prime = A*cx_e**2 + B*cx_e*cy_e + C*cy_e**2 + D_c*cx_e + E*cy_e + F

    if abs(f_prime) < 1e-15 or lam1 == 0 or lam2 == 0:
        return {"cx": cx_e, "cy": cy_e, "a": 1.0, "b": 1.0, "angle": angle}

    a_sq = -f_prime / lam1
    b_sq = -f_prime / lam2
    if a_sq <= 0 or b_sq <= 0:
        return {"cx": cx_e, "cy": cy_e, "a": 1.0, "b": 1.0, "angle": angle}

    semi_a = math.sqrt(a_sq)
    semi_b = math.sqrt(b_sq)
    if semi_a < semi_b:
        semi_a, semi_b = semi_b, semi_a
        angle += math.pi / 2

    # Normalize angle
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi

    return {"cx": float(cx_e), "cy": float(cy_e), "a": float(semi_a), "b": float(semi_b), "angle": float(angle)}


def render_board(
    img_w: int,
    img_h: int,
    markers: list[tuple[int, int, float, float]],
    codebook: list[int],
    H: np.ndarray,
    outer_radius_mm: float,
    inner_radius_mm: float,
    code_band_outer_mm: float,
    code_band_inner_mm: float,
    stress_inner_confusion: bool = False,
) -> np.ndarray:
    """Render the board into a float64 image [0, 1].

    For each pixel, determine if it falls inside any marker's rings or code band
    by inverse-projecting through H.
    """
    img = np.ones((img_h, img_w), dtype=np.float64) * 0.85  # light gray background

    # Inverse homography for backward mapping
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return img

    # Precompute marker info
    marker_info = []
    for idx, (q, r, mx, my) in enumerate(markers):
        code_id = idx % len(codebook)
        cw = codebook[code_id]
        # Project center to get approximate image location for culling
        ix, iy = project_point(H, mx, my)
        marker_info.append((mx, my, cw, ix, iy, code_id))

    # Vectorized rendering: process all pixels
    ys, xs = np.mgrid[0:img_h, 0:img_w]
    xs_flat = xs.ravel().astype(np.float64)
    ys_flat = ys.ravel().astype(np.float64)
    ones = np.ones_like(xs_flat)

    # Inverse project all pixels to board coords
    pts_img = np.vstack([xs_flat, ys_flat, ones])  # 3 x N
    pts_board = H_inv @ pts_img  # 3 x N
    bx = pts_board[0] / pts_board[2]
    by = pts_board[1] / pts_board[2]

    img_flat = img.ravel().copy()

    for mx, my, cw, ix, iy, code_id in marker_info:
        # Rough culling: skip markers far from image
        if ix < -100 or ix > img_w + 100 or iy < -100 or iy > img_h + 100:
            continue

        dx = bx - mx
        dy = by - my
        dist_sq = dx * dx + dy * dy

        # Outer ring edge: dark ring at outer_radius
        ring_width = outer_radius_mm * (0.16 if stress_inner_confusion else 0.12)
        outer_mask = (dist_sq >= (outer_radius_mm - ring_width) ** 2) & \
                     (dist_sq <= (outer_radius_mm + ring_width) ** 2)
        img_flat[outer_mask] = 0.1  # dark

        # Inner ring edge: dark ring at inner_radius
        inner_mask = (dist_sq >= (inner_radius_mm - ring_width) ** 2) & \
                     (dist_sq <= (inner_radius_mm + ring_width) ** 2)
        img_flat[inner_mask] = 0.1  # dark

        # Code band: between code_band_inner and code_band_outer
        code_mask = (dist_sq >= code_band_inner_mm ** 2) & \
                    (dist_sq <= code_band_outer_mm ** 2)

        if np.any(code_mask):
            # Compute angle for code band pixels
            angles = np.arctan2(dy[code_mask], dx[code_mask])  # [-pi, pi]
            # Map to sector index [0, 15]
            sector = ((angles / (2 * math.pi) + 0.5) * 16).astype(int) % 16
            # Look up code bits
            bits = np.array([(cw >> s) & 1 for s in range(16)])
            if stress_inner_confusion:
                pixel_vals = np.where(bits[sector] == 1, 1.0, 0.0)
            else:
                pixel_vals = np.where(bits[sector] == 1, 0.9, 0.15)
            img_flat[code_mask] = pixel_vals

    return img_flat.reshape(img_h, img_w)


def apply_anisotropic_blur(
    img: np.ndarray,
    blur_px: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply anisotropic Gaussian blur with random orientation."""
    if blur_px < 0.5:
        return img

    # Random anisotropy: sigma_x != sigma_y
    ratio = rng.uniform(0.5, 2.0)
    sigma_x = blur_px * math.sqrt(ratio)
    sigma_y = blur_px / math.sqrt(ratio)
    angle = rng.uniform(0, math.pi)

    # Build rotated 2D Gaussian kernel
    ksize = int(6 * max(sigma_x, sigma_y)) | 1  # odd
    ksize = max(ksize, 3)
    half = ksize // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)

    # Rotate coordinates
    ca, sa = math.cos(angle), math.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y

    kernel = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2))
    kernel /= kernel.sum()

    # Apply via FFT convolution (no scipy needed)
    return fft_convolve2d(img, kernel)


def fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution via FFT (zero-padded)."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    # Pad to avoid wrap-around
    fh = ih + kh - 1
    fw = iw + kw - 1
    # Optimal FFT size (power of 2)
    fh2 = 1 << int(math.ceil(math.log2(fh)))
    fw2 = 1 << int(math.ceil(math.log2(fw)))

    img_fft = np.fft.rfft2(img, s=(fh2, fw2))
    ker_fft = np.fft.rfft2(kernel, s=(fh2, fw2))
    result = np.fft.irfft2(img_fft * ker_fft, s=(fh2, fw2))

    # Crop to original size, centered
    pad_h = kh // 2
    pad_w = kw // 2
    return result[pad_h:pad_h + ih, pad_w:pad_w + iw]


def apply_illumination_gradient(
    img: np.ndarray,
    rng: np.random.RandomState,
    strength: float = 0.15,
) -> np.ndarray:
    """Apply a mild linear illumination gradient."""
    h, w = img.shape
    gx = rng.uniform(-strength, strength)
    gy = rng.uniform(-strength, strength)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    x = x / w - 0.5
    y = y / h - 0.5
    gradient = 1.0 + gx * x + gy * y
    return np.clip(img * gradient, 0, 1)


def apply_noise(
    img: np.ndarray,
    rng: np.random.RandomState,
    sigma: float = 0.02,
) -> np.ndarray:
    """Add Gaussian noise."""
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 1)


# ── PNG writing (no PIL dependency) ─────────────────────────────────────

def write_png_gray(path: str, img: np.ndarray) -> None:
    """Write a grayscale float64 image [0, 1] as 8-bit PNG."""
    h, w = img.shape
    data = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Build raw PNG data
    raw = b""
    for row in range(h):
        raw += b"\x00"  # filter: none
        raw += data[row].tobytes()

    def chunk(ctype: bytes, cdata: bytes) -> bytes:
        c = ctype + cdata
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(cdata)) + c + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    compressed = zlib.compress(raw, 6)

    with open(path, "wb") as f:
        f.write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b""))


# ── Main generator ──────────────────────────────────────────────────────

def generate_one_sample(
    idx: int,
    out_dir: Path,
    img_w: int,
    img_h: int,
    board_mm: float,
    pitch_mm: float,
    n_markers: Optional[int],
    codebook: list[int],
    blur_px: float,
    projective: bool,
    tilt_strength: float,
    seed: int,
    stress_inner_confusion: bool = False,
) -> dict:
    """Generate one synthetic image + ground truth."""
    rng = np.random.RandomState(seed + idx)

    markers = generate_hex_lattice(board_mm, pitch_mm, n_markers)

    # Marker geometry (in mm)
    outer_radius = pitch_mm * 0.6
    inner_radius = pitch_mm * 0.4
    code_band_outer = pitch_mm * 0.58
    code_band_inner = pitch_mm * 0.42

    tilt = float(tilt_strength) if projective else 0.0
    H = make_random_homography(rng, img_w, img_h, board_mm, tilt_strength=tilt)

    # Render
    img = render_board(
        img_w, img_h, markers, codebook, H,
        outer_radius, inner_radius, code_band_outer, code_band_inner,
        stress_inner_confusion=stress_inner_confusion,
    )

    # Post-processing
    img = apply_anisotropic_blur(img, blur_px, rng)
    img = apply_illumination_gradient(img, rng)
    img = apply_noise(img, rng)

    # Crop (optional random crop to simulate partial visibility)
    # For now, just keep full image

    # Save image
    img_name = f"img_{idx:04d}.png"
    img_path = out_dir / img_name
    write_png_gray(str(img_path), img)

    # Build ground truth
    gt_markers = []
    for i, (q, r, mx, my) in enumerate(markers):
        code_id = i % len(codebook)
        # True projected center
        ix, iy = project_point(H, mx, my)

        # Projected outer ellipse
        outer_ell = project_ellipse_params(H, mx, my, outer_radius)
        inner_ell = project_ellipse_params(H, mx, my, inner_radius)

        visible = (0 <= ix < img_w) and (0 <= iy < img_h)

        gt_markers.append({
            "id": code_id,
            "q": q,
            "r": r,
            "board_xy_mm": [float(mx), float(my)],
            "true_image_center": [float(ix), float(iy)],
            "outer_ellipse": outer_ell,
            "inner_ellipse": inner_ell,
            "visible": visible,
        })

    gt = {
        "image_file": img_name,
        "image_size": [img_w, img_h],
        "board_mm": board_mm,
        "pitch_mm": pitch_mm,
        "outer_radius_mm": outer_radius,
        "inner_radius_mm": inner_radius,
        "homography": H.tolist(),
        "blur_px": blur_px,
        "projective": projective,
        "tilt_strength": tilt,
        "stress_inner_confusion": stress_inner_confusion,
        "seed": seed + idx,
        "n_markers": len(markers),
        "markers": gt_markers,
        "codebook_ref": {
            "bits": 16,
            "n": len(codebook),
        },
    }

    gt_name = f"gt_{idx:04d}.json"
    gt_path = out_dir / gt_name
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ringgrid datasets")
    parser.add_argument("--out_dir", type=str, default="tools/out/synth_001")
    parser.add_argument("--n_images", type=int, default=3)
    parser.add_argument("--img_w", type=int, default=1280)
    parser.add_argument("--img_h", type=int, default=960)
    parser.add_argument("--board_mm", type=float, default=200.0)
    parser.add_argument("--pitch_mm", type=float, default=8.0)
    parser.add_argument("--n_markers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blur_px", type=float, default=1.0)
    parser.add_argument(
        "--stress-inner-confusion",
        action="store_true",
        help="Stress case: increase code-band contrast and edge thickness to confuse inner edge sampling",
    )
    parser.add_argument("--projective", action="store_true", default=True)
    parser.add_argument("--no_projective", dest="projective", action="store_false")
    parser.add_argument(
        "--tilt_strength",
        type=float,
        default=0.3,
        help=(
            "Projective tilt strength used by the homography sampler (0 disables perspective). "
            "Larger values produce stronger projective distortion."
        ),
    )
    parser.add_argument("--codebook", type=str, default="tools/codebook.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook = load_codebook(args.codebook)
    print(f"Loaded codebook: {len(codebook)} codewords")

    lattice = generate_hex_lattice(args.board_mm, args.pitch_mm, args.n_markers)
    print(f"Hex lattice: {len(lattice)} markers (pitch={args.pitch_mm}mm, board={args.board_mm}mm)")

    # Emit board_spec.json alongside generated images
    board_spec = {
        "name": f"ringgrid_{int(args.board_mm)}mm_hex",
        "board_size_mm": [args.board_mm, args.board_mm],
        "pitch_mm": args.pitch_mm,
        "origin_mm": [0.0, 0.0],
        "n_markers": len(lattice),
        "markers": [
            {"id": i, "q": q, "r": r, "xy_mm": [round(x, 4), round(y, 4)]}
            for i, (q, r, x, y) in enumerate(lattice)
        ],
    }
    board_spec_path = out_dir / "board_spec.json"
    with open(board_spec_path, "w") as f:
        json.dump(board_spec, f, indent=2)
    print(f"Board spec written to {board_spec_path}")

    for i in range(args.n_images):
        gt = generate_one_sample(
            idx=i,
            out_dir=out_dir,
            img_w=args.img_w,
            img_h=args.img_h,
            board_mm=args.board_mm,
            pitch_mm=args.pitch_mm,
            n_markers=args.n_markers,
            codebook=codebook,
            blur_px=args.blur_px,
            projective=args.projective,
            tilt_strength=args.tilt_strength,
            seed=args.seed,
            stress_inner_confusion=args.stress_inner_confusion,
        )
        vis = sum(1 for m in gt["markers"] if m["visible"])
        print(f"  [{i+1}/{args.n_images}] {gt['image_file']}: "
              f"{gt['n_markers']} markers ({vis} visible)")

    print(f"\nDataset written to {out_dir}/")


if __name__ == "__main__":
    main()
