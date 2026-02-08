#!/usr/bin/env python3
"""Generate synthetic ringgrid calibration target images with ground truth.

Renders a hex (triangular) lattice board with two-edge ring markers, each
carrying a 16-sector binary code band. Applies projective warp, anisotropic
Gaussian blur, illumination gradient, and noise to simulate Scheimpflug-like
imaging conditions.

Usage:
    python tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 10
    # Also emits ready-to-print calibration targets:
    python tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 0 --print_png --print_svg  # includes scale bar

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


def infer_parametric_layout(markers: list[tuple[int, int, float, float]]) -> tuple[int, int]:
    """Infer (rows, long_row_cols) from an explicit (q, r, x, y) marker list.

    Raises ValueError when the marker set is not representable by the runtime
    target schema (`rows + long_row_cols` alternating row lengths).
    """
    if not markers:
        raise ValueError("empty marker set")

    rows_to_q: dict[int, list[int]] = {}
    for q, r, _x, _y in markers:
        rows_to_q.setdefault(r, []).append(q)

    row_keys = sorted(rows_to_q.keys())
    for i in range(1, len(row_keys)):
        if row_keys[i] != row_keys[i - 1] + 1:
            raise ValueError("row indices are not contiguous")

    row_counts = [len(rows_to_q[r]) for r in row_keys]
    long_row_cols = max(row_counts)
    short_row_cols = long_row_cols - 1

    if len(row_keys) > 1 and short_row_cols < 1:
        raise ValueError("need long_row_cols >= 2 for multi-row board")

    for r in row_keys:
        qs = sorted(rows_to_q[r])
        if any(qs[i] != qs[i - 1] + 1 for i in range(1, len(qs))):
            raise ValueError(f"row r={r} has non-contiguous q values")

        expected_cols = (
            long_row_cols
            if len(row_keys) == 1 or ((r + long_row_cols - 1) & 1) == 0
            else short_row_cols
        )
        if len(qs) != expected_cols:
            raise ValueError(
                f"row r={r} has {len(qs)} markers; expected {expected_cols}"
            )

    return len(row_keys), long_row_cols


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


def project_circle_points(
    H: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n: int = 64,
) -> np.ndarray:
    """Project sampled circle points through homography H into working coordinates."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_board = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.ones(n),
    ])
    pts_img = (H @ pts_board.T).T
    return pts_img[:, :2] / pts_img[:, 2:3]


def camera_has_distortion(camera: Optional[dict]) -> bool:
    if not camera:
        return False
    d = camera["distortion"]
    return any(abs(float(d[k])) > 1e-15 for k in ("k1", "k2", "p1", "p2", "k3"))


def distort_normalized(
    x: np.ndarray | float,
    y: np.ndarray | float,
    distortion: dict[str, float],
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Apply Brown-Conrady radial-tangential distortion in normalized coordinates."""
    k1 = float(distortion["k1"])
    k2 = float(distortion["k2"])
    p1 = float(distortion["p1"])
    p2 = float(distortion["p2"])
    k3 = float(distortion["k3"])

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x * radial + x_tan, y * radial + y_tan


def distort_pixel(camera: dict, x_u: float, y_u: float) -> Optional[tuple[float, float]]:
    """Map working (undistorted) pixel coordinates to distorted image pixel coordinates."""
    k = camera["intrinsics"]
    fx = float(k["fx"])
    fy = float(k["fy"])
    cx = float(k["cx"])
    cy = float(k["cy"])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        return None

    xn = (x_u - cx) / fx
    yn = (y_u - cy) / fy
    xd, yd = distort_normalized(xn, yn, camera["distortion"])
    x_d = fx * xd + cx
    y_d = fy * yd + cy
    if not (math.isfinite(x_d) and math.isfinite(y_d)):
        return None
    return float(x_d), float(y_d)


def undistort_pixel(
    camera: dict,
    x_d: np.ndarray,
    y_d: np.ndarray,
    max_iters: int = 15,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert radial-tangential distortion by fixed-point iteration."""
    k = camera["intrinsics"]
    fx = float(k["fx"])
    fy = float(k["fy"])
    cx = float(k["cx"])
    cy = float(k["cy"])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("invalid camera intrinsics for undistortion")

    xd = (x_d - cx) / fx
    yd = (y_d - cy) / fy
    x = xd.copy()
    y = yd.copy()

    d = camera["distortion"]
    k1 = float(d["k1"])
    k2 = float(d["k2"])
    p1 = float(d["p1"])
    p2 = float(d["p2"])
    k3 = float(d["k3"])

    for _ in range(max(1, int(max_iters))):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial = np.where(np.abs(radial) < 1e-12, np.nan, radial)

        dx_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x_next = (xd - dx_tan) / radial
        y_next = (yd - dy_tan) / radial

        diff = np.nanmax((x_next - x) ** 2 + (y_next - y) ** 2)
        x = x_next
        y = y_next
        if not np.isfinite(diff) or math.sqrt(float(diff)) <= max(0.0, eps):
            break

    x_u = fx * x + cx
    y_u = fy * y + cy
    return x_u, y_u


def bilinear_sample_gray(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample grayscale image at float coordinates; out-of-bounds -> 0."""
    h, w = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    inb = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.zeros_like(x, dtype=np.float64)
    if not np.any(inb):
        return out

    fx = x - x0
    fy = y - y0
    idx = np.where(inb)[0]

    p00 = img[y0[idx], x0[idx]]
    p10 = img[y0[idx], x1[idx]]
    p01 = img[y1[idx], x0[idx]]
    p11 = img[y1[idx], x1[idx]]

    out[idx] = (
        (1.0 - fx[idx]) * (1.0 - fy[idx]) * p00
        + fx[idx] * (1.0 - fy[idx]) * p10
        + (1.0 - fx[idx]) * fy[idx] * p01
        + fx[idx] * fy[idx] * p11
    )
    return out


def apply_radial_tangential_distortion(img_working: np.ndarray, camera: Optional[dict]) -> np.ndarray:
    """Warp working-frame image into distorted image frame using camera model."""
    if not camera or not camera_has_distortion(camera):
        return img_working

    h, w = img_working.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xd = xx.ravel()
    yd = yy.ravel()
    xu, yu = undistort_pixel(camera, xd, yd)
    vals = bilinear_sample_gray(img_working, xu, yu)
    return vals.reshape(h, w)


def project_ellipse_params(
    H: np.ndarray,
    cx: float, cy: float,
    radius: float,
) -> dict:
    """Project a circle (center, radius) through homography H.

    Returns approximate ellipse parameters in image coords.
    Uses sample-based approach: project points on the circle, fit ellipse.
    """
    pts_img = project_circle_points(H, cx, cy, radius, n=64)
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
    stress_outer_confusion: bool = False,
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
        outer_ring_width = outer_radius_mm * (0.16 if stress_inner_confusion else 0.12)
        inner_ring_width = outer_ring_width
        outer_ring_val = 0.1
        inner_ring_val = 0.1

        if stress_outer_confusion:
            # Stress case: make the outer boundary weaker so the sampler is
            # tempted to lock onto the (strong) code-band boundary instead.
            outer_ring_width = outer_radius_mm * 0.06
            outer_ring_val = 0.35
            inner_ring_width = outer_radius_mm * (0.18 if stress_inner_confusion else 0.14)
            inner_ring_val = 0.05

        outer_mask = (dist_sq >= (outer_radius_mm - outer_ring_width) ** 2) & \
                     (dist_sq <= (outer_radius_mm + outer_ring_width) ** 2)
        img_flat[outer_mask] = outer_ring_val

        # Inner ring edge: dark ring at inner_radius
        inner_mask = (dist_sq >= (inner_radius_mm - inner_ring_width) ** 2) & \
                     (dist_sq <= (inner_radius_mm + inner_ring_width) ** 2)
        img_flat[inner_mask] = inner_ring_val

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
            if stress_inner_confusion or stress_outer_confusion:
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
    if strength <= 0.0:
        return img
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
    if sigma <= 0.0:
        return img
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 1)


# ── PNG writing (no PIL dependency) ─────────────────────────────────────

def write_png_gray(path: str, img: np.ndarray, dpi: Optional[float] = None) -> None:
    """Write a grayscale image as 8-bit PNG.

    - Float images are interpreted as [0, 1].
    - uint8 images are written as-is.
    - If dpi is provided, embeds a pHYs chunk for print scaling.
    """
    h, w = img.shape
    if img.dtype == np.uint8:
        data = img
    else:
        data = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Build raw PNG data (scanlines with per-row filter byte).
    raw = bytearray()
    raw_extend = raw.extend
    for row in range(h):
        raw.append(0)  # filter: none
        raw_extend(data[row].tobytes())

    def chunk(ctype: bytes, cdata: bytes) -> bytes:
        c = ctype + cdata
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(cdata)) + c + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    compressed = zlib.compress(bytes(raw), 6)

    chunks = [chunk(b"IHDR", ihdr)]
    if dpi is not None:
        if dpi <= 0:
            raise ValueError("dpi must be > 0")
        ppm = int(round(dpi * 1000.0 / 25.4))  # pixels per meter
        phys = struct.pack(">IIB", ppm, ppm, 1)  # unit: meter
        chunks.append(chunk(b"pHYs", phys))
    chunks.append(chunk(b"IDAT", compressed))
    chunks.append(chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(sig + b"".join(chunks))


# ── Print-target export ─────────────────────────────────────────────────

def _svg_fmt(x: float) -> str:
    # Keep files readable while preserving geometry.
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _svg_annulus_path(cx: float, cy: float, r_outer: float, r_inner: float) -> str:
    # Even-odd filled annulus: two closed circles in one path.
    ox = cx + r_outer
    ix = cx + r_inner
    return (
        f"M {_svg_fmt(ox)} {_svg_fmt(cy)} "
        f"A {_svg_fmt(r_outer)} {_svg_fmt(r_outer)} 0 1 1 {_svg_fmt(cx - r_outer)} {_svg_fmt(cy)} "
        f"A {_svg_fmt(r_outer)} {_svg_fmt(r_outer)} 0 1 1 {_svg_fmt(ox)} {_svg_fmt(cy)} Z "
        f"M {_svg_fmt(ix)} {_svg_fmt(cy)} "
        f"A {_svg_fmt(r_inner)} {_svg_fmt(r_inner)} 0 1 0 {_svg_fmt(cx - r_inner)} {_svg_fmt(cy)} "
        f"A {_svg_fmt(r_inner)} {_svg_fmt(r_inner)} 0 1 0 {_svg_fmt(ix)} {_svg_fmt(cy)} Z"
    )


def _svg_annular_sector_path(
    cx: float,
    cy: float,
    r_outer: float,
    r_inner: float,
    angle0_rad: float,
    angle1_rad: float,
) -> str:
    x0o = cx + r_outer * math.cos(angle0_rad)
    y0o = cy + r_outer * math.sin(angle0_rad)
    x1o = cx + r_outer * math.cos(angle1_rad)
    y1o = cy + r_outer * math.sin(angle1_rad)

    x1i = cx + r_inner * math.cos(angle1_rad)
    y1i = cy + r_inner * math.sin(angle1_rad)
    x0i = cx + r_inner * math.cos(angle0_rad)
    y0i = cy + r_inner * math.sin(angle0_rad)

    # SVG coordinates have +y downward, so increasing angles sweep clockwise.
    # Outer arc: angle0 -> angle1 (sweep=1). Inner arc: angle1 -> angle0 (sweep=0).
    return (
        f"M {_svg_fmt(x0o)} {_svg_fmt(y0o)} "
        f"A {_svg_fmt(r_outer)} {_svg_fmt(r_outer)} 0 0 1 {_svg_fmt(x1o)} {_svg_fmt(y1o)} "
        f"L {_svg_fmt(x1i)} {_svg_fmt(y1i)} "
        f"A {_svg_fmt(r_inner)} {_svg_fmt(r_inner)} 0 0 0 {_svg_fmt(x0i)} {_svg_fmt(y0i)} Z"
    )


def _scale_bar_params(
    board_mm: float,
    pitch_mm: float,
    markers: list[tuple[int, int, float, float]],
    outer_draw_extent_mm: float,
    margin_mm: float,
) -> dict:
    """Pick scale-bar geometry and placement (all in mm)."""
    half = board_mm / 2.0
    canvas_mm = board_mm + 2.0 * margin_mm

    inset_x_mm = max(2.0, 0.5 * pitch_mm)
    inset_y_mm = max(1.0, 0.25 * pitch_mm)

    # Try to fit the bar below the lowest marker drawings.
    if markers:
        max_my = max(m[3] for m in markers)
        marker_bottom_mm = margin_mm + half + max_my + outer_draw_extent_mm
    else:
        marker_bottom_mm = margin_mm + half

    # Height: keep it readable, but try to fit without overlap.
    bar_h_mm = min(4.0, max(2.0, 0.4 * pitch_mm))
    clearance_mm = max(0.5, 0.2 * bar_h_mm)
    available_mm = canvas_mm - inset_y_mm - bar_h_mm - (marker_bottom_mm + clearance_mm)
    if available_mm < 0:
        # Shrink to fit if possible; otherwise keep a minimum and accept overlap.
        bar_h_mm = max(1.0, canvas_mm - inset_y_mm - (marker_bottom_mm + clearance_mm))
        bar_h_mm = min(bar_h_mm, 4.0)

    # Length: aim for ~50% of board width, capped at 100 mm, rounded to 10 mm.
    usable_w_mm = max(1.0, board_mm - 2.0 * inset_x_mm)
    target_len_mm = min(100.0, 0.5 * usable_w_mm)
    bar_len_mm = int(round(target_len_mm / 10.0)) * 10
    bar_len_mm = max(10, bar_len_mm)
    while bar_len_mm > usable_w_mm and bar_len_mm >= 10:
        bar_len_mm -= 10
    bar_len_mm = max(10, bar_len_mm)

    tick_step_mm = 10 if bar_len_mm >= 50 else (5 if bar_len_mm >= 20 else 1)
    tick_w_mm = max(0.2, 0.08 * bar_h_mm)
    font_size_mm = max(1.2, 0.7 * bar_h_mm)

    x0_mm = margin_mm + inset_x_mm
    y0_mm = canvas_mm - inset_y_mm - bar_h_mm

    return {
        "x0_mm": x0_mm,
        "y0_mm": y0_mm,
        "bar_len_mm": float(bar_len_mm),
        "bar_h_mm": float(bar_h_mm),
        "tick_step_mm": float(tick_step_mm),
        "tick_w_mm": float(tick_w_mm),
        "label": f"{bar_len_mm} mm",
        "font_size_mm": float(font_size_mm),
    }


def _append_scale_bar_svg(
    lines: list[str],
    board_mm: float,
    pitch_mm: float,
    markers: list[tuple[int, int, float, float]],
    outer_draw_extent_mm: float,
    margin_mm: float,
) -> None:
    p = _scale_bar_params(board_mm, pitch_mm, markers, outer_draw_extent_mm, margin_mm)
    x0 = p["x0_mm"]
    y0 = p["y0_mm"]
    w = p["bar_len_mm"]
    h = p["bar_h_mm"]
    tick = p["tick_step_mm"]
    tick_w = p["tick_w_mm"]
    label = p["label"]
    fs = p["font_size_mm"]

    lines.append('<g id="scale_bar">')
    lines.append(
        f'<rect x="{_svg_fmt(x0)}" y="{_svg_fmt(y0)}" width="{_svg_fmt(w)}" height="{_svg_fmt(h)}" '
        f'fill="black"/>'
    )
    # Ticks: white lines inside the bar at fixed mm intervals.
    n_ticks = int(round(w / tick))
    for i in range(n_ticks + 1):
        tx = x0 + i * tick
        lines.append(
            f'<line x1="{_svg_fmt(tx)}" y1="{_svg_fmt(y0)}" x2="{_svg_fmt(tx)}" y2="{_svg_fmt(y0 + h)}" '
            f'stroke="white" stroke-width="{_svg_fmt(tick_w)}"/>'
        )
    # Label centered on the bar.
    lines.append(
        f'<text x="{_svg_fmt(x0 + w / 2)}" y="{_svg_fmt(y0 + 0.75 * h)}" '
        f'fill="white" font-size="{_svg_fmt(fs)}" font-family="monospace" text-anchor="middle">{label}</text>'
    )
    lines.append("</g>")


def write_print_target_svg(
    out_path: Path,
    board_mm: float,
    pitch_mm: float,
    n_markers: Optional[int],
    codebook: list[int],
    margin_mm: float = 0.0,
) -> None:
    """Write a ready-to-print SVG calibration target (board coordinates in mm)."""
    if board_mm <= 0:
        raise ValueError("--board_mm must be > 0")
    if pitch_mm <= 0:
        raise ValueError("--pitch_mm must be > 0")
    if margin_mm < 0:
        raise ValueError("--print_margin_mm must be >= 0")
    if not codebook:
        raise ValueError("codebook is empty")

    markers = generate_hex_lattice(board_mm, pitch_mm, n_markers)

    # Geometry (in mm), matching synthetic defaults.
    outer_radius = pitch_mm * 0.6
    inner_radius = pitch_mm * 0.4
    code_band_outer = pitch_mm * 0.58
    code_band_inner = pitch_mm * 0.42
    ring_half_thickness = outer_radius * 0.12
    ring_stroke = 2.0 * ring_half_thickness
    outer_draw_extent = outer_radius + ring_half_thickness

    size_mm = board_mm + 2.0 * margin_mm
    half = board_mm / 2.0

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_svg_fmt(size_mm)}mm" height="{_svg_fmt(size_mm)}mm" '
        f'viewBox="0 0 {_svg_fmt(size_mm)} {_svg_fmt(size_mm)}">'
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    dtheta = 2.0 * math.pi / 16.0
    for idx, (_q, _r, mx, my) in enumerate(markers):
        code_id = idx % len(codebook)
        cw = codebook[code_id]

        cx = mx + half + margin_mm
        cy = my + half + margin_mm

        lines.append(f'<g id="m{idx}" data-id="{code_id}">')
        lines.append(
            f'<circle cx="{_svg_fmt(cx)}" cy="{_svg_fmt(cy)}" r="{_svg_fmt(outer_radius)}" '
            f'fill="none" stroke="black" stroke-width="{_svg_fmt(ring_stroke)}"/>'
        )
        lines.append(
            f'<circle cx="{_svg_fmt(cx)}" cy="{_svg_fmt(cy)}" r="{_svg_fmt(inner_radius)}" '
            f'fill="none" stroke="black" stroke-width="{_svg_fmt(ring_stroke)}"/>'
        )

        # Erase ring overlap inside the code band.
        annulus_d = _svg_annulus_path(cx, cy, code_band_outer, code_band_inner)
        lines.append(f'<path d="{annulus_d}" fill="white" fill-rule="evenodd"/>')

        # Draw black sectors where bit == 0. Bit==1 stays white.
        for s in range(16):
            if ((cw >> s) & 1) == 1:
                continue
            a0 = -math.pi + s * dtheta
            a1 = a0 + dtheta
            sector_d = _svg_annular_sector_path(cx, cy, code_band_outer, code_band_inner, a0, a1)
            lines.append(f'<path d="{sector_d}" fill="black"/>')

        lines.append("</g>")

    _append_scale_bar_svg(
        lines,
        board_mm=board_mm,
        pitch_mm=pitch_mm,
        markers=markers,
        outer_draw_extent_mm=outer_draw_extent,
        margin_mm=margin_mm,
    )
    lines.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_print_target_raster(
    board_mm: float,
    pitch_mm: float,
    n_markers: Optional[int],
    codebook: list[int],
    dpi: float,
    margin_mm: float = 0.0,
) -> np.ndarray:
    """Render a top-down, ready-to-print calibration target as an 8-bit image."""
    if board_mm <= 0:
        raise ValueError("--board_mm must be > 0")
    if pitch_mm <= 0:
        raise ValueError("--pitch_mm must be > 0")
    if margin_mm < 0:
        raise ValueError("--print_margin_mm must be >= 0")
    if dpi <= 0:
        raise ValueError("--print_dpi must be > 0")
    if not codebook:
        raise ValueError("codebook is empty")

    ppmm = dpi / 25.4
    size_mm = board_mm + 2.0 * margin_mm
    w_px = int(round(size_mm * ppmm))
    h_px = int(round(size_mm * ppmm))
    w_px = max(w_px, 1)
    h_px = max(h_px, 1)

    bg = np.uint8(255)
    fg = np.uint8(0)
    img = np.full((h_px, w_px), bg, dtype=np.uint8)

    markers = generate_hex_lattice(board_mm, pitch_mm, n_markers)

    # Geometry (in mm), matching synthetic defaults.
    outer_radius_mm = pitch_mm * 0.6
    inner_radius_mm = pitch_mm * 0.4
    code_band_outer_mm = pitch_mm * 0.58
    code_band_inner_mm = pitch_mm * 0.42
    ring_half_thickness_mm = outer_radius_mm * 0.12

    outer_r = outer_radius_mm * ppmm
    inner_r = inner_radius_mm * ppmm
    code_r_outer = code_band_outer_mm * ppmm
    code_r_inner = code_band_inner_mm * ppmm
    ring_w = ring_half_thickness_mm * ppmm

    half = board_mm / 2.0
    bound = outer_r + ring_w + 2.0

    outer_min_sq = (outer_r - ring_w) ** 2
    outer_max_sq = (outer_r + ring_w) ** 2
    inner_min_sq = (inner_r - ring_w) ** 2
    inner_max_sq = (inner_r + ring_w) ** 2
    code_min_sq = code_r_inner ** 2
    code_max_sq = code_r_outer ** 2

    two_pi = 2.0 * math.pi

    for idx, (_q, _r, mx, my) in enumerate(markers):
        code_id = idx % len(codebook)
        cw = int(codebook[code_id])

        cx = (mx + half + margin_mm) * ppmm
        cy = (my + half + margin_mm) * ppmm

        x0 = int(max(math.floor(cx - bound), 0))
        x1 = int(min(math.ceil(cx + bound) + 1, w_px))
        y0 = int(max(math.floor(cy - bound), 0))
        y1 = int(min(math.ceil(cy + bound) + 1, h_px))
        if x0 >= x1 or y0 >= y1:
            continue

        patch = img[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        dx = xx - cx
        dy = yy - cy
        dist_sq = dx * dx + dy * dy

        outer_mask = (dist_sq >= outer_min_sq) & (dist_sq <= outer_max_sq)
        patch[outer_mask] = fg

        inner_mask = (dist_sq >= inner_min_sq) & (dist_sq <= inner_max_sq)
        patch[inner_mask] = fg

        code_mask = (dist_sq >= code_min_sq) & (dist_sq <= code_max_sq)
        if np.any(code_mask):
            angles = np.arctan2(dy[code_mask], dx[code_mask])  # [-pi, pi], +y down
            sector = ((angles / two_pi + 0.5) * 16).astype(np.int32) % 16
            bits = np.right_shift(cw, sector) & 1
            patch[code_mask] = np.where(bits == 1, bg, fg).astype(np.uint8)

    # Scale bar (black bar with white ticks + label).
    outer_draw_extent_mm = outer_radius_mm + ring_half_thickness_mm
    _draw_scale_bar_raster(
        img,
        board_mm=board_mm,
        pitch_mm=pitch_mm,
        markers=markers,
        outer_draw_extent_mm=outer_draw_extent_mm,
        margin_mm=margin_mm,
        dpi=dpi,
    )

    return img


_FONT_5X7: dict[str, list[int]] = {
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "3": [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    "m": [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101],
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
}


def _draw_text_5x7_u8(
    img: np.ndarray,
    x0: int,
    y0: int,
    text: str,
    scale: int,
    value: int,
) -> None:
    """Draw ASCII text using a tiny built-in 5x7 bitmap font."""
    if scale <= 0:
        return
    h, w = img.shape
    cursor_x = x0
    spacing = scale
    for ch in text:
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for row, bits in enumerate(glyph):
            for col in range(5):
                if ((bits >> (4 - col)) & 1) == 0:
                    continue
                px0 = cursor_x + col * scale
                py0 = y0 + row * scale
                px1 = px0 + scale
                py1 = py0 + scale
                if px1 <= 0 or py1 <= 0 or px0 >= w or py0 >= h:
                    continue
                sx0 = max(px0, 0)
                sy0 = max(py0, 0)
                sx1 = min(px1, w)
                sy1 = min(py1, h)
                img[sy0:sy1, sx0:sx1] = np.uint8(value)
        cursor_x += 5 * scale + spacing


def _draw_scale_bar_raster(
    img: np.ndarray,
    board_mm: float,
    pitch_mm: float,
    markers: list[tuple[int, int, float, float]],
    outer_draw_extent_mm: float,
    margin_mm: float,
    dpi: float,
) -> None:
    p = _scale_bar_params(
        board_mm=board_mm,
        pitch_mm=pitch_mm,
        markers=markers,
        outer_draw_extent_mm=outer_draw_extent_mm,
        margin_mm=margin_mm,
    )

    ppmm = dpi / 25.4
    x0 = int(round(p["x0_mm"] * ppmm))
    y0 = int(round(p["y0_mm"] * ppmm))
    bar_w = int(round(p["bar_len_mm"] * ppmm))
    bar_h = int(round(p["bar_h_mm"] * ppmm))
    tick_step = int(round(p["tick_step_mm"] * ppmm))
    tick_w = max(1, int(round(p["tick_w_mm"] * ppmm)))
    label = str(p["label"])

    h, w = img.shape
    if bar_w <= 0 or bar_h <= 0:
        return

    x1 = min(x0 + bar_w, w)
    y1 = min(y0 + bar_h, h)
    x0c = max(x0, 0)
    y0c = max(y0, 0)
    if x0c >= x1 or y0c >= y1:
        return

    # Black bar.
    img[y0c:y1, x0c:x1] = np.uint8(0)

    # White ticks.
    if tick_step > 0:
        for tx in range(x0, x0 + bar_w + 1, tick_step):
            for dx in range(-(tick_w // 2), tick_w - (tick_w // 2)):
                col = tx + dx
                if 0 <= col < w:
                    img[y0c:y1, col] = np.uint8(255)

    # White label centered on the bar.
    desired_text_h = max(7, int(round(bar_h * 0.7)))
    scale = max(1, desired_text_h // 7)
    text_h = 7 * scale
    text_w = len(label) * (5 * scale + scale) - scale
    text_x = x0 + (bar_w - text_w) // 2
    text_y = y0 + (bar_h - text_h) // 2
    _draw_text_5x7_u8(img, text_x, text_y, label, scale=scale, value=255)


# ── Main generator ──────────────────────────────────────────────────────

def camera_from_args(args: argparse.Namespace) -> Optional[dict]:
    """Build camera model dict from CLI args, or return None."""
    intr = [args.cam_fx, args.cam_fy, args.cam_cx, args.cam_cy]
    has_any_intr = any(v is not None for v in intr)
    has_all_intr = all(v is not None for v in intr)
    has_dist_coeff = any(
        abs(float(v)) > 1e-15
        for v in (args.cam_k1, args.cam_k2, args.cam_p1, args.cam_p2, args.cam_k3)
    )

    if has_any_intr and not has_all_intr:
        raise ValueError("camera intrinsics require all of --cam-fx --cam-fy --cam-cx --cam-cy")
    if not has_any_intr and has_dist_coeff:
        raise ValueError("non-zero distortion requires camera intrinsics")
    if not has_any_intr:
        return None

    fx = float(args.cam_fx)
    fy = float(args.cam_fy)
    cx = float(args.cam_cx)
    cy = float(args.cam_cy)
    if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError("camera intrinsics must be finite")
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("camera intrinsics fx/fy must be non-zero")

    return {
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "distortion": {
            "k1": float(args.cam_k1),
            "k2": float(args.cam_k2),
            "p1": float(args.cam_p1),
            "p2": float(args.cam_p2),
            "k3": float(args.cam_k3),
        },
    }


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
    illum_strength: float,
    noise_sigma: float,
    projective: bool,
    tilt_strength: float,
    seed: int,
    camera: Optional[dict] = None,
    stress_inner_confusion: bool = False,
    stress_outer_confusion: bool = False,
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
        stress_outer_confusion=stress_outer_confusion,
    )

    # Geometric distortion (working -> image frame), then photometric effects.
    img = apply_radial_tangential_distortion(img, camera)

    # Post-processing
    img = apply_anisotropic_blur(img, blur_px, rng)
    img = apply_illumination_gradient(img, rng, strength=illum_strength)
    img = apply_noise(img, rng, sigma=noise_sigma)

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
        # True projected center in working frame and image frame.
        wx, wy = project_point(H, mx, my)
        if camera:
            dxy = distort_pixel(camera, wx, wy)
            ix, iy = dxy if dxy is not None else (float("nan"), float("nan"))
        else:
            ix, iy = wx, wy

        # Circle projections in working frame.
        outer_pts_work = project_circle_points(H, mx, my, outer_radius, n=96)
        inner_pts_work = project_circle_points(H, mx, my, inner_radius, n=96)
        outer_ell_work = fit_ellipse_from_points(outer_pts_work)
        inner_ell_work = fit_ellipse_from_points(inner_pts_work)

        # Approximate projected curves in image frame after distortion.
        if camera and camera_has_distortion(camera):
            outer_pts_img = []
            for p in outer_pts_work:
                qd = distort_pixel(camera, float(p[0]), float(p[1]))
                if qd is not None:
                    outer_pts_img.append(qd)
            inner_pts_img = []
            for p in inner_pts_work:
                qd = distort_pixel(camera, float(p[0]), float(p[1]))
                if qd is not None:
                    inner_pts_img.append(qd)
            if len(outer_pts_img) >= 6:
                outer_ell_img = fit_ellipse_from_points(np.asarray(outer_pts_img, dtype=np.float64))
            else:
                outer_ell_img = outer_ell_work
            if len(inner_pts_img) >= 6:
                inner_ell_img = fit_ellipse_from_points(np.asarray(inner_pts_img, dtype=np.float64))
            else:
                inner_ell_img = inner_ell_work
        else:
            outer_ell_img = outer_ell_work
            inner_ell_img = inner_ell_work

        visible = (0 <= ix < img_w) and (0 <= iy < img_h)

        gt_markers.append({
            "id": code_id,
            "q": q,
            "r": r,
            "board_xy_mm": [float(mx), float(my)],
            "true_working_center": [float(wx), float(wy)],
            "true_image_center": [float(ix), float(iy)],
            "outer_ellipse": outer_ell_img,
            "inner_ellipse": inner_ell_img,
            "outer_ellipse_working": outer_ell_work,
            "inner_ellipse_working": inner_ell_work,
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
        "illum_strength": illum_strength,
        "noise_sigma": noise_sigma,
        "projective": projective,
        "tilt_strength": tilt,
        "camera": camera,
        "stress_inner_confusion": stress_inner_confusion,
        "stress_outer_confusion": stress_outer_confusion,
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
    parser.add_argument("--out_dir", type=str, default="tools/out/synth_002")
    parser.add_argument("--n_images", type=int, default=3)
    parser.add_argument("--img_w", type=int, default=1280)
    parser.add_argument("--img_h", type=int, default=960)
    parser.add_argument("--board_mm", type=float, default=200.0)
    parser.add_argument("--pitch_mm", type=float, default=8.0)
    parser.add_argument("--n_markers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blur_px", type=float, default=1.0)
    parser.add_argument(
        "--illum_strength",
        type=float,
        default=0.15,
        help="Linear illumination gradient strength (0 disables).",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.02,
        help="Additive Gaussian noise sigma in normalized image intensity units (0 disables).",
    )
    parser.add_argument("--cam-fx", type=float, default=None, help="Camera fx (pixels) for synthetic radial-tangential distortion.")
    parser.add_argument("--cam-fy", type=float, default=None, help="Camera fy (pixels) for synthetic radial-tangential distortion.")
    parser.add_argument("--cam-cx", type=float, default=None, help="Camera cx (pixels) for synthetic radial-tangential distortion.")
    parser.add_argument("--cam-cy", type=float, default=None, help="Camera cy (pixels) for synthetic radial-tangential distortion.")
    parser.add_argument("--cam-k1", type=float, default=0.0, help="Synthetic radial distortion k1.")
    parser.add_argument("--cam-k2", type=float, default=0.0, help="Synthetic radial distortion k2.")
    parser.add_argument("--cam-p1", type=float, default=0.0, help="Synthetic tangential distortion p1.")
    parser.add_argument("--cam-p2", type=float, default=0.0, help="Synthetic tangential distortion p2.")
    parser.add_argument("--cam-k3", type=float, default=0.0, help="Synthetic radial distortion k3.")
    parser.add_argument(
        "--stress-inner-confusion",
        action="store_true",
        help="Stress case: increase code-band contrast and edge thickness to confuse inner edge sampling",
    )
    parser.add_argument(
        "--stress-outer-confusion",
        action="store_true",
        help="Stress case: weaken outer boundary and increase code-band contrast to confuse outer edge selection",
    )
    parser.add_argument("--projective", action="store_true", default=True)
    parser.add_argument("--no_projective", dest="projective", action="store_false")
    parser.add_argument(
        "--tilt_strength",
        type=float,
        default=0.9,
        help=(
            "Projective tilt strength used by the homography sampler (0 disables perspective). "
            "Larger values produce stronger projective distortion."
        ),
    )
    parser.add_argument("--codebook", type=str, default="tools/codebook.json")
    parser.add_argument(
        "--print",
        dest="print_all",
        action="store_true",
        help="Write ready-to-print calibration target (both PNG + SVG) into out_dir",
    )
    parser.add_argument("--print_png", action="store_true", help="Write ready-to-print calibration target PNG into out_dir")
    parser.add_argument("--print_svg", action="store_true", help="Write ready-to-print calibration target SVG into out_dir")
    parser.add_argument(
        "--print_dpi",
        type=float,
        default=600.0,
        help="DPI for the print PNG (also embedded as PNG pHYs metadata)",
    )
    parser.add_argument(
        "--print_margin_mm",
        type=float,
        default=0.0,
        help="Extra white margin (mm) added around the board in print outputs",
    )
    parser.add_argument(
        "--print_basename",
        type=str,
        default="target_print",
        help="Basename for print output files (without extension)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        camera = camera_from_args(args)
    except ValueError as e:
        parser.error(str(e))

    if args.print_all:
        args.print_png = True
        args.print_svg = True

    codebook = load_codebook(args.codebook)
    print(f"Loaded codebook: {len(codebook)} codewords")

    lattice = generate_hex_lattice(args.board_mm, args.pitch_mm, args.n_markers)
    print(f"Hex lattice: {len(lattice)} markers (pitch={args.pitch_mm}mm, board={args.board_mm}mm)")
    try:
        rows, long_row_cols = infer_parametric_layout(lattice)
    except ValueError as e:
        parser.error(
            "generated lattice is not representable by target schema "
            f"(rows + long_row_cols): {e}. "
            "Use board/pitch settings that produce contiguous alternating rows."
        )

    # Emit board_spec.json alongside generated images
    marker_outer_radius = args.pitch_mm * 0.6
    marker_inner_radius = args.pitch_mm * 0.4
    marker_code_band_outer_radius = args.pitch_mm * 0.58
    marker_code_band_inner_radius = args.pitch_mm * 0.42
    board_spec = {
        "schema": "ringgrid.target.v1",
        "name": f"ringgrid_{int(args.board_mm)}mm_hex",
        "rows": rows,
        "long_row_cols": long_row_cols,
        "board_size_mm": [args.board_mm, args.board_mm],
        "pitch_mm": args.pitch_mm,
        "origin_mm": [0.0, 0.0],
        "marker_outer_radius_mm": marker_outer_radius,
        "marker_inner_radius_mm": marker_inner_radius,
        "marker_code_band_outer_radius_mm": marker_code_band_outer_radius,
        "marker_code_band_inner_radius_mm": marker_code_band_inner_radius,
    }
    board_spec_path = out_dir / "board_spec.json"
    with open(board_spec_path, "w") as f:
        json.dump(board_spec, f, indent=2)
    print(f"Board spec written to {board_spec_path}")

    if args.print_svg:
        svg_path = out_dir / f"{args.print_basename}.svg"
        write_print_target_svg(
            svg_path,
            board_mm=args.board_mm,
            pitch_mm=args.pitch_mm,
            n_markers=args.n_markers,
            codebook=codebook,
            margin_mm=args.print_margin_mm,
        )
        size_mm = args.board_mm + 2.0 * args.print_margin_mm
        print(f"Print SVG written to {svg_path} ({size_mm:.2f}mm x {size_mm:.2f}mm)")

    if args.print_png:
        png_path = out_dir / f"{args.print_basename}.png"
        img_u8 = render_print_target_raster(
            board_mm=args.board_mm,
            pitch_mm=args.pitch_mm,
            n_markers=args.n_markers,
            codebook=codebook,
            dpi=args.print_dpi,
            margin_mm=args.print_margin_mm,
        )
        write_png_gray(str(png_path), img_u8, dpi=args.print_dpi)
        print(
            f"Print PNG written to {png_path} ({args.print_dpi:.1f} dpi, "
            f"{img_u8.shape[1]}x{img_u8.shape[0]} px)"
        )

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
            illum_strength=args.illum_strength,
            noise_sigma=args.noise_sigma,
            projective=args.projective,
            tilt_strength=args.tilt_strength,
            seed=args.seed,
            camera=camera,
            stress_inner_confusion=args.stress_inner_confusion,
            stress_outer_confusion=args.stress_outer_confusion,
        )
        vis = sum(1 for m in gt["markers"] if m["visible"])
        print(f"  [{i+1}/{args.n_images}] {gt['image_file']}: "
              f"{gt['n_markers']} markers ({vis} visible)")

    print(f"\nDataset written to {out_dir}/")


if __name__ == "__main__":
    main()
