#!/usr/bin/env python3
"""Coverage score of a ringgrid detection on a real image (no ground truth).

Answers "did the detector find every ring it should have, with the right
label?" for any target (hex | rect lattice, coded | plain, with or without
origin dots) from nothing but a detection JSON (``DetectionResult`` as the CLI
writes it) and the target spec (``ringgrid.target.v6``):

1. Fit a board-to-image model to the labelled markers' centres: a homography
   by RANSAC (normalized DLT), refined by Levenberg-Marquardt on the image
   reprojection error, plus — as a second model — the homography with one
   division-model radial term ``k1`` about the image centre.
2. Predict every board cell's outer ring by projecting its outer circle (the
   boundary is sampled, so the radial term applies too). A cell is *expected*
   when the whole predicted outer ellipse lies inside the image with
   ``--margin`` px to spare.
3. Locate every detected marker on the lattice (nearest cell to its centre
   under the model) and compare with its reported label.

Per image it reports ``n_expected``, ``n_detected_expected`` (expected cells
carrying a detection whose label agrees with the fit within the label gate),
the ``missing`` cells, ``wrong_label_suspects`` (a labelled marker whose
centre is more than ``--label-gate`` lattice pitches from its labelled cell:
almost certainly the wrong cell), ``centre_outliers`` (right cell, but the
centre is more than ``--centre-gate-px`` off the fitted model), and the H and
H+k1 RMS / max residuals.

The model can be fitted to a *different* detection (``--reference``): an
independent run — a relaxed probe configuration, say — whose labelled markers
define the expected set, so that the expected set does not depend on the
configuration under test. Labels of the scored detection are mapped into the
reference's frame first (identity when both are anchored, else the lattice
rotation + translation that agrees best with the marker locations; an
unanchored detection's markers then count as found but not as detected,
since without a board origin they carry no board position).

An image whose fitting detection has fewer than ``--min-fit`` labelled
markers, or whose markers are nearly collinear, is ``unscorable`` (and counts
as a failure: the detector produced too little to even check it).

Usage:
    python tools/score_coverage.py --pred det.json --target target_spec.json \\
        [--reference ref_det.json] [--margin 2] --out coverage.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from gen_synth_target import generate_cells  # noqa: E402
from score_detect_target import IDENTITY, ROTATIONS, best_symmetry_map  # noqa: E402
from target_json import load_target_spec  # noqa: E402

N_BOUNDARY_SAMPLES = 64
# Median ring-shape mismatch above which a self-fitted labelling is flagged
# as inconsistent with the image (see `ring_shape_errors`).
SHAPE_ERR_MAX = 0.25


# ── Lattice ──────────────────────────────────────────────────────────────


class Lattice:
    """A target's cells plus the affine (u, v) -> board mm map of its lattice."""

    def __init__(self, spec: dict):
        self.kind = spec["lattice"]["kind"]
        self.outer_radius_mm = float(spec["marker"]["outer_radius_mm"])
        cells = generate_cells(spec["lattice"])
        self.coords = [(int(u), int(v)) for u, v, _, _ in cells]
        uv1 = np.array([[u, v, 1.0] for u, v, _, _ in cells])
        xy = np.array([[x, y] for _, _, x, y in cells])
        # Both lattices are affine in their integer coordinates; solving it
        # from the cells keeps this independent of the lattice formulas.
        self.affine, *_ = np.linalg.lstsq(uv1, xy, rcond=None)  # (3, 2)
        self.affine_inv = np.linalg.inv(
            np.vstack([self.affine.T, [0.0, 0.0, 1.0]])
        )

    def xy_mm(self, coords) -> np.ndarray:
        uv = np.asarray(coords, dtype=float).reshape(-1, 2)
        return np.hstack([uv, np.ones((len(uv), 1))]) @ self.affine

    def nearest_coord(self, xy_mm) -> tuple[np.ndarray, np.ndarray]:
        """Nearest lattice coordinate (unbounded lattice) to board points."""
        xy = np.asarray(xy_mm, dtype=float).reshape(-1, 2)
        uvw = np.hstack([xy, np.ones((len(xy), 1))]) @ self.affine_inv.T
        base = np.floor(uvw[:, :2]).astype(int)
        best = np.zeros_like(base)
        best_d = np.full(len(xy), np.inf)
        # Hex Voronoi cells are not unit squares in axial coordinates: check
        # the four corners of the containing square plus their neighbours.
        for du in (-1, 0, 1, 2):
            for dv in (-1, 0, 1, 2):
                cand = base + np.array([du, dv])
                d = np.linalg.norm(self.xy_mm(cand) - xy, axis=1)
                better = d < best_d
                best[better] = cand[better]
                best_d[better] = d[better]
        return best, best_d


# ── Camera model: homography (+ one division-model radial term) ──────────


def _normalizer(points: np.ndarray) -> np.ndarray:
    """Hartley normalization: centroid to the origin, mean distance sqrt(2)."""
    c = points.mean(axis=0)
    d = np.mean(np.linalg.norm(points - c, axis=1))
    s = math.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return p[:, :2] / p[:, 2:3]


def fit_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Normalized DLT homography mapping ``src`` to ``dst`` (n >= 4)."""
    if len(src) < 4:
        return None
    ts, td = _normalizer(src), _normalizer(dst)
    s = _apply_h(ts, src)
    d = _apply_h(td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, sv, vt = np.linalg.svd(np.asarray(rows))
    if sv[-2] < 1e-12:
        return None
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    if abs(h[2, 2]) < 1e-15:
        return None
    return h / h[2, 2]


def ransac_homography(src, dst, thresh_px, iters=2000, seed=0):
    """RANSAC + DLT; returns (H, inlier mask) or (None, None)."""
    n = len(src)
    if n < 4:
        return None, None
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        h = fit_homography_dlt(src[idx], dst[idx])
        if h is None:
            continue
        with np.errstate(all="ignore"):
            err = np.linalg.norm(_apply_h(h, src) - dst, axis=1)
        mask = err < thresh_px
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
            if count == n:
                break
    if best_mask is None or best_count < 4:
        return None, None
    # Re-estimate on the consensus set until it stops growing.
    mask = best_mask
    h = None
    for _ in range(5):
        h = fit_homography_dlt(src[mask], dst[mask])
        if h is None:
            return None, None
        err = np.linalg.norm(_apply_h(h, src) - dst, axis=1)
        new_mask = err < thresh_px
        if new_mask.sum() <= mask.sum():
            break
        mask = new_mask
    return h, mask


class Model:
    """Board mm -> image px: ``distort(H · x)`` with a division-model ``k1``.

    The division model maps a distorted point to the undistorted one as
    ``p_u = c + (p_d - c) / (1 + k1 r_d^2)``, ``r`` in units of the image
    half-diagonal about the image centre ``c``; ``k1 = 0`` is the plain
    homography.
    """

    def __init__(self, h: np.ndarray, k1: float, image_size):
        self.h = h / h[2, 2]
        self.k1 = float(k1)
        w, hh = image_size
        self.c = np.array([(w - 1) / 2.0, (hh - 1) / 2.0])
        self.s = 0.5 * math.hypot(w, hh)

    def project(self, xy_mm: np.ndarray) -> np.ndarray:
        """Image points; NaN where the model is undefined."""
        p = np.hstack([xy_mm, np.ones((len(xy_mm), 1))]) @ self.h.T
        with np.errstate(all="ignore"):
            pu = p[:, :2] / p[:, 2:3]
        pu[p[:, 2] <= 0] = np.nan
        if self.k1 == 0.0:
            return pu
        q = (pu - self.c) / self.s
        ru = np.linalg.norm(q, axis=1)
        disc = 1.0 - 4.0 * self.k1 * ru * ru
        with np.errstate(all="ignore"):
            rd = np.where(
                ru > 1e-12,
                (1.0 - np.sqrt(disc)) / (2.0 * self.k1 * ru),
                0.0,
            )
            scale = np.where(ru > 1e-12, rd / ru, 1.0)
        scale[disc < 0] = np.nan
        return self.c + q * scale[:, None] * self.s

    def jacobian(self, xy_mm, eps: float = 1e-3) -> np.ndarray:
        """2x2 Jacobian d(image px) / d(board mm) at one board point."""
        xy = np.asarray(xy_mm, dtype=float)
        p0 = self.project(xy[None])[0]
        steps = (np.array([eps, 0.0]), np.array([0.0, eps]))
        return np.column_stack([(self.project((xy + d)[None])[0] - p0) / eps for d in steps])

    def unproject(self, px: np.ndarray) -> np.ndarray:
        """Board points of image points (exact inverse of :meth:`project`)."""
        q = (px - self.c) / self.s
        r2 = np.sum(q * q, axis=1)
        pu = self.c + q / (1.0 + self.k1 * r2)[:, None] * self.s
        return _apply_h(np.linalg.inv(self.h), pu)


def refine_model(src, dst, h0, k1_free, image_size, iters=50):
    """Levenberg-Marquardt on the image reprojection error.

    Parameters: the 8 free entries of H (``h33 = 1``) and, when ``k1_free``,
    the radial term. The Jacobian is numeric — a few hundred points by nine
    parameters is cheap.
    """
    params = np.append((h0 / h0[2, 2]).ravel()[:8], 0.0)
    free = list(range(8)) + ([8] if k1_free else [])

    def residual(pv):
        m = Model(np.append(pv[:8], 1.0).reshape(3, 3), pv[8], image_size)
        r = (m.project(src) - dst).ravel()
        return np.where(np.isfinite(r), r, 1e6)

    lam = 1e-3
    r = residual(params)
    cost = float(r @ r)
    for _ in range(iters):
        jac = np.empty((len(r), len(free)))
        for j, k in enumerate(free):
            step = 1e-6 * max(1.0, abs(params[k]))
            if k == 8:
                step = 1e-7
            p2 = params.copy()
            p2[k] += step
            jac[:, j] = (residual(p2) - r) / step
        jtj = jac.T @ jac
        g = jac.T @ r
        improved = False
        for _ in range(10):
            a = jtj + lam * np.diag(np.diag(jtj) + 1e-12)
            try:
                delta = np.linalg.solve(a, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            p2 = params.copy()
            p2[free] += delta
            r2 = residual(p2)
            c2 = float(r2 @ r2)
            if c2 < cost:
                params, r, lam = p2, r2, lam * 0.3
                improved = cost - c2 > 1e-12 * max(cost, 1.0)
                cost = c2
                break
            lam *= 10
        if not improved:
            break
    return Model(np.append(params[:8], 1.0).reshape(3, 3), params[8], image_size)


def residual_stats(model: Model, src, dst) -> dict:
    err = np.linalg.norm(model.project(src) - dst, axis=1)
    return {
        "rms_px": float(np.sqrt(np.mean(err**2))),
        "max_px": float(np.max(err)),
    }


# ── Detection parsing ────────────────────────────────────────────────────


def labelled_markers(det: dict):
    """(index, (u, v), centre) of every marker carrying a grid coordinate."""
    out = []
    for i, m in enumerate(det.get("detected_markers", [])):
        gc = m.get("grid_coord")
        if gc is None:
            continue
        out.append((i, (int(gc[0]), int(gc[1])), np.asarray(m["center"], dtype=float)))
    return out


def is_anchored(det: dict) -> bool:
    return det.get("board_frame") == "absolute"


def fit_models(det, lattice, image_size, ransac_px, min_fit):
    """Fit H and H+k1 to a detection's labelled markers.

    Returns ``(fit, reason)``: ``fit`` is a dict with both models and the
    inlier statistics, or ``None`` with the reason the image is unscorable.
    """
    lab = labelled_markers(det)
    if len(lab) < min_fit:
        return None, f"only {len(lab)} labelled markers (< {min_fit})"
    src = lattice.xy_mm([c for _, c, _ in lab])
    dst = np.array([p for _, _, p in lab])
    h, mask = ransac_homography(src, dst, ransac_px)
    if h is None or mask.sum() < min_fit:
        n = 0 if mask is None else int(mask.sum())
        return None, f"only {n} RANSAC inliers (< {min_fit})"
    sin, din = src[mask], dst[mask]
    # Collinear / degenerate support cannot constrain a homography.
    centred = sin - sin.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] <= 0 or sv[1] / sv[0] < 0.1:
        return None, f"degenerate support (singular value ratio {sv[1] / max(sv[0], 1e-12):.3f})"
    model_h = refine_model(sin, din, h, False, image_size)
    model_k = refine_model(sin, din, model_h.h, True, image_size)
    # The radial model is used for prediction only when it is a clear win.
    stats_h = residual_stats(model_h, sin, din)
    stats_k = residual_stats(model_k, sin, din)
    return {
        "h": model_h,
        "hk": model_k,
        "inliers": int(mask.sum()),
        "n_labelled": len(lab),
        "stats_h": stats_h,
        "stats_hk": stats_k,
    }, None


# ── Expected set ─────────────────────────────────────────────────────────


def ellipse_inside(model: Model, centres_mm, radius_mm, image_size, margin):
    """Whether each cell's projected outer circle lies inside the image."""
    w, h = image_size
    t = np.linspace(0.0, 2.0 * math.pi, N_BOUNDARY_SAMPLES, endpoint=False)
    ring = radius_mm * np.stack([np.cos(t), np.sin(t)], axis=1)
    inside = np.zeros(len(centres_mm), dtype=bool)
    for i, c in enumerate(centres_mm):
        p = model.project(c + ring)
        if not np.all(np.isfinite(p)):
            continue
        inside[i] = (
            p[:, 0].min() >= margin
            and p[:, 1].min() >= margin
            and p[:, 0].max() <= w - 1 - margin
            and p[:, 1].max() <= h - 1 - margin
        )
    return inside


def projected_semi_axes(model: Model, lattice: Lattice, coords) -> list:
    """Semi-axes (px, major first) of each cell's projected outer circle,
    from the model's local affine map at the cell centre."""
    out = []
    for xy in lattice.xy_mm(coords):
        sv = np.linalg.svd(model.jacobian(xy), compute_uv=False) * lattice.outer_radius_mm
        out.append([float(sv[0]), float(sv[1])])
    return out


def candidate_coords(lattice: Lattice, model: Model, image_size, anchored: bool):
    """Cells to test: the board's cells (anchored) or the lattice under the image."""
    if anchored:
        return list(lattice.coords)
    w, h = image_size
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
    uv, _ = lattice.nearest_coord(model.unproject(corners))
    lo, hi = uv.min(axis=0) - 2, uv.max(axis=0) + 2
    return [(u, v) for u in range(lo[0], hi[0] + 1) for v in range(lo[1], hi[1] + 1)]


def local_pitch_px(model: Model, lattice: Lattice, coords) -> np.ndarray:
    """Image distance from each cell to its +u neighbour."""
    c = np.asarray(coords, dtype=float).reshape(-1, 2)
    a = model.project(lattice.xy_mm(c))
    b = model.project(lattice.xy_mm(c + np.array([1, 0])))
    return np.linalg.norm(a - b, axis=1)


def ring_shape_errors(model: Model, lattice: Lattice, det: dict, labelled) -> np.ndarray:
    """Relative mismatch of each labelled ring's detected outer ellipse and
    the outer circle the model projects at its labelled cell.

    ``|M_pred - M_det|_F / |M_det|_F`` for the ellipses' shape matrices
    ``M = R diag(a^2, b^2) R^T`` (the model's local affine map at the cell
    for the prediction). Correct labels give a few percent (edge blur, the
    perspective within one ring); a sheared or scaled labelling — a valid
    lattice basis that is not the board's, which a homography fits just as
    tightly — mismatches every ring.
    """
    markers = det.get("detected_markers", [])
    r = lattice.outer_radius_mm
    out = []
    for i, c, _ in labelled:
        e = markers[i].get("ellipse_outer")
        if not e:
            continue
        jac = model.jacobian(lattice.xy_mm([c])[0])
        m_pred = r * r * jac @ jac.T
        ca, sa = math.cos(e["angle"]), math.sin(e["angle"])
        rot = np.array([[ca, -sa], [sa, ca]])
        m_det = rot @ np.diag([e["a"] ** 2, e["b"] ** 2]) @ rot.T
        out.append(np.linalg.norm(m_pred - m_det) / np.linalg.norm(m_det))
    return np.asarray(out)


# ── Scoring ──────────────────────────────────────────────────────────────


def map_labels(pred_lab, located, lattice_kind, identity: bool):
    """Map predicted labels into the fitting frame.

    ``located`` holds each predicted marker's nearest cell under the model.
    Returns ``(mapper, agreement)``.
    """
    if identity:
        return (lambda c: tuple(c)), None
    pairs = [(c, tuple(int(x) for x in loc)) for (_, c, _), loc in zip(pred_lab, located)]
    agreement, rot, t = best_symmetry_map(pairs, ROTATIONS[lattice_kind])
    if rot is None:
        rot, t = IDENTITY, (0, 0)
    (a, b), (cc, d) = rot

    def mapper(c):
        return (a * c[0] + b * c[1] + t[0], cc * c[0] + d * c[1] + t[1])

    return mapper, agreement


def score_coverage(
    pred: dict,
    spec: dict,
    reference: dict | None = None,
    margin: float = 2.0,
    label_gate: float = 0.25,
    centre_gate_px: float = 3.0,
    ransac_px: float = 3.0,
    min_fit: int = 8,
) -> dict:
    """Coverage score of ``pred`` (see the module docstring)."""
    lattice = Lattice(spec)
    image_size = tuple(pred.get("image_size") or (reference or {}).get("image_size"))
    fit_det = reference if reference is not None else pred
    fit, reason = fit_models(fit_det, lattice, image_size, ransac_px, min_fit)
    pred_anchored = is_anchored(pred)
    base = {
        "n_pred": len(pred.get("detected_markers", [])),
        "n_pred_labelled": len(labelled_markers(pred)),
        "board_frame": pred.get("board_frame"),
        "fit_source": "reference" if reference is not None else "pred",
    }
    if fit is None:
        return {**base, "scorable": False, "unscorable_reason": reason}

    # Predict with H+k1 when the radial term pays off (it always fits at
    # least as well; require a 10 % RMS drop so noise does not pick it).
    use_k = fit["stats_hk"]["rms_px"] < 0.9 * fit["stats_h"]["rms_px"]
    model = fit["hk"] if use_k else fit["h"]
    frame_anchored = is_anchored(fit_det)

    coords = candidate_coords(lattice, model, image_size, frame_anchored)
    centres_mm = lattice.xy_mm(coords)
    inside = ellipse_inside(model, centres_mm, lattice.outer_radius_mm, image_size, margin)
    expected = [c for c, ok in zip(coords, inside) if ok]
    expected_set = set(expected)

    # Locate every predicted labelled marker on the lattice.
    pred_lab = labelled_markers(pred)
    det_index = {}
    suspects, centre_outliers, label_agreement = [], [], None
    found_any = set()
    if pred_lab:
        centres = np.array([p for _, _, p in pred_lab])
        located, _ = lattice.nearest_coord(model.unproject(centres))
        identity = reference is None or (pred_anchored and frame_anchored)
        to_frame, label_agreement = map_labels(pred_lab, located, lattice.kind, identity)
        labels = [to_frame(c) for _, c, _ in pred_lab]
        predicted = model.project(lattice.xy_mm(labels))
        pitch = local_pitch_px(model, lattice, labels)
        for (i, c, p), lab, pp, pit, loc in zip(pred_lab, labels, predicted, pitch, located):
            err = float(np.linalg.norm(p - pp))
            found_any.add(tuple(int(x) for x in loc))
            if not np.isfinite(err) or err > label_gate * pit:
                suspects.append({
                    "index": i,
                    "label": list(c),
                    "label_in_fit_frame": list(lab),
                    "located_cell": [int(x) for x in loc],
                    "err_px": err,
                    "center": p.tolist(),
                })
                continue
            if err > centre_gate_px:
                centre_outliers.append({"index": i, "label": list(lab), "err_px": err})
            det_index[lab] = i

    # Against a reference frame an unanchored detection's labels carry no
    # board position: its markers are found, not detected.
    counts = pred_anchored or reference is None
    # Reference-free label check: under its own fit, the detection's labels
    # must also predict the rings' shapes.
    shape_err = None
    if reference is None and pred_lab:
        errs = ring_shape_errors(model, lattice, pred, pred_lab)
        shape_err = float(np.median(errs)) if len(errs) else None
    detected = sorted(c for c in expected if counts and c in det_index)
    detected_set = set(detected)
    found = sorted(c for c in expected if c in det_index or c in found_any)
    missing = sorted(c for c in expected if c not in detected_set)
    exp_px = model.project(lattice.xy_mm(expected)) if expected else np.zeros((0, 2))
    return {
        **base,
        "scorable": True,
        "pred_anchored": pred_anchored,
        "fit_anchored": frame_anchored,
        "model": "h+k1" if use_k else "h",
        "k1": fit["hk"].k1,
        "fit_inliers": fit["inliers"],
        "fit_labelled": fit["n_labelled"],
        "h_rms_px": fit["stats_h"]["rms_px"],
        "h_max_px": fit["stats_h"]["max_px"],
        "hk1_rms_px": fit["stats_hk"]["rms_px"],
        "hk1_max_px": fit["stats_hk"]["max_px"],
        "homography": model.h.tolist(),
        "n_expected": len(expected),
        "n_detected_expected": len(detected),
        "n_found_expected": len(found),
        "n_detected_unexpected": sum(1 for c in det_index if c not in expected_set),
        "n_wrong_label_suspects": len(suspects),
        "n_centre_outliers": len(centre_outliers),
        "label_agreement": label_agreement,
        "ring_shape_err_median": shape_err,
        "labels_shape_inconsistent": shape_err is not None and shape_err > SHAPE_ERR_MAX,
        "missing": [list(c) for c in missing],
        "expected": [
            {"cell": list(c), "center": p.tolist(), "outer_semi_axes_px": ax}
            for c, p, ax in zip(expected, exp_px, projected_semi_axes(model, lattice, expected))
        ],
        "wrong_label_suspects": suspects,
        "centre_outliers": centre_outliers,
    }


def summary_line(r: dict) -> str:
    if not r["scorable"]:
        return f"UNSCORABLE ({r['unscorable_reason']}); {r['n_pred']} markers"
    def px(key):
        v = r.get(key)
        return "-" if v is None else f"{v:.3f}"

    return (
        f"expected {r['n_expected']}  detected {r['n_detected_expected']}  "
        f"missing {len(r['missing'])}  wrong-label {r['n_wrong_label_suspects']}  "
        f"centre-outliers {r['n_centre_outliers']}  "
        f"H rms/max {px('h_rms_px')}/{px('h_max_px')}px  "
        f"H+k1 rms/max {px('hk1_rms_px')}/{px('hk1_max_px')}px"
        + ("  LABELS INCONSISTENT WITH RING SHAPES" if r.get("labels_shape_inconsistent") else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="detection JSON to score")
    ap.add_argument("--target", required=True, help="target spec JSON (ringgrid.target.v6)")
    ap.add_argument("--reference", help="detection JSON to fit the model to (default: --pred)")
    ap.add_argument("--margin", type=float, default=2.0, help="image margin for expected rings (px)")
    ap.add_argument("--label-gate", type=float, default=0.25,
                    help="wrong-label gate, in local lattice pitches (default 0.25)")
    ap.add_argument("--centre-gate-px", type=float, default=3.0,
                    help="centre-outlier gate for correctly labelled markers (px)")
    ap.add_argument("--ransac-px", type=float, default=3.0, help="RANSAC inlier threshold (px)")
    ap.add_argument("--min-fit", type=int, default=8, help="minimum labelled markers to fit")
    ap.add_argument("--out", help="write the full score JSON here")
    args = ap.parse_args()

    spec = load_target_spec(args.target)
    with open(args.pred) as f:
        pred = json.load(f)
    reference = None
    if args.reference:
        with open(args.reference) as f:
            reference = json.load(f)
    result = score_coverage(
        pred, spec, reference,
        margin=args.margin, label_gate=args.label_gate,
        centre_gate_px=args.centre_gate_px, ransac_px=args.ransac_px,
        min_fit=args.min_fit,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(summary_line(result))


if __name__ == "__main__":
    main()
