"""Synthetic checks for the coverage scorer (``tools/score_coverage.py``).

A plain rect board is projected through a known homography (optionally with a
division-model radial term); the "detection" is built from the exact
projections, so every expected count is known in closed form.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

np = pytest.importorskip("numpy")
sc = pytest.importorskip("score_coverage")

IMAGE_SIZE = (800, 600)
SPEC = {
    "schema": "ringgrid.target.v6",
    "name": "synthetic",
    "lattice": {"kind": "rect", "rows": 30, "cols": 30, "pitch_mm": 2.0},
    "marker": {"outer_radius_mm": 0.8, "inner_radius_mm": 0.4},
    "coding": {"kind": "plain"},
    "fiducials": {"dot_radius_mm": 0.2},
}


def board_to_image(k1: float = 0.0) -> "sc.Model":
    """A mildly oblique view (~30 px pitch) of the board's centre."""
    s = 15.0
    h = np.array([
        [s * 1.0, s * 0.15, 400.0 - s * 29.0],
        [s * -0.05, s * 0.8, 300.0 - s * 0.8 * 29.0],
        [0.0005, 0.002, 1.0],
    ])
    # Re-centre: put board centre (29, 29) mm near the image centre.
    p = h @ np.array([29.0, 29.0, 1.0])
    p = p[:2] / p[2]
    shift = np.array([[1.0, 0.0, 400.0 - p[0]], [0.0, 1.0, 300.0 - p[1]], [0.0, 0.0, 1.0]])
    return sc.Model(shift @ h, k1, IMAGE_SIZE)


def ellipse_of(model, xy_mm, r_mm):
    """Outer ellipse (a, b, angle) of a circle under the model's local affine map."""
    p0 = model.project(xy_mm[None])[0]
    jac = model.jacobian(xy_mm, eps=1e-4)
    m = r_mm * r_mm * jac @ jac.T
    w, v = np.linalg.eigh(m)
    return {
        "cx": p0[0], "cy": p0[1],
        "a": math.sqrt(w[1]), "b": math.sqrt(w[0]),
        "angle": math.atan2(v[1, 1], v[0, 1]),
    }


def synthetic_detection(model, noise_px=0.0, seed=0):
    """Every fully visible ring, exactly labelled, in the absolute frame.

    Centres are exact by default: noise would move the fitted model and with
    it rings that sit right at the image margin in or out of the expected set.
    """
    lattice = sc.Lattice(SPEC)
    inside = sc.ellipse_inside(
        model, lattice.xy_mm(lattice.coords), lattice.outer_radius_mm, IMAGE_SIZE, 2.0
    )
    rng = np.random.default_rng(seed)
    markers = []
    for c, ok in zip(lattice.coords, inside):
        if not ok:
            continue
        xy = lattice.xy_mm([c])[0]
        centre = model.project(xy[None])[0] + rng.normal(0.0, noise_px, 2)
        markers.append({
            "grid_coord": list(c),
            "board_xy_mm": xy.tolist(),
            "center": centre.tolist(),
            "confidence": 1.0,
            "ellipse_outer": ellipse_of(model, xy, lattice.outer_radius_mm),
        })
    det = {
        "detected_markers": markers,
        "image_size": list(IMAGE_SIZE),
        "board_frame": "absolute",
    }
    return det, int(inside.sum())


def test_complete_detection_scores_full_coverage():
    det, n_visible = synthetic_detection(board_to_image())
    assert n_visible > 100
    r = sc.score_coverage(det, SPEC)
    assert r["scorable"]
    assert r["n_expected"] == n_visible
    assert r["n_detected_expected"] == n_visible
    assert r["missing"] == []
    assert r["n_wrong_label_suspects"] == 0
    assert r["h_rms_px"] < 0.1
    assert not r["labels_shape_inconsistent"]


def test_missing_and_wrong_labels_are_reported():
    det, n_visible = synthetic_detection(board_to_image())
    markers = det["detected_markers"]
    dropped = [tuple(m["grid_coord"]) for m in markers[10:15]]
    del markers[10:15]
    # One marker labelled with its right-hand neighbour's cell.
    wrong = markers[40]
    true_cell = tuple(wrong["grid_coord"])
    wrong["grid_coord"] = [true_cell[0] + 1, true_cell[1]]
    r = sc.score_coverage(det, SPEC)
    assert r["n_expected"] == n_visible
    assert r["n_wrong_label_suspects"] == 1
    assert r["wrong_label_suspects"][0]["located_cell"] == list(true_cell)
    # The 5 dropped cells plus the mislabelled marker's own cell.
    assert r["n_detected_expected"] == n_visible - 6
    assert {tuple(c) for c in r["missing"]} == set(dropped) | {true_cell}


def test_radial_term_is_recovered():
    k1 = -0.08
    det, _ = synthetic_detection(board_to_image(k1), noise_px=0.02)
    r = sc.score_coverage(det, SPEC)
    assert r["model"] == "h+k1"
    assert r["hk1_rms_px"] < 0.05 < r["h_rms_px"]
    assert r["k1"] == pytest.approx(k1, abs=0.01)


def test_too_few_labels_are_unscorable():
    det, _ = synthetic_detection(board_to_image())
    det["detected_markers"] = det["detected_markers"][:5]
    r = sc.score_coverage(det, SPEC)
    assert not r["scorable"]


def test_sheared_labelling_fails_the_ring_shape_check():
    """A labelling in another basis of the same point lattice fits a homography
    exactly as well, but no longer predicts the rings' shapes."""
    det, _ = synthetic_detection(board_to_image())
    for m in det["detected_markers"]:
        u, v = m["grid_coord"]
        m["grid_coord"] = [u + v, v]
        del m["board_xy_mm"]
    det["board_frame"] = "relative_canonical"
    r = sc.score_coverage(det, SPEC)
    assert r["scorable"] and r["h_rms_px"] < 0.1
    assert r["labels_shape_inconsistent"]


def test_unanchored_detection_against_a_reference():
    """Relative labels (rotated + shifted) map onto the reference frame; they
    count as found, not as detected, since they carry no board position."""
    ref, n_visible = synthetic_detection(board_to_image())
    det, _ = synthetic_detection(board_to_image(), seed=1)
    for m in det["detected_markers"]:
        u, v = m["grid_coord"]
        m["grid_coord"] = [-v + 3, u + 7]
        del m["board_xy_mm"]
    det["board_frame"] = "relative_canonical"
    r = sc.score_coverage(det, SPEC, reference=ref)
    assert r["n_expected"] == n_visible
    assert r["label_agreement"] == pytest.approx(1.0)
    assert r["n_wrong_label_suspects"] == 0
    assert r["n_found_expected"] == n_visible
    assert r["n_detected_expected"] == 0
