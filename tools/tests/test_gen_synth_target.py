"""Parity guards for the synthetic target generator and scorer.

``tools/gen_synth_target.py`` renders synthetic boards *and* emits the target
spec the detector is scored against, so its cell layout and origin-dot
placement must agree with the library's exactly. Since schema v6 the spec no
longer carries dot coordinates — only the size — which means a drift between
the rendered image and the derived positions would no longer be visible in the
spec at all. These tests are what catch it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

ringgrid = pytest.importorskip("ringgrid")
gen_synth_target = pytest.importorskip("gen_synth_target")
score_detect_target = pytest.importorskip("score_detect_target")


# (rows, cols, pitch_mm) — square and oblong, even and odd, coarse and fine.
RECT_GEOMETRIES = [
    (24, 24, 14.0),
    (12, 12, 14.0),
    (8, 8, 14.0),
    (5, 7, 10.0),
    (13, 9, 6.5),
    (4, 4, 14.0),
]

# (rows, long_row_cols, pitch_mm) — odd/even row counts, both row parities.
HEX_GEOMETRIES = [
    (15, 14, 8.0),
    (37, 34, 8.0),
    (8, 7, 6.0),
    (9, 10, 5.0),
]


@pytest.mark.parametrize(("rows", "cols", "pitch_mm"), RECT_GEOMETRIES)
def test_rect_dots_match_library_placement(rows: int, cols: int, pitch_mm: float) -> None:
    target = ringgrid.TargetLayout.plain_rect(
        pitch_mm, rows, cols, 0.35 * pitch_mm, 0.18 * pitch_mm, dots=True
    )
    library = target.fiducial_dots_mm()
    generated = gen_synth_target.origin_dots_mm(target.lattice.to_dict())

    assert len(generated) == len(library) == 3
    for got, want in zip(generated, library):
        assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize(("rows", "long_row_cols", "pitch_mm"), HEX_GEOMETRIES)
def test_hex_dots_match_library_placement(
    rows: int, long_row_cols: int, pitch_mm: float
) -> None:
    target = ringgrid.TargetLayout.plain_hex(
        pitch_mm, rows, long_row_cols, 0.6 * pitch_mm, 0.4 * pitch_mm, dots=True
    )
    library = target.fiducial_dots_mm()
    generated = gen_synth_target.origin_dots_mm(target.lattice.to_dict())

    assert len(generated) == len(library) == 3
    for got, want in zip(generated, library):
        # The library stores f32 positions.
        assert got == pytest.approx(want, abs=1e-4)


def test_rect_24x24_placement_matches_the_printed_board() -> None:
    """The frozen triad physical `rect_24x24` boards are printed with."""
    assert gen_synth_target.rect_origin_dots_mm(24, 24, 14.0) == [
        [161.0, 161.0],
        [147.0, 161.0],
        [161.0, 175.0],
    ]


@pytest.mark.parametrize(("rows", "cols"), [(24, 24), (12, 12), (5, 7)])
def test_rect_lattice_coords_are_centered(rows: int, cols: int) -> None:
    """Cell coordinates must match the library's centered rect convention, or
    scored ground truth would disagree with every detection."""
    cells = gen_synth_target.generate_rect_lattice(rows, cols, 14.0)
    coords = {(u, v) for u, v, _, _ in cells}

    assert len(cells) == rows * cols
    assert (0, 0) in coords
    assert min(u for u, _ in coords) == -((cols - 1) // 2)
    assert min(v for _, v in coords) == -((rows - 1) // 2)

    spec_cells = {(u, v): (x, y) for u, v, x, y in cells}
    # Cell (0, 0) sits where the library puts it.
    assert spec_cells[(0, 0)] == (
        14.0 * ((cols - 1) // 2),
        14.0 * ((rows - 1) // 2),
    )


@pytest.mark.parametrize(("rows", "long_row_cols", "pitch_mm"), HEX_GEOMETRIES)
def test_hex_lattice_matches_library_layout(
    rows: int, long_row_cols: int, pitch_mm: float
) -> None:
    """Hex cells: alternating row lengths, first cell at the origin, axial
    coordinates consistent with positions, and the library's cell count."""
    cells = gen_synth_target.generate_hex_lattice(rows, long_row_cols, pitch_mm)
    rows_seen = sorted({r for _, r, _, _ in cells})

    assert cells[0][2:] == (0.0, 0.0)
    assert rows_seen == list(range(-(rows // 2), rows - rows // 2))
    assert (0, 0) in {(q, r) for q, r, _, _ in cells}
    # Library cell count: exceeding it would trip codebook-capacity validation.
    n_long = sum(1 for r in rows_seen if ((r + long_row_cols - 1) & 1) == 0)
    assert len(cells) == n_long * long_row_cols + (rows - n_long) * (long_row_cols - 1)

    # Positions follow the pointy-top axial formula (up to the origin shift).
    q0, r0, _, _ = cells[0]
    sqrt3 = math.sqrt(3.0)
    for q, r, x, y in cells:
        dq, dr = q - q0, r - r0
        assert x == pytest.approx(pitch_mm * sqrt3 * (dq + 0.5 * dr))
        assert y == pytest.approx(pitch_mm * 1.5 * dr)


@pytest.mark.parametrize("kind", ["rect", "hex"])
def test_scorer_lattice_maps_are_isometries(kind: str) -> None:
    """Every rotation (and rotation x reflection) the scorer tries must map the
    lattice's nearest-neighbour steps onto themselves — and rotations must keep
    orientation while reflections reverse it."""
    if kind == "rect":
        steps = {(1, 0), (0, 1), (-1, 0), (0, -1)}
    else:
        steps = {(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)}

    def apply(m, p):
        (a, b), (c, d) = m
        return (a * p[0] + b * p[1], c * p[0] + d * p[1])

    def det(m):
        (a, b), (c, d) = m
        return a * d - b * c

    rotations = score_detect_target.ROTATIONS[kind]
    reflection = score_detect_target.REFLECTIONS[kind]
    mirrored = [score_detect_target.compose(r, reflection) for r in rotations]
    assert len(set(rotations + mirrored)) == 2 * len(rotations)
    for m in rotations + mirrored:
        assert {apply(m, s) for s in steps} == steps
    assert all(det(m) == 1 for m in rotations)
    assert all(det(m) == -1 for m in mirrored)
