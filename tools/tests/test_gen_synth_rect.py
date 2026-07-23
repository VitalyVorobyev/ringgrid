"""Parity guard for the synthetic rect generator's origin-dot placement.

``tools/gen_synth_rect.py`` renders synthetic boards *and* emits the target spec
the detector is scored against, so its dot placement must agree with the
library's exactly. Since schema v6 the spec no longer carries dot coordinates —
only the size — which means a drift between the rendered image and the derived
positions would no longer be visible in the spec at all. This test is what
catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

ringgrid = pytest.importorskip("ringgrid")
gen_synth_rect = pytest.importorskip("gen_synth_rect")


# (rows, cols, pitch_mm) — square and oblong, even and odd, coarse and fine.
GEOMETRIES = [
    (24, 24, 14.0),
    (12, 12, 14.0),
    (8, 8, 14.0),
    (5, 7, 10.0),
    (13, 9, 6.5),
    (4, 4, 14.0),
]


@pytest.mark.parametrize(("rows", "cols", "pitch_mm"), GEOMETRIES)
def test_generator_dots_match_library_placement(
    rows: int, cols: int, pitch_mm: float
) -> None:
    target = ringgrid.TargetLayout.plain_rect(
        pitch_mm, rows, cols, 0.35 * pitch_mm, 0.18 * pitch_mm, dots=True
    )
    library = target.fiducial_dots_mm()
    generated = gen_synth_rect.origin_dots_mm(rows, cols, pitch_mm)

    assert len(generated) == len(library) == 3
    for got, want in zip(generated, library):
        assert got == pytest.approx(want, abs=1e-6)


def test_rect_24x24_placement_matches_the_printed_board() -> None:
    """The frozen triad physical `rect_24x24` boards are printed with."""
    assert gen_synth_rect.origin_dots_mm(24, 24, 14.0) == [
        [161.0, 161.0],
        [147.0, 161.0],
        [161.0, 175.0],
    ]


@pytest.mark.parametrize(("rows", "cols"), [(24, 24), (12, 12), (5, 7)])
def test_generator_lattice_coords_are_centered(rows: int, cols: int) -> None:
    """Cell coordinates must match the library's centered rect convention, or
    scored ground truth would disagree with every detection."""
    cells = gen_synth_rect.generate_rect_lattice(rows, cols, 14.0)
    coords = {(u, v) for u, v, _, _ in cells}

    assert len(cells) == rows * cols
    assert (0, 0) in coords
    assert min(u for u, _ in coords) == -((cols - 1) // 2)
    assert min(v for _, v in coords) == -((rows - 1) // 2)

    target = ringgrid.TargetLayout.plain_rect(14.0, rows, cols, 5.0, 2.5, dots=False)
    spec_cells = {
        (u, v): (x, y)
        for u, v, x, y in cells
    }
    # Cell (0, 0) sits where the library puts it.
    assert spec_cells[(0, 0)] == (
        14.0 * ((cols - 1) // 2),
        14.0 * ((rows - 1) // 2),
    )
    assert target.lattice.rows == rows and target.lattice.cols == cols
