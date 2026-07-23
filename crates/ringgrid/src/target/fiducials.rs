//! Origin fiducials: filled dots that anchor the board frame.

use projective_grid::Coord;

use super::error::TargetValidationError;
use super::lattice::{LatticeGeometry, point_set_invariant_under, rotational_symmetries};
use super::ring::{MarkerCoding, RingGeometry};

/// Filled circular dots that define the target's origin and orientation.
///
/// Dots are printed dark on the white background, in the gaps of the marker
/// lattice around cell `(0, 0)`. **Only the dot size is stored** — the
/// positions are derived from the lattice and read back via
/// [`TargetLayout::fiducial_dots_mm`](crate::TargetLayout::fiducial_dots_mm),
/// so they cannot go stale when the target's dimensions change. (Before schema
/// v6 this type carried absolute `dots_mm` coordinates, which silently
/// remained "valid" when copied onto a board they no longer fit.)
///
/// The derived pattern breaks every rotational symmetry of the cell lattice, so
/// a detector can resolve board orientation uniquely. Reflections need not be
/// broken: an opaque planar target always maps to the image through an
/// orientation-preserving homography.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginFiducials {
    /// Dot radius in millimeters.
    pub dot_radius_mm: f32,
}

impl OriginFiducials {
    /// Fiducials with an automatically-sized dot.
    ///
    /// The radius is derived from the gap clearance and the printed marker
    /// extent (targeting a legible `~0.1 × pitch`, shrunk to fit tight gaps).
    /// Positions are derived from the lattice and need no input; read them back
    /// with [`TargetLayout::fiducial_dots_mm`](crate::TargetLayout::fiducial_dots_mm).
    ///
    /// This is the single source of truth for automatic fiducials: the
    /// [`plain_rect`](crate::TargetLayout::plain_rect) /
    /// [`plain_hex`](crate::TargetLayout::plain_hex) constructors,
    /// [`TargetLayout::with_auto_fiducials`](crate::TargetLayout::with_auto_fiducials),
    /// the [`rect_24x24`](crate::TargetLayout::rect_24x24) preset, and the
    /// recipe `fiducials = "auto"` option all route through it.
    ///
    /// # Errors
    ///
    /// Returns [`TargetValidationError::AutoFiducialsDoNotFit`] when the markers
    /// are packed too tightly to admit a dot in the gaps, or propagates a
    /// placement failure for a board too small to hold the triad.
    pub fn auto(
        lattice: &LatticeGeometry,
        marker: RingGeometry,
        coding: &MarkerCoding,
    ) -> Result<Self, TargetValidationError> {
        let raw = lattice.generate_cells()?;
        let positions: Vec<[f32; 2]> = raw.iter().map(|(_, xy)| *xy).collect();
        let outer_draw = coding.outer_draw_radius_mm(&marker);
        let pitch = f64::from(lattice.pitch_mm());

        let dots = origin_dot_positions_mm(lattice)?;

        // Clearance: the smallest distance from any dot to the nearest cell.
        let clearance = dots
            .iter()
            .map(|&d| nearest_cell_distance([f64::from(d[0]), f64::from(d[1])], &positions))
            .fold(f64::INFINITY, f64::min);

        // Largest dot that still clears every marker, with a small safety margin.
        let margin = 0.02 * pitch;
        let max_fit = clearance - f64::from(outer_draw) - margin;
        if max_fit <= 0.0 {
            return Err(TargetValidationError::AutoFiducialsDoNotFit {
                pitch_mm: lattice.pitch_mm(),
                max_dot_radius_mm: max_fit as f32,
            });
        }
        // Prefer a legible dot (~0.1 pitch), never exceeding the gap.
        let dot_radius_mm = (0.1 * pitch).min(0.6 * max_fit) as f32;

        let fiducials = OriginFiducials { dot_radius_mm };
        fiducials.validate(lattice, &positions, outer_draw)?;
        Ok(fiducials)
    }

    /// Validate dot geometry against the cell lattice.
    ///
    /// `cells_mm` are all cell centers, `outer_draw_radius_mm` the outermost
    /// drawn marker radius.
    pub(crate) fn validate(
        &self,
        lattice: &LatticeGeometry,
        cells_mm: &[[f32; 2]],
        outer_draw_radius_mm: f32,
    ) -> Result<(), TargetValidationError> {
        if !self.dot_radius_mm.is_finite() || self.dot_radius_mm <= 0.0 {
            return Err(TargetValidationError::InvalidDotRadius {
                dot_radius_mm: self.dot_radius_mm,
            });
        }

        let dots_mm = origin_dot_positions_mm(lattice)?;

        // Dots must not touch any marker's drawn extent — otherwise both
        // rendering and dot detection are ill-defined.
        let min_clearance = f64::from(outer_draw_radius_mm) + f64::from(self.dot_radius_mm);
        let min_clearance_sq = min_clearance * min_clearance;
        for (index, dot) in dots_mm.iter().enumerate() {
            let hit = cells_mm.iter().any(|cell| {
                let dx = f64::from(dot[0]) - f64::from(cell[0]);
                let dy = f64::from(dot[1]) - f64::from(cell[1]);
                dx * dx + dy * dy < min_clearance_sq
            });
            if hit {
                return Err(TargetValidationError::DotOverlapsMarker { index });
            }
        }

        // The dot pattern must break every rotational symmetry of the cell
        // set, otherwise orientation stays ambiguous. Structurally guaranteed
        // by the L-triad, but re-checked so a future placement change cannot
        // silently ship an unanchorable target.
        let dots: Vec<[f64; 2]> = dots_mm
            .iter()
            .map(|&[x, y]| [f64::from(x), f64::from(y)])
            .collect();
        let tol = (f64::from(self.dot_radius_mm) * 0.1).max(1e-6);
        for rotation in rotational_symmetries(lattice.kind(), cells_mm) {
            if point_set_invariant_under(&dots, &rotation, tol) {
                return Err(TargetValidationError::FiducialsRotationallySymmetric {
                    angle_deg: rotation.angle_rad.to_degrees() as f32,
                });
            }
        }

        Ok(())
    }
}

/// Board-millimeter positions of the origin-dot triad for `lattice`.
///
/// The **single source of truth** for where origin dots go. Three lattice gaps
/// forming an asymmetric L anchored at cell `(0, 0)`, which both lattices place
/// near the board center:
///
/// - **Rect:** the gap diagonally at `(+pitch/2, +pitch/2)` from cell `(0, 0)`,
///   plus that gap's neighbours one pitch in `-x` and one pitch in `+y`.
/// - **Hex:** the three adjacent triangle holes around cell `(0, 0)` at 30°,
///   90° and 150° (clearance `pitch`).
///
/// Two properties make this placement right:
///
/// - **Anchorable:** the dots sit inside the densely-labeled interior, so a
///   detector's board→image homography *interpolates* to them rather than
///   extrapolating to a corner it may not have labeled (hex grid labeling in
///   particular has limited boundary recall).
/// - **Orientation-resolving:** an L of three gaps is not invariant under any
///   lattice rotation, so it breaks every rotational symmetry.
///
/// # Errors
///
/// Returns [`TargetValidationError::OriginDotsOutsideBoard`] when the board is
/// too small to hold the triad within its marker field (rect boards need at
/// least 3 columns and 4 rows).
pub(crate) fn origin_dot_positions_mm(
    lattice: &LatticeGeometry,
) -> Result<Vec<[f32; 2]>, TargetValidationError> {
    let cells = lattice.generate_cells()?;
    let origin = cells
        .iter()
        .find(|(coord, _)| *coord == Coord::new(0, 0))
        .map(|(_, xy)| [f64::from(xy[0]), f64::from(xy[1])])
        .ok_or(TargetValidationError::OriginDotsOutsideBoard)?;
    let pitch = f64::from(lattice.pitch_mm());

    let dots = match lattice {
        LatticeGeometry::Rect(_) => {
            let h = 0.5 * pitch;
            let anchor = [origin[0] + h, origin[1] + h];
            [
                anchor,
                [anchor[0] - pitch, anchor[1]],
                [anchor[0], anchor[1] + pitch],
            ]
        }
        LatticeGeometry::Hex(_) => {
            let mut holes = [[0.0; 2]; 3];
            for (i, deg) in [30.0_f64, 90.0, 150.0].iter().enumerate() {
                let (sin, cos) = deg.to_radians().sin_cos();
                holes[i] = [origin[0] + pitch * cos, origin[1] + pitch * sin];
            }
            holes
        }
    };

    // Dots outside the marker field would be extrapolated by the anchoring
    // homography — exactly what this placement exists to avoid.
    let (min_xy, max_xy) = cell_bounds(&cells);
    if dots
        .iter()
        .any(|d| d[0] < min_xy[0] || d[0] > max_xy[0] || d[1] < min_xy[1] || d[1] > max_xy[1])
    {
        return Err(TargetValidationError::OriginDotsOutsideBoard);
    }

    Ok(dots.map(|[x, y]| [x as f32, y as f32]).to_vec())
}

/// Axis-aligned bounds of the cell centers, in board mm.
fn cell_bounds(cells: &[(Coord, [f32; 2])]) -> ([f64; 2], [f64; 2]) {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for (_, xy) in cells {
        min[0] = min[0].min(f64::from(xy[0]));
        min[1] = min[1].min(f64::from(xy[1]));
        max[0] = max[0].max(f64::from(xy[0]));
        max[1] = max[1].max(f64::from(xy[1]));
    }
    (min, max)
}

/// Distance from a point to the nearest cell center (board mm).
fn nearest_cell_distance(p: [f64; 2], cells_mm: &[[f32; 2]]) -> f64 {
    cells_mm
        .iter()
        .map(|&[x, y]| {
            let dx = p[0] - f64::from(x);
            let dy = p[1] - f64::from(y);
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::lattice::{HexGeometry, RectGeometry};

    fn plain(outer: f32, inner: f32) -> (RingGeometry, MarkerCoding) {
        (
            RingGeometry {
                outer_radius_mm: outer,
                inner_radius_mm: inner,
            },
            MarkerCoding::Plain,
        )
    }

    fn cell_positions(lattice: &LatticeGeometry) -> Vec<[f32; 2]> {
        lattice
            .generate_cells()
            .expect("valid geometry")
            .into_iter()
            .map(|(_, xy)| xy)
            .collect()
    }

    /// Auto dots must pass validation and break every actual rotational
    /// symmetry of the cell set — the core anchoring invariant.
    fn assert_auto_breaks_symmetry(
        lattice: LatticeGeometry,
        marker: RingGeometry,
        coding: MarkerCoding,
    ) {
        let fid = OriginFiducials::auto(&lattice, marker, &coding).expect("auto fiducials fit");
        let cells = cell_positions(&lattice);
        // `auto` already calls validate() internally; re-assert explicitly.
        fid.validate(&lattice, &cells, coding.outer_draw_radius_mm(&marker))
            .expect("auto fiducials valid");

        let dots: Vec<[f64; 2]> = origin_dot_positions_mm(&lattice)
            .expect("dots derivable")
            .iter()
            .map(|&[x, y]| [f64::from(x), f64::from(y)])
            .collect();
        let tol = (f64::from(fid.dot_radius_mm) * 0.1).max(1e-6);
        for rotation in rotational_symmetries(lattice.kind(), &cells) {
            assert!(
                !point_set_invariant_under(&dots, &rotation, tol),
                "auto dots invariant under {:.0}° rotation",
                rotation.angle_rad.to_degrees()
            );
        }
    }

    #[test]
    fn auto_rect_symmetric_patch_is_anchored() {
        // A square 4×4 patch has full C4 symmetry; the dots must break it.
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 4,
            cols: 4,
            pitch_mm: 14.0,
        });
        let (marker, coding) = plain(5.6, 2.8);
        assert_auto_breaks_symmetry(lattice, marker, coding);
    }

    #[test]
    fn auto_hex_patch_is_anchored() {
        let lattice = LatticeGeometry::Hex(HexGeometry {
            rows: 15,
            long_row_cols: 14,
            pitch_mm: 8.0,
        });
        let (marker, coding) = plain(4.8, 3.2);
        assert_auto_breaks_symmetry(lattice, marker, coding);
    }

    /// **The frozen-print guarantee.** Physical boards printed from the
    /// `rect_24x24` preset carry this exact triad; derived placement must
    /// reproduce it, or those boards stop anchoring.
    #[test]
    fn rect_24x24_derived_dots_match_the_printed_board() {
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 24,
            cols: 24,
            pitch_mm: 14.0,
        });
        let dots = origin_dot_positions_mm(&lattice).expect("derivable");
        assert_eq!(dots, vec![[161.0, 161.0], [147.0, 161.0], [161.0, 175.0]]);

        let (marker, coding) = plain(5.6, 2.8);
        let fid = OriginFiducials::auto(&lattice, marker, &coding).expect("fit");
        // Historic preset radius: Ø2.8 mm dots.
        assert!(
            (fid.dot_radius_mm - 1.4).abs() < 1e-4,
            "dot radius {} != 1.4",
            fid.dot_radius_mm
        );
    }

    /// Positions are lattice-relative, so resizing the board moves them with
    /// it instead of leaving them stranded in absolute millimeters.
    #[test]
    fn dots_track_the_lattice_when_dimensions_change() {
        let dots_of = |rows, cols, pitch_mm| {
            origin_dot_positions_mm(&LatticeGeometry::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm,
            }))
            .expect("derivable")
        };
        // Half the board: dots stay in the interior rather than off the field.
        assert_eq!(
            dots_of(12, 12, 14.0),
            vec![[77.0, 77.0], [63.0, 77.0], [77.0, 91.0]]
        );
        // Same cell count, different pitch: dots scale with it.
        assert_eq!(
            dots_of(12, 12, 7.0),
            vec![[38.5, 38.5], [31.5, 38.5], [38.5, 45.5]]
        );
    }

    #[test]
    fn auto_rejects_markers_too_large_for_gaps() {
        // Outer radius near the packing limit leaves no room for a dot.
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 4,
            cols: 4,
            pitch_mm: 10.0,
        });
        // pitch/√2 ≈ 7.07 gap clearance; a 6.9 mm outer radius overruns it.
        let (marker, coding) = plain(6.9, 2.0);
        assert!(matches!(
            OriginFiducials::auto(&lattice, marker, &coding),
            Err(TargetValidationError::AutoFiducialsDoNotFit { .. })
        ));
    }

    /// A board too small to hold the triad fails loudly rather than placing
    /// dots beyond the marker field, where the homography would extrapolate.
    #[test]
    fn rejects_boards_too_small_for_the_triad() {
        for (rows, cols) in [(3, 3), (4, 2), (1, 1)] {
            let lattice = LatticeGeometry::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm: 14.0,
            });
            assert!(
                matches!(
                    origin_dot_positions_mm(&lattice),
                    Err(TargetValidationError::OriginDotsOutsideBoard)
                ),
                "{rows}x{cols} should reject auto dots"
            );
        }
    }
}
