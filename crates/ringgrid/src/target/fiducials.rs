//! Origin fiducials: filled dots that anchor the board frame.

use projective_grid::LatticeKind;

use super::error::TargetValidationError;
use super::lattice::{LatticeGeometry, point_set_invariant_under, rotational_symmetries};
use super::ring::{MarkerCoding, RingGeometry};

/// Filled circular dots that define the target's origin and orientation.
///
/// Dots are printed dark on the white background, placed in the gaps of the
/// marker lattice (board-frame millimeters, same frame as the cell centers).
/// The dot pattern must break every rotational symmetry of the cell lattice
/// so a detector can resolve the board orientation uniquely. Reflections need
/// not be broken: an opaque planar target always maps to the image through an
/// orientation-preserving homography.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OriginFiducials {
    /// Dot radius in millimeters.
    pub dot_radius_mm: f32,
    /// Dot centers in board-frame millimeters.
    pub dots_mm: Vec<[f32; 2]>,
}

impl OriginFiducials {
    /// Automatically place origin fiducials in the lattice gaps near the board
    /// center.
    ///
    /// Produces an asymmetric triad of dots (an L) in the gaps around the
    /// centermost cell. Two properties make this the right placement:
    ///
    /// - **Anchorable:** the dots sit inside the densely-labeled interior, so a
    ///   detector's board→image homography *interpolates* to them rather than
    ///   extrapolating to a corner it may not have labeled (hex grid labeling in
    ///   particular has limited boundary recall).
    /// - **Orientation-resolving:** an L of three gaps is not invariant under
    ///   any lattice rotation, so it breaks every rotational symmetry — the
    ///   invariant `validate` enforces.
    ///
    /// The dot radius is derived from the gap clearance and the printed marker
    /// extent (targeting a legible `~0.1 × pitch`, shrunk to fit tight gaps).
    ///
    /// This is the single source of truth for automatic dot placement:
    /// [`TargetLayout::with_auto_fiducials`](crate::TargetLayout::with_auto_fiducials)
    /// and the recipe `dots: auto` option both route through it. (The
    /// [`TargetLayout::rect_24x24`](crate::TargetLayout::rect_24x24) preset keeps
    /// its own frozen dot geometry for print compatibility and does *not* use
    /// this placement.)
    ///
    /// # Errors
    ///
    /// Returns [`TargetValidationError::AutoFiducialsDoNotFit`] when the markers
    /// are packed too tightly to admit a dot in the gaps, or propagates a
    /// `validate` failure if the resulting pattern is degenerate (e.g. a lattice
    /// so small or symmetric that the triad cannot break its symmetry).
    pub fn auto(
        lattice: &LatticeGeometry,
        marker: RingGeometry,
        coding: &MarkerCoding,
    ) -> Result<Self, TargetValidationError> {
        let raw = lattice.generate_cells()?;
        let positions: Vec<[f32; 2]> = raw.iter().map(|(_, xy)| *xy).collect();
        let outer_draw = coding.outer_draw_radius_mm(&marker);
        let pitch = f64::from(lattice.pitch_mm());

        let holes = gap_triad_near_center(lattice, &positions);

        // Clearance: the smallest distance from any hole to the nearest cell.
        let clearance = holes
            .iter()
            .map(|&h| nearest_cell_distance(h, &positions))
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

        let fiducials = OriginFiducials {
            dot_radius_mm,
            dots_mm: holes.map(|[x, y]| [x as f32, y as f32]).to_vec(),
        };
        fiducials.validate(lattice.kind(), &positions, outer_draw)?;
        Ok(fiducials)
    }

    /// Validate dot geometry against the cell lattice.
    ///
    /// `cells_mm` are all cell centers, `outer_draw_radius_mm` the outermost
    /// drawn marker radius, `kind` the underlying lattice kind.
    pub(crate) fn validate(
        &self,
        kind: LatticeKind,
        cells_mm: &[[f32; 2]],
        outer_draw_radius_mm: f32,
    ) -> Result<(), TargetValidationError> {
        if !self.dot_radius_mm.is_finite() || self.dot_radius_mm <= 0.0 {
            return Err(TargetValidationError::InvalidDotRadius {
                dot_radius_mm: self.dot_radius_mm,
            });
        }
        if self.dots_mm.is_empty() {
            return Err(TargetValidationError::EmptyFiducialDots);
        }
        for (index, dot) in self.dots_mm.iter().enumerate() {
            if !dot[0].is_finite() || !dot[1].is_finite() {
                return Err(TargetValidationError::NonFiniteDot { index });
            }
        }

        // Dots must not touch any marker's drawn extent — otherwise both
        // rendering and dot detection are ill-defined.
        let min_clearance = f64::from(outer_draw_radius_mm) + f64::from(self.dot_radius_mm);
        let min_clearance_sq = min_clearance * min_clearance;
        for (index, dot) in self.dots_mm.iter().enumerate() {
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
        // set, otherwise orientation stays ambiguous.
        let dots: Vec<[f64; 2]> = self
            .dots_mm
            .iter()
            .map(|&[x, y]| [f64::from(x), f64::from(y)])
            .collect();
        let tol = (f64::from(self.dot_radius_mm) * 0.1).max(1e-6);
        for rotation in rotational_symmetries(kind, cells_mm) {
            if point_set_invariant_under(&dots, &rotation, tol) {
                return Err(TargetValidationError::FiducialsRotationallySymmetric {
                    angle_deg: rotation.angle_rad.to_degrees() as f32,
                });
            }
        }

        Ok(())
    }
}

/// Three lattice-gap centers clustered around the board's centermost cell, in
/// board-frame mm, forming an asymmetric L.
///
/// Anchored at the cell nearest the board centroid (`c0`), so the triad lands
/// in the densely-labeled interior. For a **rect** lattice the holes are three
/// of `c0`'s four surrounding 2×2-block gaps (clearance `pitch/√2`); for a
/// **hex** lattice they are three adjacent triangle holes around `c0` at 30°,
/// 90°, 150° (clearance `pitch`). Three-of-N is never rotation-invariant, so
/// the pattern breaks every lattice symmetry.
fn gap_triad_near_center(lattice: &LatticeGeometry, cells_mm: &[[f32; 2]]) -> [[f64; 2]; 3] {
    let pitch = f64::from(lattice.pitch_mm());
    let c0 = centermost_cell(cells_mm);
    match lattice {
        LatticeGeometry::Rect(_) => {
            let h = 0.5 * pitch;
            [
                [c0[0] + h, c0[1] + h],
                [c0[0] - h, c0[1] + h],
                [c0[0] + h, c0[1] - h],
            ]
        }
        LatticeGeometry::Hex(_) => {
            // Adjacent triangle holes sit at pitch distance, angles 30/90/150.
            let mut holes = [[0.0; 2]; 3];
            for (i, deg) in [30.0_f64, 90.0, 150.0].iter().enumerate() {
                let (sin, cos) = deg.to_radians().sin_cos();
                holes[i] = [c0[0] + pitch * cos, c0[1] + pitch * sin];
            }
            holes
        }
    }
}

/// The cell center nearest the board centroid (board mm).
fn centermost_cell(cells_mm: &[[f32; 2]]) -> [f64; 2] {
    let n = cells_mm.len().max(1) as f64;
    let mut centroid = [0.0f64; 2];
    for &[x, y] in cells_mm {
        centroid[0] += f64::from(x);
        centroid[1] += f64::from(y);
    }
    centroid[0] /= n;
    centroid[1] /= n;
    cells_mm
        .iter()
        .map(|&[x, y]| [f64::from(x), f64::from(y)])
        .min_by(|a, b| {
            let da = (a[0] - centroid[0]).powi(2) + (a[1] - centroid[1]).powi(2);
            let db = (b[0] - centroid[0]).powi(2) + (b[1] - centroid[1]).powi(2);
            da.total_cmp(&db)
        })
        .unwrap_or([0.0, 0.0])
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
        fid.validate(lattice.kind(), &cells, coding.outer_draw_radius_mm(&marker))
            .expect("auto fiducials valid");

        let dots: Vec<[f64; 2]> = fid
            .dots_mm
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
        // A square 4×4 patch has full C4 symmetry; corner dots must break it.
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

    #[test]
    fn auto_rect_24x24_matches_legacy_dot_radius() {
        // Regression: the 24×24 preset historically used Ø2.8 mm dots
        // (radius 1.4). Auto placement must reproduce that legible size.
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 24,
            cols: 24,
            pitch_mm: 14.0,
        });
        let (marker, coding) = plain(5.6, 2.8);
        let fid = OriginFiducials::auto(&lattice, marker, &coding).expect("fit");
        assert!(
            (fid.dot_radius_mm - 1.4).abs() < 1e-4,
            "dot radius {} != 1.4",
            fid.dot_radius_mm
        );
        // Dots cluster around the board center (~161 mm on a 24×14 mm board).
        for dot in &fid.dots_mm {
            assert!(
                dot[0] > 140.0 && dot[0] < 175.0 && dot[1] > 140.0 && dot[1] < 175.0,
                "dot {dot:?} should sit near the board center"
            );
        }
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
}
