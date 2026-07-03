//! Origin fiducials: filled dots that anchor the board frame.

use projective_grid::LatticeKind;

use super::error::TargetValidationError;
use super::lattice::{point_set_invariant_under, rotational_symmetries};

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
