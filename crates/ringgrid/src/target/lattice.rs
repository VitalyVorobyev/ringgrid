//! Lattice geometry: cell arrangements for hex and rectangular targets.

use projective_grid::{Coord, LatticeKind};

use super::error::TargetValidationError;

/// Hex-lattice geometry: axial rows alternating between long and short rows.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HexGeometry {
    /// Number of marker rows.
    pub rows: usize,
    /// Number of columns in the longest (even-offset) rows.
    pub long_row_cols: usize,
    /// Axial lattice pitch in millimeters (see [`LatticeGeometry::min_center_spacing_mm`]
    /// for the resulting nearest-neighbor distance).
    pub pitch_mm: f32,
}

/// Rectangular (square-lattice) geometry: `rows × cols` cells at uniform pitch.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RectGeometry {
    /// Number of marker rows.
    pub rows: usize,
    /// Number of marker columns.
    pub cols: usize,
    /// Center-to-center spacing between adjacent markers in millimeters.
    pub pitch_mm: f32,
}

/// Lattice arrangement of marker cells.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LatticeGeometry {
    /// Hexagonal lattice (axial rows with alternating long/short rows).
    Hex(HexGeometry),
    /// Rectangular lattice (`rows × cols` grid).
    Rect(RectGeometry),
}

impl LatticeGeometry {
    /// The projective-grid lattice kind matching this geometry.
    pub fn kind(&self) -> LatticeKind {
        match self {
            Self::Hex(_) => LatticeKind::Hex,
            Self::Rect(_) => LatticeKind::Square,
        }
    }

    /// Lattice pitch in millimeters.
    pub fn pitch_mm(&self) -> f32 {
        match self {
            Self::Hex(h) => h.pitch_mm,
            Self::Rect(r) => r.pitch_mm,
        }
    }

    /// Minimum center-to-center spacing between adjacent markers (mm).
    pub fn min_center_spacing_mm(&self) -> f32 {
        match self {
            // Hex nearest-neighbor distance in this axial layout.
            Self::Hex(h) => h.pitch_mm * f32::sqrt(3.0),
            Self::Rect(r) => r.pitch_mm,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), TargetValidationError> {
        let pitch_mm = self.pitch_mm();
        if !pitch_mm.is_finite() || pitch_mm <= 0.0 {
            return Err(TargetValidationError::InvalidPitch { pitch_mm });
        }
        match *self {
            Self::Hex(HexGeometry {
                rows,
                long_row_cols,
                ..
            }) => {
                if rows == 0 {
                    return Err(TargetValidationError::InvalidRows { rows });
                }
                if long_row_cols == 0 {
                    return Err(TargetValidationError::InvalidLongRowCols { long_row_cols });
                }
                if rows > 1 && long_row_cols < 2 {
                    return Err(TargetValidationError::InvalidLongRowColsForRows {
                        rows,
                        long_row_cols,
                    });
                }
            }
            Self::Rect(RectGeometry { rows, cols, .. }) => {
                if rows == 0 {
                    return Err(TargetValidationError::InvalidRows { rows });
                }
                if cols == 0 {
                    return Err(TargetValidationError::InvalidCols { cols });
                }
            }
        }
        Ok(())
    }

    /// Generate cell coordinates and positions in generation order.
    ///
    /// The first generated cell is normalized to `[0, 0]` mm; downstream
    /// consumers rely on this anchoring (see `BoardLayout` docs).
    pub(crate) fn generate_cells(&self) -> Result<Vec<(Coord, [f32; 2])>, TargetValidationError> {
        let mut cells = match *self {
            Self::Hex(HexGeometry {
                rows,
                long_row_cols,
                pitch_mm,
            }) => generate_hex_cells(rows, long_row_cols, pitch_mm)?,
            Self::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm,
            }) => generate_rect_cells(rows, cols, pitch_mm),
        };
        normalize_cell_origin(&mut cells);
        Ok(cells)
    }
}

/// Generate hex cells row by row.
///
/// This is the single source of truth for hex marker placement; `BoardLayout`
/// delegates here. The math is kept in f64 and the generation order (top row
/// first, left to right) is load-bearing: sequential IDs and the `[0, 0]`
/// anchor both derive from it.
fn generate_hex_cells(
    rows: usize,
    long_row_cols: usize,
    pitch_mm: f32,
) -> Result<Vec<(Coord, [f32; 2])>, TargetValidationError> {
    let short_row_cols = long_row_cols.saturating_sub(1);
    let mut cells = Vec::new();
    let row_mid = (rows as i32) / 2;

    for row_idx in 0..rows {
        let r = row_idx as i32 - row_mid;
        let n_cols = if rows == 1 || ((r + long_row_cols as i32 - 1) & 1) == 0 {
            long_row_cols
        } else {
            short_row_cols
        };

        if n_cols == 0 {
            return Err(TargetValidationError::DerivedZeroColumns {
                row_index: row_idx,
                rows,
                long_row_cols,
            });
        }

        let q_start = -((r + n_cols as i32 - 1) / 2);
        for col_idx in 0..n_cols {
            let q = q_start + col_idx as i32;
            cells.push((Coord::new(q, r), hex_axial_to_xy_mm(q, r, pitch_mm)));
        }
    }

    Ok(cells)
}

fn generate_rect_cells(rows: usize, cols: usize, pitch_mm: f32) -> Vec<(Coord, [f32; 2])> {
    let pitch = f64::from(pitch_mm);
    let mut cells = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let x = (pitch * col as f64) as f32;
            let y = (pitch * row as f64) as f32;
            cells.push((Coord::new(col as i32, row as i32), [x, y]));
        }
    }
    cells
}

fn hex_axial_to_xy_mm(q: i32, r: i32, pitch_mm: f32) -> [f32; 2] {
    let qf = q as f64;
    let rf = r as f64;
    let pitch = pitch_mm as f64;
    let x = pitch * (f64::sqrt(3.0) * qf + 0.5 * f64::sqrt(3.0) * rf);
    let y = pitch * (1.5 * rf);
    [x as f32, y as f32]
}

fn normalize_cell_origin(cells: &mut [(Coord, [f32; 2])]) {
    let Some(anchor) = cells.first().map(|(_, xy)| *xy) else {
        return;
    };
    for (_, xy) in cells {
        xy[0] -= anchor[0];
        xy[1] -= anchor[1];
    }
}

/// A rotational symmetry of the finite cell set.
///
/// Rotating every cell center by `angle_rad` about `center_mm` maps the cell
/// set onto itself. Only orientation-preserving symmetries are considered:
/// an opaque planar target seen by a camera always yields an
/// orientation-preserving homography, so reflections can never cause
/// labeling ambiguity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LatticeRotation {
    /// Rotation angle in radians (counter-clockwise in board frame).
    pub angle_rad: f64,
    /// Rotation center in board millimeters.
    pub center_mm: [f64; 2],
}

impl LatticeRotation {
    /// Apply the rotation to a point in board millimeters.
    pub fn apply(&self, xy_mm: [f64; 2]) -> [f64; 2] {
        let (sin, cos) = self.angle_rad.sin_cos();
        let dx = xy_mm[0] - self.center_mm[0];
        let dy = xy_mm[1] - self.center_mm[1];
        [
            self.center_mm[0] + cos * dx - sin * dy,
            self.center_mm[1] + sin * dx + cos * dy,
        ]
    }
}

/// Compute the non-identity rotational symmetries of a finite cell set.
///
/// Candidate angles come from the underlying lattice (90° steps for square,
/// 60° steps for hex); each candidate is verified numerically against the
/// actual cell positions, so finite-patch effects (e.g. a non-square rect
/// grid only admitting 180°) are handled exactly.
pub(crate) fn rotational_symmetries(kind: LatticeKind, cells: &[[f32; 2]]) -> Vec<LatticeRotation> {
    if cells.is_empty() {
        return Vec::new();
    }

    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for &[x, y] in cells {
        min[0] = min[0].min(f64::from(x));
        min[1] = min[1].min(f64::from(y));
        max[0] = max[0].max(f64::from(x));
        max[1] = max[1].max(f64::from(y));
    }
    let center = [0.5 * (min[0] + max[0]), 0.5 * (min[1] + max[1])];
    let span = (max[0] - min[0]).max(max[1] - min[1]);
    let tol = (span * 1e-6).max(1e-6);

    let steps: usize = match kind {
        LatticeKind::Hex => 6,
        _ => 4,
    };

    (1..steps)
        .map(|k| LatticeRotation {
            angle_rad: 2.0 * std::f64::consts::PI * (k as f64) / (steps as f64),
            center_mm: center,
        })
        .filter(|rot| {
            let points: Vec<[f64; 2]> = cells
                .iter()
                .map(|&[x, y]| [f64::from(x), f64::from(y)])
                .collect();
            point_set_invariant_under(&points, rot, tol)
        })
        .collect()
}

/// Check whether a point set maps onto itself under a rotation, within `tol`.
pub(crate) fn point_set_invariant_under(
    points: &[[f64; 2]],
    rotation: &LatticeRotation,
    tol: f64,
) -> bool {
    points.iter().all(|&p| {
        let rp = rotation.apply(p);
        points
            .iter()
            .any(|&q| (q[0] - rp[0]).abs() <= tol && (q[1] - rp[1]).abs() <= tol)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_cells_are_row_major_from_origin() {
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 2,
            cols: 3,
            pitch_mm: 14.0,
        });
        let cells = lattice.generate_cells().expect("valid geometry");
        assert_eq!(cells.len(), 6);
        assert_eq!(cells[0], (Coord::new(0, 0), [0.0, 0.0]));
        assert_eq!(cells[1], (Coord::new(1, 0), [14.0, 0.0]));
        assert_eq!(cells[3], (Coord::new(0, 1), [0.0, 14.0]));
        assert_eq!(cells[5], (Coord::new(2, 1), [28.0, 14.0]));
    }

    #[test]
    fn square_rect_has_all_c4_rotations() {
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 4,
            cols: 4,
            pitch_mm: 10.0,
        });
        let cells = lattice.generate_cells().expect("valid geometry");
        let positions: Vec<[f32; 2]> = cells.iter().map(|(_, xy)| *xy).collect();
        let syms = rotational_symmetries(LatticeKind::Square, &positions);
        assert_eq!(syms.len(), 3, "square patch admits 90/180/270");
    }

    #[test]
    fn oblong_rect_has_only_half_turn() {
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows: 3,
            cols: 5,
            pitch_mm: 10.0,
        });
        let cells = lattice.generate_cells().expect("valid geometry");
        let positions: Vec<[f32; 2]> = cells.iter().map(|(_, xy)| *xy).collect();
        let syms = rotational_symmetries(LatticeKind::Square, &positions);
        assert_eq!(syms.len(), 1, "oblong patch admits only 180");
        assert!((syms[0].angle_rad - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn rotation_apply_round_trips() {
        let rot = LatticeRotation {
            angle_rad: std::f64::consts::FRAC_PI_2,
            center_mm: [10.0, 20.0],
        };
        let p = rot.apply([13.0, 20.0]);
        assert!((p[0] - 10.0).abs() < 1e-12);
        assert!((p[1] - 23.0).abs() < 1e-12);
    }
}
