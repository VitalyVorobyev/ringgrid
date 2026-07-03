//! Ring geometry and marker coding style.

use super::error::TargetValidationError;
use crate::marker::codebook::CODEBOOK_N;

/// Ring radii shared by every marker on the target, in millimeters.
///
/// For [`MarkerCoding::Coded16`] markers these are the *centerline* radii of
/// the stroked outer and inner rings; for [`MarkerCoding::Plain`] markers they
/// bound the filled annulus directly.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RingGeometry {
    /// Outer ring radius in millimeters.
    pub outer_radius_mm: f32,
    /// Inner ring radius in millimeters.
    pub inner_radius_mm: f32,
}

impl RingGeometry {
    pub(crate) fn validate(&self) -> Result<(), TargetValidationError> {
        if !self.outer_radius_mm.is_finite() || self.outer_radius_mm <= 0.0 {
            return Err(TargetValidationError::InvalidOuterRadius {
                marker_outer_radius_mm: self.outer_radius_mm,
            });
        }
        if !self.inner_radius_mm.is_finite() || self.inner_radius_mm <= 0.0 {
            return Err(TargetValidationError::InvalidInnerRadius {
                marker_inner_radius_mm: self.inner_radius_mm,
            });
        }
        if self.inner_radius_mm >= self.outer_radius_mm {
            return Err(TargetValidationError::InnerRadiusNotSmallerThanOuter {
                marker_inner_radius_mm: self.inner_radius_mm,
                marker_outer_radius_mm: self.outer_radius_mm,
            });
        }
        Ok(())
    }
}

/// Parameters of the 16-sector coded ring style.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CodedRingSpec {
    /// Stroke width of the inner and outer rings in millimeters.
    pub ring_width_mm: f32,
    /// Optional optimized codebook-ID assignment. When present,
    /// `id_assignment[i]` is the codebook ID for the i-th cell (in generation
    /// order). When absent, IDs are assigned sequentially (0, 1, 2, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_assignment: Option<Vec<usize>>,
}

/// Marker coding style: how (and whether) markers encode their identity.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarkerCoding {
    /// Two stroked rings with a 16-sector code band between them.
    Coded16(CodedRingSpec),
    /// A plain filled annulus between the inner and outer radii; markers
    /// carry no identity and are labeled by lattice position instead.
    Plain,
}

impl MarkerCoding {
    /// Whether markers encode a decodable identity.
    pub fn is_coded(&self) -> bool {
        matches!(self, Self::Coded16(_))
    }

    /// Half of the ring stroke width (0 for plain markers).
    pub(crate) fn ring_half_width_mm(&self) -> f32 {
        match self {
            Self::Coded16(spec) => 0.5 * spec.ring_width_mm,
            Self::Plain => 0.0,
        }
    }

    /// Outermost drawn radius of a marker: stroked rings overshoot the
    /// centerline radius by half the stroke width; plain annuli do not.
    pub(crate) fn outer_draw_radius_mm(&self, ring: &RingGeometry) -> f32 {
        ring.outer_radius_mm + self.ring_half_width_mm()
    }

    /// Code band bounds `(inner_mm, outer_mm)` for coded markers.
    pub(crate) fn code_band_bounds_mm(&self, ring: &RingGeometry) -> Option<(f32, f32)> {
        match self {
            Self::Coded16(_) => {
                let half = self.ring_half_width_mm();
                Some((ring.inner_radius_mm + half, ring.outer_radius_mm - half))
            }
            Self::Plain => None,
        }
    }

    pub(crate) fn validate(
        &self,
        ring: &RingGeometry,
        n_cells: usize,
    ) -> Result<(), TargetValidationError> {
        match self {
            Self::Coded16(spec) => {
                if !spec.ring_width_mm.is_finite() || spec.ring_width_mm <= 0.0 {
                    return Err(TargetValidationError::InvalidRingWidth {
                        marker_ring_width_mm: spec.ring_width_mm,
                    });
                }

                let half = self.ring_half_width_mm();
                let inner_ring_outer_edge_mm = ring.inner_radius_mm + half;
                let outer_ring_inner_edge_mm = ring.outer_radius_mm - half;
                if inner_ring_outer_edge_mm >= outer_ring_inner_edge_mm {
                    return Err(TargetValidationError::NonPositiveCodeBandGap {
                        inner_ring_outer_edge_mm,
                        outer_ring_inner_edge_mm,
                    });
                }

                if n_cells > CODEBOOK_N {
                    return Err(TargetValidationError::CodebookCapacityExceeded {
                        n_cells,
                        codebook_len: CODEBOOK_N,
                    });
                }

                if let Some(assignment) = &spec.id_assignment {
                    if assignment.len() != n_cells {
                        return Err(TargetValidationError::IdAssignmentLength {
                            expected: n_cells,
                            got: assignment.len(),
                        });
                    }
                    let mut seen = std::collections::HashSet::new();
                    for (position, &id) in assignment.iter().enumerate() {
                        if id >= CODEBOOK_N {
                            return Err(TargetValidationError::IdAssignmentOutOfRange {
                                id,
                                position,
                                codebook_len: CODEBOOK_N,
                            });
                        }
                        if !seen.insert(id) {
                            return Err(TargetValidationError::IdAssignmentDuplicate {
                                id,
                                position,
                            });
                        }
                    }
                }
                Ok(())
            }
            Self::Plain => Ok(()),
        }
    }
}
