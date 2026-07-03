//! Validation and load errors for target layout specifications.

/// Validation failures for a target layout specification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TargetValidationError {
    /// Validation failed: unsupported target schema version.
    UnsupportedSchema {
        /// Schema string found in the file.
        found: String,
        /// Schema strings the loader accepts.
        expected: &'static str,
    },
    /// Validation failed: target name is empty.
    EmptyName,
    /// Validation failed: pitch is non-positive or non-finite.
    InvalidPitch {
        /// The invalid pitch value.
        pitch_mm: f32,
    },
    /// Validation failed: row count is zero.
    InvalidRows {
        /// The invalid row count.
        rows: usize,
    },
    /// Validation failed: rectangular column count is zero.
    InvalidCols {
        /// The invalid column count.
        cols: usize,
    },
    /// Validation failed: long-row column count is zero.
    InvalidLongRowCols {
        /// The invalid column count.
        long_row_cols: usize,
    },
    /// Validation failed: long-row columns must exceed short-row columns derived from row count.
    InvalidLongRowColsForRows {
        /// Total number of rows.
        rows: usize,
        /// Column count for the longest row.
        long_row_cols: usize,
    },
    /// Validation failed: outer radius is non-positive or non-finite.
    InvalidOuterRadius {
        /// The invalid outer radius value.
        marker_outer_radius_mm: f32,
    },
    /// Validation failed: inner radius is non-positive or non-finite.
    InvalidInnerRadius {
        /// The invalid inner radius value.
        marker_inner_radius_mm: f32,
    },
    /// Validation failed: ring width is non-positive or non-finite.
    InvalidRingWidth {
        /// The invalid ring width value.
        marker_ring_width_mm: f32,
    },
    /// Validation failed: inner radius must be strictly less than outer radius.
    InnerRadiusNotSmallerThanOuter {
        /// The inner radius value.
        marker_inner_radius_mm: f32,
        /// The outer radius value.
        marker_outer_radius_mm: f32,
    },
    /// Validation failed: code band gap between inner and outer rings is non-positive.
    NonPositiveCodeBandGap {
        /// Outer edge of the inner ring in mm.
        inner_ring_outer_edge_mm: f32,
        /// Inner edge of the outer ring in mm.
        outer_ring_inner_edge_mm: f32,
    },
    /// Validation failed: outer diameter exceeds minimum center-to-center spacing.
    OuterDiameterExceedsMinCenterSpacing {
        /// Outer diameter in mm.
        marker_outer_diameter_mm: f32,
        /// Minimum center spacing in mm.
        min_center_spacing_mm: f32,
    },
    /// Validation failed: marker draw diameter exceeds minimum center-to-center spacing.
    MarkerDrawDiameterExceedsMinCenterSpacing {
        /// Marker draw diameter in mm.
        marker_draw_diameter_mm: f32,
        /// Minimum center spacing in mm.
        min_center_spacing_mm: f32,
    },
    /// Validation failed: a row has zero columns after applying hex-lattice offset.
    DerivedZeroColumns {
        /// Index of the problematic row.
        row_index: usize,
        /// Total number of rows.
        rows: usize,
        /// Column count for the longest row.
        long_row_cols: usize,
    },
    /// Validation failed: coded target has more cells than the embedded codebook.
    CodebookCapacityExceeded {
        /// Number of cells requiring distinct codes.
        n_cells: usize,
        /// Number of codewords in the embedded base codebook.
        codebook_len: usize,
    },
    /// Validation failed: `id_assignment` length does not match marker count.
    IdAssignmentLength {
        /// Expected length (marker count).
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// Validation failed: `id_assignment` contains duplicate IDs.
    IdAssignmentDuplicate {
        /// The duplicated codebook ID.
        id: usize,
        /// Position index where the duplicate was found.
        position: usize,
    },
    /// Validation failed: `id_assignment` contains an ID beyond the codebook.
    IdAssignmentOutOfRange {
        /// The out-of-range codebook ID.
        id: usize,
        /// Position index where it was found.
        position: usize,
        /// Number of codewords in the embedded base codebook.
        codebook_len: usize,
    },
    /// Validation failed: fiducial dot radius is non-positive or non-finite.
    InvalidDotRadius {
        /// The invalid dot radius value.
        dot_radius_mm: f32,
    },
    /// Validation failed: fiducials are present but contain no dots.
    EmptyFiducialDots,
    /// Validation failed: a fiducial dot has a non-finite coordinate.
    NonFiniteDot {
        /// Index of the problematic dot in `dots_mm`.
        index: usize,
    },
    /// Validation failed: a fiducial dot disk overlaps a marker's drawn extent.
    DotOverlapsMarker {
        /// Index of the problematic dot in `dots_mm`.
        index: usize,
    },
    /// Validation failed: the fiducial dot pattern is invariant under a
    /// rotational symmetry of the cell lattice, so it cannot resolve the
    /// target's orientation.
    FiducialsRotationallySymmetric {
        /// The rotation angle (degrees) under which the dots are invariant.
        angle_deg: f32,
    },
}

impl std::fmt::Display for TargetValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema { found, expected } => write!(
                f,
                "unsupported target schema '{}' (expected {})",
                found, expected
            ),
            Self::EmptyName => f.write_str("target name must not be empty"),
            Self::InvalidPitch { pitch_mm } => {
                write!(f, "pitch_mm must be finite and > 0 (got {pitch_mm})")
            }
            Self::InvalidRows { rows } => write!(f, "rows must be >= 1 (got {rows})"),
            Self::InvalidCols { cols } => write!(f, "cols must be >= 1 (got {cols})"),
            Self::InvalidLongRowCols { long_row_cols } => {
                write!(f, "long_row_cols must be >= 1 (got {long_row_cols})")
            }
            Self::InvalidLongRowColsForRows {
                rows,
                long_row_cols,
            } => write!(
                f,
                "long_row_cols must be >= 2 when rows > 1 (got rows={}, long_row_cols={})",
                rows, long_row_cols
            ),
            Self::InvalidOuterRadius {
                marker_outer_radius_mm,
            } => write!(
                f,
                "marker_outer_radius_mm must be finite and > 0 (got {marker_outer_radius_mm})"
            ),
            Self::InvalidInnerRadius {
                marker_inner_radius_mm,
            } => write!(
                f,
                "marker_inner_radius_mm must be finite and > 0 (got {marker_inner_radius_mm})"
            ),
            Self::InvalidRingWidth {
                marker_ring_width_mm,
            } => write!(
                f,
                "marker_ring_width_mm must be finite and > 0 (got {marker_ring_width_mm})"
            ),
            Self::InnerRadiusNotSmallerThanOuter {
                marker_inner_radius_mm,
                marker_outer_radius_mm,
            } => write!(
                f,
                "marker_inner_radius_mm must be < marker_outer_radius_mm (inner={}, outer={})",
                marker_inner_radius_mm, marker_outer_radius_mm
            ),
            Self::NonPositiveCodeBandGap {
                inner_ring_outer_edge_mm,
                outer_ring_inner_edge_mm,
            } => write!(
                f,
                "marker geometry leaves no code band between rings (inner ring outer edge={inner_ring_outer_edge_mm:.4}mm, outer ring inner edge={outer_ring_inner_edge_mm:.4}mm)"
            ),
            Self::OuterDiameterExceedsMinCenterSpacing {
                marker_outer_diameter_mm,
                min_center_spacing_mm,
            } => write!(
                f,
                "marker outer diameter ({marker_outer_diameter_mm:.4}mm) must be smaller than minimum center spacing ({min_center_spacing_mm:.4}mm)"
            ),
            Self::MarkerDrawDiameterExceedsMinCenterSpacing {
                marker_draw_diameter_mm,
                min_center_spacing_mm,
            } => write!(
                f,
                "printed marker diameter including ring stroke ({marker_draw_diameter_mm:.4}mm) must be smaller than minimum center spacing ({min_center_spacing_mm:.4}mm)"
            ),
            Self::DerivedZeroColumns {
                row_index,
                rows,
                long_row_cols,
            } => write!(
                f,
                "derived row has zero columns at row {} (rows={}, long_row_cols={})",
                row_index, rows, long_row_cols
            ),
            Self::CodebookCapacityExceeded {
                n_cells,
                codebook_len,
            } => write!(
                f,
                "coded target has {n_cells} cells but the embedded codebook holds only {codebook_len} codewords"
            ),
            Self::IdAssignmentLength { expected, got } => write!(
                f,
                "id_assignment length ({got}) does not match marker count ({expected})"
            ),
            Self::IdAssignmentDuplicate { id, position } => write!(
                f,
                "id_assignment contains duplicate ID {id} at position {position}"
            ),
            Self::IdAssignmentOutOfRange {
                id,
                position,
                codebook_len,
            } => write!(
                f,
                "id_assignment contains ID {id} at position {position}, beyond the embedded codebook ({codebook_len} codewords)"
            ),
            Self::InvalidDotRadius { dot_radius_mm } => {
                write!(
                    f,
                    "dot_radius_mm must be finite and > 0 (got {dot_radius_mm})"
                )
            }
            Self::EmptyFiducialDots => {
                f.write_str("fiducials present but dots_mm is empty (omit fiducials instead)")
            }
            Self::NonFiniteDot { index } => {
                write!(f, "fiducial dot {index} has a non-finite coordinate")
            }
            Self::DotOverlapsMarker { index } => {
                write!(f, "fiducial dot {index} overlaps a marker's drawn extent")
            }
            Self::FiducialsRotationallySymmetric { angle_deg } => write!(
                f,
                "fiducial dots are invariant under a {angle_deg:.0}° lattice rotation and cannot resolve orientation"
            ),
        }
    }
}

impl std::error::Error for TargetValidationError {}

/// Load-time failures for target layout JSON.
#[derive(Debug)]
#[non_exhaustive]
pub enum TargetLoadError {
    /// File I/O error while reading the layout JSON.
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// JSON deserialization failed.
    JsonParse(serde_json::Error),
    /// Deserialized values failed validation.
    Validation(TargetValidationError),
}

impl std::fmt::Display for TargetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Io(err) => write!(f, "failed to read target JSON: {err}"),
            Self::JsonParse(err) => write!(f, "failed to parse target JSON: {err}"),
            Self::Validation(err) => write!(f, "invalid target spec: {err}"),
        }
    }
}

impl std::error::Error for TargetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "std")]
            Self::Io(err) => Some(err),
            Self::JsonParse(err) => Some(err),
            Self::Validation(err) => Some(err),
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for TargetLoadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for TargetLoadError {
    fn from(value: serde_json::Error) -> Self {
        Self::JsonParse(value)
    }
}

impl From<TargetValidationError> for TargetLoadError {
    fn from(value: TargetValidationError) -> Self {
        Self::Validation(value)
    }
}
