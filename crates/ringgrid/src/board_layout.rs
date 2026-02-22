//! Runtime target layout specification.
//!
//! Target JSON follows a parametric schema (`ringgrid.target.v3`): marker
//! locations are generated at runtime from `(rows, long_row_cols, pitch_mm)`.
//! Per-marker coordinate lists are intentionally not part of the runtime schema.

use std::collections::HashMap;
use std::path::Path;

const TARGET_SCHEMA_V3: &str = "ringgrid.target.v3";

const DEFAULT_NAME: &str = "ringgrid_200mm_hex";
const DEFAULT_PITCH_MM: f32 = 8.0;
const DEFAULT_ROWS: usize = 15;
const DEFAULT_LONG_ROW_COLS: usize = 14;
const DEFAULT_OUTER_RADIUS_MM: f32 = 4.8;
const DEFAULT_INNER_RADIUS_MM: f32 = 3.2;

/// Validation failures for a board layout specification.
#[derive(Debug, Clone)]
pub enum BoardLayoutValidationError {
    UnsupportedSchema {
        found: String,
        expected: &'static str,
    },
    EmptyName,
    InvalidPitch {
        pitch_mm: f32,
    },
    InvalidRows {
        rows: usize,
    },
    InvalidLongRowCols {
        long_row_cols: usize,
    },
    InvalidLongRowColsForRows {
        rows: usize,
        long_row_cols: usize,
    },
    InvalidOuterRadius {
        marker_outer_radius_mm: f32,
    },
    InvalidInnerRadius {
        marker_inner_radius_mm: f32,
    },
    InnerRadiusNotSmallerThanOuter {
        marker_inner_radius_mm: f32,
        marker_outer_radius_mm: f32,
    },
    OuterDiameterExceedsMinCenterSpacing {
        marker_outer_diameter_mm: f32,
        min_center_spacing_mm: f32,
    },
    DerivedZeroColumns {
        row_index: usize,
        rows: usize,
        long_row_cols: usize,
    },
}

impl std::fmt::Display for BoardLayoutValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema { found, expected } => write!(
                f,
                "unsupported target schema '{}' (expected '{}')",
                found, expected
            ),
            Self::EmptyName => f.write_str("target name must not be empty"),
            Self::InvalidPitch { pitch_mm } => {
                write!(f, "pitch_mm must be finite and > 0 (got {pitch_mm})")
            }
            Self::InvalidRows { rows } => write!(f, "rows must be >= 1 (got {rows})"),
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
            Self::InnerRadiusNotSmallerThanOuter {
                marker_inner_radius_mm,
                marker_outer_radius_mm,
            } => write!(
                f,
                "marker_inner_radius_mm must be < marker_outer_radius_mm (inner={}, outer={})",
                marker_inner_radius_mm, marker_outer_radius_mm
            ),
            Self::OuterDiameterExceedsMinCenterSpacing {
                marker_outer_diameter_mm,
                min_center_spacing_mm,
            } => write!(
                f,
                "marker outer diameter ({marker_outer_diameter_mm:.4}mm) must be smaller than minimum center spacing ({min_center_spacing_mm:.4}mm)"
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
        }
    }
}

impl std::error::Error for BoardLayoutValidationError {}

/// Load-time failures for board layout JSON.
#[derive(Debug)]
pub enum BoardLayoutLoadError {
    Io(std::io::Error),
    JsonParse(serde_json::Error),
    Validation(BoardLayoutValidationError),
}

impl std::fmt::Display for BoardLayoutLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read target JSON: {err}"),
            Self::JsonParse(err) => write!(f, "failed to parse target JSON: {err}"),
            Self::Validation(err) => write!(f, "invalid target spec: {err}"),
        }
    }
}

impl std::error::Error for BoardLayoutLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::JsonParse(err) => Some(err),
            Self::Validation(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for BoardLayoutLoadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for BoardLayoutLoadError {
    fn from(value: serde_json::Error) -> Self {
        Self::JsonParse(value)
    }
}

impl From<BoardLayoutValidationError> for BoardLayoutLoadError {
    fn from(value: BoardLayoutValidationError) -> Self {
        Self::Validation(value)
    }
}
/// A single marker's position on the calibration board.
///
/// Each marker has a unique `id` (codebook index 0–892), a physical position
/// `xy_mm` on the board, and optional hex-lattice axial coordinates `(q, r)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoardMarker {
    /// Unique marker ID (codebook index, 0–892).
    pub id: usize,
    /// Position on the board in millimeters `[x, y]`.
    pub xy_mm: [f32; 2],
    /// Hex-lattice axial coordinate q (column offset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q: Option<i16>,
    /// Hex-lattice axial coordinate r (row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r: Option<i16>,
}

/// Runtime board layout used by the detector.
///
/// Describes the physical hex-lattice arrangement of ring markers: their
/// positions in millimeters, ring radii, and lattice parameters. Load from
/// a JSON file conforming to `ringgrid.target.v3` schema, or use the
/// built-in default via [`BoardLayout::default()`].
///
/// # Example
///
/// ```no_run
/// use ringgrid::BoardLayout;
/// use std::path::Path;
///
/// let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
/// println!("{} markers, pitch={} mm", board.n_markers(), board.pitch_mm);
/// ```
#[derive(Debug, Clone)]
pub struct BoardLayout {
    pub name: String,
    pub pitch_mm: f32,
    pub rows: usize,
    pub long_row_cols: usize,
    pub marker_outer_radius_mm: f32,
    pub marker_inner_radius_mm: f32,
    markers: Vec<BoardMarker>,

    /// Fast lookup: marker ID -> index into `markers`.
    id_to_idx: HashMap<usize, usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BoardLayoutSpecV3 {
    schema: String,
    name: String,
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    marker_outer_radius_mm: f32,
    marker_inner_radius_mm: f32,
}

impl BoardLayout {
    /// Look up board coordinates (x, y) in mm for a given marker ID.
    pub fn xy_mm(&self, id: usize) -> Option<[f32; 2]> {
        self.id_to_idx.get(&id).map(|&idx| self.markers[idx].xy_mm)
    }

    /// Look up a marker by ID.
    pub fn marker(&self, id: usize) -> Option<&BoardMarker> {
        self.id_to_idx.get(&id).map(|&idx| &self.markers[idx])
    }

    /// Borrow all markers as a read-only slice.
    pub fn markers(&self) -> &[BoardMarker] {
        &self.markers
    }

    /// Look up a marker by storage index.
    pub fn marker_by_index(&self, index: usize) -> Option<&BoardMarker> {
        self.markers.get(index)
    }

    /// Total number of markers on the board.
    pub fn n_markers(&self) -> usize {
        self.markers.len()
    }

    /// Marker outer radius in board units (mm).
    pub fn marker_outer_radius_mm(&self) -> f32 {
        self.marker_outer_radius_mm
    }

    /// Marker inner radius in board units (mm).
    pub fn marker_inner_radius_mm(&self) -> f32 {
        self.marker_inner_radius_mm
    }

    /// Iterator over all marker IDs present on the board.
    pub fn marker_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.markers.iter().map(|m| m.id)
    }

    /// Maximum marker ID present on the board.
    pub fn max_marker_id(&self) -> usize {
        self.markers.iter().map(|m| m.id).max().unwrap_or(0)
    }

    /// Axis-aligned marker bounds in board mm.
    ///
    /// Returns `(min_xy, max_xy)` over marker centers.
    pub fn marker_bounds_mm(&self) -> Option<([f32; 2], [f32; 2])> {
        let first = self.markers.first()?;
        let mut min_x = first.xy_mm[0];
        let mut max_x = first.xy_mm[0];
        let mut min_y = first.xy_mm[1];
        let mut max_y = first.xy_mm[1];

        for m in &self.markers[1..] {
            min_x = min_x.min(m.xy_mm[0]);
            max_x = max_x.max(m.xy_mm[0]);
            min_y = min_y.min(m.xy_mm[1]);
            max_y = max_y.max(m.xy_mm[1]);
        }

        Some(([min_x, min_y], [max_x, max_y]))
    }

    /// Axis-aligned marker span in board mm (`[width, height]`).
    pub fn marker_span_mm(&self) -> Option<[f32; 2]> {
        self.marker_bounds_mm()
            .map(|(min_xy, max_xy)| [max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]])
    }

    /// Load a board layout from a JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self, BoardLayoutLoadError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Load a board layout from a JSON string.
    pub fn from_json_str(data: &str) -> Result<Self, BoardLayoutLoadError> {
        let spec: BoardLayoutSpecV3 = serde_json::from_str(data)?;
        Self::from_layout_spec(spec).map_err(Into::into)
    }

    fn from_layout_spec(spec: BoardLayoutSpecV3) -> Result<Self, BoardLayoutValidationError> {
        if spec.schema != TARGET_SCHEMA_V3 {
            return Err(BoardLayoutValidationError::UnsupportedSchema {
                found: spec.schema,
                expected: TARGET_SCHEMA_V3,
            });
        }

        validate_layout_spec(&spec)?;
        let markers = generate_markers(spec.rows, spec.long_row_cols, spec.pitch_mm)?;
        let id_to_idx = markers.iter().enumerate().map(|(i, m)| (m.id, i)).collect();

        Ok(Self {
            name: spec.name,
            pitch_mm: spec.pitch_mm,
            rows: spec.rows,
            long_row_cols: spec.long_row_cols,
            marker_outer_radius_mm: spec.marker_outer_radius_mm,
            marker_inner_radius_mm: spec.marker_inner_radius_mm,
            markers,
            id_to_idx,
        })
    }
}

impl Default for BoardLayout {
    fn default() -> Self {
        let spec = BoardLayoutSpecV3 {
            schema: TARGET_SCHEMA_V3.to_string(),
            name: DEFAULT_NAME.to_string(),
            pitch_mm: DEFAULT_PITCH_MM,
            rows: DEFAULT_ROWS,
            long_row_cols: DEFAULT_LONG_ROW_COLS,
            marker_outer_radius_mm: DEFAULT_OUTER_RADIUS_MM,
            marker_inner_radius_mm: DEFAULT_INNER_RADIUS_MM,
        };

        Self::from_layout_spec(spec).expect("default board spec must be valid")
    }
}

fn validate_layout_spec(spec: &BoardLayoutSpecV3) -> Result<(), BoardLayoutValidationError> {
    if spec.name.trim().is_empty() {
        return Err(BoardLayoutValidationError::EmptyName);
    }

    if !spec.pitch_mm.is_finite() || spec.pitch_mm <= 0.0 {
        return Err(BoardLayoutValidationError::InvalidPitch {
            pitch_mm: spec.pitch_mm,
        });
    }

    if spec.rows == 0 {
        return Err(BoardLayoutValidationError::InvalidRows { rows: spec.rows });
    }

    if spec.long_row_cols == 0 {
        return Err(BoardLayoutValidationError::InvalidLongRowCols {
            long_row_cols: spec.long_row_cols,
        });
    }

    if spec.rows > 1 && spec.long_row_cols < 2 {
        return Err(BoardLayoutValidationError::InvalidLongRowColsForRows {
            rows: spec.rows,
            long_row_cols: spec.long_row_cols,
        });
    }

    if !spec.marker_outer_radius_mm.is_finite() || spec.marker_outer_radius_mm <= 0.0 {
        return Err(BoardLayoutValidationError::InvalidOuterRadius {
            marker_outer_radius_mm: spec.marker_outer_radius_mm,
        });
    }

    if !spec.marker_inner_radius_mm.is_finite() || spec.marker_inner_radius_mm <= 0.0 {
        return Err(BoardLayoutValidationError::InvalidInnerRadius {
            marker_inner_radius_mm: spec.marker_inner_radius_mm,
        });
    }

    if spec.marker_inner_radius_mm >= spec.marker_outer_radius_mm {
        return Err(BoardLayoutValidationError::InnerRadiusNotSmallerThanOuter {
            marker_inner_radius_mm: spec.marker_inner_radius_mm,
            marker_outer_radius_mm: spec.marker_outer_radius_mm,
        });
    }

    let min_center_spacing = hex_row_spacing_mm(spec.pitch_mm);
    if spec.marker_outer_radius_mm * 2.0 >= min_center_spacing {
        return Err(
            BoardLayoutValidationError::OuterDiameterExceedsMinCenterSpacing {
                marker_outer_diameter_mm: spec.marker_outer_radius_mm * 2.0,
                min_center_spacing_mm: min_center_spacing,
            },
        );
    }

    Ok(())
}

fn generate_markers(
    rows: usize,
    long_row_cols: usize,
    pitch_mm: f32,
) -> Result<Vec<BoardMarker>, BoardLayoutValidationError> {
    let short_row_cols = long_row_cols.saturating_sub(1);
    let mut markers = Vec::new();
    let row_mid = (rows as i32) / 2;

    for row_idx in 0..rows {
        let r = row_idx as i32 - row_mid;
        let n_cols = if rows == 1 || ((r + long_row_cols as i32 - 1) & 1) == 0 {
            long_row_cols
        } else {
            short_row_cols
        };

        if n_cols == 0 {
            return Err(BoardLayoutValidationError::DerivedZeroColumns {
                row_index: row_idx,
                rows,
                long_row_cols,
            });
        }

        let q_start = -((r + n_cols as i32 - 1) / 2);
        for col_idx in 0..n_cols {
            let q = q_start + col_idx as i32;
            let xy = hex_axial_to_xy_mm(q, r, pitch_mm);
            markers.push(BoardMarker {
                id: markers.len(),
                xy_mm: xy,
                q: i16::try_from(q).ok(),
                r: i16::try_from(r).ok(),
            });
        }
    }

    normalize_marker_origin(&mut markers);
    Ok(markers)
}

fn hex_axial_to_xy_mm(q: i32, r: i32, pitch_mm: f32) -> [f32; 2] {
    let qf = q as f64;
    let rf = r as f64;
    let pitch = pitch_mm as f64;
    let x = pitch * (f64::sqrt(3.0) * qf + 0.5 * f64::sqrt(3.0) * rf);
    let y = pitch * (1.5 * rf);
    [x as f32, y as f32]
}

fn normalize_marker_origin(markers: &mut [BoardMarker]) {
    let Some(anchor) = markers.first().map(|m| m.xy_mm) else {
        return;
    };
    for marker in markers {
        marker.xy_mm[0] -= anchor[0];
        marker.xy_mm[1] -= anchor[1];
    }
}

fn hex_row_spacing_mm(pitch_mm: f32) -> f32 {
    // Hex nearest-neighbor distance in this axial layout.
    pitch_mm * f32::sqrt(3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_json_path(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ringgrid_{prefix}_{}_{}.json",
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn default_board_has_expected_shape() {
        let board = BoardLayout::default();
        assert_eq!(board.rows, 15);
        assert_eq!(board.long_row_cols, 14);
        assert_eq!(board.n_markers(), 203);
        assert_eq!(board.xy_mm(0), Some([0.0, 0.0]));
        assert_eq!(board.xy_mm(20), Some([90.06664, 12.0]));
    }

    #[test]
    fn default_board_lookup_stays_consistent() {
        let board = BoardLayout::default();
        for id in 0..board.n_markers() {
            let xy = board.xy_mm(id).expect("valid id");
            let marker = board.marker_by_index(id).expect("marker index");
            assert_eq!(xy, marker.xy_mm);
        }
        assert_eq!(board.xy_mm(999), None);
    }

    #[test]
    fn default_board_anchor_is_top_left_marker() {
        let board = BoardLayout::default();
        assert_eq!(board.xy_mm(0), Some([0.0, 0.0]));
        let anchor = board.marker_by_index(0).expect("marker 0").xy_mm;
        let min_y = board
            .markers()
            .iter()
            .map(|m| m.xy_mm[1])
            .fold(f32::INFINITY, f32::min);
        let min_x_at_min_y = board
            .markers()
            .iter()
            .filter(|m| (m.xy_mm[1] - min_y).abs() < 1e-6)
            .map(|m| m.xy_mm[0])
            .fold(f32::INFINITY, f32::min);
        assert!((anchor[1] - min_y).abs() < 1e-6);
        assert!((anchor[0] - min_x_at_min_y).abs() < 1e-6);
    }

    #[test]
    fn from_json_requires_v3_schema() {
        let raw = r#"{
            "schema":"ringgrid.target.v2",
            "name":"x",
            "pitch_mm":8.0,
            "rows":1,
            "long_row_cols":1,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2
        }"#;
        let spec: BoardLayoutSpecV3 = serde_json::from_str(raw).expect("valid json");
        let err = BoardLayout::from_layout_spec(spec).expect_err("expected error");
        assert!(matches!(
            err,
            BoardLayoutValidationError::UnsupportedSchema { .. }
        ));
    }

    #[test]
    fn from_json_rejects_marker_list_field() {
        let raw = r#"{
            "schema":"ringgrid.target.v3",
            "name":"x",
            "pitch_mm":8.0,
            "rows":1,
            "long_row_cols":1,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,
            "markers":[{"id":0,"xy_mm":[0.0,0.0]}]
        }"#;
        let parsed: Result<BoardLayoutSpecV3, _> = serde_json::from_str(raw);
        assert!(parsed.is_err());
    }

    #[test]
    fn from_json_rejects_legacy_fields() {
        let raw = r#"{
            "schema":"ringgrid.target.v3",
            "name":"x",
            "pitch_mm":8.0,
            "rows":3,
            "long_row_cols":4,
            "origin_mm":[0.0,0.0],
            "board_size_mm":[200.0,200.0],
            "marker_code_band_outer_radius_mm":4.64,
            "marker_code_band_inner_radius_mm":3.36,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2
        }"#;
        let parsed: Result<BoardLayoutSpecV3, _> = serde_json::from_str(raw);
        assert!(parsed.is_err());
    }

    #[test]
    fn marker_span_is_positive() {
        let board = BoardLayout::default();
        let span = board.marker_span_mm().expect("span");
        assert!(span[0] > 0.0);
        assert!(span[1] > 0.0);
    }

    #[test]
    fn from_json_file_maps_io_error_to_typed_variant() {
        let missing = temp_json_path("missing_board");
        let err = BoardLayout::from_json_file(&missing).expect_err("expected io error");
        assert!(matches!(err, BoardLayoutLoadError::Io(_)));
    }

    #[test]
    fn from_json_file_maps_parse_error_to_typed_variant() {
        let path = temp_json_path("bad_json");
        std::fs::write(&path, "{ this is not valid json").expect("write temp json");

        let err = BoardLayout::from_json_file(&path).expect_err("expected parse error");
        assert!(matches!(err, BoardLayoutLoadError::JsonParse(_)));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn from_json_file_maps_validation_error_to_typed_variant() {
        let path = temp_json_path("bad_schema");
        let raw = r#"{
            "schema":"ringgrid.target.v2",
            "name":"x",
            "pitch_mm":8.0,
            "rows":1,
            "long_row_cols":1,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2
        }"#;
        std::fs::write(&path, raw).expect("write temp json");

        let err = BoardLayout::from_json_file(&path).expect_err("expected validation error");
        assert!(matches!(
            err,
            BoardLayoutLoadError::Validation(BoardLayoutValidationError::UnsupportedSchema { .. })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn from_json_str_loads_valid_spec() {
        let raw = r#"{
            "schema":"ringgrid.target.v3",
            "name":"x",
            "pitch_mm":8.0,
            "rows":3,
            "long_row_cols":4,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2
        }"#;

        let board = BoardLayout::from_json_str(raw).expect("valid board json");
        assert_eq!(board.name, "x");
        assert_eq!(board.rows, 3);
        assert_eq!(board.long_row_cols, 4);
        assert!(board.n_markers() > 0);
    }
}
