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
/// A single marker's position on the calibration board.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoardMarker {
    pub id: usize,
    pub xy_mm: [f32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q: Option<i16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r: Option<i16>,
}

/// Runtime board layout used by the detector.
#[derive(Debug, Clone)]
pub struct BoardLayout {
    pub name: String,
    pub pitch_mm: f32,
    pub rows: usize,
    pub long_row_cols: usize,
    pub marker_outer_radius_mm: f32,
    pub marker_inner_radius_mm: f32,
    pub markers: Vec<BoardMarker>,

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
    /// Build the internal ID->index lookup table.
    /// Must be called after manual marker modifications.
    pub fn build_index(&mut self) {
        self.id_to_idx = self
            .markers
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id, i))
            .collect();
    }

    /// Look up board coordinates (x, y) in mm for a given marker ID.
    pub fn xy_mm(&self, id: usize) -> Option<[f32; 2]> {
        self.id_to_idx.get(&id).map(|&idx| self.markers[idx].xy_mm)
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
    pub fn from_json_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let spec: BoardLayoutSpecV3 = serde_json::from_str(&data)?;
        Self::from_layout_spec(spec).map_err(Into::into)
    }

    fn from_layout_spec(spec: BoardLayoutSpecV3) -> Result<Self, String> {
        if spec.schema != TARGET_SCHEMA_V3 {
            return Err(format!(
                "unsupported target schema '{}' (expected '{}')",
                spec.schema, TARGET_SCHEMA_V3
            ));
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

fn validate_layout_spec(spec: &BoardLayoutSpecV3) -> Result<(), String> {
    if spec.name.trim().is_empty() {
        return Err("target name must not be empty".to_string());
    }

    if !spec.pitch_mm.is_finite() || spec.pitch_mm <= 0.0 {
        return Err("pitch_mm must be finite and > 0".to_string());
    }

    if spec.rows == 0 {
        return Err("rows must be >= 1".to_string());
    }

    if spec.long_row_cols == 0 {
        return Err("long_row_cols must be >= 1".to_string());
    }

    if spec.rows > 1 && spec.long_row_cols < 2 {
        return Err("long_row_cols must be >= 2 when rows > 1".to_string());
    }

    if !spec.marker_outer_radius_mm.is_finite() || spec.marker_outer_radius_mm <= 0.0 {
        return Err("marker_outer_radius_mm must be finite and > 0".to_string());
    }

    if !spec.marker_inner_radius_mm.is_finite() || spec.marker_inner_radius_mm <= 0.0 {
        return Err("marker_inner_radius_mm must be finite and > 0".to_string());
    }

    if spec.marker_inner_radius_mm >= spec.marker_outer_radius_mm {
        return Err("marker_inner_radius_mm must be < marker_outer_radius_mm".to_string());
    }

    let min_center_spacing = hex_row_spacing_mm(spec.pitch_mm);
    if spec.marker_outer_radius_mm * 2.0 >= min_center_spacing {
        return Err(format!(
            "marker outer diameter ({:.4}mm) must be smaller than minimum center spacing ({:.4}mm)",
            spec.marker_outer_radius_mm * 2.0,
            min_center_spacing
        ));
    }

    Ok(())
}

fn generate_markers(
    rows: usize,
    long_row_cols: usize,
    pitch_mm: f32,
) -> Result<Vec<BoardMarker>, String> {
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
            return Err("derived row has zero columns; increase long_row_cols".to_string());
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
            let marker = &board.markers[id];
            assert_eq!(xy, marker.xy_mm);
        }
        assert_eq!(board.xy_mm(999), None);
    }

    #[test]
    fn default_board_anchor_is_top_left_marker() {
        let board = BoardLayout::default();
        assert_eq!(board.xy_mm(0), Some([0.0, 0.0]));
        let anchor = board.markers[0].xy_mm;
        let min_y = board
            .markers
            .iter()
            .map(|m| m.xy_mm[1])
            .fold(f32::INFINITY, f32::min);
        let min_x_at_min_y = board
            .markers
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
        assert!(err.contains("unsupported target schema"));
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
}
