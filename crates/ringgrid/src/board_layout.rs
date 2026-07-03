//! Legacy flat hex board layout — a facade over the compositional target
//! model in [`crate::target`].
//!
//! `BoardLayout` describes the classic hex-lattice coded target through the
//! flat `ringgrid.target.v4` schema. New code should use
//! [`TargetLayout`](crate::TargetLayout), which additionally models
//! rectangular lattices, plain (uncoded) rings, and origin fiducials, and
//! reads/writes the compositional `ringgrid.target.v5` schema. This facade
//! remains for one release; all validation and cell generation delegate to
//! the target module, so both types produce bit-identical hex geometry.

// The deprecated facade's own impls, helpers, and tests necessarily use the
// deprecated types.
#![allow(deprecated)]

use std::collections::HashMap;
#[cfg(feature = "std")]
use std::path::Path;

use crate::target::{
    BoardSpecV4, CodedRingSpec, HexGeometry, LatticeGeometry, MarkerCoding, RingGeometry,
    TARGET_SCHEMA_V4, TargetLayout,
};

/// Validation failures for a board layout specification.
///
/// Alias of [`TargetValidationError`](crate::TargetValidationError); kept for
/// source compatibility with pre-0.8 code.
#[deprecated(since = "0.8.0", note = "use `TargetValidationError` instead")]
pub type BoardLayoutValidationError = crate::target::TargetValidationError;

/// Load-time failures for board layout JSON.
///
/// Alias of [`TargetLoadError`](crate::TargetLoadError); kept for source
/// compatibility with pre-0.8 code.
#[deprecated(since = "0.8.0", note = "use `TargetLoadError` instead")]
pub type BoardLayoutLoadError = crate::target::TargetLoadError;

/// A single marker's position on the calibration board.
///
/// Each marker has a unique `id` (codebook index in the active profile), a
/// physical position `xy_mm` on the board, and optional hex-lattice axial
/// coordinates `(q, r)`.
#[deprecated(
    since = "0.8.0",
    note = "use `TargetLayout` (compositional target model, `ringgrid.target.v5` schema) instead; this v4 facade will be removed after 0.8"
)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoardMarker {
    /// Unique marker ID (codebook index in the active profile).
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

/// Runtime hex board layout (legacy facade).
///
/// Describes the physical hex-lattice arrangement of coded ring markers:
/// their positions in millimeters, ring radii, ring width, and lattice
/// parameters. Load from a JSON file conforming to the `ringgrid.target.v4`
/// schema, or use the built-in default via [`BoardLayout::default()`].
///
/// # Example
///
/// ```no_run
/// use ringgrid::BoardLayout;
/// use std::path::Path;
///
/// let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
/// println!("{} markers, pitch={} mm", board.n_markers(), board.pitch_mm());
/// ```
///
/// Scalar geometry fields are not publicly mutable: they are coupled to a
/// derived marker cache that an in-place mutation would silently desync.
/// Construct via [`BoardLayout::default`], [`BoardLayout::new`],
/// [`BoardLayout::with_name`], or [`BoardLayout::from_json_file`]; read the
/// geometry through the accessor methods.
#[deprecated(
    since = "0.8.0",
    note = "use `TargetLayout` (compositional target model, `ringgrid.target.v5` schema) instead; this v4 facade will be removed after 0.8"
)]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BoardLayout {
    pub(crate) name: String,
    pub(crate) pitch_mm: f32,
    pub(crate) rows: usize,
    pub(crate) long_row_cols: usize,
    pub(crate) marker_outer_radius_mm: f32,
    pub(crate) marker_inner_radius_mm: f32,
    pub(crate) marker_ring_width_mm: f32,
    markers: Vec<BoardMarker>,

    /// Fast lookup: marker ID -> index into `markers`.
    id_to_idx: HashMap<usize, usize>,
}

impl BoardLayout {
    /// Construct a board layout from direct geometry arguments.
    ///
    /// Uses a deterministic geometry-derived name so the layout can round-trip
    /// through the canonical `ringgrid.target.v4` JSON schema without requiring
    /// the caller to supply a name up front.
    pub fn new(
        pitch_mm: f32,
        rows: usize,
        long_row_cols: usize,
        marker_outer_radius_mm: f32,
        marker_inner_radius_mm: f32,
        marker_ring_width_mm: f32,
    ) -> Result<Self, BoardLayoutValidationError> {
        Self::with_name(
            crate::target::layout::hex_generated_name(
                pitch_mm,
                rows,
                long_row_cols,
                marker_outer_radius_mm,
                marker_inner_radius_mm,
                marker_ring_width_mm,
            ),
            pitch_mm,
            rows,
            long_row_cols,
            marker_outer_radius_mm,
            marker_inner_radius_mm,
            marker_ring_width_mm,
        )
    }

    /// Construct a named board layout from direct geometry arguments.
    pub fn with_name<S: Into<String>>(
        name: S,
        pitch_mm: f32,
        rows: usize,
        long_row_cols: usize,
        marker_outer_radius_mm: f32,
        marker_inner_radius_mm: f32,
        marker_ring_width_mm: f32,
    ) -> Result<Self, BoardLayoutValidationError> {
        Self::from_layout_spec(BoardSpecV4 {
            schema: TARGET_SCHEMA_V4.to_string(),
            name: name.into(),
            pitch_mm,
            rows,
            long_row_cols,
            marker_outer_radius_mm,
            marker_inner_radius_mm,
            marker_ring_width_mm,
            id_assignment: None,
        })
    }

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

    /// Human-readable name of the target layout.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Center-to-center spacing between adjacent markers (mm).
    pub fn pitch_mm(&self) -> f32 {
        self.pitch_mm
    }

    /// Number of marker rows on the board.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns in the longest (even-indexed) row.
    pub fn long_row_cols(&self) -> usize {
        self.long_row_cols
    }

    /// Marker outer radius in board units (mm).
    pub fn marker_outer_radius_mm(&self) -> f32 {
        self.marker_outer_radius_mm
    }

    /// Marker inner radius in board units (mm).
    pub fn marker_inner_radius_mm(&self) -> f32 {
        self.marker_inner_radius_mm
    }

    /// Marker ring width in board units (mm).
    pub fn marker_ring_width_mm(&self) -> f32 {
        self.marker_ring_width_mm
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

    /// Load a board layout from a JSON file (`ringgrid.target.v4`).
    #[cfg(feature = "std")]
    pub fn from_json_file(path: &Path) -> Result<Self, BoardLayoutLoadError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Load a board layout from a JSON string (`ringgrid.target.v4`).
    pub fn from_json_str(data: &str) -> Result<Self, BoardLayoutLoadError> {
        let spec: BoardSpecV4 = serde_json::from_str(data)?;
        Self::from_layout_spec(spec).map_err(Into::into)
    }

    /// Serialize the layout as canonical `ringgrid.target.v4` JSON.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_layout_spec())
            .expect("board layout JSON serialization must succeed")
    }

    /// Write the canonical `ringgrid.target.v4` JSON representation to disk.
    #[cfg(feature = "std")]
    pub fn write_json_file(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, format!("{}\n", self.to_json_string()))
    }

    /// Validation and marker generation delegate to the target module so the
    /// hex geometry has exactly one source of truth.
    pub(crate) fn from_layout_spec(spec: BoardSpecV4) -> Result<Self, BoardLayoutValidationError> {
        let target = spec.into_layout()?;
        Ok(Self::from_hex_target(&target))
    }

    /// Extract the flat facade fields from a validated hex coded target.
    fn from_hex_target(target: &TargetLayout) -> Self {
        let LatticeGeometry::Hex(hex) = *target.lattice() else {
            unreachable!("BoardLayout only wraps hex targets")
        };
        let MarkerCoding::Coded16(spec) = target.coding() else {
            unreachable!("BoardLayout only wraps coded targets")
        };
        let markers: Vec<BoardMarker> = target
            .cells()
            .iter()
            .map(|cell| BoardMarker {
                id: cell.id.expect("coded target cells always carry IDs"),
                xy_mm: cell.xy_mm,
                q: i16::try_from(cell.coord.u).ok(),
                r: i16::try_from(cell.coord.v).ok(),
            })
            .collect();
        let id_to_idx = markers.iter().enumerate().map(|(i, m)| (m.id, i)).collect();

        Self {
            name: target.name().to_string(),
            pitch_mm: hex.pitch_mm,
            rows: hex.rows,
            long_row_cols: hex.long_row_cols,
            marker_outer_radius_mm: target.ring().outer_radius_mm,
            marker_inner_radius_mm: target.ring().inner_radius_mm,
            marker_ring_width_mm: spec.ring_width_mm,
            markers,
            id_to_idx,
        }
    }

    fn to_layout_spec(&self) -> BoardSpecV4 {
        let is_sequential = self.markers.iter().enumerate().all(|(i, m)| m.id == i);
        let id_assignment = if is_sequential {
            None
        } else {
            Some(self.markers.iter().map(|m| m.id).collect())
        };
        BoardSpecV4 {
            schema: TARGET_SCHEMA_V4.to_string(),
            name: self.name.clone(),
            pitch_mm: self.pitch_mm,
            rows: self.rows,
            long_row_cols: self.long_row_cols,
            marker_outer_radius_mm: self.marker_outer_radius_mm,
            marker_inner_radius_mm: self.marker_inner_radius_mm,
            marker_ring_width_mm: self.marker_ring_width_mm,
            id_assignment,
        }
    }
}

impl Default for BoardLayout {
    fn default() -> Self {
        Self::from_hex_target(&TargetLayout::default_hex())
    }
}

impl From<BoardLayout> for TargetLayout {
    fn from(board: BoardLayout) -> Self {
        let is_sequential = board.markers.iter().enumerate().all(|(i, m)| m.id == i);
        let id_assignment = if is_sequential {
            None
        } else {
            Some(board.markers.iter().map(|m| m.id).collect())
        };
        TargetLayout::new(
            board.name,
            LatticeGeometry::Hex(HexGeometry {
                rows: board.rows,
                long_row_cols: board.long_row_cols,
                pitch_mm: board.pitch_mm,
            }),
            RingGeometry {
                outer_radius_mm: board.marker_outer_radius_mm,
                inner_radius_mm: board.marker_inner_radius_mm,
            },
            MarkerCoding::Coded16(CodedRingSpec {
                ring_width_mm: board.marker_ring_width_mm,
                id_assignment,
            }),
            None,
        )
        .expect("a validated BoardLayout is always a valid hex coded target")
    }
}

impl From<&BoardLayout> for TargetLayout {
    fn from(board: &BoardLayout) -> Self {
        board.clone().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(feature = "std")]
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
    fn default_board_min_center_spacing_matches_hex_pitch() {
        let target = TargetLayout::from(&BoardLayout::default());
        let expected = 8.0 * f32::sqrt(3.0);
        assert!((target.min_center_spacing_mm() - expected).abs() < 1.0e-6);
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
    fn from_json_requires_v4_schema() {
        let raw = r#"{
            "schema":"ringgrid.target.v2",
            "name":"x",
            "pitch_mm":8.0,
            "rows":1,
            "long_row_cols":1,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,
            "marker_ring_width_mm":1.152
        }"#;
        let err = BoardLayout::from_json_str(raw).expect_err("expected error");
        assert!(matches!(
            err,
            BoardLayoutLoadError::Validation(BoardLayoutValidationError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn from_json_rejects_marker_list_field() {
        let raw = r#"{
            "schema":"ringgrid.target.v4",
            "name":"x",
            "pitch_mm":8.0,
            "rows":1,
            "long_row_cols":1,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,
            "marker_ring_width_mm":1.152,
            "markers":[{"id":0,"xy_mm":[0.0,0.0]}]
        }"#;
        assert!(BoardLayout::from_json_str(raw).is_err());
    }

    #[test]
    fn from_json_rejects_legacy_fields() {
        let raw = r#"{
            "schema":"ringgrid.target.v4",
            "name":"x",
            "pitch_mm":8.0,
            "rows":3,
            "long_row_cols":4,
            "origin_mm":[0.0,0.0],
            "board_size_mm":[200.0,200.0],
            "marker_code_band_outer_radius_mm":4.64,
            "marker_code_band_inner_radius_mm":3.36,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,
            "marker_ring_width_mm":1.152
        }"#;
        assert!(BoardLayout::from_json_str(raw).is_err());
    }

    #[test]
    fn marker_span_is_positive() {
        let board = BoardLayout::default();
        let span = board.marker_span_mm().expect("span");
        assert!(span[0] > 0.0);
        assert!(span[1] > 0.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_json_file_maps_io_error_to_typed_variant() {
        let missing = temp_json_path("missing_board");
        let err = BoardLayout::from_json_file(&missing).expect_err("expected io error");
        assert!(matches!(err, BoardLayoutLoadError::Io(_)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_json_file_maps_parse_error_to_typed_variant() {
        let path = temp_json_path("bad_json");
        std::fs::write(&path, "{ this is not valid json").expect("write temp json");

        let err = BoardLayout::from_json_file(&path).expect_err("expected parse error");
        assert!(matches!(err, BoardLayoutLoadError::JsonParse(_)));

        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "std")]
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
            "marker_inner_radius_mm":3.2,
            "marker_ring_width_mm":1.152
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
            "schema":"ringgrid.target.v4",
            "name":"x",
            "pitch_mm":8.0,
            "rows":3,
            "long_row_cols":4,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,
            "marker_ring_width_mm":1.152
        }"#;

        let board = BoardLayout::from_json_str(raw).expect("valid board json");
        assert_eq!(board.name, "x");
        assert_eq!(board.rows, 3);
        assert_eq!(board.long_row_cols, 4);
        assert!(board.n_markers() > 0);
    }

    #[test]
    fn direct_constructor_matches_round_trip_json() {
        let board = BoardLayout::with_name("fixture_compact_hex", 8.0, 3, 4, 4.8, 3.2, 1.152)
            .expect("valid direct geometry");

        let json = board.to_json_string();
        let reloaded = BoardLayout::from_json_str(&json).expect("round-trip json");

        assert_eq!(reloaded.name, "fixture_compact_hex");
        assert_eq!(reloaded.rows, 3);
        assert_eq!(reloaded.long_row_cols, 4);
        assert_eq!(reloaded.marker_outer_radius_mm, 4.8);
        assert_eq!(reloaded.marker_inner_radius_mm, 3.2);
        assert!((reloaded.marker_ring_width_mm - 1.152).abs() < 1e-6);
        assert_eq!(reloaded.markers().len(), board.markers().len());
        assert_eq!(reloaded.xy_mm(0), Some([0.0, 0.0]));
    }

    #[test]
    fn direct_constructor_uses_deterministic_default_name() {
        let board = BoardLayout::new(8.0, 3, 4, 4.8, 3.2, 1.152).expect("valid direct geometry");
        assert_eq!(board.name, "ringgrid_hex_r3_c4_p8.000_o4.800_i3.200_w1.152");
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_json_file_creates_parent_dirs_and_round_trips() {
        let path = temp_json_path("round_trip");
        let nested = path.with_file_name("nested").join("board.json");
        let board = BoardLayout::with_name("fixture_compact_hex", 8.0, 3, 4, 4.8, 3.2, 1.152)
            .expect("valid direct geometry");

        board
            .write_json_file(&nested)
            .expect("write nested board json");
        let loaded = BoardLayout::from_json_file(&nested).expect("load nested board json");

        assert_eq!(loaded.name, board.name);
        assert_eq!(loaded.markers().len(), board.markers().len());

        let _ = std::fs::remove_file(&nested);
        let _ = std::fs::remove_dir(nested.parent().expect("nested parent"));
    }

    #[test]
    fn direct_constructor_reuses_layout_validation() {
        assert!(matches!(
            BoardLayout::new(8.0, 0, 4, 4.8, 3.2, 1.152),
            Err(BoardLayoutValidationError::InvalidRows { rows: 0 })
        ));
        assert!(matches!(
            BoardLayout::new(8.0, 3, 1, 4.8, 3.2, 1.152),
            Err(BoardLayoutValidationError::InvalidLongRowColsForRows {
                rows: 3,
                long_row_cols: 1,
            })
        ));
        assert!(matches!(
            BoardLayout::new(8.0, 3, 4, 4.8, 4.8, 1.152),
            Err(BoardLayoutValidationError::InnerRadiusNotSmallerThanOuter {
                marker_inner_radius_mm: 4.8,
                marker_outer_radius_mm: 4.8,
            })
        ));
        assert!(matches!(
            BoardLayout::new(8.0, 3, 4, 4.8, 4.1, 1.152),
            Err(BoardLayoutValidationError::NonPositiveCodeBandGap { .. })
        ));
        assert!(matches!(
            BoardLayout::new(f32::NAN, 3, 4, 4.8, 3.2, 1.152),
            Err(BoardLayoutValidationError::InvalidPitch { .. })
        ));
        assert!(matches!(
            BoardLayout::new(5.0, 3, 4, 4.0, 2.0, 1.152),
            Err(BoardLayoutValidationError::MarkerDrawDiameterExceedsMinCenterSpacing { .. })
        ));
        assert!(matches!(
            BoardLayout::new(8.0, 3, 4, 4.8, 3.2, 0.0),
            Err(BoardLayoutValidationError::InvalidRingWidth { .. })
        ));
    }

    #[test]
    fn conversion_to_target_layout_round_trips() {
        let board = BoardLayout::default();
        let target: TargetLayout = (&board).into();
        assert_eq!(target.n_cells(), board.n_markers());
        for (cell, marker) in target.cells().iter().zip(board.markers()) {
            assert_eq!(cell.id, Some(marker.id));
            assert_eq!(cell.xy_mm, marker.xy_mm);
        }
    }

    // ── id_assignment tests ───────────────────────────────────────

    #[cfg(feature = "std")]
    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[cfg(feature = "std")]
    #[test]
    fn id_assignment_loads_and_remaps() {
        let path = repo_root().join("tools/board/board_spec_optimized.json");
        let board = BoardLayout::from_json_file(&path).unwrap();
        assert_eq!(board.n_markers(), 203);

        // Read raw JSON to get the assignment array
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let assignment: Vec<usize> = raw["id_assignment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        // Each marker's ID should match the assignment
        for (i, &expected_id) in assignment.iter().enumerate() {
            let marker = board.marker_by_index(i).unwrap();
            assert_eq!(
                marker.id, expected_id,
                "marker at index {i}: expected ID {expected_id}, got {}",
                marker.id
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn id_assignment_roundtrip() {
        let path = repo_root().join("tools/board/board_spec_optimized.json");
        let board = BoardLayout::from_json_file(&path).unwrap();

        // Serialize and deserialize
        let json = board.to_json_string();
        let board2 = BoardLayout::from_json_str(&json).unwrap();

        assert_eq!(board.n_markers(), board2.n_markers());
        for i in 0..board.n_markers() {
            let m1 = board.marker_by_index(i).unwrap();
            let m2 = board2.marker_by_index(i).unwrap();
            assert_eq!(m1.id, m2.id, "ID mismatch at index {i}");
            assert!(
                (m1.xy_mm[0] - m2.xy_mm[0]).abs() < 1e-4
                    && (m1.xy_mm[1] - m2.xy_mm[1]).abs() < 1e-4,
                "position mismatch at index {i}"
            );
        }
    }

    #[test]
    fn id_assignment_rejects_wrong_length() {
        // Build a small valid board JSON, then add wrong-length id_assignment
        let board = BoardLayout::default();
        let json = board.to_json_string();
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        val["id_assignment"] = serde_json::json!([0, 1, 2]); // only 3, need 203
        let bad_json = serde_json::to_string(&val).unwrap();
        let err = BoardLayout::from_json_str(&bad_json).unwrap_err();
        assert!(
            err.to_string().contains("id_assignment length"),
            "expected IdAssignmentLength error, got: {err}"
        );
    }

    #[test]
    fn id_assignment_rejects_duplicates() {
        let board = BoardLayout::default();
        let json = board.to_json_string();
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        // Create assignment with correct length but duplicate ID at positions 0 and 1
        let n = board.n_markers();
        let mut ids: Vec<usize> = (0..n).collect();
        ids[1] = ids[0]; // duplicate
        val["id_assignment"] = serde_json::json!(ids);
        let bad_json = serde_json::to_string(&val).unwrap();
        let err = BoardLayout::from_json_str(&bad_json).unwrap_err();
        assert!(
            err.to_string().contains("duplicate ID"),
            "expected IdAssignmentDuplicate error, got: {err}"
        );
    }

    #[test]
    fn sequential_board_omits_id_assignment() {
        let board = BoardLayout::default();
        let json = board.to_json_string();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            val.get("id_assignment").is_none(),
            "sequential board should not include id_assignment in JSON"
        );
    }
}
