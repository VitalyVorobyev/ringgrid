//! Runtime board layout specification.
//!
//! Replaces direct compile-time access to `board_spec` constants with a runtime struct
//! that can be loaded from JSON or constructed from the embedded defaults.

use std::collections::HashMap;

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

/// Board layout: marker positions and physical geometry.
///
/// Can be deserialized from JSON (compatible with `tools/board/board_spec.json`)
/// or constructed from the embedded `board_spec` constants via `Default`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoardLayout {
    pub name: String,
    pub pitch_mm: f32,
    pub board_size_mm: [f32; 2],
    pub marker_outer_radius_mm: f32,
    pub marker_inner_radius_mm: f32,
    pub markers: Vec<BoardMarker>,

    /// Fast lookup: marker ID → index into `markers`.
    #[serde(skip)]
    id_to_idx: HashMap<usize, usize>,
}

impl BoardLayout {
    /// Build the internal ID→index lookup table.
    /// Must be called after deserialization or manual construction.
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

    /// Iterator over all marker IDs present on the board.
    pub fn marker_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.markers.iter().map(|m| m.id)
    }

    /// Maximum marker ID present on the board.
    pub fn max_marker_id(&self) -> usize {
        self.markers.iter().map(|m| m.id).max().unwrap_or(0)
    }

    /// Load a board layout from a JSON file.
    pub fn from_json_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let mut layout: Self = serde_json::from_str(&data)?;
        layout.build_index();
        Ok(layout)
    }
}

impl Default for BoardLayout {
    fn default() -> Self {
        use crate::board_spec;

        let markers: Vec<BoardMarker> = (0..board_spec::BOARD_N)
            .map(|id| BoardMarker {
                id,
                xy_mm: board_spec::BOARD_XY_MM[id],
                q: Some(board_spec::BOARD_QR[id][0]),
                r: Some(board_spec::BOARD_QR[id][1]),
            })
            .collect();

        let id_to_idx: HashMap<usize, usize> =
            markers.iter().enumerate().map(|(i, m)| (m.id, i)).collect();

        Self {
            name: board_spec::BOARD_NAME.to_string(),
            pitch_mm: board_spec::BOARD_PITCH_MM,
            board_size_mm: board_spec::BOARD_SIZE_MM,
            marker_outer_radius_mm: board_spec::MARKER_OUTER_RADIUS_MM,
            marker_inner_radius_mm: board_spec::MARKER_INNER_RADIUS_MM,
            markers,
            id_to_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_board_has_correct_count() {
        let board = BoardLayout::default();
        assert_eq!(board.n_markers(), 203);
    }

    #[test]
    fn default_board_lookup_matches_board_spec() {
        let board = BoardLayout::default();
        for id in 0..board.n_markers() {
            let expected = crate::board_spec::xy_mm(id);
            let actual = board.xy_mm(id);
            assert_eq!(expected, actual, "mismatch at id={id}");
        }
        assert_eq!(board.xy_mm(999), None);
    }

    #[test]
    fn serde_roundtrip() {
        let board = BoardLayout::default();
        let json = serde_json::to_string(&board).unwrap();
        let mut restored: BoardLayout = serde_json::from_str(&json).unwrap();
        restored.build_index();
        assert_eq!(restored.n_markers(), board.n_markers());
        assert_eq!(restored.xy_mm(0), board.xy_mm(0));
        assert_eq!(restored.xy_mm(202), board.xy_mm(202));
    }
}
