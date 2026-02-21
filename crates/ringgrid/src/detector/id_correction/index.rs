use std::collections::HashMap;

use crate::board_layout::BoardLayout;
use crate::detector::marker_build::DetectedMarker;

/// Six axial neighbor direction offsets for a hex lattice.
pub(super) const HEX_NEIGHBORS: [(i16, i16); 6] =
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

pub(super) struct BoardIndex {
    /// board ID → [x_mm, y_mm] board-space position.
    pub(super) id_to_xy: HashMap<usize, [f32; 2]>,
    /// board ID → list of neighboring board IDs (those that exist on the board).
    pub(super) board_neighbors: HashMap<usize, Vec<usize>>,
    /// Board pitch in mm.
    pub(super) pitch_mm: f64,
}

impl BoardIndex {
    pub(super) fn build(board: &BoardLayout) -> Self {
        let mut qr_to_id: HashMap<(i16, i16), usize> = HashMap::new();
        let mut id_to_qr: HashMap<usize, (i16, i16)> = HashMap::new();
        let mut id_to_xy: HashMap<usize, [f32; 2]> = HashMap::new();

        for m in board.markers() {
            if let (Some(q), Some(r)) = (m.q, m.r) {
                qr_to_id.insert((q, r), m.id);
                id_to_qr.insert(m.id, (q, r));
            }
            id_to_xy.insert(m.id, m.xy_mm);
        }

        let mut board_neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&id, &(q, r)) in &id_to_qr {
            let neighbors = HEX_NEIGHBORS
                .iter()
                .filter_map(|&(dq, dr)| qr_to_id.get(&(q + dq, r + dr)).copied())
                .collect();
            board_neighbors.insert(id, neighbors);
        }

        Self {
            id_to_xy,
            board_neighbors,
            pitch_mm: board.pitch_mm as f64,
        }
    }

    /// Find the nearest board marker to `xy_mm` within `tolerance_mm`.
    pub(super) fn nearest_within(&self, xy_mm: [f64; 2], tolerance_mm: f64) -> Option<usize> {
        let tol2 = tolerance_mm * tolerance_mm;
        let mut best_id = None;
        let mut best_d2 = tol2;
        for (&id, &bxy) in &self.id_to_xy {
            let dx = xy_mm[0] - bxy[0] as f64;
            let dy = xy_mm[1] - bxy[1] as f64;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best_id = Some(id);
            }
        }
        best_id
    }
}

#[inline]
pub(super) fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Median of a mutable slice.
fn median_f64(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    } else {
        v[n / 2]
    }
}

/// Estimate board pitch in image pixels from pairs of board-adjacent decoded markers.
pub(super) fn estimate_pitch_px(
    markers: &[DetectedMarker],
    board_index: &BoardIndex,
) -> Option<f64> {
    let pitch_mm = board_index.pitch_mm;
    let mut samples: Vec<f64> = Vec::new();

    for (i, m1) in markers.iter().enumerate() {
        let id1 = match m1.id {
            Some(id) if board_index.id_to_xy.contains_key(&id) => id,
            _ => continue,
        };
        let neighbors = match board_index.board_neighbors.get(&id1) {
            Some(n) => n,
            None => continue,
        };
        for m2 in &markers[i + 1..] {
            let id2 = match m2.id {
                Some(id) => id,
                None => continue,
            };
            if !neighbors.contains(&id2) {
                continue;
            }
            let img_dist = dist2(m1.center, m2.center).sqrt();
            if img_dist > 1.0 {
                samples.push(img_dist / pitch_mm);
            }
        }
    }

    if samples.is_empty() {
        return None;
    }
    Some(median_f64(&mut samples))
}
