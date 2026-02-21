use std::collections::HashMap;

use crate::board_layout::BoardLayout;

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

    /// True if `id_b` is a one-hop hex neighbor of `id_a`.
    pub(super) fn are_neighbors(&self, id_a: usize, id_b: usize) -> bool {
        self.board_neighbors
            .get(&id_a)
            .is_some_and(|nbrs| nbrs.contains(&id_b))
    }

    /// Return `k` nearest board IDs to `xy_mm`, sorted by distance then ID.
    pub(super) fn nearest_k_ids(&self, xy_mm: [f64; 2], k: usize) -> Vec<(usize, f64)> {
        if k == 0 {
            return Vec::new();
        }
        let mut ranked: Vec<(usize, f64)> = self
            .id_to_xy
            .iter()
            .map(|(&id, &bxy)| {
                let dx = xy_mm[0] - f64::from(bxy[0]);
                let dy = xy_mm[1] - f64::from(bxy[1]);
                (id, dx * dx + dy * dy)
            })
            .collect();
        ranked.sort_by(|(ida, d2a), (idb, d2b)| d2a.total_cmp(d2b).then_with(|| ida.cmp(idb)));
        ranked.truncate(k);
        ranked
    }
}

#[inline]
pub(super) fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}
