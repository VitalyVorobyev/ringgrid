use std::collections::HashMap;

use crate::target::TargetLayout;

/// Six axial neighbor direction offsets for a hex lattice.
pub(super) const HEX_NEIGHBORS: [(i32, i32); 6] =
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

pub(super) struct BoardIndex {
    /// board ID → [x_mm, y_mm] board-space position.
    pub(super) id_to_xy: HashMap<usize, [f32; 2]>,
    /// board ID → list of neighboring board IDs (those that exist on the board).
    pub(super) board_neighbors: HashMap<usize, Vec<usize>>,
    /// Board pitch in mm.
    pub(super) pitch_mm: f64,
    /// Center-to-center distance (mm) between board-adjacent markers.
    ///
    /// On the hex lattice this is `pitch_mm · √3`, not the axial pitch.
    /// Scale-vote predictions must use this metric: conflating it with
    /// `pitch_mm` shortened predicted one-hop deltas by √3, pushing them
    /// outside the acceptance tolerance (the scale fallback silently never
    /// fired on hex boards).
    pub(super) neighbor_spacing_mm: f64,
}

impl BoardIndex {
    pub(super) fn build(target: &TargetLayout) -> Self {
        let mut qr_to_id: HashMap<(i32, i32), usize> = HashMap::new();
        let mut id_to_qr: HashMap<usize, (i32, i32)> = HashMap::new();
        let mut id_to_xy: HashMap<usize, [f32; 2]> = HashMap::new();

        for cell in target.cells() {
            let Some(id) = cell.id else { continue };
            qr_to_id.insert((cell.coord.u, cell.coord.v), id);
            id_to_qr.insert(id, (cell.coord.u, cell.coord.v));
            id_to_xy.insert(id, cell.xy_mm);
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
            pitch_mm: target.pitch_mm() as f64,
            neighbor_spacing_mm: f64::from(target.lattice().min_center_spacing_mm()),
        }
    }

    /// Find the nearest board marker to `xy_mm` within `tolerance_mm`.
    pub(super) fn nearest_within(&self, xy_mm: [f64; 2], tolerance_mm: f64) -> Option<usize> {
        let tol2 = tolerance_mm * tolerance_mm;
        let mut best_id: Option<usize> = None;
        let mut best_d2 = tol2;
        for (&id, &bxy) in &self.id_to_xy {
            let dx = xy_mm[0] - bxy[0] as f64;
            let dy = xy_mm[1] - bxy[1] as f64;
            let d2 = dx * dx + dy * dy;
            // Strict `<` keeps the initial `tol2` an exclusive bound; the
            // equidistant tie-break (lower id wins) makes the result
            // deterministic regardless of `HashMap` iteration order, matching
            // `nearest_k_ids`.
            if d2 < best_d2 || (d2 == best_d2 && best_id.is_some_and(|b| id < b)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn index_with(positions: &[(usize, [f32; 2])]) -> BoardIndex {
        BoardIndex {
            id_to_xy: positions.iter().copied().collect(),
            board_neighbors: HashMap::new(),
            pitch_mm: 1.0,
            neighbor_spacing_mm: 1.0,
        }
    }

    #[test]
    fn nearest_within_breaks_ties_by_lower_id() {
        // ids 7 and 3 are exactly equidistant from the query at [1, 0]; the
        // lower id (3) must always win, regardless of `HashMap` iteration order.
        // A fresh index per iteration re-seeds the map's `RandomState` so this
        // exercises order-independence rather than one fixed ordering.
        for _ in 0..64 {
            let idx = index_with(&[(7, [0.0, 0.0]), (3, [2.0, 0.0]), (42, [9.0, 9.0])]);
            assert_eq!(idx.nearest_within([1.0, 0.0], 5.0), Some(3));
        }
    }

    #[test]
    fn nearest_within_respects_tolerance() {
        let idx = index_with(&[(1, [0.0, 0.0])]);
        assert_eq!(idx.nearest_within([0.5, 0.0], 1.0), Some(1));
        assert_eq!(idx.nearest_within([5.0, 0.0], 1.0), None);
    }

    #[test]
    fn nearest_k_ids_ranks_by_distance_then_id() {
        let idx = index_with(&[(7, [0.0, 0.0]), (3, [2.0, 0.0]), (42, [9.0, 9.0])]);
        let ranked = idx.nearest_k_ids([0.0, 0.0], 2);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 7);
        assert!((ranked[0].1 - 0.0).abs() < 1e-9);
        assert_eq!(ranked[1].0, 3);
        assert!((ranked[1].1 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn nearest_k_ids_empty_for_k_zero() {
        let idx = index_with(&[(1, [0.0, 0.0])]);
        assert!(idx.nearest_k_ids([0.0, 0.0], 0).is_empty());
    }

    #[test]
    fn are_neighbors_reflects_adjacency_list() {
        let mut board_neighbors = HashMap::new();
        board_neighbors.insert(1usize, vec![2usize, 3usize]);
        board_neighbors.insert(2usize, vec![1usize]);
        let idx = BoardIndex {
            id_to_xy: HashMap::new(),
            board_neighbors,
            pitch_mm: 1.0,
            neighbor_spacing_mm: 1.0,
        };
        assert!(idx.are_neighbors(1, 2));
        assert!(idx.are_neighbors(1, 3));
        assert!(
            !idx.are_neighbors(2, 3),
            "adjacency is looked up per source id"
        );
        assert!(!idx.are_neighbors(9, 1), "unknown id has no neighbors");
    }

    #[test]
    fn dist2_is_squared_euclidean() {
        assert!((dist2([0.0, 0.0], [3.0, 4.0]) - 25.0).abs() < 1e-12);
        assert_eq!(dist2([1.5, -2.0], [1.5, -2.0]), 0.0);
    }
}
