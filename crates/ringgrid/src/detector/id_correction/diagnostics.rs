use std::cmp::Ordering;

use super::consistency::{consistency_evidence_for_id, should_clear_by_consistency};
use super::index::dist2;
use super::vote::{VoteOutcome, gather_trusted_neighbors_local_scale, vote_for_candidate};
use super::workspace::{IdCorrectionWorkspace, is_soft_locked_assignment, marker_center_is_finite};

pub(super) fn estimate_adjacent_spacing_px(ws: &IdCorrectionWorkspace<'_>) -> Option<f64> {
    let mut dists = Vec::<f64>::new();
    for i in 0..ws.markers.len() {
        let Some(id_i) = ws.markers[i].id else {
            continue;
        };
        if !ws.board_index.id_to_xy.contains_key(&id_i) || !marker_center_is_finite(&ws.markers[i])
        {
            continue;
        }
        for j in (i + 1)..ws.markers.len() {
            let Some(id_j) = ws.markers[j].id else {
                continue;
            };
            if !ws.board_index.id_to_xy.contains_key(&id_j)
                || !marker_center_is_finite(&ws.markers[j])
            {
                continue;
            }
            if !ws.board_index.are_neighbors(id_i, id_j) {
                continue;
            }
            let d = dist2(ws.markers[i].center, ws.markers[j].center).sqrt();
            if d.is_finite() && d > 1.0 {
                dists.push(d);
            }
        }
    }
    if dists.is_empty() {
        return None;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = dists.len() / 2;
    Some(if dists.len().is_multiple_of(2) {
        0.5 * (dists[mid - 1] + dists[mid])
    } else {
        dists[mid]
    })
}

pub(super) fn diagnose_unverified_reasons(
    ws: &mut IdCorrectionWorkspace<'_>,
    final_outer_mul: f64,
) {
    let tolerance_mm = ws.board_index.pitch_mm * 0.6;
    for i in 0..ws.markers.len() {
        if ws.trust[i].is_trusted() || !marker_center_is_finite(&ws.markers[i]) {
            continue;
        }
        let neighbors = gather_trusted_neighbors_local_scale(
            i,
            ws.markers,
            &ws.trust,
            &ws.board_index,
            &ws.outer_radii_px,
            final_outer_mul,
        );
        if neighbors.is_empty() {
            ws.stats.n_unverified_no_neighbors += 1;
            continue;
        }
        let effective_min_votes = ws.config.effective_min_votes(ws.markers[i].id.is_some());
        let out = vote_for_candidate(
            ws.markers[i].center,
            ws.outer_radii_px[i],
            &neighbors,
            &ws.board_index,
            tolerance_mm,
            effective_min_votes,
            ws.config.min_vote_weight_frac,
        );
        match out {
            VoteOutcome::NoVotes | VoteOutcome::InsufficientVotes { .. } => {
                ws.stats.n_unverified_no_votes += 1;
            }
            VoteOutcome::GateRejected { .. } => {
                ws.stats.n_unverified_gate_rejects += 1;
            }
            VoteOutcome::Candidate { .. } => {}
        }
    }
}

pub(super) fn count_inconsistent_remaining(ws: &IdCorrectionWorkspace<'_>) -> usize {
    let mut n = 0usize;
    for i in 0..ws.markers.len() {
        let Some(id) = ws.markers[i].id else {
            continue;
        };
        if !ws.board_index.id_to_xy.contains_key(&id) {
            n += 1;
            continue;
        }
        let evidence = consistency_evidence_for_id(ws, i, id);
        let is_soft_locked = is_soft_locked_assignment(
            &ws.markers[i],
            ws.config.soft_lock_exact_decode,
            ws.codebook_min_cyclic_dist,
        );
        if should_clear_by_consistency(evidence, is_soft_locked, ws.config) {
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::config::IdCorrectionConfig;
    use crate::detector::id_correction::index::BoardIndex;
    use crate::detector::marker_build::MarkerRecord;
    use crate::marker::codec::Codebook;
    use crate::target::TargetLayout;

    fn marker(id: Option<usize>, center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            id,
            center,
            confidence: 0.9,
            ..MarkerRecord::default()
        }
    }

    fn min_cyclic() -> usize {
        Codebook::default().min_cyclic_dist()
    }

    /// A board-adjacent id pair from the default hex layout (deterministic).
    fn adjacent_pair(board_index: &BoardIndex) -> (usize, usize) {
        let (&id_a, nbrs) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| !nbrs.is_empty())
            .expect("hex board has an adjacent pair");
        (id_a, nbrs[0])
    }

    #[test]
    fn adjacent_spacing_is_the_pair_center_distance() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();
        let board_index = BoardIndex::build(&board);
        let (id_a, id_b) = adjacent_pair(&board_index);

        // Two board-adjacent markers 40 px apart: the median (single) adjacent
        // spacing must be exactly that separation.
        let mut markers = vec![
            marker(Some(id_a), [10.0, 10.0]),
            marker(Some(id_b), [50.0, 10.0]),
        ];
        let ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());

        let spacing = estimate_adjacent_spacing_px(&ws).expect("adjacent pair yields a spacing");
        assert!((spacing - 40.0).abs() < 1e-9, "spacing = {spacing}");
    }

    #[test]
    fn adjacent_spacing_is_none_without_adjacent_pairs() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();
        // A single decoded marker has no adjacent partner → no spacing sample.
        let mut markers = vec![marker(Some(0), [10.0, 10.0])];
        let ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());
        assert!(estimate_adjacent_spacing_px(&ws).is_none());
    }

    #[test]
    fn count_inconsistent_flags_off_board_ids() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();
        let off_board = board.n_cells() + 100_000;
        let mut markers = vec![marker(Some(off_board), [10.0, 10.0])];
        let ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());
        assert_eq!(count_inconsistent_remaining(&ws), 1);
    }

    #[test]
    fn count_inconsistent_ignores_isolated_valid_marker() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();
        // A valid on-board id with no neighbors: below min_neighbors, so it is
        // never counted as inconsistent (avoids clearing isolated-but-correct).
        let mut markers = vec![marker(Some(0), [10.0, 10.0])];
        let ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());
        assert_eq!(count_inconsistent_remaining(&ws), 0);
    }
}
