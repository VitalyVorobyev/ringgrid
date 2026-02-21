use crate::detector::marker_build::DetectedMarker;

use super::index::dist2;
use super::types::{ConsistencyEvidence, ScrubStage, Trust};
use super::vote::{gather_trusted_neighbors_local_scale, vote_for_candidate, VoteOutcome};
use super::workspace::{
    clear_marker_id, is_soft_locked_assignment, marker_center_is_finite, IdCorrectionWorkspace,
};

pub(super) fn local_edge_neighbor_ids(
    marker_index: usize,
    markers: &[DetectedMarker],
    board_index: &super::index::BoardIndex,
    outer_radii_px: &[f64],
    outer_mul: f64,
) -> Vec<usize> {
    let center_i = markers[marker_index].center;
    let radius_i = outer_radii_px[marker_index];
    let mut out = Vec::<usize>::new();
    for (j, m) in markers.iter().enumerate() {
        if j == marker_index {
            continue;
        }
        let Some(id_j) = m.id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id_j) || !marker_center_is_finite(m) {
            continue;
        }
        let radius_j = outer_radii_px[j];
        let gate = outer_mul * 0.5 * (radius_i + radius_j);
        if gate <= 0.0 || !gate.is_finite() {
            continue;
        }
        if dist2(center_i, m.center) <= gate * gate {
            out.push(id_j);
        }
    }
    out
}

fn anchor_edge_support_counts(
    ws: &IdCorrectionWorkspace<'_>,
    marker_index: usize,
    assumed_id: usize,
) -> (usize, usize) {
    let neighbors = local_edge_neighbor_ids(
        marker_index,
        ws.markers,
        &ws.board_index,
        &ws.outer_radii_px,
        ws.config.consistency_outer_mul,
    );
    let mut support = 0usize;
    let mut contradiction = 0usize;
    for id_j in neighbors {
        let is_anchor = ws
            .markers
            .iter()
            .enumerate()
            .find_map(|(j, m)| (m.id == Some(id_j)).then_some(ws.trust[j]))
            .is_some_and(|t| t.is_anchor());
        if !is_anchor {
            continue;
        }
        if ws.board_index.are_neighbors(assumed_id, id_j) {
            support += 1;
        } else {
            contradiction += 1;
        }
    }
    (support, contradiction)
}

pub(super) fn consistency_evidence_for_id(
    ws: &IdCorrectionWorkspace<'_>,
    marker_index: usize,
    assumed_id: usize,
) -> ConsistencyEvidence {
    let neighbor_ids = local_edge_neighbor_ids(
        marker_index,
        ws.markers,
        &ws.board_index,
        &ws.outer_radii_px,
        ws.config.consistency_outer_mul,
    );

    let mut support_edges = 0usize;
    let mut contradiction_edges = 0usize;
    for &neighbor_id in &neighbor_ids {
        if ws.board_index.are_neighbors(assumed_id, neighbor_id) {
            support_edges += 1;
        } else {
            contradiction_edges += 1;
        }
    }

    let n_neighbors = support_edges + contradiction_edges;
    let contradiction_frac = if n_neighbors == 0 {
        0.0
    } else {
        contradiction_edges as f64 / n_neighbors as f64
    };

    let vote_neighbors = gather_trusted_neighbors_local_scale(
        marker_index,
        ws.markers,
        &ws.trust,
        &ws.board_index,
        &ws.outer_radii_px,
        ws.config.consistency_outer_mul,
    );
    let vote = vote_for_candidate(
        ws.markers[marker_index].center,
        ws.outer_radii_px[marker_index],
        &vote_neighbors,
        &ws.board_index,
        ws.board_index.pitch_mm * 0.6,
        ws.config.min_votes,
        ws.config.min_vote_weight_frac,
    );

    let (vote_mismatch, vote_winner_frac) = match vote {
        VoteOutcome::Candidate {
            id,
            winner_weight_frac,
            ..
        } if id != assumed_id => (true, winner_weight_frac),
        _ => (false, 0.0),
    };

    ConsistencyEvidence {
        n_neighbors,
        support_edges,
        contradiction_edges,
        contradiction_frac,
        vote_mismatch,
        vote_winner_frac,
    }
}

pub(super) fn should_clear_by_consistency(
    evidence: ConsistencyEvidence,
    soft_locked: bool,
    config: &crate::detector::config::IdCorrectionConfig,
) -> bool {
    if evidence.n_neighbors < config.consistency_min_neighbors {
        return false;
    }
    if soft_locked {
        evidence.support_edges == 0 && evidence.contradiction_edges >= 2
    } else {
        let strong_vote_mismatch = evidence.vote_mismatch && evidence.vote_winner_frac >= 0.60;
        evidence.support_edges < config.consistency_min_support_edges
            || evidence.contradiction_frac > f64::from(config.consistency_max_contradiction_frac)
            || strong_vote_mismatch
    }
}

pub(super) fn scrub_inconsistent_ids(
    ws: &mut IdCorrectionWorkspace<'_>,
    stage: ScrubStage,
) -> usize {
    let mut to_clear = Vec::<usize>::new();
    for i in 0..ws.markers.len() {
        let Some(id) = ws.markers[i].id else {
            continue;
        };
        if !ws.board_index.id_to_xy.contains_key(&id) {
            to_clear.push(i);
            continue;
        }
        let evidence = consistency_evidence_for_id(ws, i, id);
        let (support_anchor, contradiction_anchor) = anchor_edge_support_counts(ws, i, id);
        let recovered_two_neighbor_contradiction = matches!(stage, ScrubStage::Post)
            && matches!(
                ws.trust[i],
                Trust::RecoveredLocal | Trust::RecoveredHomography
            )
            && ((evidence.support_edges == 0
                && evidence.contradiction_edges >= 2
                && evidence.vote_mismatch
                && evidence.vote_winner_frac >= 0.60)
                || (contradiction_anchor >= 1 && support_anchor == 0));
        if recovered_two_neighbor_contradiction {
            to_clear.push(i);
            continue;
        }
        let is_soft_locked =
            is_soft_locked_assignment(&ws.markers[i], ws.config.soft_lock_exact_decode);
        let soft_locked_anchor_contradiction = matches!(stage, ScrubStage::Post)
            && is_soft_locked
            && support_anchor == 0
            && contradiction_anchor >= 2;
        let soft_locked_contradiction_dominated = matches!(stage, ScrubStage::Post)
            && is_soft_locked
            && evidence.contradiction_edges >= 2
            && evidence.contradiction_frac
                > f64::from(ws.config.consistency_max_contradiction_frac);
        if soft_locked_anchor_contradiction || soft_locked_contradiction_dominated {
            to_clear.push(i);
            continue;
        }
        if should_clear_by_consistency(evidence, is_soft_locked, ws.config) {
            to_clear.push(i);
        }
    }

    let mut cleared = 0usize;
    for i in to_clear {
        if clear_marker_id(
            i,
            ws.markers,
            &mut ws.trust,
            &mut ws.stats,
            ws.config.soft_lock_exact_decode,
            stage,
        ) {
            cleared += 1;
        }
    }
    cleared
}

pub(super) fn candidate_passes_local_consistency_gate(
    ws: &IdCorrectionWorkspace<'_>,
    marker_index: usize,
    candidate_id: usize,
) -> bool {
    let neighbor_ids = local_edge_neighbor_ids(
        marker_index,
        ws.markers,
        &ws.board_index,
        &ws.outer_radii_px,
        ws.config.consistency_outer_mul,
    );
    let mut support_edges = 0usize;
    let mut contradiction_edges = 0usize;
    for id in neighbor_ids {
        if ws.board_index.are_neighbors(candidate_id, id) {
            support_edges += 1;
        } else {
            contradiction_edges += 1;
        }
    }
    let total = support_edges + contradiction_edges;
    if support_edges < 1 || total == 0 {
        return false;
    }
    let contradiction_frac = contradiction_edges as f64 / total as f64;
    contradiction_frac <= f64::from(ws.config.consistency_max_contradiction_frac)
}
