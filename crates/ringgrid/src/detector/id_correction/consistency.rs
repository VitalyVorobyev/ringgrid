use crate::detector::marker_build::MarkerRecord;

use super::index::dist2;
use super::types::{ConsistencyEvidence, ScrubStage, Trust};
use super::vote::{VoteOutcome, gather_trusted_neighbors_local_scale, vote_for_candidate};
use super::workspace::{
    IdCorrectionWorkspace, clear_marker_id, is_soft_locked_assignment, marker_center_is_finite,
};

pub(super) fn local_edge_neighbor_ids(
    marker_index: usize,
    markers: &[MarkerRecord],
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

/// Count `(support, contradiction)` board-adjacency edges to decoded neighbors
/// whose trust state satisfies `keep`. Support = the neighbor is one board hop
/// from `assumed_id`; contradiction = the neighbor is decoded but not adjacent.
///
/// Filtering by trust is what keeps confirmation/scrub decisions anchored to
/// ground truth: counting *any* decoded neighbor would let a chain of mutually
/// board-adjacent untrusted decodes vouch for each other.
fn edge_support_counts(
    ws: &IdCorrectionWorkspace<'_>,
    marker_index: usize,
    assumed_id: usize,
    keep: impl Fn(Trust) -> bool,
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
        let kept = ws
            .markers
            .iter()
            .enumerate()
            .find_map(|(j, m)| (m.id == Some(id_j)).then_some(ws.trust[j]))
            .is_some_and(&keep);
        if !kept {
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
        let (support_anchor, contradiction_anchor) =
            edge_support_counts(ws, i, id, Trust::is_anchor);
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
        let is_soft_locked = is_soft_locked_assignment(
            &ws.markers[i],
            ws.config.soft_lock_exact_decode,
            ws.codebook_min_cyclic_dist,
        );
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
            ws.codebook_min_cyclic_dist,
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

/// Promote decoded IDs the voting stages could not reach but whose local
/// neighborhood structurally confirms them.
///
/// In sparse/partial/blurry views a correctly-decoded but non-exact marker may
/// have no affine and no adjacent voting pair (⇒ zero votes), so it is never
/// promoted to trusted and would be cleared by `cleanup_unverified_markers`
/// despite being correct. This pass trusts such a marker when its decoded ID is
/// structurally clean: at least one board-adjacent neighbor supports it and no
/// neighbor contradicts it. It is precision-first — it never confirms an ID that
/// a confident local vote actively disputes (`strong_vote_mismatch`).
pub(super) fn confirm_ids_by_consistency(ws: &mut IdCorrectionWorkspace<'_>) -> usize {
    if !ws.config.confirm_by_consistency {
        return 0;
    }
    let mut confirmed = 0usize;
    for i in 0..ws.markers.len() {
        if ws.trust[i].is_trusted() {
            continue;
        }
        let Some(id) = ws.markers[i].id else {
            continue;
        };
        // Only confirm genuine decodes whose ID exists on the board.
        if ws.markers[i].decode.is_none() || !ws.board_index.id_to_xy.contains_key(&id) {
            continue;
        }
        let evidence = consistency_evidence_for_id(ws, i, id);
        // Precision: the support must come from *trusted* neighbors. Counting any
        // decoded neighbor (as `evidence.support_edges` does) would let a chain of
        // mutually board-adjacent untrusted decodes self-confirm — each seeing the
        // others as support — keeping false IDs that cleanup would otherwise clear.
        let (trusted_support, _) = edge_support_counts(ws, i, id, Trust::is_trusted);
        let strong_vote_mismatch = evidence.vote_mismatch && evidence.vote_winner_frac >= 0.60;
        if evidence.n_neighbors >= ws.config.consistency_min_neighbors
            && trusted_support >= ws.config.consistency_min_support_edges
            && evidence.contradiction_edges == 0
            && !strong_vote_mismatch
        {
            ws.trust[i] = Trust::ConfirmedConsistent;
            ws.stats.n_confirmed_by_consistency += 1;
            confirmed += 1;
        }
    }
    confirmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::config::IdCorrectionConfig;

    fn evidence(
        n_neighbors: usize,
        support_edges: usize,
        contradiction_edges: usize,
        vote_mismatch: bool,
        vote_winner_frac: f64,
    ) -> ConsistencyEvidence {
        let contradiction_frac = if n_neighbors == 0 {
            0.0
        } else {
            contradiction_edges as f64 / n_neighbors as f64
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

    // --- not soft-locked: a marker is precision-first cleared on weak structure ---

    #[test]
    fn keeps_marker_below_min_neighbors() {
        // Default `consistency_min_neighbors == 1`: zero neighbors ⇒ never clear,
        // even with no support (avoids clearing isolated-but-correct markers).
        let cfg = IdCorrectionConfig::default();
        assert!(!should_clear_by_consistency(
            evidence(0, 0, 0, false, 0.0),
            false,
            &cfg
        ));
    }

    #[test]
    fn keeps_structurally_clean_marker() {
        let cfg = IdCorrectionConfig::default();
        // 3 supporting neighbors, no contradictions, no vote dispute ⇒ keep.
        assert!(!should_clear_by_consistency(
            evidence(3, 3, 0, false, 0.0),
            false,
            &cfg
        ));
    }

    #[test]
    fn clears_marker_with_no_support() {
        let cfg = IdCorrectionConfig::default();
        // support_edges (0) < consistency_min_support_edges (1) ⇒ clear.
        assert!(should_clear_by_consistency(
            evidence(2, 0, 2, false, 0.0),
            false,
            &cfg
        ));
    }

    #[test]
    fn clears_marker_with_contradiction_majority() {
        let cfg = IdCorrectionConfig::default();
        // contradiction_frac 3/4 = 0.75 > 0.5 even though one neighbor supports.
        assert!(should_clear_by_consistency(
            evidence(4, 1, 3, false, 0.0),
            false,
            &cfg
        ));
    }

    #[test]
    fn clears_on_strong_vote_mismatch_only() {
        let cfg = IdCorrectionConfig::default();
        // Clean edges, but a confident local vote (>= 0.60) disputes the id.
        assert!(should_clear_by_consistency(
            evidence(2, 2, 0, true, 0.70),
            false,
            &cfg
        ));
        // A weak vote mismatch (< 0.60) is not enough to clear a clean marker.
        assert!(!should_clear_by_consistency(
            evidence(2, 2, 0, true, 0.55),
            false,
            &cfg
        ));
    }

    // --- soft-locked exact decodes: only strict structural contradiction clears ---

    #[test]
    fn soft_locked_marker_survives_partial_contradiction() {
        let cfg = IdCorrectionConfig::default();
        // One support edge present ⇒ a soft-locked exact decode is protected even
        // with two contradictions (requires support == 0 to clear).
        assert!(!should_clear_by_consistency(
            evidence(3, 1, 2, false, 0.0),
            true,
            &cfg
        ));
    }

    #[test]
    fn soft_locked_marker_cleared_on_strict_contradiction() {
        let cfg = IdCorrectionConfig::default();
        // Zero support and >= 2 contradictions ⇒ even a soft-locked decode clears.
        assert!(should_clear_by_consistency(
            evidence(2, 0, 2, false, 0.0),
            true,
            &cfg
        ));
    }
}
