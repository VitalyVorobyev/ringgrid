use std::cmp::Ordering;
use std::collections::HashMap;

use crate::detector::marker_build::DetectedMarker;

use super::index::{dist2, BoardIndex};
use super::math::{affine_to_board, fit_local_affine};
use super::types::Trust;

/// Detailed voting outcome for a query marker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum VoteOutcome {
    /// Candidate ID accepted by vote count and weight-fraction gates.
    Candidate {
        id: usize,
        winner_weight_frac: f64,
        n_votes: usize,
        n_candidates: usize,
    },
    /// No votes were cast (no usable prediction or no in-tolerance candidate).
    NoVotes,
    /// Some votes were cast but fewer than `min_votes`.
    InsufficientVotes { got: usize, needed: usize },
    /// Winner exists but does not satisfy the weight-fraction gate.
    GateRejected {
        winner_id: usize,
        winner_weight_frac: f64,
        min_required: f64,
    },
}

/// Per-neighbor vote record.
pub(super) struct NeighborInfo {
    pub(super) id: usize,
    pub(super) center: [f64; 2],
    /// `board_xy` in mm.
    pub(super) board_xy: [f64; 2],
    /// Neighbor local outer radius in pixels.
    pub(super) outer_radius_px: f64,
    pub(super) confidence: f64,
}

#[inline]
fn finite_radius_or(radius_px: f64, fallback: f64) -> f64 {
    if radius_px.is_finite() && radius_px > 0.0 {
        radius_px
    } else {
        fallback.max(1.0)
    }
}

#[inline]
fn local_scale_gate_px(radius_i: f64, radius_j: f64, outer_mul: f64) -> f64 {
    outer_mul * 0.5 * (radius_i + radius_j)
}

/// Collect trusted neighbor info for marker at index `i` using local-scale
/// pairwise gating.
pub(super) fn gather_trusted_neighbors_local_scale(
    i: usize,
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    outer_radii_px: &[f64],
    outer_mul: f64,
) -> Vec<NeighborInfo> {
    let center_i = markers[i].center;
    let radius_i = outer_radii_px[i];
    let mut out = Vec::new();
    for (j, m) in markers.iter().enumerate() {
        if j == i {
            continue;
        }
        if !trust[j].is_trusted() {
            continue;
        }
        if !(m.center[0].is_finite() && m.center[1].is_finite()) {
            continue;
        }
        let id_j = match m.id {
            Some(id) if board_index.id_to_xy.contains_key(&id) => id,
            _ => continue,
        };
        let radius_j = outer_radii_px[j];
        let gate = local_scale_gate_px(radius_i, radius_j, outer_mul);
        if gate <= 0.0 || !gate.is_finite() {
            continue;
        }
        if dist2(center_i, m.center) > gate * gate {
            continue;
        }
        let bxy = board_index.id_to_xy[&id_j];
        out.push(NeighborInfo {
            id: id_j,
            center: m.center,
            board_xy: [f64::from(bxy[0]), f64::from(bxy[1])],
            outer_radius_px: radius_j,
            confidence: f64::from(m.confidence),
        });
    }
    out
}

fn local_pitch_ratio_from_adjacent_neighbors(
    neighbors: &[NeighborInfo],
    board_index: &BoardIndex,
) -> Option<f64> {
    let mut ratios = Vec::<f64>::new();
    for a in 0..neighbors.len() {
        for b in (a + 1)..neighbors.len() {
            let na = &neighbors[a];
            let nb = &neighbors[b];
            if !board_index.are_neighbors(na.id, nb.id) {
                continue;
            }
            let mean_radius = 0.5 * (na.outer_radius_px + nb.outer_radius_px);
            if !mean_radius.is_finite() || mean_radius <= 0.0 {
                continue;
            }
            let img_dist = dist2(na.center, nb.center).sqrt();
            if img_dist <= 1.0 || !img_dist.is_finite() {
                continue;
            }
            ratios.push(img_dist / mean_radius);
        }
    }
    if ratios.is_empty() {
        return None;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = ratios.len() / 2;
    Some(if ratios.len().is_multiple_of(2) {
        0.5 * (ratios[mid - 1] + ratios[mid])
    } else {
        ratios[mid]
    })
}

/// Cast votes for the best candidate board ID for a query marker.
///
/// Uses a local 2D affine (≥ 3 neighbors) or local-radius scale fallback
/// derived from nearby trusted board-adjacent pairs.
pub(super) fn vote_for_candidate(
    center_q: [f64; 2],
    query_outer_radius_px: f64,
    neighbors: &[NeighborInfo],
    board_index: &BoardIndex,
    tolerance_mm: f64,
    min_votes: usize,
    min_vote_weight_frac: f32,
) -> VoteOutcome {
    const AFFINE_VOTE_WEIGHT: f64 = 0.8;
    const SCALE_VOTE_WEIGHT: f64 = 1.0;

    // Shared predicted board position from the local affine (if ≥ 3 neighbors available).
    let affine_predicted_board: Option<[f64; 2]> = if neighbors.len() >= 3 {
        let board_pts: Vec<[f64; 2]> = neighbors.iter().map(|n| n.board_xy).collect();
        let image_pts: Vec<[f64; 2]> = neighbors.iter().map(|n| n.center).collect();
        fit_local_affine(&board_pts, &image_pts)
            .as_ref()
            .and_then(|aff| affine_to_board(aff, center_q))
    } else {
        None
    };

    // Keep local-ratio votes even when affine exists. This makes voting robust
    // against local affine bias under distortion.
    let local_ratio = local_pitch_ratio_from_adjacent_neighbors(neighbors, board_index);

    let query_radius = finite_radius_or(query_outer_radius_px, 1.0);
    let mut votes: HashMap<usize, f64> = HashMap::new();
    let mut n_votes: usize = 0;

    for n in neighbors {
        if let Some(pb) = affine_predicted_board {
            if let Some(candidate_id) = board_index.nearest_within(pb, tolerance_mm) {
                *votes.entry(candidate_id).or_insert(0.0) += n.confidence * AFFINE_VOTE_WEIGHT;
                n_votes += 1;
            }
        }

        if let Some(ratio) = local_ratio {
            let mean_radius = 0.5 * (query_radius + n.outer_radius_px);
            let one_hop_pitch_px = ratio * mean_radius;
            if !one_hop_pitch_px.is_finite() || one_hop_pitch_px <= 1e-9 {
                continue;
            } else {
                let delta_img = [center_q[0] - n.center[0], center_q[1] - n.center[1]];
                let pitch_mm = board_index.pitch_mm;
                let pb = [
                    n.board_xy[0] + delta_img[0] / one_hop_pitch_px * pitch_mm,
                    n.board_xy[1] + delta_img[1] / one_hop_pitch_px * pitch_mm,
                ];
                if let Some(candidate_id) = board_index.nearest_within(pb, tolerance_mm) {
                    *votes.entry(candidate_id).or_insert(0.0) += n.confidence * SCALE_VOTE_WEIGHT;
                    n_votes += 1;
                }
            }
        }
    }

    if votes.is_empty() {
        return VoteOutcome::NoVotes;
    }
    if n_votes < min_votes {
        return VoteOutcome::InsufficientVotes {
            got: n_votes,
            needed: min_votes,
        };
    }

    let total_weight: f64 = votes.values().sum();
    let winner = votes.iter().max_by(|(id_a, w_a), (id_b, w_b)| {
        w_a.partial_cmp(w_b)
            .unwrap_or(Ordering::Equal)
            // Deterministic tie-break: lower candidate ID wins.
            .then_with(|| id_b.cmp(id_a))
    });
    let Some((&winner_id, &winner_weight)) = winner else {
        return VoteOutcome::NoVotes;
    };
    let winner_weight_frac = if total_weight > 0.0 && total_weight.is_finite() {
        winner_weight / total_weight
    } else {
        0.0
    };

    if winner_weight_frac >= f64::from(min_vote_weight_frac) {
        VoteOutcome::Candidate {
            id: winner_id,
            winner_weight_frac,
            n_votes,
            n_candidates: votes.len(),
        }
    } else {
        VoteOutcome::GateRejected {
            winner_id,
            winner_weight_frac,
            min_required: f64::from(min_vote_weight_frac),
        }
    }
}

/// If two markers share the same decoded ID, clear the lower-confidence one's ID.
pub(super) fn resolve_id_conflicts(markers: &mut [DetectedMarker]) -> usize {
    // Map: id → index of the highest-confidence marker with that id.
    let mut best: HashMap<usize, usize> = HashMap::new();
    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            best.entry(id).and_modify(|prev| {
                if markers[i].confidence > markers[*prev].confidence {
                    *prev = i;
                }
            });
            best.entry(id).or_insert(i);
        }
    }
    let mut n_cleared = 0usize;
    for (i, m) in markers.iter_mut().enumerate() {
        if let Some(id) = m.id {
            if best.get(&id).copied() != Some(i) {
                m.id = None;
                n_cleared += 1;
            }
        }
    }
    n_cleared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board_layout::BoardLayout;
    use crate::conic::Ellipse;
    use crate::detector::id_correction::index::BoardIndex;

    #[test]
    fn vote_tie_break_is_deterministic() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let id0 = 0usize;
        let id1 = 1usize;
        let id2 = board_index.board_neighbors[&id0][0];
        let neighbors = vec![
            NeighborInfo {
                id: id0,
                center: [0.0, 0.0],
                board_xy: board_index.id_to_xy[&id0].map(f64::from),
                outer_radius_px: 22.0,
                confidence: 0.5,
            },
            NeighborInfo {
                id: id1,
                center: [0.0, 0.0],
                board_xy: board_index.id_to_xy[&id1].map(f64::from),
                outer_radius_px: 22.0,
                confidence: 1.0,
            },
            NeighborInfo {
                id: id2,
                center: [44.0, 0.0],
                board_xy: board_index.id_to_xy[&id2].map(f64::from),
                outer_radius_px: 22.0,
                confidence: 0.5,
            },
        ];

        let out = vote_for_candidate(
            [0.0, 0.0],
            22.0,
            &neighbors,
            &board_index,
            board_index.pitch_mm,
            1,
            0.0,
        );

        match out {
            VoteOutcome::Candidate { id, .. } => assert_eq!(id, id0),
            other => panic!("expected candidate outcome, got {other:?}"),
        }
    }

    #[test]
    fn vote_reports_no_votes_without_affine_or_local_ratio() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let neighbors = vec![NeighborInfo {
            id: 0,
            center: [100.0, 100.0],
            board_xy: board_index.id_to_xy[&0].map(f64::from),
            outer_radius_px: 22.0,
            confidence: 1.0,
        }];

        let out = vote_for_candidate([110.0, 110.0], 22.0, &neighbors, &board_index, 5.0, 1, 0.5);
        assert!(matches!(out, VoteOutcome::NoVotes));
    }

    #[test]
    fn local_scale_neighbor_gate_uses_pairwise_radii() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let markers = vec![
            DetectedMarker {
                id: Some(0),
                center: [100.0, 100.0],
                ellipse_outer: Some(Ellipse {
                    cx: 100.0,
                    cy: 100.0,
                    a: 8.0,
                    b: 8.0,
                    angle: 0.0,
                }),
                ..DetectedMarker::default()
            },
            DetectedMarker {
                id: Some(1),
                center: [112.0, 100.0],
                ellipse_outer: Some(Ellipse {
                    cx: 112.0,
                    cy: 100.0,
                    a: 8.0,
                    b: 8.0,
                    angle: 0.0,
                }),
                ..DetectedMarker::default()
            },
            DetectedMarker {
                id: Some(2),
                center: [142.0, 100.0],
                ellipse_outer: Some(Ellipse {
                    cx: 142.0,
                    cy: 100.0,
                    a: 20.0,
                    b: 20.0,
                    angle: 0.0,
                }),
                ..DetectedMarker::default()
            },
        ];
        let outer_radii = vec![8.0, 8.0, 20.0];
        let trust = vec![Trust::AnchorWeak, Trust::AnchorWeak, Trust::AnchorWeak];

        let n0 = gather_trusted_neighbors_local_scale(
            0,
            &markers,
            &trust,
            &board_index,
            &outer_radii,
            2.0,
        );
        assert_eq!(n0.len(), 1, "only marker 1 should be local-scale reachable");
        assert_eq!(n0[0].id, 1);

        // With larger local multiplier, marker 2 becomes reachable.
        let n0_wide = gather_trusted_neighbors_local_scale(
            0,
            &markers,
            &trust,
            &board_index,
            &outer_radii,
            4.0,
        );
        assert_eq!(n0_wide.len(), 2);
        assert!(n0_wide.iter().any(|n| n.id == 2));
    }
}
