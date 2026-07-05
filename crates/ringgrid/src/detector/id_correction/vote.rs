use std::cmp::Ordering;
use std::collections::HashMap;

use crate::detector::marker_build::MarkerRecord;

use super::index::{BoardIndex, dist2};
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
    markers: &[MarkerRecord],
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

/// Local board→image similarity estimated from trusted board-adjacent pairs:
/// a radius-relative pitch ratio plus the in-plane rotation angle.
struct LocalFrame {
    /// Median of `image_distance / mean_outer_radius` over adjacent pairs.
    pitch_ratio: f64,
    /// cos of the board→image rotation angle.
    cos_rot: f64,
    /// sin of the board→image rotation angle.
    sin_rot: f64,
}

/// Estimate the local similarity frame from board-adjacent neighbor pairs.
///
/// Each adjacent pair contributes a scale sample (image distance relative to
/// the pair's mean outer radius) and a rotation sample (angle between the
/// image-space and board-space deltas). The rotation is aggregated as a
/// circular mean; when the pairs' rotations contradict each other (short
/// resultant) the frame is unconstrained and `None` is returned — casting
/// axis-aligned scale votes on a rotated board would predict systematically
/// wrong cells (a 60°-rotated hex neighborhood lands exactly on the wrong
/// lattice site).
fn local_frame_from_adjacent_neighbors(
    neighbors: &[NeighborInfo],
    board_index: &BoardIndex,
) -> Option<LocalFrame> {
    let mut ratios = Vec::<f64>::new();
    let (mut rot_x, mut rot_y) = (0.0f64, 0.0f64);
    let mut n_pairs = 0usize;
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
            let delta_img = [nb.center[0] - na.center[0], nb.center[1] - na.center[1]];
            let img_dist = (delta_img[0] * delta_img[0] + delta_img[1] * delta_img[1]).sqrt();
            if img_dist <= 1.0 || !img_dist.is_finite() {
                continue;
            }
            let delta_board = [
                nb.board_xy[0] - na.board_xy[0],
                nb.board_xy[1] - na.board_xy[1],
            ];
            let board_dist =
                (delta_board[0] * delta_board[0] + delta_board[1] * delta_board[1]).sqrt();
            if board_dist <= 1e-9 || !board_dist.is_finite() {
                continue;
            }
            ratios.push(img_dist / mean_radius);
            // Rotation sample: angle(delta_img) − angle(delta_board), summed
            // as a unit vector for a circular mean.
            let theta = delta_img[1].atan2(delta_img[0]) - delta_board[1].atan2(delta_board[0]);
            rot_x += theta.cos();
            rot_y += theta.sin();
            n_pairs += 1;
        }
    }
    if ratios.is_empty() {
        return None;
    }
    // Contradictory rotation samples (e.g. mislabeled pairs) cancel out; a
    // short resultant means the local orientation is unconstrained.
    let resultant = (rot_x * rot_x + rot_y * rot_y).sqrt();
    if !resultant.is_finite() || resultant < 0.5 * n_pairs as f64 {
        return None;
    }
    let (cos_rot, sin_rot) = (rot_x / resultant, rot_y / resultant);

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = ratios.len() / 2;
    let pitch_ratio = if ratios.len().is_multiple_of(2) {
        0.5 * (ratios[mid - 1] + ratios[mid])
    } else {
        ratios[mid]
    };
    Some(LocalFrame {
        pitch_ratio,
        cos_rot,
        sin_rot,
    })
}

/// Median neighbor confidence (deterministic; falls back to 0 for empty input).
fn median_confidence(neighbors: &[NeighborInfo]) -> f64 {
    if neighbors.is_empty() {
        return 0.0;
    }
    let mut confs: Vec<f64> = neighbors.iter().map(|n| n.confidence).collect();
    confs.sort_by(|a, b| a.total_cmp(b));
    let mid = confs.len() / 2;
    if confs.len().is_multiple_of(2) {
        0.5 * (confs[mid - 1] + confs[mid])
    } else {
        confs[mid]
    }
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

    // Keep local-frame votes even when affine exists. This makes voting robust
    // against local affine bias under distortion.
    let local_frame = local_frame_from_adjacent_neighbors(neighbors, board_index);

    let query_radius = finite_radius_or(query_outer_radius_px, 1.0);
    let mut votes: HashMap<usize, f64> = HashMap::new();
    let mut n_votes: usize = 0;

    // The affine prediction is ONE joint hypothesis derived from all
    // neighbors, so it casts one vote (weighted by the median neighbor
    // confidence). Casting it once per neighbor inflated `n_votes` and let a
    // single ill-conditioned affine satisfy `min_votes` with no corroboration.
    if let Some(pb) = affine_predicted_board
        && let Some(candidate_id) = board_index.nearest_within(pb, tolerance_mm)
    {
        *votes.entry(candidate_id).or_insert(0.0) +=
            median_confidence(neighbors) * AFFINE_VOTE_WEIGHT;
        n_votes += 1;
    }

    if let Some(frame) = &local_frame {
        for n in neighbors {
            let mean_radius = 0.5 * (query_radius + n.outer_radius_px);
            let one_hop_pitch_px = frame.pitch_ratio * mean_radius;
            if !one_hop_pitch_px.is_finite() || one_hop_pitch_px <= 1e-9 {
                continue;
            }
            let delta_img = [center_q[0] - n.center[0], center_q[1] - n.center[1]];
            // `one_hop_pitch_px` is the image-space distance of one board hop,
            // so the mm conversion must use the board-adjacent center spacing
            // (√3·pitch on hex), not the axial pitch — see `BoardIndex`.
            let hop_mm = board_index.neighbor_spacing_mm;
            // Undo the locally-estimated board→image rotation before scaling
            // back to board millimetres; the previous axis-aligned prediction
            // silently assumed an unrotated board.
            let delta_board_px = [
                frame.cos_rot * delta_img[0] + frame.sin_rot * delta_img[1],
                -frame.sin_rot * delta_img[0] + frame.cos_rot * delta_img[1],
            ];
            let pb = [
                n.board_xy[0] + delta_board_px[0] / one_hop_pitch_px * hop_mm,
                n.board_xy[1] + delta_board_px[1] / one_hop_pitch_px * hop_mm,
            ];
            if let Some(candidate_id) = board_index.nearest_within(pb, tolerance_mm) {
                *votes.entry(candidate_id).or_insert(0.0) += n.confidence * SCALE_VOTE_WEIGHT;
                n_votes += 1;
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
pub(super) fn resolve_id_conflicts(markers: &mut [MarkerRecord]) -> usize {
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
        if let Some(id) = m.id
            && best.get(&id).copied() != Some(i)
        {
            m.id = None;
            n_cleared += 1;
        }
    }
    n_cleared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::detector::id_correction::index::BoardIndex;
    use crate::target::TargetLayout;

    fn neighbor(conf: f64) -> NeighborInfo {
        NeighborInfo {
            id: 0,
            center: [0.0, 0.0],
            board_xy: [0.0, 0.0],
            outer_radius_px: 10.0,
            confidence: conf,
        }
    }

    #[test]
    fn finite_radius_or_falls_back_on_non_positive_or_nan() {
        assert!((finite_radius_or(5.0, 2.0) - 5.0).abs() < 1e-12);
        // Non-positive radius uses the fallback (floored at 1.0).
        assert!((finite_radius_or(0.0, 3.0) - 3.0).abs() < 1e-12);
        assert!((finite_radius_or(-4.0, 3.0) - 3.0).abs() < 1e-12);
        // A tiny/degenerate fallback is floored to 1.0.
        assert!((finite_radius_or(f64::NAN, 0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn local_scale_gate_is_mul_times_mean_radius() {
        // mul · 0.5 · (r_i + r_j) = 2 · 0.5 · 20 = 20.
        assert!((local_scale_gate_px(8.0, 12.0, 2.0) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn median_confidence_handles_odd_even_and_empty() {
        assert_eq!(median_confidence(&[]), 0.0);
        let odd = [neighbor(0.2), neighbor(0.8), neighbor(0.5)];
        assert!((median_confidence(&odd) - 0.5).abs() < 1e-12);
        let even = [neighbor(0.2), neighbor(0.8)];
        assert!((median_confidence(&even) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn vote_tie_break_is_deterministic() {
        let board = TargetLayout::default_hex();
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
        let board = TargetLayout::default_hex();
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

    /// Find `n` pairwise non-board-adjacent, non-collinear ids (deterministic).
    fn non_adjacent_ids(board_index: &BoardIndex, n: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = board_index.id_to_xy.keys().copied().collect();
        ids.sort_unstable();
        let mut picked = Vec::<usize>::new();
        for id in ids {
            if picked
                .iter()
                .all(|&p| !board_index.are_neighbors(p, id) && p != id)
            {
                // Reject collinear triples so a local affine stays well-posed.
                if picked.len() >= 2 {
                    let a = board_index.id_to_xy[&picked[0]];
                    let b = board_index.id_to_xy[&picked[1]];
                    let c = board_index.id_to_xy[&id];
                    let cross = (f64::from(b[0]) - f64::from(a[0]))
                        * (f64::from(c[1]) - f64::from(a[1]))
                        - (f64::from(b[1]) - f64::from(a[1])) * (f64::from(c[0]) - f64::from(a[0]));
                    if cross.abs() < 1e-6 {
                        continue;
                    }
                }
                picked.push(id);
                if picked.len() == n {
                    break;
                }
            }
        }
        assert_eq!(picked.len(), n, "board must supply {n} non-adjacent ids");
        picked
    }

    #[test]
    fn affine_hypothesis_casts_a_single_vote() {
        // Three trusted neighbors, none board-adjacent to each other: the
        // local frame is unconstrained (no scale votes) and the only evidence
        // is ONE joint affine hypothesis. It must count as one vote — before
        // the fix it cast one vote per neighbor, satisfying min_votes on its
        // own with zero corroboration.
        let board = TargetLayout::default_hex();
        let board_index = BoardIndex::build(&board);
        let picked = non_adjacent_ids(&board_index, 4);
        let (n_ids, query_id) = (&picked[..3], picked[3]);

        // Identity board→image mapping (1 px per mm).
        let neighbors: Vec<NeighborInfo> = n_ids
            .iter()
            .map(|&id| NeighborInfo {
                id,
                center: board_index.id_to_xy[&id].map(f64::from),
                board_xy: board_index.id_to_xy[&id].map(f64::from),
                outer_radius_px: 4.0,
                confidence: 0.9,
            })
            .collect();
        let center_q = board_index.id_to_xy[&query_id].map(f64::from);
        let tolerance = board_index.pitch_mm * 0.6;

        // min_votes = 1: the affine vote alone recovers the query id.
        let out = vote_for_candidate(center_q, 4.0, &neighbors, &board_index, tolerance, 1, 0.0);
        match out {
            VoteOutcome::Candidate { id, n_votes, .. } => {
                assert_eq!(id, query_id);
                assert_eq!(n_votes, 1, "one joint hypothesis = one vote");
            }
            other => panic!("expected candidate, got {other:?}"),
        }

        // min_votes = 2: a lone affine hypothesis is insufficient evidence.
        let out = vote_for_candidate(center_q, 4.0, &neighbors, &board_index, tolerance, 2, 0.0);
        assert!(
            matches!(out, VoteOutcome::InsufficientVotes { got: 1, needed: 2 }),
            "uncorroborated affine must not satisfy min_votes=2, got {out:?}"
        );
    }

    #[test]
    fn scale_votes_follow_locally_estimated_rotation() {
        // Board rotated 50° in the image. The old axis-aligned scale
        // prediction assumed zero rotation; on a hex lattice a ~50–60° error
        // lands the prediction near the WRONG lattice site. The local frame
        // estimated from the adjacent pair must vote for the correct cell.
        let board = TargetLayout::default_hex();
        let board_index = BoardIndex::build(&board);

        // An adjacent pair (a, b) plus a query cell adjacent to `a`.
        let (&id_a, nbrs_a) = board_index
            .board_neighbors
            .iter()
            .min_by_key(|(id, nbrs)| (usize::MAX - nbrs.len(), **id))
            .expect("hex board has neighbors");
        assert!(nbrs_a.len() >= 2);
        let id_b = nbrs_a[0];
        let query_id = nbrs_a[1];

        let theta = 50f64.to_radians();
        let (s, c) = theta.sin_cos();
        let k = 4.0; // px per mm
        let img = |id: usize| {
            let b = board_index.id_to_xy[&id].map(f64::from);
            [k * (c * b[0] - s * b[1]), k * (s * b[0] + c * b[1])]
        };

        let radius = 0.5 * board_index.pitch_mm * k / 1.5;
        let neighbors: Vec<NeighborInfo> = [id_a, id_b]
            .iter()
            .map(|&id| NeighborInfo {
                id,
                center: img(id),
                board_xy: board_index.id_to_xy[&id].map(f64::from),
                outer_radius_px: radius,
                confidence: 0.9,
            })
            .collect();

        let center_q = img(query_id);
        let tolerance = board_index.pitch_mm * 0.6;
        let out = vote_for_candidate(
            center_q,
            radius,
            &neighbors,
            &board_index,
            tolerance,
            1,
            0.0,
        );
        match out {
            VoteOutcome::Candidate { id, .. } => {
                assert_eq!(
                    id, query_id,
                    "rotation-aware prediction must hit the true cell"
                );
            }
            other => panic!("expected candidate on rotated board, got {other:?}"),
        }
    }

    #[test]
    fn local_scale_neighbor_gate_uses_pairwise_radii() {
        let board = TargetLayout::default_hex();
        let board_index = BoardIndex::build(&board);
        let markers = vec![
            MarkerRecord {
                id: Some(0),
                center: [100.0, 100.0],
                ellipse_outer: Some(Ellipse {
                    cx: 100.0,
                    cy: 100.0,
                    a: 8.0,
                    b: 8.0,
                    angle: 0.0,
                }),
                ..MarkerRecord::default()
            },
            MarkerRecord {
                id: Some(1),
                center: [112.0, 100.0],
                ellipse_outer: Some(Ellipse {
                    cx: 112.0,
                    cy: 100.0,
                    a: 8.0,
                    b: 8.0,
                    angle: 0.0,
                }),
                ..MarkerRecord::default()
            },
            MarkerRecord {
                id: Some(2),
                center: [142.0, 100.0],
                ellipse_outer: Some(Ellipse {
                    cx: 142.0,
                    cy: 100.0,
                    a: 20.0,
                    b: 20.0,
                    angle: 0.0,
                }),
                ..MarkerRecord::default()
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
