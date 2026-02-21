use std::collections::HashMap;

use crate::detector::marker_build::DetectedMarker;

use super::index::{dist2, BoardIndex};
use super::math::{affine_to_board, fit_local_affine};
use super::Trust;

/// Per-neighbor vote record.
pub(super) struct NeighborInfo {
    pub(super) center: [f64; 2],
    /// `board_xy` in mm.
    pub(super) board_xy: [f64; 2],
    pub(super) confidence: f64,
}

/// Collect trusted neighbor info for marker at index `i`.
pub(super) fn gather_trusted_neighbors(
    i: usize,
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    search_radius_sq: f64,
) -> Vec<NeighborInfo> {
    let center_i = markers[i].center;
    let mut out = Vec::new();
    for (j, m) in markers.iter().enumerate() {
        if j == i {
            continue;
        }
        if trust[j] != Trust::Trusted && trust[j] != Trust::Tentative {
            continue;
        }
        let id_j = match m.id {
            Some(id) if board_index.id_to_xy.contains_key(&id) => id,
            _ => continue,
        };
        if dist2(center_i, m.center) > search_radius_sq {
            continue;
        }
        let bxy = board_index.id_to_xy[&id_j];
        out.push(NeighborInfo {
            center: m.center,
            board_xy: [bxy[0] as f64, bxy[1] as f64],
            confidence: m.confidence as f64,
        });
    }
    out
}

/// Cast votes for the best candidate board ID for a query marker at `center_q`.
///
/// Uses a local 2D affine (≥ 3 neighbors) or per-neighbor scale prediction
/// (< 3 neighbors). Returns the winning ID, or `None` if the vote did not
/// reach `min_votes` and `min_vote_weight_frac`.
pub(super) fn vote_for_candidate(
    center_q: [f64; 2],
    neighbors: &[NeighborInfo],
    board_index: &BoardIndex,
    tolerance_mm: f64,
    pitch_px: Option<f64>,
    min_votes: usize,
    min_vote_weight_frac: f32,
) -> Option<usize> {
    // Shared predicted board position from the local affine (if ≥ 3 neighbors available).
    // All neighbors share this single prediction when the affine succeeds.
    let affine_predicted_board: Option<[f64; 2]> = if neighbors.len() >= 3 {
        let board_pts: Vec<[f64; 2]> = neighbors.iter().map(|n| n.board_xy).collect();
        let image_pts: Vec<[f64; 2]> = neighbors.iter().map(|n| n.center).collect();
        fit_local_affine(&board_pts, &image_pts)
            .as_ref()
            .and_then(|aff| affine_to_board(aff, center_q))
    } else {
        None
    };

    let mut votes: HashMap<usize, f64> = HashMap::new();
    let mut n_votes: usize = 0;

    for n in neighbors {
        let predicted_board = if let Some(pb) = affine_predicted_board {
            // All neighbors share the affine prediction.
            Some(pb)
        } else {
            // Per-neighbor scale-based fallback.
            pitch_px.map(|scale| {
                let delta_img = [center_q[0] - n.center[0], center_q[1] - n.center[1]];
                let pitch_mm = board_index.pitch_mm;
                [
                    n.board_xy[0] + delta_img[0] / scale * pitch_mm,
                    n.board_xy[1] + delta_img[1] / scale * pitch_mm,
                ]
            })
        };

        if let Some(pb) = predicted_board {
            if let Some(candidate_id) = board_index.nearest_within(pb, tolerance_mm) {
                *votes.entry(candidate_id).or_insert(0.0) += n.confidence;
                n_votes += 1;
            }
        }
    }

    if n_votes < min_votes || votes.is_empty() {
        return None;
    }

    let total_weight: f64 = votes.values().sum();
    let (&winner_id, &winner_weight) = votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;

    if winner_weight / total_weight >= min_vote_weight_frac as f64 {
        Some(winner_id)
    } else {
        None
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
