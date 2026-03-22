//! Non-maximum suppression and greedy distance suppression for proposals.

use std::collections::HashMap;

use super::config::ProposalConfig;
use super::Proposal;

/// Build a table of linear index offsets for all integer points inside a
/// circular neighbourhood of `nms_radius`.
pub(crate) fn build_nms_offsets(stride: usize, nms_radius: f32) -> Vec<isize> {
    let nms_r = nms_radius.ceil() as i32;
    let nms_r_sq = nms_radius * nms_radius;
    let mut nms_offsets = Vec::new();
    for dy in -nms_r..=nms_r {
        for dx in -nms_r..=nms_r {
            if dx == 0 && dy == 0 {
                continue;
            }
            if (dx * dx + dy * dy) as f32 > nms_r_sq {
                continue;
            }
            nms_offsets.push(dy as isize * stride as isize + dx as isize);
        }
    }
    nms_offsets
}

/// Extract local maxima from the smoothed accumulator.
///
/// Uses `config.nms_radius()` (the capped internal radius) for the
/// local-max check and `config.min_vote_frac` for thresholding.
pub(crate) fn extract_proposals_from_smoothed(
    smoothed_data: &[f32],
    w: u32,
    h: u32,
    config: &ProposalConfig,
) -> Vec<Proposal> {
    let stride = w as usize;
    let max_val = smoothed_data.iter().cloned().fold(0.0f32, f32::max);
    if max_val < 1e-6 {
        return Vec::new();
    }

    let nms_radius = config.nms_radius();
    let vote_threshold = config.min_vote_frac * max_val;
    let nms_r = nms_radius.ceil() as i32;
    let nms_offsets = build_nms_offsets(stride, nms_radius);

    let mut proposals = Vec::new();
    for y in nms_r..(h as i32 - nms_r) {
        for x in nms_r..(w as i32 - nms_r) {
            let idx = y as usize * stride + x as usize;
            let val = smoothed_data[idx];
            if val < vote_threshold {
                continue;
            }
            // Check if local maximum within nms_radius
            let mut is_max = true;
            for &off in &nms_offsets {
                let nidx = idx.wrapping_add_signed(off);
                if smoothed_data[nidx] > val || (smoothed_data[nidx] == val && nidx < idx) {
                    is_max = false;
                    break;
                }
            }
            if is_max {
                proposals.push(Proposal {
                    x: x as f32,
                    y: y as f32,
                    score: val,
                });
            }
        }
    }

    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    truncate_proposals(&mut proposals, config.max_candidates);
    proposals
}

/// Greedy distance suppression: keep proposals in score order, rejecting
/// any that fall within `min_distance_px` of an already-accepted proposal.
pub(crate) fn suppress_proposals_by_distance(
    proposals: &[Proposal],
    min_distance_px: f32,
) -> Vec<Proposal> {
    if proposals.is_empty() || !min_distance_px.is_finite() || min_distance_px <= 0.0 {
        return proposals.to_vec();
    }

    let cell_size = min_distance_px;
    let min_distance_sq = min_distance_px * min_distance_px;
    let mut accepted: Vec<Proposal> = Vec::with_capacity(proposals.len());
    let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();

    for &proposal in proposals {
        let cell_x = (proposal.x / cell_size).floor() as i32;
        let cell_y = (proposal.y / cell_size).floor() as i32;
        let mut keep = true;

        'neighbors: for ny in (cell_y - 1)..=(cell_y + 1) {
            for nx in (cell_x - 1)..=(cell_x + 1) {
                let Some(indices) = grid.get(&(nx, ny)) else {
                    continue;
                };
                for &accepted_idx in indices {
                    let other = accepted[accepted_idx];
                    let dx = proposal.x - other.x;
                    let dy = proposal.y - other.y;
                    if dx * dx + dy * dy < min_distance_sq {
                        keep = false;
                        break 'neighbors;
                    }
                }
            }
        }

        if keep {
            let accepted_idx = accepted.len();
            accepted.push(proposal);
            grid.entry((cell_x, cell_y)).or_default().push(accepted_idx);
        }
    }

    accepted
}

pub(crate) fn truncate_proposals(proposals: &mut Vec<Proposal>, max_candidates: Option<usize>) {
    if let Some(max_candidates) = max_candidates {
        proposals.truncate(max_candidates.min(proposals.len()));
    }
}
