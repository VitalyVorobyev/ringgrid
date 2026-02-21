//! Structural ID verification and correction using hex neighborhood consensus.
//!
//! ## Algorithm
//!
//! 1. **Build hex index** — precompute `(q,r) → id` and board neighbors from
//!    [`BoardLayout`].
//!
//! 2. **Estimate pitch** — median of `image_dist / pitch_mm` over all pairs of
//!    decoded markers whose board IDs are hex-adjacent.
//!
//! 3. **Bootstrap seeds** — mark markers with high decode margin
//!    (`margin >= CODEBOOK_MIN_CYCLIC_DIST`) as trusted. Requires ≥ 2 seeds.
//!
//! 4. **Iterative propagation** — for each unverified marker, gather trusted
//!    neighbors within `neighbor_search_radius_px`. Fit a local 2D affine from
//!    board-mm to image-px (using ≥ 3 neighbors); fall back to per-neighbor
//!    scale-based prediction with fewer neighbors. Each neighbor casts a
//!    weighted vote for the nearest board marker to the predicted position.
//!    Accept the winning candidate if it has ≥ `min_votes` (or `min_votes_recover`
//!    for `id=None` markers) votes and ≥ `min_vote_weight_frac` of total weight.
//!
//! 5. **Final cleanup** — markers still unverified after all iterations have
//!    their IDs cleared (or are removed if `remove_unverified = true`). ID
//!    conflicts (two markers claiming the same board ID) resolve by clearing the
//!    lower-confidence duplicate.

mod index;
mod math;
mod vote;

use crate::board_layout::BoardLayout;
use crate::detector::config::IdCorrectionConfig;
use crate::detector::marker_build::DetectedMarker;
use crate::marker::codebook::CODEBOOK_MIN_CYCLIC_DIST;

use index::{estimate_pitch_px, BoardIndex};
use vote::{gather_trusted_neighbors, resolve_id_conflicts, vote_for_candidate};

// ── Internal types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trust {
    Trusted,
    Tentative,
    Unverified,
}

// ── Public output ─────────────────────────────────────────────────────────────

/// Statistics produced by the ID verification and correction stage.
#[derive(Debug, Clone, Default)]
pub(crate) struct IdCorrectionStats {
    /// Markers whose decoded ID was replaced with a different, verified ID.
    pub n_ids_corrected: usize,
    /// Markers whose id was `None` and received a new ID from neighbor consensus.
    pub n_ids_recovered: usize,
    /// Markers whose ID was cleared (`id = None`) after failed verification.
    pub n_ids_cleared: usize,
    /// Markers removed entirely (only when `remove_unverified = true`).
    pub n_markers_removed: usize,
    /// Markers confirmed as structurally consistent with the board layout.
    pub n_verified: usize,
    /// Number of iterative correction passes executed.
    pub n_iterations: usize,
    /// Estimated board pitch in image pixels (median over adjacent-pair distances).
    pub pitch_px_estimated: Option<f64>,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Verify and correct marker IDs using the board's hex neighborhood structure.
///
/// Mutates `markers` in-place. Returns statistics about the corrections made.
pub(crate) fn verify_and_correct_ids(
    markers: &mut Vec<DetectedMarker>,
    board: &BoardLayout,
    config: &IdCorrectionConfig,
) -> IdCorrectionStats {
    let mut stats = IdCorrectionStats::default();

    if markers.is_empty() {
        return stats;
    }

    // ── 1. Build board index ─────────────────────────────────────────────────
    let board_index = BoardIndex::build(board);
    let pitch_mm = board_index.pitch_mm;

    // ── 2. Estimate image-space pitch ────────────────────────────────────────
    let pitch_px = estimate_pitch_px(markers, &board_index);
    stats.pitch_px_estimated = pitch_px;

    let search_radius_px = config
        .neighbor_search_radius_px
        .or_else(|| pitch_px.map(|p| p * 2.5))
        .unwrap_or(pitch_mm * 5.0); // rough fallback if no pairs found
    let search_radius_sq = search_radius_px * search_radius_px;

    // Snap-to-board tolerance: a board position must be within this many mm of
    // a real board marker to count as a valid vote.
    let tolerance_mm = pitch_mm * 0.6;

    // ── 3. Bootstrap trusted seeds ───────────────────────────────────────────
    let n = markers.len();
    let mut trust = vec![Trust::Unverified; n];

    let n_seeds: usize = markers
        .iter()
        .enumerate()
        .filter(|(i, m)| {
            let id = match m.id {
                Some(id) if board_index.id_to_xy.contains_key(&id) => id,
                _ => return false,
            };
            let high_margin = m
                .decode
                .as_ref()
                .is_some_and(|d| usize::from(d.margin) >= CODEBOOK_MIN_CYCLIC_DIST);
            let high_confidence = m.confidence >= config.seed_min_decode_confidence;
            let _ = id;
            if high_margin || high_confidence {
                trust[*i] = Trust::Trusted;
                true
            } else {
                false
            }
        })
        .count();

    if n_seeds < 2 {
        tracing::debug!(n_seeds, "id_correction: too few seeds, skipping");
        return stats;
    }

    tracing::debug!(
        n_seeds,
        n_markers = n,
        search_radius_px,
        "id_correction: starting iterative correction"
    );

    // ── 4. Iterative propagation ─────────────────────────────────────────────
    for iter in 0..config.max_iters {
        let mut corrections: Vec<(usize, Option<usize>)> = Vec::new();

        for i in 0..n {
            if trust[i] == Trust::Trusted || trust[i] == Trust::Tentative {
                continue;
            }

            let neighbors =
                gather_trusted_neighbors(i, markers, &trust, &board_index, search_radius_sq);
            if neighbors.is_empty() {
                continue;
            }

            // Use a lower vote threshold for id=None markers: a single high-confidence
            // neighbor is sufficient evidence when there is no existing wrong ID to protect.
            let effective_min_votes = if markers[i].id.is_none() {
                config.min_votes_recover
            } else {
                config.min_votes
            };

            let candidate = vote_for_candidate(
                markers[i].center,
                &neighbors,
                &board_index,
                tolerance_mm,
                pitch_px,
                effective_min_votes,
                config.min_vote_weight_frac,
            );

            if let Some(cid) = candidate {
                // Do not overwrite a higher-confidence trusted marker's ID.
                let taken_by_higher = markers.iter().enumerate().any(|(j, mj)| {
                    j != i
                        && mj.id == Some(cid)
                        && trust[j] == Trust::Trusted
                        && mj.confidence >= markers[i].confidence
                });
                if !taken_by_higher {
                    corrections.push((i, Some(cid)));
                }
            }
        }

        // Count every Unverified→Tentative promotion, including same-ID confirmations
        // that do not change the ID but advance the trusted frontier for the next pass.
        let mut n_promoted = 0usize;
        for (i, new_id) in corrections {
            if markers[i].id != new_id {
                match markers[i].id {
                    None => {
                        stats.n_ids_recovered += 1;
                    }
                    Some(_) => {
                        stats.n_ids_corrected += 1;
                    }
                }
                markers[i].id = new_id;
            }
            trust[i] = Trust::Tentative;
            n_promoted += 1;
        }

        // Promote tentative → trusted so newly confirmed markers can vote in the next pass.
        for t in trust.iter_mut() {
            if *t == Trust::Tentative {
                *t = Trust::Trusted;
            }
        }

        stats.n_iterations = iter + 1;
        if n_promoted == 0 && iter > 0 {
            break;
        }
    }

    stats.n_verified = trust.iter().filter(|&&t| t == Trust::Trusted).count();

    // ── 5. Final cleanup ─────────────────────────────────────────────────────
    if config.remove_unverified {
        let mut i = 0;
        while i < markers.len() {
            if trust[i] == Trust::Unverified && markers[i].id.is_some() {
                markers.remove(i);
                trust.remove(i);
                stats.n_markers_removed += 1;
            } else {
                i += 1;
            }
        }
    } else {
        for i in 0..markers.len() {
            if trust[i] == Trust::Unverified && markers[i].id.is_some() {
                markers[i].id = None;
                stats.n_ids_cleared += 1;
            }
        }
    }

    // Resolve any ID conflicts introduced by corrections.
    let _ = resolve_id_conflicts(markers);

    stats
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use index::{estimate_pitch_px, BoardIndex, HEX_NEIGHBORS};
    use math::{affine_to_board, fit_local_affine, solve_3x3};

    #[test]
    fn hex_neighbor_offsets_cover_all_six_directions() {
        let set: std::collections::HashSet<(i16, i16)> = HEX_NEIGHBORS.iter().copied().collect();
        assert_eq!(set.len(), 6, "all six neighbor offsets must be distinct");
        // Each direction must have an opposite.
        for &(q, r) in &HEX_NEIGHBORS {
            assert!(set.contains(&(-q, -r)), "missing opposite of ({q}, {r})");
        }
    }

    #[test]
    fn solve_3x3_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [3.0, 7.0, 2.0];
        let x = solve_3x3(&a, &b).expect("must solve identity");
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn solve_3x3_rejects_singular() {
        let a = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]; // rank-2
        let b = [1.0, 2.0, 0.0];
        assert!(solve_3x3(&a, &b).is_none());
    }

    #[test]
    fn fit_affine_roundtrip() {
        // Translation + scale affine: image = 2*board + [100, 200]
        let board_pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let image_pts = board_pts.map(|[x, y]| [2.0 * x + 100.0, 2.0 * y + 200.0]);
        let aff = fit_local_affine(&board_pts, &image_pts).expect("affine must fit");
        // Forward check: board → image
        let query_board = [0.5, 0.5];
        let expected_img = [101.0, 201.0];
        let img_x = aff[0][0] * query_board[0] + aff[0][1] * query_board[1] + aff[0][2];
        let img_y = aff[1][0] * query_board[0] + aff[1][1] * query_board[1] + aff[1][2];
        assert!((img_x - expected_img[0]).abs() < 1e-8);
        assert!((img_y - expected_img[1]).abs() < 1e-8);
        // Inverse check: image → board
        let recovered = affine_to_board(&aff, expected_img).expect("must invert");
        assert!((recovered[0] - query_board[0]).abs() < 1e-8);
        assert!((recovered[1] - query_board[1]).abs() < 1e-8);
    }

    #[test]
    fn pitch_estimation_from_adjacent_pair() {
        // Two markers at board IDs 0 and 1 (adjacent), pitch_mm = 1.0
        // Place them 50 px apart in image space.
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        // Find two adjacent board IDs.
        let (id0, id1) = board_index
            .board_neighbors
            .iter()
            .find_map(|(&id, nbrs)| nbrs.first().map(|&nb| (id, nb)))
            .expect("board must have neighbors");

        let pitch_mm = board_index.pitch_mm;
        let m0 = DetectedMarker {
            id: Some(id0),
            center: [0.0, 0.0],
            ..DetectedMarker::default()
        };
        let m1 = DetectedMarker {
            id: Some(id1),
            center: [50.0, 0.0], // 50 px apart
            ..DetectedMarker::default()
        };

        let pitch_px = estimate_pitch_px(&[m0, m1], &board_index);
        let expected = 50.0 / pitch_mm;
        let got = pitch_px.expect("must estimate pitch");
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn single_wrong_id_corrected_by_neighbors() {
        // Use default board (203 markers).
        // Set up 4 markers: 3 trusted with correct IDs around a central position,
        // 1 unverified with a wrong ID. The 3 trusted markers should vote for the
        // correct ID.
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);

        // Pick a marker ID that has at least 3 board neighbors.
        let (center_id, neighbor_ids) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| nbrs.len() >= 3)
            .map(|(&id, nbrs)| (id, nbrs.clone()))
            .expect("board must have markers with >= 3 neighbors");

        let pitch_mm = board_index.pitch_mm;
        // Synthetic scale: 10 px/mm.
        let scale = 10.0;

        // Position the center marker in image space.
        let center_xy_mm = board_index.id_to_xy[&center_id];
        let center_img = [
            center_xy_mm[0] as f64 * scale,
            center_xy_mm[1] as f64 * scale,
        ];

        // Build 3 trusted neighbor markers.
        let mut markers: Vec<DetectedMarker> = Vec::new();
        for &nid in &neighbor_ids[..3] {
            let bxy = board_index.id_to_xy[&nid];
            let img = [bxy[0] as f64 * scale, bxy[1] as f64 * scale];
            markers.push(DetectedMarker {
                id: Some(nid),
                center: img,
                confidence: 0.95,
                decode: Some(crate::marker::decode::DecodeMetrics {
                    observed_word: 0,
                    best_id: nid,
                    best_rotation: 0,
                    best_dist: 0,
                    margin: CODEBOOK_MIN_CYCLIC_DIST as u8,
                    decode_confidence: 1.0,
                }),
                ..DetectedMarker::default()
            });
        }

        // Add the "bad" marker: it's at the correct position for center_id but
        // has a wrong decoded ID (any ID that is NOT center_id).
        let wrong_id = board_index
            .id_to_xy
            .keys()
            .copied()
            .find(|&id| id != center_id && !neighbor_ids.contains(&id))
            .expect("board must have other markers");

        markers.push(DetectedMarker {
            id: Some(wrong_id),
            center: center_img,
            confidence: 0.5, // lower confidence than neighbors
            ..DetectedMarker::default()
        });

        // Run correction.
        let config = IdCorrectionConfig {
            enable: true,
            neighbor_search_radius_px: Some(pitch_mm * scale * 3.0),
            min_votes: 2,
            min_vote_weight_frac: 0.5,
            ..IdCorrectionConfig::default()
        };
        let stats = verify_and_correct_ids(&mut markers, &board, &config);

        // The bad marker should have been corrected to center_id.
        let corrected_marker = markers.iter().find(|m| m.center == center_img);
        assert!(
            corrected_marker.is_some(),
            "corrected marker must still be present"
        );
        assert_eq!(
            corrected_marker.unwrap().id,
            Some(center_id),
            "marker should be corrected to {center_id}"
        );
        assert!(
            stats.n_ids_corrected >= 1,
            "at least one correction expected"
        );
    }

    #[test]
    fn no_id_marker_is_not_cleared_when_unverified() {
        // A marker with id=None that has no neighbors should remain with id=None.
        let board = BoardLayout::default();
        let mut markers = vec![DetectedMarker {
            id: None,
            center: [500.0, 500.0],
            ..DetectedMarker::default()
        }];
        let config = IdCorrectionConfig::default();
        let stats = verify_and_correct_ids(&mut markers, &board, &config);
        assert_eq!(markers[0].id, None);
        assert_eq!(stats.n_ids_cleared, 0);
    }
}
