use crate::board_layout::BoardLayout;
use crate::detector::config::IdCorrectionConfig;
use crate::detector::marker_build::MarkerRecord;
use crate::marker::codec::{Codebook, CodebookProfile};

use super::bootstrap::bootstrap_trust_anchors;
use super::cleanup::{cleanup_unverified_markers, finalize_correction_stats};
use super::consistency::{confirm_ids_by_consistency, scrub_inconsistent_ids};
use super::diagnostics::diagnose_unverified_reasons;
use super::homography::{fit_anchor_homography_for_local_stage, run_homography_fallback};
use super::local::{run_adaptive_local_recovery, run_post_consistency_refill};
use super::types::{IdCorrectionStats, ScrubStage};
use super::workspace::IdCorrectionWorkspace;

/// Verify and correct marker IDs using the board's hex neighborhood structure.
///
/// Mutates `markers` in-place. Returns statistics about the corrections made.
pub(crate) fn verify_and_correct_ids(
    markers: &mut Vec<MarkerRecord>,
    board: &BoardLayout,
    config: &IdCorrectionConfig,
    codebook_profile: CodebookProfile,
) -> IdCorrectionStats {
    let codebook_min_cyclic_dist = Codebook::from_profile(codebook_profile).min_cyclic_dist();
    let mut ws = IdCorrectionWorkspace::new(markers, board, config, codebook_min_cyclic_dist);
    if !config.enable || ws.markers.is_empty() {
        return ws.stats;
    }

    let final_outer_mul = ws.final_outer_mul();
    let n_seeds = bootstrap_trust_anchors(&mut ws);
    if n_seeds < 2 {
        tracing::debug!(n_seeds, "id_correction: too few seeds, skipping");
        return ws.stats;
    }

    tracing::debug!(
        n_seeds,
        n_markers = ws.markers.len(),
        outer_muls = ?ws.outer_muls,
        "id_correction: starting local-scale consistency-first correction",
    );

    let n_pre_cleared = scrub_inconsistent_ids(&mut ws, ScrubStage::Pre);
    tracing::debug!(
        n_pre_cleared,
        "id_correction pre-consistency scrub complete",
    );

    ws.anchor_h = fit_anchor_homography_for_local_stage(&ws);

    run_adaptive_local_recovery(&mut ws);

    if ws.trust.iter().any(|t| !t.is_trusted()) {
        run_homography_fallback(&mut ws);
    }

    run_post_consistency_refill(&mut ws);

    // Promote correct non-exact decodes that the voting stages could not reach
    // but whose neighborhood structurally confirms them, so cleanup does not
    // drop them in sparse/partial views.
    let n_confirmed = confirm_ids_by_consistency(&mut ws);
    tracing::debug!(n_confirmed, "id_correction confirm-by-consistency complete");

    diagnose_unverified_reasons(&mut ws, final_outer_mul);
    cleanup_unverified_markers(&mut ws);
    finalize_correction_stats(&mut ws);
    ws.stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::detector::id_correction::index::BoardIndex;
    use crate::marker::codec::Codebook;
    use crate::marker::decode::DecodeMetrics;

    fn marker_with_id(
        id: usize,
        center: [f64; 2],
        conf: f32,
        dist: u8,
        margin: u8,
    ) -> MarkerRecord {
        MarkerRecord {
            id: Some(id),
            center,
            confidence: conf,
            decode: Some(DecodeMetrics {
                observed_word: 0,
                best_id: id,
                best_rotation: 0,
                best_dist: dist,
                margin,
                decode_confidence: conf,
            }),
            ellipse_outer: Some(Ellipse {
                cx: center[0],
                cy: center[1],
                a: 22.0,
                b: 22.0,
                angle: 0.0,
            }),
            ..MarkerRecord::default()
        }
    }

    fn marker_no_id(center: [f64; 2], conf: f32) -> MarkerRecord {
        MarkerRecord {
            id: None,
            center,
            confidence: conf,
            ellipse_outer: Some(Ellipse {
                cx: center[0],
                cy: center[1],
                a: 22.0,
                b: 22.0,
                angle: 0.0,
            }),
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn workspace_constructs_parallel_state_vectors() {
        let board = BoardLayout::default();
        let mut markers = vec![
            marker_no_id([10.0, 20.0], 0.3),
            marker_no_id([30.0, 40.0], 0.4),
        ];
        let cfg = IdCorrectionConfig::default();

        let ws = IdCorrectionWorkspace::new(
            &mut markers,
            &board,
            &cfg,
            Codebook::default().min_cyclic_dist(),
        );
        assert_eq!(ws.markers.len(), 2);
        assert_eq!(ws.trust.len(), 2);
        assert_eq!(ws.outer_radii_px.len(), 2);
        assert!(!ws.outer_muls.is_empty());
    }

    #[test]
    fn verify_skips_when_no_seed_anchors() {
        let board = BoardLayout::default();
        let mut markers = vec![
            marker_no_id([10.0, 20.0], 0.2),
            marker_no_id([30.0, 40.0], 0.2),
        ];

        let stats = verify_and_correct_ids(
            &mut markers,
            &board,
            &IdCorrectionConfig::default(),
            CodebookProfile::Base,
        );
        assert_eq!(stats.n_verified, 0);
        assert_eq!(stats.n_ids_recovered, 0);
        assert!(markers.iter().all(|m| m.id.is_none()));
    }

    #[test]
    fn pre_scrub_then_local_recovery_pipeline_path() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (&center_id, neighbors) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| nbrs.len() >= 3)
            .expect("board must have marker with >=3 neighbors");
        let wrong_id = board_index
            .id_to_xy
            .keys()
            .copied()
            .find(|id| *id != center_id && !neighbors.contains(id))
            .expect("must find non-neighbor wrong id");

        let scale = 4.0f64;
        let mut markers = Vec::<MarkerRecord>::new();
        let min_cyclic_dist = Codebook::default().min_cyclic_dist() as u8;
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            markers.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                min_cyclic_dist,
            ));
        }
        let cxy = board_index.id_to_xy[&center_id];
        // wrong decoded ID at true center location
        markers.push(marker_with_id(
            wrong_id,
            [f64::from(cxy[0]) * scale, f64::from(cxy[1]) * scale],
            0.7,
            1,
            1,
        ));

        let cfg = IdCorrectionConfig {
            homography_fallback_enable: false,
            auto_search_radius_outer_muls: vec![2.4, 3.5],
            max_iters: 3,
            ..IdCorrectionConfig::default()
        };

        let stats = verify_and_correct_ids(&mut markers, &board, &cfg, CodebookProfile::Base);
        assert!(stats.n_ids_cleared_inconsistent_pre >= 1);
        assert!(markers.iter().any(|m| m.id == Some(center_id)));
    }

    #[test]
    fn confirm_by_consistency_rescues_sparse_correct_decode() {
        // Sparse/partial view: a correctly-decoded but non-exact, low-confidence
        // center marker has exactly one board-adjacent decoded neighbor. Voting
        // cannot promote it (1 neighbor ⇒ no affine, no adjacent pair ⇒ no
        // votes), so without confirm-by-consistency cleanup drops it despite the
        // ID being correct. With the promotion path it survives.
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (&center_id, neighbors) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| !nbrs.is_empty())
            .expect("board must have a marker with >=1 neighbor");
        let near_id = neighbors[0];
        let far_ids: Vec<usize> = board_index
            .id_to_xy
            .keys()
            .copied()
            .filter(|id| *id != center_id && *id != near_id)
            .take(2)
            .collect();
        assert_eq!(far_ids.len(), 2, "need two extra ids for bootstrap seeds");

        let scale = 4.0f64;
        let min_cyclic_dist = Codebook::default().min_cyclic_dist() as u8;
        let px = |id: usize| {
            let xy = board_index.id_to_xy[&id];
            [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale]
        };
        let center_px = px(center_id);

        // Markers: one near exact anchor (board-adjacent to center) + two far
        // exact anchors (out of the center's neighborhood, purely to satisfy the
        // ≥2 bootstrap seeds) + the non-exact, low-confidence center decode.
        let build = || {
            vec![
                marker_with_id(near_id, px(near_id), 0.95, 0, min_cyclic_dist),
                marker_with_id(
                    far_ids[0],
                    [center_px[0] + 200.0, center_px[1]],
                    0.95,
                    0,
                    min_cyclic_dist,
                ),
                marker_with_id(
                    far_ids[1],
                    [center_px[0], center_px[1] + 200.0],
                    0.95,
                    0,
                    min_cyclic_dist,
                ),
                marker_with_id(center_id, center_px, 0.5, 1, 1),
            ]
        };
        let center_of = |markers: &[MarkerRecord]| {
            markers
                .iter()
                .find(|m| m.center == center_px)
                .expect("center marker present")
                .id
        };

        // Disabled: the correct id is cleared (reproduces the bug).
        let mut off = build();
        let cfg_off = IdCorrectionConfig {
            homography_fallback_enable: false,
            auto_search_radius_outer_muls: vec![2.4, 3.5],
            max_iters: 3,
            confirm_by_consistency: false,
            ..IdCorrectionConfig::default()
        };
        let stats_off = verify_and_correct_ids(&mut off, &board, &cfg_off, CodebookProfile::Base);
        assert_eq!(
            center_of(&off),
            None,
            "without confirm, correct id is dropped"
        );
        assert_eq!(stats_off.n_confirmed_by_consistency, 0);

        // Enabled: the correct id survives via confirm-by-consistency.
        let mut on = build();
        let cfg_on = IdCorrectionConfig {
            confirm_by_consistency: true,
            ..cfg_off.clone()
        };
        let stats_on = verify_and_correct_ids(&mut on, &board, &cfg_on, CodebookProfile::Base);
        assert_eq!(
            center_of(&on),
            Some(center_id),
            "with confirm, the correct non-exact id is promoted and kept"
        );
        assert!(stats_on.n_confirmed_by_consistency >= 1);
    }

    #[test]
    fn deterministic_assignments_and_stats() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (&center_id, neighbors) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| nbrs.len() >= 3)
            .expect("board must have marker with >=3 neighbors");

        let scale = 4.0f64;
        let mut base = Vec::<MarkerRecord>::new();
        let min_cyclic_dist = Codebook::default().min_cyclic_dist() as u8;
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            base.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                min_cyclic_dist,
            ));
        }
        let cxy = board_index.id_to_xy[&center_id];
        base.push(marker_no_id(
            [f64::from(cxy[0]) * scale, f64::from(cxy[1]) * scale],
            0.6,
        ));

        let cfg = IdCorrectionConfig {
            homography_fallback_enable: false,
            auto_search_radius_outer_muls: vec![2.4, 2.9, 3.5],
            max_iters: 3,
            ..IdCorrectionConfig::default()
        };

        let mut run_a = base.clone();
        let mut run_b = base;
        let stats_a = verify_and_correct_ids(&mut run_a, &board, &cfg, CodebookProfile::Base);
        let stats_b = verify_and_correct_ids(&mut run_b, &board, &cfg, CodebookProfile::Base);

        assert_eq!(
            run_a.iter().map(|m| m.id).collect::<Vec<_>>(),
            run_b.iter().map(|m| m.id).collect::<Vec<_>>()
        );
        assert_eq!(stats_a.n_ids_recovered, stats_b.n_ids_recovered);
        assert_eq!(stats_a.n_ids_cleared, stats_b.n_ids_cleared);
    }
}
