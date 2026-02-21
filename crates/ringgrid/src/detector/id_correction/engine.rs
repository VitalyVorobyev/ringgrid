use crate::board_layout::BoardLayout;
use crate::detector::config::IdCorrectionConfig;
use crate::detector::marker_build::DetectedMarker;

use super::bootstrap::bootstrap_trust_anchors;
use super::cleanup::{cleanup_unverified_markers, finalize_correction_stats};
use super::consistency::scrub_inconsistent_ids;
use super::diagnostics::diagnose_unverified_reasons;
use super::homography::{fit_anchor_homography_for_local_stage, run_homography_fallback};
use super::local::{run_adaptive_local_recovery, run_post_consistency_refill};
use super::types::{IdCorrectionStats, ScrubStage};
use super::workspace::IdCorrectionWorkspace;

/// Verify and correct marker IDs using the board's hex neighborhood structure.
///
/// Mutates `markers` in-place. Returns statistics about the corrections made.
pub(crate) fn verify_and_correct_ids(
    markers: &mut Vec<DetectedMarker>,
    board: &BoardLayout,
    config: &IdCorrectionConfig,
) -> IdCorrectionStats {
    let mut ws = IdCorrectionWorkspace::new(markers, board, config);
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
    use crate::marker::codebook::CODEBOOK_MIN_CYCLIC_DIST;
    use crate::marker::decode::DecodeMetrics;

    fn marker_with_id(
        id: usize,
        center: [f64; 2],
        conf: f32,
        dist: u8,
        margin: u8,
    ) -> DetectedMarker {
        DetectedMarker {
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
            ..DetectedMarker::default()
        }
    }

    fn marker_no_id(center: [f64; 2], conf: f32) -> DetectedMarker {
        DetectedMarker {
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
            ..DetectedMarker::default()
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

        let ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg);
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

        let stats = verify_and_correct_ids(&mut markers, &board, &IdCorrectionConfig::default());
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
        let mut markers = Vec::<DetectedMarker>::new();
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            markers.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
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

        let stats = verify_and_correct_ids(&mut markers, &board, &cfg);
        assert!(stats.n_ids_cleared_inconsistent_pre >= 1);
        assert!(markers.iter().any(|m| m.id == Some(center_id)));
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
        let mut base = Vec::<DetectedMarker>::new();
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            base.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
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
        let stats_a = verify_and_correct_ids(&mut run_a, &board, &cfg);
        let stats_b = verify_and_correct_ids(&mut run_b, &board, &cfg);

        assert_eq!(
            run_a.iter().map(|m| m.id).collect::<Vec<_>>(),
            run_b.iter().map(|m| m.id).collect::<Vec<_>>()
        );
        assert_eq!(stats_a.n_ids_recovered, stats_b.n_ids_recovered);
        assert_eq!(stats_a.n_ids_cleared, stats_b.n_ids_cleared);
    }
}
