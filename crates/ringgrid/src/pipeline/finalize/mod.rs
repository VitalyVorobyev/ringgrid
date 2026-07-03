//! Finalize stage: projective centers → ID correction → global filter /
//! plain assignment → completion → final H refit → geometric verify.

mod coded;
mod common;
mod plain;

use super::time_compat::Instant;

use image::GrayImage;

use super::axis_ratio_filter::remove_axis_ratio_outliers;
use super::{
    DetectConfig, annotate_neighbor_radius_ratios, apply_projective_centers,
    try_recover_inner_as_outer, verify_and_correct_ids, warn_center_correction_without_intrinsics,
};
use crate::detector::MarkerRecord;
use crate::pipeline::{DetectionFrame, PipelineResult};
use crate::pixelmap::PixelMapper;

use coded::{finalize_global_filter_result, finalize_no_global_filter_result};
use common::{duration_ms, sync_marker_board_correspondence};
use plain::finalize_plain_result;

/// Runs `annotate_neighbor_radius_ratios`, then optionally runs
/// `try_recover_inner_as_outer` (+ re-sync + re-annotate) when recovery is
/// enabled. Called identically by both finalize paths, eliminating duplication.
fn apply_post_filter_fixup(
    gray: &GrayImage,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) {
    let k = config.advanced.inner_as_outer_recovery.k_neighbors;
    annotate_neighbor_radius_ratios(markers, k);
    if config.advanced.inner_as_outer_recovery.enable {
        try_recover_inner_as_outer(gray, markers, config, mapper);
        sync_marker_board_correspondence(markers, &config.target);
        annotate_neighbor_radius_ratios(markers, k);
    }
    let removed = remove_axis_ratio_outliers(markers);
    if removed > 0 {
        annotate_neighbor_radius_ratios(markers, k);
        tracing::info!(removed, "axis-ratio consistency filter removed markers");
    }
}

/// ID correction is a hex-neighbor BFS consensus over decoded IDs; it applies
/// only to hex coded targets. Rect coded targets rely on the global filter +
/// geometric verify instead, and plain targets carry no IDs at all.
fn id_correction_applies(config: &DetectConfig) -> bool {
    config.advanced.id_correction.enable
        && config.target.is_coded()
        && matches!(
            config.target.lattice_kind(),
            projective_grid::LatticeKind::Hex
        )
}

fn log_id_correction_summary(stats: &crate::detector::id_correction::IdCorrectionStats) {
    tracing::info!(
        n_ids_corrected = stats.n_ids_corrected,
        n_ids_recovered = stats.n_ids_recovered,
        n_recovered_local = stats.n_recovered_local,
        n_recovered_homography = stats.n_recovered_homography,
        n_homography_seeded = stats.n_homography_seeded,
        n_ids_cleared = stats.n_ids_cleared,
        n_ids_cleared_inconsistent_pre = stats.n_ids_cleared_inconsistent_pre,
        n_ids_cleared_inconsistent_post = stats.n_ids_cleared_inconsistent_post,
        n_soft_locked_cleared = stats.n_soft_locked_cleared,
        n_verified = stats.n_verified,
        n_inconsistent_remaining = stats.n_inconsistent_remaining,
        n_unverified_no_neighbors = stats.n_unverified_no_neighbors,
        n_unverified_no_votes = stats.n_unverified_no_votes,
        n_unverified_gate_rejects = stats.n_unverified_gate_rejects,
        n_iterations = stats.n_iterations,
        pitch_px = stats.pitch_px_estimated,
        "id_correction complete"
    );
}

/// Apply projective-center correction and ID correction to `fit_markers`.
///
/// Returns the corrected marker list **without** running global filter,
/// completion, or final H refit. These post-merge stages are deferred to
/// [`finalize_postmerge`] after all tier outputs have been merged by
/// `merge_multiscale_markers`.
///
/// Called once per scale tier by `detect_multiscale`.
pub(super) fn finalize_premerge(
    fit_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
) -> Vec<MarkerRecord> {
    let mut corrected_markers = fit_markers;

    if config.circle_refinement.uses_projective_center() {
        apply_projective_centers(&mut corrected_markers, config);
    }

    if id_correction_applies(config) {
        let stats = verify_and_correct_ids(
            &mut corrected_markers,
            &config.target,
            &config.advanced.id_correction,
            config.advanced.decode.codebook_profile,
        );
        log_id_correction_summary(&stats);
    }

    corrected_markers
}

/// Apply global RANSAC homography filter, completion, and final H refit to
/// the merged marker pool.
///
/// Called once after all tier outputs have been merged by
/// `merge_multiscale_markers`. Uses `config` for the board, RANSAC
/// homography, and completion parameters.
pub(super) fn finalize_postmerge(
    gray: &GrayImage,
    merged_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> PipelineResult {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    let homography_frame = if mapper.is_some() {
        DetectionFrame::Working
    } else {
        DetectionFrame::Image
    };

    if !config.advanced.use_global_filter {
        return finalize_no_global_filter_result(
            gray,
            merged_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    }

    if !config.target.is_coded() {
        return finalize_plain_result(
            gray,
            merged_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    }

    finalize_global_filter_result(
        gray,
        merged_markers,
        config,
        mapper,
        homography_frame,
        image_size,
    )
}

pub(super) fn run(
    gray: &GrayImage,
    fit_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> PipelineResult {
    let total_start = Instant::now();
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    let homography_frame = if mapper.is_some() {
        DetectionFrame::Working
    } else {
        DetectionFrame::Image
    };
    let markers_in = fit_markers.len();

    warn_center_correction_without_intrinsics(config, mapper.is_some());

    let mut corrected_markers = fit_markers;
    let projective_center_start = Instant::now();
    if config.circle_refinement.uses_projective_center() {
        apply_projective_centers(&mut corrected_markers, config);
    }
    let projective_center_elapsed = projective_center_start.elapsed();

    let id_correction_start = Instant::now();
    if id_correction_applies(config) {
        let stats = verify_and_correct_ids(
            &mut corrected_markers,
            &config.target,
            &config.advanced.id_correction,
            config.advanced.decode.codebook_profile,
        );
        log_id_correction_summary(&stats);
    }
    let id_correction_elapsed = id_correction_start.elapsed();

    let tail_start = Instant::now();
    if !config.advanced.use_global_filter {
        let result = finalize_no_global_filter_result(
            gray,
            corrected_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        );
        tracing::info!(
            markers_in,
            markers_out = result.markers.len(),
            projective_center_ms = duration_ms(projective_center_elapsed),
            id_correction_ms = duration_ms(id_correction_elapsed),
            tail_ms = duration_ms(tail_start.elapsed()),
            total_ms = duration_ms(total_start.elapsed()),
            "finalize stage timing summary"
        );
        return result;
    }
    let result = if config.target.is_coded() {
        finalize_global_filter_result(
            gray,
            corrected_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        )
    } else {
        finalize_plain_result(
            gray,
            corrected_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        )
    };
    tracing::info!(
        markers_in,
        markers_out = result.markers.len(),
        projective_center_ms = duration_ms(projective_center_elapsed),
        id_correction_ms = duration_ms(id_correction_elapsed),
        tail_ms = duration_ms(tail_start.elapsed()),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize stage timing summary"
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OffsetMapper;

    impl PixelMapper for OffsetMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some([image_xy[0] + 5.0, image_xy[1] + 7.0])
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some([working_xy[0] - 5.0, working_xy[1] - 7.0])
        }
    }

    struct RejectingMapper;

    impl PixelMapper for RejectingMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some(image_xy)
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            if working_xy[0] > 50.0 {
                None
            } else {
                Some([working_xy[0], working_xy[1]])
            }
        }
    }

    fn marker(center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            confidence: 1.0,
            center,
            ..MarkerRecord::default()
        }
    }

    fn marker_with_id(id: usize, center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            id: Some(id),
            confidence: 1.0,
            center,
            ..MarkerRecord::default()
        }
    }

    fn no_global_filter_config() -> DetectConfig {
        let mut config = DetectConfig {
            circle_refinement: crate::CircleRefinementMethod::None,
            ..DetectConfig::default()
        };
        config.advanced.use_global_filter = false;
        config
    }

    #[test]
    fn no_mapper_keeps_image_space_centers() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([12.0, 22.0])];
        let (result, _) = run(&img, markers, &no_global_filter_config(), None).split();

        assert_eq!(result.center_frame, DetectionFrame::Image);
        assert_eq!(result.homography_frame, DetectionFrame::Image);
        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [12.0, 22.0]);
        assert!(result.detected_markers[0].center_mapped.is_none());
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn no_mapper_populates_board_xy_for_valid_id() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let expected = config.target.xy_mm_of_id(0).expect("board marker 0");
        let markers = vec![marker_with_id(0, [12.0, 22.0])];
        let (result, _) = run(&img, markers, &config, None).split();

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].id, Some(0));
        assert_eq!(
            result.detected_markers[0].board_xy_mm,
            Some([expected[0] as f64, expected[1] as f64])
        );
    }

    #[test]
    fn mapper_outputs_image_center_and_preserves_mapped_center() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let expected = config.target.xy_mm_of_id(1).expect("board marker 1");
        let markers = vec![marker_with_id(1, [20.0, 30.0])];
        let mapper = OffsetMapper;
        let (result, _) = run(&img, markers, &config, Some(&mapper)).split();

        assert_eq!(result.center_frame, DetectionFrame::Image);
        assert_eq!(result.homography_frame, DetectionFrame::Working);
        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [15.0, 23.0]);
        assert_eq!(result.detected_markers[0].center_mapped, Some([20.0, 30.0]));
        assert_eq!(result.detected_markers[0].id, Some(1));
        assert_eq!(
            result.detected_markers[0].board_xy_mm,
            Some([expected[0] as f64, expected[1] as f64])
        );
    }

    #[test]
    fn mapper_drops_unmappable_centers() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([10.0, 10.0]), marker([60.0, 10.0])];
        let mapper = RejectingMapper;
        let (result, _) = run(&img, markers, &no_global_filter_config(), Some(&mapper)).split();

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [10.0, 10.0]);
        assert_eq!(result.detected_markers[0].center_mapped, Some([10.0, 10.0]));
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn invalid_id_is_cleared_and_board_xy_is_none() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let invalid_id = config.target.max_marker_id() + 1;
        let markers = vec![marker_with_id(invalid_id, [12.0, 22.0])];
        let (result, _) = run(&img, markers, &config, None).split();

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].id, None);
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn serialization_omits_none_board_xy_and_includes_finite_when_present() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let markers = vec![marker([12.0, 22.0]), marker_with_id(0, [20.0, 30.0])];
        let (result, _) = run(&img, markers, &config, None).split();
        let json = serde_json::to_value(&result).expect("serialize detection result");
        let detected = json
            .get("detected_markers")
            .and_then(|v| v.as_array())
            .expect("detected markers array");

        assert!(detected[0].get("board_xy_mm").is_none());
        let board_xy = detected[1]
            .get("board_xy_mm")
            .and_then(|v| v.as_array())
            .expect("board_xy_mm must be present for valid id");
        assert_eq!(board_xy.len(), 2);
        assert!(
            board_xy
                .iter()
                .all(|v| v.as_f64().is_some_and(|x| x.is_finite()))
        );
    }
}
