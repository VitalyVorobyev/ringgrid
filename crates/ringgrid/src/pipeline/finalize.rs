use super::time_compat::Instant;

use image::GrayImage;

use super::axis_ratio_filter::remove_axis_ratio_outliers;
use super::geometric_verify::{
    GeometricVerifyStats, annotate_h_reproj_err_px, geometric_verify_filter,
};
use super::{
    CompletionStats, DetectConfig, annotate_neighbor_radius_ratios, apply_projective_centers,
    complete_with_h, compute_h_stats, global_filter, matrix3_to_array, mean_reproj_error_px,
    refit_homography, try_recover_inner_as_outer, verify_and_correct_ids,
    warn_center_correction_without_intrinsics,
};
use crate::detector::MarkerRecord;
use crate::homography::RansacStats;
use crate::pipeline::{DetectionFrame, PipelineResult};
use crate::pixelmap::PixelMapper;
use crate::target::TargetLayout;

#[inline]
fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

struct FilterPhaseOutput {
    markers: Vec<MarkerRecord>,
    h_current: Option<nalgebra::Matrix3<f64>>,
    ransac_stats: Option<RansacStats>,
}

fn filter_with_h(fit_markers: Vec<MarkerRecord>, config: &DetectConfig) -> FilterPhaseOutput {
    if !config.advanced.use_global_filter {
        return FilterPhaseOutput {
            markers: fit_markers,
            h_current: None,
            ransac_stats: None,
        };
    }

    let (filtered, h_result, stats) = global_filter(
        &fit_markers,
        &config.advanced.ransac_homography,
        &config.target,
    );
    let h_current = h_result.as_ref().map(|r| r.h);

    FilterPhaseOutput {
        markers: filtered,
        h_current,
        ransac_stats: stats,
    }
}

fn phase_completion(
    gray: &GrayImage,
    final_markers: &mut Vec<MarkerRecord>,
    h_current: Option<&nalgebra::Matrix3<f64>>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> CompletionStats {
    if !config.advanced.completion.enable {
        return CompletionStats::default();
    }

    let Some(h) = h_current else {
        return CompletionStats::default();
    };

    complete_with_h(gray, h, final_markers, config, &config.target, mapper)
}

fn phase_final_h(
    final_markers: &[MarkerRecord],
    h_current: Option<nalgebra::Matrix3<f64>>,
    mut ransac_stats: Option<RansacStats>,
    config: &DetectConfig,
) -> (Option<nalgebra::Matrix3<f64>>, Option<RansacStats>) {
    let final_h_matrix = if final_markers.len() >= 10 {
        let h_refit = refit_homography(
            final_markers,
            &config.advanced.ransac_homography,
            &config.target,
        )
        .map(|(h, _)| h);
        match (h_current, h_refit) {
            (Some(h_cur), Some(h_new)) => {
                let cur_err = mean_reproj_error_px(&h_cur, final_markers, &config.target);
                let new_err = mean_reproj_error_px(&h_new, final_markers, &config.target);
                if new_err.is_finite() && (new_err < cur_err || !cur_err.is_finite()) {
                    Some(h_new)
                } else {
                    Some(h_cur)
                }
            }
            (None, Some(h_new)) => Some(h_new),
            (Some(h_cur), None) => Some(h_cur),
            (None, None) => None,
        }
    } else {
        h_current
    };

    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| {
            compute_h_stats(
                h,
                final_markers,
                config.advanced.ransac_homography.inlier_threshold,
                &config.target,
            )
        })
        .or_else(|| ransac_stats.take());

    (final_h_matrix, final_ransac)
}

/// Run the geometric verification gate, or — when it is disabled — only annotate
/// the reprojection diagnostic so opt-out callers can filter on it themselves.
/// Returns the gate stats (empty when disabled).
fn phase_geometric_verify(
    markers: &mut Vec<MarkerRecord>,
    final_h: Option<&nalgebra::Matrix3<f64>>,
    config: &DetectConfig,
) -> GeometricVerifyStats {
    if config.advanced.geometric_verify {
        geometric_verify_filter(
            markers,
            final_h,
            &config.target,
            config.advanced.ransac_homography.inlier_threshold,
        )
    } else {
        annotate_h_reproj_err_px(markers, final_h);
        GeometricVerifyStats::default()
    }
}

fn drop_unmappable_markers(markers: &mut Vec<MarkerRecord>, mapper: &dyn PixelMapper) -> usize {
    let before = markers.len();
    markers.retain(|m| {
        mapper
            .working_to_image_pixel(m.center)
            .map(|p| p[0].is_finite() && p[1].is_finite())
            .unwrap_or(false)
    });
    before.saturating_sub(markers.len())
}

fn drop_unmappable_markers_with_warning(
    markers: &mut Vec<MarkerRecord>,
    mapper: Option<&dyn PixelMapper>,
) {
    let Some(mapper) = mapper else {
        return;
    };
    let dropped = drop_unmappable_markers(markers, mapper);
    if dropped > 0 {
        tracing::warn!(
            dropped,
            "dropping markers whose mapped centers cannot be converted to image frame"
        );
    }
}

fn map_centers_to_image(markers: &mut [MarkerRecord], mapper: &dyn PixelMapper) {
    for marker in markers.iter_mut() {
        let center_mapped = marker.center;
        marker.center_mapped = Some(center_mapped);
        if let Some(center_image) = mapper.working_to_image_pixel(center_mapped) {
            marker.center = center_image;
        }
    }
}

fn sync_marker_board_correspondence(markers: &mut [MarkerRecord], target: &TargetLayout) -> usize {
    let mut cleared_invalid_ids = 0usize;
    for marker in markers.iter_mut() {
        match marker.id {
            Some(id) => {
                if let Some(board_xy) = target.xy_mm_of_id(id) {
                    marker.board_xy_mm = Some([board_xy[0] as f64, board_xy[1] as f64]);
                } else {
                    marker.id = None;
                    marker.board_xy_mm = None;
                    cleared_invalid_ids += 1;
                }
            }
            None => {
                marker.board_xy_mm = None;
            }
        }
    }
    cleared_invalid_ids
}

fn sync_marker_board_correspondence_with_logging(
    markers: &mut [MarkerRecord],
    target: &TargetLayout,
) {
    let cleared_invalid_ids = sync_marker_board_correspondence(markers, target);
    tracing::debug!(
        cleared_invalid_ids,
        n_markers = markers.len(),
        "synchronized marker id/board correspondence"
    );
}

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

fn finalize_no_global_filter_result(
    gray: &GrayImage,
    mut corrected_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();

    let map_to_image_start = Instant::now();
    drop_unmappable_markers_with_warning(&mut corrected_markers, mapper);
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut corrected_markers, mapper);
    }
    let map_to_image_elapsed = map_to_image_start.elapsed();

    let sync_start = Instant::now();
    sync_marker_board_correspondence_with_logging(&mut corrected_markers, &config.target);
    let sync_elapsed = sync_start.elapsed();

    let post_fixup_start = Instant::now();
    apply_post_filter_fixup(gray, &mut corrected_markers, config, mapper);
    let post_fixup_elapsed = post_fixup_start.elapsed();

    let result = PipelineResult {
        markers: corrected_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        ..PipelineResult::default()
    };

    tracing::info!(
        markers_in,
        markers_out = result.markers.len(),
        map_to_image_ms = duration_ms(map_to_image_elapsed),
        sync_board_ms = duration_ms(sync_elapsed),
        post_fixup_ms = duration_ms(post_fixup_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(no global filter) timing summary"
    );

    result
}

fn finalize_global_filter_result(
    gray: &GrayImage,
    corrected_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();

    let filter_start = Instant::now();
    let mut filter_phase = filter_with_h(corrected_markers, config);
    let filter_elapsed = filter_start.elapsed();
    let mut final_markers = filter_phase.markers;
    let markers_after_filter = final_markers.len();
    let h_current = filter_phase.h_current;

    let n_before_completion = final_markers.len();

    let completion_start = Instant::now();
    let completion_stats =
        phase_completion(gray, &mut final_markers, h_current.as_ref(), config, mapper);
    let completion_elapsed = completion_start.elapsed();
    let markers_after_completion = final_markers.len();

    if config.circle_refinement.uses_projective_center()
        && final_markers.len() > n_before_completion
    {
        apply_projective_centers(&mut final_markers[n_before_completion..], config);
    }
    drop_unmappable_markers_with_warning(&mut final_markers, mapper);

    let final_h_start = Instant::now();
    let (final_h_mat, final_ransac) = phase_final_h(
        &final_markers,
        h_current,
        filter_phase.ransac_stats.take(),
        config,
    );
    let final_h_elapsed = final_h_start.elapsed();

    // Sync id → board_xy_mm before the gate. This is id-only (no center-frame
    // dependency), so it is safe ahead of the image remap and gives the global
    // reprojection test the board positions it needs.
    let sync_start = Instant::now();
    sync_marker_board_correspondence_with_logging(&mut final_markers, &config.target);
    let sync_elapsed = sync_start.elapsed();

    // Final precision-first geometric verification, in the working frame (where
    // `final_h` was fit), over all markers including completed ones — before
    // centers are remapped to image space.
    let geom_start = Instant::now();
    let geom_stats = phase_geometric_verify(&mut final_markers, final_h_mat.as_ref(), config);
    let geom_elapsed = geom_start.elapsed();

    let final_h = final_h_mat.as_ref().map(matrix3_to_array);

    let map_to_image_start = Instant::now();
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut final_markers, mapper);
    }
    let map_to_image_elapsed = map_to_image_start.elapsed();

    tracing::info!(
        "{} markers after global filter/completion/verify",
        final_markers.len(),
    );
    tracing::debug!(
        "Completion summary: attempted={}, added={}, failed_fit={}, failed_gate={}",
        completion_stats.n_attempted,
        completion_stats.n_added,
        completion_stats.n_failed_fit,
        completion_stats.n_failed_gate
    );

    let post_fixup_start = Instant::now();
    apply_post_filter_fixup(gray, &mut final_markers, config, mapper);
    let post_fixup_elapsed = post_fixup_start.elapsed();

    let result = PipelineResult {
        markers: final_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        ..PipelineResult::default()
    };

    tracing::info!(
        markers_in,
        markers_after_filter,
        markers_after_completion,
        n_geom_removed = geom_stats.n_removed_total,
        markers_out = result.markers.len(),
        global_filter_ms = duration_ms(filter_elapsed),
        completion_ms = duration_ms(completion_elapsed),
        final_h_ms = duration_ms(final_h_elapsed),
        sync_board_ms = duration_ms(sync_elapsed),
        geom_verify_ms = duration_ms(geom_elapsed),
        map_to_image_ms = duration_ms(map_to_image_elapsed),
        post_fixup_ms = duration_ms(post_fixup_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(global filter) timing summary"
    );

    result
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

    if config.advanced.id_correction.enable {
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
    if config.advanced.id_correction.enable {
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
    let result = finalize_global_filter_result(
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
