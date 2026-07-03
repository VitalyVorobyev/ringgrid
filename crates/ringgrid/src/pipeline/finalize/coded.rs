//! Coded-target finalize path: global RANSAC homography filter, completion at
//! missing IDs, and final H refit.

use crate::pipeline::time_compat::Instant;

use image::GrayImage;

use super::apply_post_filter_fixup;
use super::common::{
    drop_unmappable_markers_with_warning, duration_ms, map_centers_to_image,
    phase_geometric_verify, sync_marker_board_correspondence_with_logging,
};
use crate::detector::MarkerRecord;
use crate::homography::RansacStats;
use crate::pipeline::{
    CompletionStats, DetectConfig, DetectionFrame, PipelineResult, apply_projective_centers,
    complete_with_h, compute_h_stats, global_filter, matrix3_to_array, mean_reproj_error_px,
    refit_homography,
};
use crate::pixelmap::PixelMapper;

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

pub(super) fn phase_final_h(
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

pub(super) fn finalize_no_global_filter_result(
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

    let board_frame = corrected_markers
        .iter()
        .any(|m| m.id.is_some())
        .then_some(crate::pipeline::BoardFrame::Absolute);
    let result = PipelineResult {
        markers: corrected_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        board_frame,
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

pub(super) fn finalize_global_filter_result(
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

    let board_frame = final_markers
        .iter()
        .any(|m| m.id.is_some())
        .then_some(crate::pipeline::BoardFrame::Absolute);
    let result = PipelineResult {
        markers: final_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        homography: final_h,
        board_frame,
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
