use std::collections::HashMap;
use std::time::Instant;

use image::GrayImage;
use nalgebra::Point2;
use projective_grid::GridIndex;
use projective_grid::hex::hex_find_inconsistent_corners;

use super::{
    CompletionStats, DetectConfig, annotate_neighbor_radius_ratios, apply_projective_centers,
    complete_with_h, compute_h_stats, global_filter, matrix3_to_array, mean_reproj_error_px,
    refit_homography, try_recover_inner_as_outer, verify_and_correct_ids,
    warn_center_correction_without_intrinsics,
};
use crate::board_layout::BoardLayout;
use crate::detector::{DetectedMarker, DetectionSource};
use crate::homography::RansacStats;
use crate::pipeline::{DetectionFrame, DetectionResult};
use crate::pixelmap::PixelMapper;

const AXIS_RATIO_RELATIVE_TOLERANCE: f64 = 0.25;

#[inline]
fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn marker_inner_outer_axis_ratio(marker: &DetectedMarker) -> Option<f64> {
    let inner = marker.ellipse_inner?;
    let outer = marker.ellipse_outer?;
    let inner_axis = inner.mean_axis();
    let outer_axis = outer.mean_axis();
    if !inner_axis.is_finite() || !outer_axis.is_finite() || inner_axis <= 0.0 || outer_axis <= 0.0
    {
        return None;
    }
    Some(inner_axis / outer_axis)
}

fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    })
}

fn clear_axis_ratio_outlier_ids(markers: &mut [DetectedMarker]) -> usize {
    let reference = median_f64(
        markers
            .iter()
            .filter(|marker| marker.id.is_some() && marker.source != DetectionSource::Completion)
            .filter_map(marker_inner_outer_axis_ratio)
            .collect(),
    );
    let Some(reference) = reference.filter(|ratio| ratio.is_finite() && *ratio > 0.0) else {
        return 0;
    };

    let mut cleared = 0usize;
    for marker in markers.iter_mut() {
        let Some(id) = marker.id else {
            continue;
        };
        let Some(ratio) = marker_inner_outer_axis_ratio(marker) else {
            continue;
        };
        let rel_err = (ratio - reference).abs() / reference;
        if rel_err > AXIS_RATIO_RELATIVE_TOLERANCE {
            tracing::warn!(
                id,
                observed_ratio = ratio,
                reference_ratio = reference,
                rel_err,
                "clearing marker id due to inner/outer axis-ratio inconsistency"
            );
            marker.id = None;
            marker.board_xy_mm = None;
            marker.fit.h_reproj_err_px = None;
            cleared += 1;
        }
    }
    cleared
}

struct FilterPhaseOutput {
    markers: Vec<DetectedMarker>,
    h_current: Option<nalgebra::Matrix3<f64>>,
    ransac_stats: Option<RansacStats>,
}

fn filter_with_h(fit_markers: Vec<DetectedMarker>, config: &DetectConfig) -> FilterPhaseOutput {
    if !config.use_global_filter {
        return FilterPhaseOutput {
            markers: fit_markers,
            h_current: None,
            ransac_stats: None,
        };
    }

    let (filtered, h_result, stats) =
        global_filter(&fit_markers, &config.ransac_homography, &config.board);
    let h_current = h_result.as_ref().map(|r| r.h);

    FilterPhaseOutput {
        markers: filtered,
        h_current,
        ransac_stats: stats,
    }
}

/// Build a hex grid map from decoded markers, mapping `(q, r)` → image center.
pub(crate) fn build_hex_grid_map(
    markers: &[DetectedMarker],
    board: &BoardLayout,
) -> HashMap<GridIndex, Point2<f32>> {
    markers
        .iter()
        .filter_map(|m| {
            let id = m.id?;
            let bm = board.marker(id)?;
            let q = bm.q? as i32;
            let r = bm.r? as i32;
            Some((
                GridIndex { i: q, j: r },
                Point2::new(m.center[0] as f32, m.center[1] as f32),
            ))
        })
        .collect()
}

/// Remove markers whose image-space positions are inconsistent with their hex
/// neighbors (midpoint prediction). Returns the number of markers removed.
fn topology_filter(
    markers: &mut Vec<DetectedMarker>,
    board: &BoardLayout,
    threshold_px: f32,
) -> usize {
    let grid = build_hex_grid_map(markers, board);
    let inconsistent = hex_find_inconsistent_corners(&grid, threshold_px);
    if inconsistent.is_empty() {
        return 0;
    }
    let bad_indices: std::collections::HashSet<GridIndex> =
        inconsistent.iter().map(|(idx, _)| *idx).collect();
    let before = markers.len();
    markers.retain(|m| {
        let Some(id) = m.id else { return true };
        let Some(bm) = board.marker(id) else {
            return true;
        };
        let (Some(q), Some(r)) = (bm.q, bm.r) else {
            return true;
        };
        !bad_indices.contains(&GridIndex {
            i: q as i32,
            j: r as i32,
        })
    });
    let removed = before - markers.len();
    tracing::debug!(
        removed,
        threshold_px,
        "topology filter: removed spatially inconsistent markers"
    );
    removed
}

fn phase_completion(
    gray: &GrayImage,
    final_markers: &mut Vec<DetectedMarker>,
    h_current: Option<&nalgebra::Matrix3<f64>>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> CompletionStats {
    if !config.completion.enable {
        return CompletionStats::default();
    }

    let Some(h) = h_current else {
        return CompletionStats::default();
    };

    complete_with_h(gray, h, final_markers, config, &config.board, mapper)
}

fn phase_final_h(
    final_markers: &[DetectedMarker],
    h_current: Option<nalgebra::Matrix3<f64>>,
    mut ransac_stats: Option<RansacStats>,
    config: &DetectConfig,
) -> (Option<[[f64; 3]; 3]>, Option<RansacStats>) {
    let final_h_matrix = if final_markers.len() >= 10 {
        let h_refit = refit_homography(final_markers, &config.ransac_homography, &config.board)
            .map(|(h, _)| h);
        match (h_current, h_refit) {
            (Some(h_cur), Some(h_new)) => {
                let cur_err = mean_reproj_error_px(&h_cur, final_markers, &config.board);
                let new_err = mean_reproj_error_px(&h_new, final_markers, &config.board);
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

    let final_h = final_h_matrix.as_ref().map(matrix3_to_array);
    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| {
            compute_h_stats(
                h,
                final_markers,
                config.ransac_homography.inlier_threshold,
                &config.board,
            )
        })
        .or_else(|| ransac_stats.take());

    (final_h, final_ransac)
}

fn drop_unmappable_markers(markers: &mut Vec<DetectedMarker>, mapper: &dyn PixelMapper) -> usize {
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
    markers: &mut Vec<DetectedMarker>,
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

fn map_centers_to_image(markers: &mut [DetectedMarker], mapper: &dyn PixelMapper) {
    for marker in markers.iter_mut() {
        let center_mapped = marker.center;
        marker.center_mapped = Some(center_mapped);
        if let Some(center_image) = mapper.working_to_image_pixel(center_mapped) {
            marker.center = center_image;
        }
    }
}

fn sync_marker_board_correspondence(markers: &mut [DetectedMarker], board: &BoardLayout) -> usize {
    let mut cleared_invalid_ids = 0usize;
    for marker in markers.iter_mut() {
        match marker.id {
            Some(id) => {
                if let Some(board_xy) = board.xy_mm(id) {
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
    markers: &mut [DetectedMarker],
    board: &BoardLayout,
) {
    let cleared_invalid_ids = sync_marker_board_correspondence(markers, board);
    tracing::debug!(
        cleared_invalid_ids,
        n_markers = markers.len(),
        "synchronized marker id/board correspondence"
    );
}

fn annotate_h_reprojection_and_adjust_confidence(
    markers: &mut [DetectedMarker],
    final_h: Option<[[f64; 3]; 3]>,
    alpha: f32,
) {
    let Some(h) = final_h else {
        return;
    };

    for marker in markers {
        let Some(board_xy) = marker.board_xy_mm.as_ref() else {
            continue;
        };
        let x = board_xy[0];
        let y = board_xy[1];
        let pw = h[0][0] * x + h[0][1] * y + h[0][2];
        let ph = h[1][0] * x + h[1][1] * y + h[1][2];
        let pz = h[2][0] * x + h[2][1] * y + h[2][2];
        if pz.abs() <= 1e-15 {
            continue;
        }

        let px = pw / pz;
        let py = ph / pz;
        let dx = px - marker.center[0];
        let dy = py - marker.center[1];
        let err = (dx * dx + dy * dy).sqrt() as f32;
        marker.fit.h_reproj_err_px = Some(err);
        if alpha > 0.0 {
            marker.confidence *= 1.0 / (1.0 + alpha * err);
        }
    }
}

/// Runs `annotate_neighbor_radius_ratios`, then optionally runs
/// `try_recover_inner_as_outer` (+ re-sync + re-annotate) when recovery is
/// enabled. Called identically by both finalize paths, eliminating duplication.
fn apply_post_filter_fixup(
    gray: &GrayImage,
    markers: &mut [DetectedMarker],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) {
    let k = config.inner_as_outer_recovery.k_neighbors;
    annotate_neighbor_radius_ratios(markers, k);
    if config.inner_as_outer_recovery.enable {
        try_recover_inner_as_outer(gray, markers, config, mapper);
        sync_marker_board_correspondence(markers, &config.board);
        annotate_neighbor_radius_ratios(markers, k);
    }
    let cleared = clear_axis_ratio_outlier_ids(markers);
    if cleared > 0 {
        sync_marker_board_correspondence(markers, &config.board);
        annotate_neighbor_radius_ratios(markers, k);
        tracing::info!(cleared, "axis-ratio consistency filter cleared marker ids");
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
    mut corrected_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> DetectionResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();

    let map_to_image_start = Instant::now();
    drop_unmappable_markers_with_warning(&mut corrected_markers, mapper);
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut corrected_markers, mapper);
    }
    let map_to_image_elapsed = map_to_image_start.elapsed();

    let sync_start = Instant::now();
    sync_marker_board_correspondence_with_logging(&mut corrected_markers, &config.board);
    let sync_elapsed = sync_start.elapsed();

    let post_fixup_start = Instant::now();
    apply_post_filter_fixup(gray, &mut corrected_markers, config, mapper);
    let post_fixup_elapsed = post_fixup_start.elapsed();

    let result = DetectionResult {
        detected_markers: corrected_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        ..DetectionResult::default()
    };

    tracing::info!(
        markers_in,
        markers_out = result.detected_markers.len(),
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
    corrected_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> DetectionResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();

    let filter_start = Instant::now();
    let mut filter_phase = filter_with_h(corrected_markers, config);
    let filter_elapsed = filter_start.elapsed();
    let mut final_markers = filter_phase.markers;
    let markers_after_filter = final_markers.len();
    let h_current = filter_phase.h_current;

    let topology_start = Instant::now();
    let n_topology_removed = if let Some(threshold) = config.topology_filter_threshold_px {
        topology_filter(&mut final_markers, &config.board, threshold)
    } else {
        0
    };
    let topology_elapsed = topology_start.elapsed();

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
    let (final_h, final_ransac) = phase_final_h(
        &final_markers,
        h_current,
        filter_phase.ransac_stats.take(),
        config,
    );
    let final_h_elapsed = final_h_start.elapsed();

    let map_to_image_start = Instant::now();
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut final_markers, mapper);
    }
    let map_to_image_elapsed = map_to_image_start.elapsed();

    let sync_start = Instant::now();
    sync_marker_board_correspondence_with_logging(&mut final_markers, &config.board);
    let sync_elapsed = sync_start.elapsed();

    // Annotate per-marker H-reprojection error and apply confidence soft-penalty
    // now that board_xy_mm and image-space centers are both available.
    let reproj_annotate_start = Instant::now();
    annotate_h_reprojection_and_adjust_confidence(
        &mut final_markers,
        final_h,
        config.h_reproj_confidence_alpha,
    );
    let reproj_annotate_elapsed = reproj_annotate_start.elapsed();

    tracing::info!(
        "{} markers after global filter/completion",
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

    let result = DetectionResult {
        detected_markers: final_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        ..DetectionResult::default()
    };

    tracing::info!(
        markers_in,
        markers_after_filter,
        n_topology_removed,
        markers_after_completion,
        markers_out = result.detected_markers.len(),
        global_filter_ms = duration_ms(filter_elapsed),
        topology_ms = duration_ms(topology_elapsed),
        completion_ms = duration_ms(completion_elapsed),
        final_h_ms = duration_ms(final_h_elapsed),
        map_to_image_ms = duration_ms(map_to_image_elapsed),
        sync_board_ms = duration_ms(sync_elapsed),
        reproj_annotate_ms = duration_ms(reproj_annotate_elapsed),
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
    fit_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
) -> Vec<DetectedMarker> {
    let mut corrected_markers = fit_markers;

    if config.circle_refinement.uses_projective_center() {
        apply_projective_centers(&mut corrected_markers, config);
    }

    if config.id_correction.enable {
        let stats = verify_and_correct_ids(
            &mut corrected_markers,
            &config.board,
            &config.id_correction,
            config.decode.codebook_profile,
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
    merged_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    let homography_frame = if mapper.is_some() {
        DetectionFrame::Working
    } else {
        DetectionFrame::Image
    };

    if !config.use_global_filter {
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
    fit_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
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
    if config.id_correction.enable {
        let stats = verify_and_correct_ids(
            &mut corrected_markers,
            &config.board,
            &config.id_correction,
            config.decode.codebook_profile,
        );
        log_id_correction_summary(&stats);
    }
    let id_correction_elapsed = id_correction_start.elapsed();

    let tail_start = Instant::now();
    if !config.use_global_filter {
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
            markers_out = result.detected_markers.len(),
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
        markers_out = result.detected_markers.len(),
        projective_center_ms = duration_ms(projective_center_elapsed),
        id_correction_ms = duration_ms(id_correction_elapsed),
        tail_ms = duration_ms(tail_start.elapsed()),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize stage timing summary"
    );
    result
}

#[cfg(test)]
mod axis_ratio_tests {
    use super::*;
    use crate::conic::Ellipse;

    fn marker_with_ratio(id: usize, ratio: f64, source: DetectionSource) -> DetectedMarker {
        let outer = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 20.0,
            b: 20.0,
            angle: 0.0,
        };
        let inner = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 20.0 * ratio,
            b: 20.0 * ratio,
            angle: 0.0,
        };
        DetectedMarker {
            id: Some(id),
            confidence: 1.0,
            center: [id as f64, 0.0],
            ellipse_outer: Some(outer),
            ellipse_inner: Some(inner),
            source,
            ..DetectedMarker::default()
        }
    }

    #[test]
    fn axis_ratio_filter_clears_strong_outliers() {
        let mut markers = vec![
            marker_with_ratio(0, 0.50, DetectionSource::FitDecoded),
            marker_with_ratio(1, 0.49, DetectionSource::FitDecoded),
            marker_with_ratio(2, 0.51, DetectionSource::SeededPass),
            marker_with_ratio(3, 0.30, DetectionSource::Completion),
        ];
        let cleared = clear_axis_ratio_outlier_ids(&mut markers);
        assert_eq!(cleared, 1);
        assert_eq!(markers[3].id, None);
    }

    #[test]
    fn axis_ratio_filter_keeps_in_family_markers() {
        let mut markers = vec![
            marker_with_ratio(0, 0.50, DetectionSource::FitDecoded),
            marker_with_ratio(1, 0.49, DetectionSource::FitDecoded),
            marker_with_ratio(2, 0.52, DetectionSource::Completion),
        ];
        let cleared = clear_axis_ratio_outlier_ids(&mut markers);
        assert_eq!(cleared, 0);
        assert!(markers.iter().all(|marker| marker.id.is_some()));
    }
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

    fn marker(center: [f64; 2]) -> DetectedMarker {
        DetectedMarker {
            confidence: 1.0,
            center,
            ..DetectedMarker::default()
        }
    }

    fn marker_with_id(id: usize, center: [f64; 2]) -> DetectedMarker {
        DetectedMarker {
            id: Some(id),
            confidence: 1.0,
            center,
            ..DetectedMarker::default()
        }
    }

    fn no_global_filter_config() -> DetectConfig {
        DetectConfig {
            use_global_filter: false,
            circle_refinement: crate::CircleRefinementMethod::None,
            ..DetectConfig::default()
        }
    }

    #[test]
    fn no_mapper_keeps_image_space_centers() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([12.0, 22.0])];
        let result = run(&img, markers, &no_global_filter_config(), None);

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
        let expected = config.board.xy_mm(0).expect("board marker 0");
        let markers = vec![marker_with_id(0, [12.0, 22.0])];
        let result = run(&img, markers, &config, None);

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
        let expected = config.board.xy_mm(1).expect("board marker 1");
        let markers = vec![marker_with_id(1, [20.0, 30.0])];
        let mapper = OffsetMapper;
        let result = run(&img, markers, &config, Some(&mapper));

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
        let result = run(&img, markers, &no_global_filter_config(), Some(&mapper));

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [10.0, 10.0]);
        assert_eq!(result.detected_markers[0].center_mapped, Some([10.0, 10.0]));
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn invalid_id_is_cleared_and_board_xy_is_none() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let invalid_id = config.board.max_marker_id() + 1;
        let markers = vec![marker_with_id(invalid_id, [12.0, 22.0])];
        let result = run(&img, markers, &config, None);

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].id, None);
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn serialization_omits_none_board_xy_and_includes_finite_when_present() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let markers = vec![marker([12.0, 22.0]), marker_with_id(0, [20.0, 30.0])];
        let result = run(&img, markers, &config, None);
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
