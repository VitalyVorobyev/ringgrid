//! Fallbacks for views the grid-first path cannot label or anchor.
//!
//! - [`finalize_decode_anchored_fallback`] (coded targets): ID correction
//!   (hex lattices), then a global RANSAC homography over decoded markers,
//!   completion at missing IDs, final H refit and geometric verify. With fewer
//!   than four consistent decodes it returns the markers without a homography.
//! - [`finalize_unlabeled_result`]: markers pass through with their decoded IDs
//!   and no homography (plain targets without a grid, and the
//!   `use_global_filter = false` opt-out).

use crate::pipeline::time_compat::{Instant, duration_ms};

use image::GrayImage;

use super::common::phase_final_h;
use super::common::{
    drop_unmappable_markers_with_warning, map_centers_to_image, phase_geometric_verify,
    sync_marker_board_correspondence_with_logging,
};
use super::{annotate_radius_ratios, apply_marker_fixup};
use crate::detector::MarkerRecord;
use crate::homography::RansacStats;
use crate::pipeline::{
    CompletionScope, CompletionStats, DetectConfig, DetectionFrame, PipelineResult,
    apply_projective_centers, complete_with_h, global_filter, matrix3_to_array,
    verify_and_correct_ids,
};
use crate::pixelmap::PixelMapper;

struct FilterPhaseOutput {
    markers: Vec<MarkerRecord>,
    h_current: Option<nalgebra::Matrix3<f64>>,
    ransac_stats: Option<RansacStats>,
}

/// Global RANSAC homography filter. Only reached when
/// `advanced.use_global_filter` is on — `finalize_postmerge` owns that switch.
fn filter_with_h(fit_markers: Vec<MarkerRecord>, config: &DetectConfig) -> FilterPhaseOutput {
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
    let Some(h) = h_current else {
        return CompletionStats::default();
    };

    complete_with_h(
        gray,
        h,
        final_markers,
        &mut Vec::new(),
        config,
        CompletionScope::Coded,
        mapper,
    )
}

pub(super) fn finalize_unlabeled_result(
    gray: &GrayImage,
    mut corrected_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();

    let fixup_start = Instant::now();
    apply_marker_fixup(gray, &mut corrected_markers, config, mapper);
    let fixup_elapsed = fixup_start.elapsed();

    let map_to_image_start = Instant::now();
    drop_unmappable_markers_with_warning(&mut corrected_markers, mapper);
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut corrected_markers, mapper);
    }
    let map_to_image_elapsed = map_to_image_start.elapsed();

    let sync_start = Instant::now();
    sync_marker_board_correspondence_with_logging(&mut corrected_markers, &config.target);
    let sync_elapsed = sync_start.elapsed();
    annotate_radius_ratios(&mut corrected_markers, config);

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
        fixup_ms = duration_ms(fixup_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(unlabeled) timing summary"
    );

    result
}

/// Decode-anchored fallback for coded views without an anchored grid. ID
/// correction runs here only: it is the one stage that can still recover IDs
/// when the lattice labeler finds no grid (hex targets only — it is a
/// hex-neighbor consensus).
pub(super) fn finalize_decode_anchored_fallback(
    gray: &GrayImage,
    mut corrected_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();

    phase_id_correction(&mut corrected_markers, config);

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

    if final_markers.len() > n_before_completion {
        apply_projective_centers(&mut final_markers[n_before_completion..], config);
    }

    let fixup_start = Instant::now();
    apply_marker_fixup(gray, &mut final_markers, config, mapper);
    let fixup_elapsed = fixup_start.elapsed();
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

    annotate_radius_ratios(&mut final_markers, config);

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
        fixup_ms = duration_ms(fixup_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(decode-anchored fallback) timing summary"
    );

    result
}

/// Hex ID correction over the decoded markers, when it applies.
fn phase_id_correction(markers: &mut Vec<MarkerRecord>, config: &DetectConfig) {
    if !id_correction_applies(config) {
        return;
    }
    let stats = verify_and_correct_ids(
        markers,
        &config.target,
        &config.advanced.id_correction,
        config.advanced.decode.codebook_profile,
    );
    log_id_correction_summary(&stats);
}

/// ID correction is a hex-neighbor BFS consensus over decoded IDs; it applies
/// only to hex coded targets (rect coded fallbacks rely on the global filter +
/// geometric verify alone, plain targets carry no IDs).
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
