//! Fallbacks for views the grid-first path cannot label or anchor.
//!
//! - [`finalize_decode_anchored_fallback`] (coded targets): ID correction
//!   (hex lattices), then a global RANSAC homography over decoded markers that
//!   the rings' own decodes must corroborate, completion grown from the
//!   identified patch, final H refit and geometric verify. Without such a
//!   homography it fails closed: markers without IDs.
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
    CompletionScope, DetectConfig, DetectionFrame, PipelineResult, apply_projective_centers,
    complete_with_h, global_filter, matrix3_to_array, verify_and_correct_ids,
};
use crate::pixelmap::PixelMapper;

struct FilterPhaseOutput {
    markers: Vec<MarkerRecord>,
    h: nalgebra::Matrix3<f64>,
    ransac_stats: Option<RansacStats>,
}

/// Whether a marker's ID is its own ring's accepted decode — independent
/// evidence — rather than one ID correction derived from its neighbors.
fn id_is_own_decode(m: &MarkerRecord) -> bool {
    m.id.is_some() && m.decode.as_ref().is_some_and(|d| Some(d.best_id) == m.id)
}

/// Global RANSAC homography over the decoded markers, kept only when at least
/// `ransac_homography.min_inliers` of its inliers carry their own ring's
/// decode. IDs that ID correction recovered from neighbors are not independent
/// evidence: around a few misread seeds they form a self-consistent wrong
/// labeling that RANSAC fits tightly (precision harness: every wrong fallback
/// homography had 1–4 own-decode inliers; 7 of 15 inliers at 0.09 px is
/// typical). `None` when no homography is accepted. Only reached when
/// `advanced.use_global_filter` is on — `finalize_postmerge` owns that switch.
fn filter_with_h(markers: &[MarkerRecord], config: &DetectConfig) -> Option<FilterPhaseOutput> {
    let ransac = &config.advanced.ransac_homography;
    let (filtered, h_result, stats) = global_filter(markers, ransac, &config.target);
    let h = h_result?.h;
    let n_own = filtered.iter().filter(|m| id_is_own_decode(m)).count();
    tracing::info!(
        n_own,
        n_inliers = filtered.len(),
        "fallback homography support"
    );
    (n_own >= ransac.min_inliers).then_some(FilterPhaseOutput {
        markers: filtered,
        h,
        ransac_stats: stats,
    })
}

/// Fail-closed result when no homography corroborates the decodes: a single
/// decode cannot be trusted (an unreadable code band yields a valid-looking
/// word often enough), so every marker is returned without an ID, board
/// position or homography.
fn finalize_without_homography(
    gray: &GrayImage,
    markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let mut result =
        finalize_unlabeled_result(gray, markers, config, mapper, homography_frame, image_size);
    for m in &mut result.markers {
        m.id = None;
        m.board_xy_mm = None;
        m.grid_coord = None;
    }
    result.board_frame = None;
    result
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

    let Some(mut filter_phase) = filter_with_h(&corrected_markers, config) else {
        tracing::info!("decode-anchored fallback: no corroborated homography; no IDs");
        return finalize_without_homography(
            gray,
            corrected_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    };
    let mut final_markers = filter_phase.markers;
    let markers_after_filter = final_markers.len();
    let h_current = filter_phase.h;

    // Completion grows from the identified patch only, one lattice ring per
    // round: a whole-board sweep seeds far cells from the homography's
    // extrapolation, which a patch of decodes clustered on the near side of a
    // steep view gets wrong by a lattice row (precision harness, rect 65°).
    let completion_start = Instant::now();
    let n_before_completion = final_markers.len();
    complete_with_h(
        gray,
        &h_current,
        &mut final_markers,
        &mut Vec::new(),
        config,
        CompletionScope::CodedPatch,
        mapper,
    );
    let completion_elapsed = completion_start.elapsed();
    let markers_after_completion = final_markers.len();
    if final_markers.len() > n_before_completion {
        apply_projective_centers(&mut final_markers[n_before_completion..], config);
    }

    apply_marker_fixup(gray, &mut final_markers, config, mapper);
    drop_unmappable_markers_with_warning(&mut final_markers, mapper);

    let (final_h_mat, final_ransac) = phase_final_h(
        &final_markers,
        Some(h_current),
        filter_phase.ransac_stats.take(),
        config,
    );

    // Sync id → board_xy_mm before the gate. This is id-only (no center-frame
    // dependency), so it is safe ahead of the image remap and gives the global
    // reprojection test the board positions it needs.
    sync_marker_board_correspondence_with_logging(&mut final_markers, &config.target);

    // Final precision-first geometric verification, in the working frame (where
    // `final_h` was fit), over all markers including completed ones — before
    // centers are remapped to image space.
    let geom_stats = phase_geometric_verify(&mut final_markers, final_h_mat.as_ref(), config);
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut final_markers, mapper);
    }
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
        homography: final_h_mat.as_ref().map(matrix3_to_array),
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
        completion_ms = duration_ms(completion_elapsed),
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
