//! Plain-target finalize path: grid assignment, origin anchoring,
//! coordinate-keyed completion, and the plain-frame output contract.

use crate::pipeline::time_compat::Instant;

use image::GrayImage;

use super::apply_post_filter_fixup;
use super::coded::{finalize_no_global_filter_result, phase_final_h};
use super::common::{
    drop_unmappable_markers_with_warning, duration_ms, map_centers_to_image, phase_geometric_verify,
};
use crate::detector::MarkerRecord;
use crate::pipeline::{
    CompletionStats, DetectConfig, DetectionFrame, PipelineResult, apply_projective_centers,
    complete_plain_with_h, matrix3_to_array,
};
use crate::pixelmap::PixelMapper;

/// Resolve the plain-target board origin and, on success, remap every labeled
/// marker to absolute board coordinates and replace the frame homography with
/// the anchored board homography. Returns whether the origin was resolved.
fn phase_plain_anchor(
    gray: &GrayImage,
    markers: &mut [MarkerRecord],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    h_current: &mut nalgebra::Matrix3<f64>,
) -> bool {
    let Some(res) = crate::pipeline::anchor::resolve_origin(gray, markers, &config.target, mapper)
    else {
        return false;
    };
    for m in markers.iter_mut() {
        if let Some(c) = m.grid_coord {
            let board = res.coord_map.apply(projective_grid::Coord::new(c[0], c[1]));
            m.grid_coord = Some([board.u, board.v]);
            m.board_xy_mm = config
                .target
                .cell_xy_mm(board)
                .map(|xy| [f64::from(xy[0]), f64::from(xy[1])]);
        }
    }
    *h_current = res.h;
    true
}

/// Coordinate-keyed completion plus projective-center correction for the newly
/// completed markers only — the plain counterpart of `phase_completion`.
fn phase_plain_completion(
    gray: &GrayImage,
    h_current: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    anchored: bool,
    mapper: Option<&dyn PixelMapper>,
) -> CompletionStats {
    let n_before = markers.len();
    let stats = complete_plain_with_h(
        gray,
        h_current,
        markers,
        config,
        &config.target,
        anchored,
        mapper,
    );
    if config.circle_refinement.uses_projective_center() && markers.len() > n_before {
        apply_projective_centers(&mut markers[n_before..], config);
    }
    stats
}

/// Public plain-frame contract: millimeter positions only in the absolute
/// board frame — a wrong millimeter position is worse than none. Returns the
/// frame of the labeled outputs.
fn enforce_plain_frame_contract(
    markers: &mut [MarkerRecord],
    anchored: bool,
) -> crate::pipeline::BoardFrame {
    if anchored {
        return crate::pipeline::BoardFrame::Absolute;
    }
    for m in markers.iter_mut() {
        m.board_xy_mm = None;
    }
    crate::pipeline::BoardFrame::RelativeCanonical
}

/// Plain-target finalize: grid assignment → origin anchor → completion →
/// final H refit → geometric verify.
///
/// Mirrors [`finalize_global_filter_result`], with the decode-driven stages
/// replaced by their coordinate-keyed counterparts: `assign_plain_grid` plays
/// the global filter's role (label + keep homography inliers), and
/// `resolve_origin` decides whether outputs are absolute board-frame or stay
/// in the canonical relative frame (`board_xy_mm` cleared).
///
/// [`finalize_global_filter_result`]: super::coded::finalize_global_filter_result
pub(super) fn finalize_plain_result(
    gray: &GrayImage,
    corrected_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();
    let mut final_markers = corrected_markers;

    // Grid assignment: label centers with lattice coords, fit the frame H in
    // f64, keep labeling/homography inliers only.
    let assign_start = Instant::now();
    let Some(assignment) = crate::pipeline::assign::assign_plain_grid(
        &mut final_markers,
        &config.target,
        &config.advanced.ransac_homography,
    ) else {
        // No labeling ⇒ markers pass through unlabeled, like the coded path
        // when too few markers decode for the global filter.
        return finalize_no_global_filter_result(
            gray,
            final_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    };
    let assign_elapsed = assign_start.elapsed();
    let markers_after_assign = final_markers.len();
    let mut h_current = assignment.h;
    let mut ransac_stats = Some(assignment.ransac);

    // Origin resolution: remap relative labels to absolute board cells when
    // the fiducial dots verify at their predicted positions.
    let anchor_start = Instant::now();
    let anchored = phase_plain_anchor(gray, &mut final_markers, config, mapper, &mut h_current);
    let anchor_elapsed = anchor_start.elapsed();

    // Completion at missing cells (full board when anchored, patch bbox when
    // not), then projective centers for the newly added markers only.
    let completion_start = Instant::now();
    let completion_stats = phase_plain_completion(
        gray,
        &h_current,
        &mut final_markers,
        config,
        anchored,
        mapper,
    );
    let completion_elapsed = completion_start.elapsed();
    drop_unmappable_markers_with_warning(&mut final_markers, mapper);

    // Final H refit over all labeled markers (frame correspondences).
    let final_h_start = Instant::now();
    let (final_h_mat, final_ransac) =
        phase_final_h(&final_markers, Some(h_current), ransac_stats.take(), config);
    let final_h_elapsed = final_h_start.elapsed();

    // Precision-first geometric verification in the working frame, over all
    // labeled markers including completed ones.
    let geom_start = Instant::now();
    let geom_stats = phase_geometric_verify(&mut final_markers, final_h_mat.as_ref(), config);
    let geom_elapsed = geom_start.elapsed();

    let final_h = final_h_mat.as_ref().map(matrix3_to_array);

    let map_to_image_start = Instant::now();
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut final_markers, mapper);
    }
    let map_to_image_elapsed = map_to_image_start.elapsed();

    let post_fixup_start = Instant::now();
    apply_post_filter_fixup(gray, &mut final_markers, config, mapper);
    let post_fixup_elapsed = post_fixup_start.elapsed();

    let board_frame = Some(enforce_plain_frame_contract(&mut final_markers, anchored));

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
        markers_after_assign,
        n_completed = completion_stats.n_added,
        n_geom_removed = geom_stats.n_removed_total,
        markers_out = result.markers.len(),
        anchored,
        assign_ms = duration_ms(assign_elapsed),
        anchor_ms = duration_ms(anchor_elapsed),
        completion_ms = duration_ms(completion_elapsed),
        final_h_ms = duration_ms(final_h_elapsed),
        geom_verify_ms = duration_ms(geom_elapsed),
        map_to_image_ms = duration_ms(map_to_image_elapsed),
        post_fixup_ms = duration_ms(post_fixup_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(plain) timing summary"
    );

    result
}
