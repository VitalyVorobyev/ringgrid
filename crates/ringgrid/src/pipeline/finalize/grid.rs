//! Grid-first finalize path: grid labeling, origin anchoring, cell
//! identification, coordinate-keyed completion, and the board-frame output
//! contract.
//!
//! Plain targets always take this path. Coded targets take it under the
//! temporary grid-first A/B switch (`super::grid_first_enabled`): every fitted
//! ring — decoded or not — is labeled by the lattice, decoded IDs serve only as
//! origin evidence, and each anchored marker takes its ID from its cell.

use crate::pipeline::time_compat::{Instant, duration_ms};

use image::GrayImage;
use projective_grid::Coord;

use super::coded::{
    finalize_global_filter_result, finalize_no_global_filter_result, phase_final_h,
};
use super::common::{
    drop_unmappable_markers_with_warning, map_centers_to_image, phase_geometric_verify,
    sync_marker_board_correspondence_with_logging,
};
use super::{annotate_radius_ratios, apply_marker_fixup};
use crate::detector::MarkerRecord;
use crate::pipeline::anchor::CoordMap;
use crate::pipeline::{
    CompletionScope, CompletionStats, DetectConfig, DetectionFrame, PipelineResult,
    apply_projective_centers, complete_with_h, matrix3_to_array,
};
use crate::pixelmap::PixelMapper;
use crate::target::TargetLayout;

/// Outcome counts of [`identify_cells`].
#[derive(Debug, Default)]
struct IdentifyStats {
    /// Labeled markers whose anchored cell lies off the board (removed).
    n_off_board: usize,
    /// Undecoded markers that took their ID from their anchored cell.
    n_id_from_cell: usize,
    /// Decoded markers whose decoded ID contradicts their anchored cell; they
    /// keep the cell's ID (the ring is an observation of that cell — the code
    /// band was misread).
    n_id_contradicted: usize,
}

/// Remap every labeled marker to its absolute board cell and position, and —
/// on coded targets — give it the cell's ID. Markers labeled outside the board
/// are removed: the labeling only reaches them through a lattice error.
fn identify_cells(
    markers: &mut Vec<MarkerRecord>,
    coord_map: CoordMap,
    target: &TargetLayout,
) -> IdentifyStats {
    let mut stats = IdentifyStats::default();
    markers.retain_mut(|m| {
        let Some(c) = m.grid_coord else {
            return true;
        };
        let board = coord_map.apply(Coord::new(c[0], c[1]));
        let Some(xy) = target.cell_xy_mm(board) else {
            stats.n_off_board += 1;
            return false;
        };
        m.grid_coord = Some([board.u, board.v]);
        m.board_xy_mm = Some([f64::from(xy[0]), f64::from(xy[1])]);
        if let Some(cell_id) = target.id_of(board) {
            match m.id {
                None => stats.n_id_from_cell += 1,
                Some(id) if id != cell_id => stats.n_id_contradicted += 1,
                Some(_) => {}
            }
            m.id = Some(cell_id);
        }
        true
    });
    stats
}

/// Resolve the board origin and, on success, identify every labeled marker
/// with its absolute board cell and replace the frame homography with the
/// anchored board homography. Returns whether the origin was resolved.
fn phase_anchor(
    gray: &GrayImage,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    h_current: &mut nalgebra::Matrix3<f64>,
) -> bool {
    let Some(res) = crate::pipeline::anchor::resolve_origin(gray, markers, &config.target, mapper)
    else {
        return false;
    };
    let stats = identify_cells(markers, res.coord_map, &config.target);
    if config.target.is_coded() {
        tracing::info!(?stats, "grid-first cell identification");
    }
    *h_current = res.h;
    true
}

/// Coordinate-keyed completion plus projective-center correction for the newly
/// completed markers only.
fn phase_grid_completion(
    gray: &GrayImage,
    h_current: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    anchored: bool,
    mapper: Option<&dyn PixelMapper>,
) -> CompletionStats {
    let scope = if config.target.is_coded() {
        CompletionScope::Coded
    } else {
        CompletionScope::plain(anchored)
    };
    let n_before = markers.len();
    let stats = complete_with_h(gray, h_current, markers, config, scope, mapper);
    if markers.len() > n_before {
        apply_projective_centers(&mut markers[n_before..], config);
    }
    stats
}

/// Public board-frame contract: millimeter positions only in the absolute
/// board frame — a wrong millimeter position is worse than none. Returns the
/// frame of the labeled outputs.
fn enforce_frame_contract(
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

/// Result when no grid can be labeled (or, on coded targets, anchored): coded
/// targets fall back to the decode-anchored global filter, plain markers pass
/// through unlabeled.
fn unlabeled_fallback(
    gray: &GrayImage,
    markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let finalize = if config.target.is_coded() {
        finalize_global_filter_result
    } else {
        finalize_no_global_filter_result
    };
    finalize(gray, markers, config, mapper, homography_frame, image_size)
}

/// Grid-first finalize: grid assignment → origin anchor (+ cell
/// identification) → completion → marker fixup → final H refit → geometric
/// verify.
///
/// `assign_plain_grid` labels every fitted ring and keeps the frame-homography
/// inliers; `resolve_origin` decides whether outputs are absolute board-frame
/// or stay in the canonical relative frame (`board_xy_mm` cleared). Coded
/// targets require the absolute frame: without it they fall back to the
/// decode-anchored global filter on the unlabeled input.
pub(super) fn finalize_grid_result(
    gray: &GrayImage,
    corrected_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    homography_frame: DetectionFrame,
    image_size: [u32; 2],
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = corrected_markers.len();
    // Labeling drops unlabeled rings in place; keep the input for the coded
    // fallback when the origin then fails to anchor.
    let fallback_input = config.target.is_coded().then(|| corrected_markers.clone());
    let mut final_markers = corrected_markers;

    let assign_start = Instant::now();
    let Some(assignment) = crate::pipeline::assign::assign_plain_grid(
        &mut final_markers,
        &config.target,
        &config.advanced.ransac_homography,
    ) else {
        return unlabeled_fallback(
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

    let anchor_start = Instant::now();
    let anchored = phase_anchor(gray, &mut final_markers, config, mapper, &mut h_current);
    let anchor_elapsed = anchor_start.elapsed();
    if let (false, Some(input)) = (anchored, fallback_input) {
        tracing::info!("grid-first: origin unresolved; decode-anchored fallback");
        return finalize_global_filter_result(
            gray,
            input,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    }

    let completion_start = Instant::now();
    let completion_stats = phase_grid_completion(
        gray,
        &h_current,
        &mut final_markers,
        config,
        anchored,
        mapper,
    );
    let completion_elapsed = completion_start.elapsed();
    // Coded only (ID-keyed recovery + axis-ratio filter), before the final H.
    apply_marker_fixup(gray, &mut final_markers, config, mapper);
    drop_unmappable_markers_with_warning(&mut final_markers, mapper);

    let (final_h_mat, final_ransac) = phase_final_h(
        &final_markers,
        Some(h_current),
        Some(assignment.ransac),
        config,
    );
    // Coded only: re-derive cell/board positions from (possibly recovered) IDs.
    sync_marker_board_correspondence_with_logging(&mut final_markers, &config.target);

    // Precision-first geometric verification in the working frame, over all
    // labeled markers including completed ones.
    let geom_stats = phase_geometric_verify(&mut final_markers, final_h_mat.as_ref(), config);
    if let Some(mapper) = mapper {
        map_centers_to_image(&mut final_markers, mapper);
    }
    annotate_radius_ratios(&mut final_markers, config);
    let board_frame = Some(enforce_frame_contract(&mut final_markers, anchored));

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
        markers_after_assign,
        n_completed = completion_stats.n_added,
        n_geom_removed = geom_stats.n_removed_total,
        markers_out = result.markers.len(),
        anchored,
        assign_ms = duration_ms(assign_elapsed),
        anchor_ms = duration_ms(anchor_elapsed),
        completion_ms = duration_ms(completion_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(grid) timing summary"
    );

    result
}
