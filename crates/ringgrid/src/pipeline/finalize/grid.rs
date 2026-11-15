//! Grid-first finalize path: grid labeling, origin anchoring, cell
//! identification, coordinate-keyed completion, and the board-frame output
//! contract.
//!
//! Plain targets always take this path. Coded targets take it under the
//! temporary grid-first A/B switch (`super::grid_first_enabled`): every fitted
//! ring — decoded or not — is labeled by the lattice, decoded IDs serve only as
//! origin evidence, and each anchored marker takes its ID from its cell.

use crate::pipeline::time_compat::{Instant, duration_ms};

use std::collections::HashSet;

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
use crate::detector::{DetectionSource, MarkerRecord};
use crate::homography::fit_homography_ransac;
use crate::pipeline::anchor::CoordMap;
use crate::pipeline::assign::{assign_plain_grid, grid_components};
use crate::pipeline::{
    CompletionScope, DetectConfig, DetectionFrame, PipelineResult, apply_projective_centers,
    complete_with_h, marker_cell_coord, matrix3_to_array,
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

/// Let decoded rings the labeler dropped join the anchored grid on the same
/// terms the decode-anchored global filter accepts a decode: a RANSAC
/// homography over every identified marker plus every dropped ring whose
/// decoded cell is still free, at the configured inlier threshold. Dropped
/// rings that are inliers join with their decoded cell, and the RANSAC
/// homography (fit over the whole view, not just the labeled patch, whose
/// homography extrapolates poorly) replaces `h`. Identified markers stay
/// regardless — geometric verify judges them. Returns the number joined.
fn join_decoded_unlabeled(
    markers: &mut Vec<MarkerRecord>,
    pool: &mut Vec<MarkerRecord>,
    h: &mut nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> usize {
    let target = &config.target;
    let mut present: HashSet<Coord> = markers
        .iter()
        .filter_map(|m| marker_cell_coord(m, target))
        .collect();
    // Pool candidates: decoded onto a free board cell (first claim wins).
    let mut candidates: Vec<(usize, Coord, [f64; 2])> = Vec::new();
    for (i, m) in pool.iter().enumerate() {
        let Some(c) = m.id.and_then(|id| target.coord_of_id(id)) else {
            continue;
        };
        if let Some(xy) = target.cell_xy_mm(c).filter(|_| present.insert(c)) {
            candidates.push((i, c, [f64::from(xy[0]), f64::from(xy[1])]));
        }
    }
    if candidates.is_empty() {
        return 0;
    }
    let identified = markers
        .iter()
        .filter_map(|m| Some((m.board_xy_mm?, m.center)));
    let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = identified
        .chain(candidates.iter().map(|&(i, _, xy)| (xy, pool[i].center)))
        .unzip();
    let Ok(fit) = fit_homography_ransac(&src, &dst, &config.advanced.ransac_homography) else {
        return 0;
    };
    *h = fit.h;
    let n_identified = src.len() - candidates.len();
    let mut joined: Vec<usize> = Vec::new();
    for (k, &(i, c, xy)) in candidates.iter().enumerate() {
        if fit
            .inlier_mask
            .get(n_identified + k)
            .copied()
            .unwrap_or(false)
        {
            pool[i].grid_coord = Some([c.u, c.v]);
            pool[i].board_xy_mm = Some(xy);
            joined.push(i);
        }
    }
    // Remove back to front so pool indices stay valid.
    joined.sort_unstable();
    for &i in joined.iter().rev() {
        markers.push(pool.swap_remove(i));
    }
    tracing::info!(
        n_candidates = candidates.len(),
        n_joined = joined.len(),
        "grid-first: decoded unlabeled rings joined"
    );
    joined.len()
}

/// Coordinate-keyed completion plus projective-center correction for the newly
/// refitted markers only (adopted rings were corrected before labeling).
///
/// Coded targets grow the identified patch outward first, adopting the rings
/// the labeler dropped before refitting, then sweep the rest of the board once
/// (a growth stall — a lattice ring with nothing fittable — must not hide the
/// cells beyond it). Plain targets keep their pre-A/B behavior: one scope, no
/// pool.
fn phase_grid_completion(
    gray: &GrayImage,
    h_current: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<MarkerRecord>,
    mut unlabeled: Vec<MarkerRecord>,
    config: &DetectConfig,
    anchored: bool,
    mapper: Option<&dyn PixelMapper>,
) -> usize {
    let n_before = markers.len();
    let mut complete = |scope, pool: &mut Vec<MarkerRecord>| {
        complete_with_h(gray, h_current, markers, pool, config, scope, mapper)
    };
    if config.target.is_coded() {
        complete(CompletionScope::CodedPatch, &mut unlabeled);
        complete(CompletionScope::Coded, &mut unlabeled);
    } else {
        complete(CompletionScope::plain(anchored), &mut Vec::new());
    }
    // Refits first (stable: a pool-less run keeps its order), then correct them.
    let added = &mut markers[n_before..];
    added.sort_by_key(|m| m.source != DetectionSource::Completion);
    let n_refit = added
        .iter()
        .take_while(|m| m.source == DetectionSource::Completion)
        .count();
    if n_refit > 0 {
        apply_projective_centers(&mut added[..n_refit], config);
    }
    added.len()
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

/// A labeled (and, when evidence allows, anchored) view.
struct Labeled {
    markers: Vec<MarkerRecord>,
    unlabeled: Vec<MarkerRecord>,
    h: nalgebra::Matrix3<f64>,
    ransac: crate::homography::RansacStats,
    anchored: bool,
}

/// Grid labeling plus origin anchoring.
///
/// Plain targets label the primary grid component, anchored by their dots or
/// left in the relative frame. Coded targets try every component the lattice
/// labeler returns, largest first, and keep the first one its own decoded IDs
/// anchor; the rings of every other component go to the unlabeled pool, where
/// the decoded-join and completion-adoption steps place them through the
/// anchored homography. `Err` hands back the untouched input when nothing can
/// be labeled (or, coded, anchored).
fn phase_label_anchor(
    gray: &GrayImage,
    mut markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> Result<Labeled, Vec<MarkerRecord>> {
    let (target, ransac) = (&config.target, &config.advanced.ransac_homography);
    if !target.is_coded() {
        let Some(assignment) = assign_plain_grid(&mut markers, target, ransac) else {
            return Err(markers);
        };
        let mut h = assignment.h;
        let anchored = phase_anchor(gray, &mut markers, config, mapper, &mut h);
        return Ok(Labeled {
            markers,
            unlabeled: assignment.unlabeled,
            h,
            ransac: assignment.ransac,
            anchored,
        });
    }
    let components = grid_components(&markers, target, ransac);
    let n_components = components.len();
    for (rank, component) in components.into_iter().enumerate() {
        let n_labeled = component.len();
        let mut trial = markers.clone();
        let assignment = component.apply(&mut trial, target);
        let mut h = assignment.h;
        if phase_anchor(gray, &mut trial, config, mapper, &mut h) {
            tracing::info!(
                rank,
                n_components,
                n_labeled,
                "grid-first: component anchored"
            );
            return Ok(Labeled {
                markers: trial,
                unlabeled: assignment.unlabeled,
                h,
                ransac: assignment.ransac,
                anchored: true,
            });
        }
    }
    tracing::info!(
        n_components,
        "grid-first: no component anchored; decode-anchored fallback"
    );
    Err(markers)
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
/// [`phase_label_anchor`] labels the fitted rings and decides whether outputs
/// are absolute board-frame or stay in the canonical relative frame
/// (`board_xy_mm` cleared). Coded targets require the absolute frame: without
/// it they fall back to the decode-anchored global filter on the input.
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

    let label_start = Instant::now();
    let Labeled {
        markers: mut final_markers,
        mut unlabeled,
        h: mut h_current,
        ransac,
        anchored,
    } = match phase_label_anchor(gray, corrected_markers, config, mapper) {
        Ok(labeled) => labeled,
        Err(input) => {
            return unlabeled_fallback(gray, input, config, mapper, homography_frame, image_size);
        }
    };
    let label_elapsed = label_start.elapsed();
    let markers_after_assign = final_markers.len();

    if config.target.is_coded() {
        join_decoded_unlabeled(&mut final_markers, &mut unlabeled, &mut h_current, config);
    }

    let completion_start = Instant::now();
    let n_completed = phase_grid_completion(
        gray,
        &h_current,
        &mut final_markers,
        unlabeled,
        config,
        anchored,
        mapper,
    );
    let completion_elapsed = completion_start.elapsed();
    // Coded only (ID-keyed recovery + axis-ratio filter), before the final H.
    apply_marker_fixup(gray, &mut final_markers, config, mapper);
    drop_unmappable_markers_with_warning(&mut final_markers, mapper);

    let (final_h_mat, final_ransac) =
        phase_final_h(&final_markers, Some(h_current), Some(ransac), config);
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
        n_completed,
        n_geom_removed = geom_stats.n_removed_total,
        markers_out = result.markers.len(),
        anchored,
        label_anchor_ms = duration_ms(label_elapsed),
        completion_ms = duration_ms(completion_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize(grid) timing summary"
    );

    result
}
