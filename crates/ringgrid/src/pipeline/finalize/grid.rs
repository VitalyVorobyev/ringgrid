//! Grid-first finalize path, for every target: grid labeling, origin
//! anchoring, cell identification, coordinate-keyed completion, and the
//! board-frame output contract.
//!
//! Every fitted ring — decoded or not — is labeled by the lattice; decoded IDs
//! serve only as origin evidence (dots on plain targets), and each anchored
//! marker takes its ID from its cell. Views without a labeled (coded: and
//! anchored) grid go to [`super::fallback`].

use crate::pipeline::time_compat::{Instant, duration_ms};

use std::collections::HashSet;

use image::GrayImage;
use projective_grid::Coord;

use super::common::{
    drop_unmappable_markers_with_warning, map_centers_to_image, phase_final_h,
    phase_geometric_verify, sync_marker_board_correspondence_with_logging,
};
use super::fallback::{finalize_decode_anchored_fallback, finalize_unlabeled_result};
use super::{annotate_radius_ratios, apply_marker_fixup, meets_output_contract};
use crate::detector::completion::decode_contradicts_cell;
use crate::detector::{DetectionSource, MarkerRecord};
use crate::homography::fit_homography_ransac;
use crate::pipeline::anchor::CoordMap;
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
    /// Decoded markers whose decode names another board cell than their
    /// anchored one; they leave the labeling (see [`identify_cells`]).
    n_id_contradicted: usize,
}

/// Remap every labeled marker to its absolute board cell and position, and —
/// on coded targets — give it the cell's ID. Markers labeled outside the board
/// are removed: the labeling only reaches them through a lattice error.
///
/// A marker whose own decode names a *different board cell* is not given the
/// cell's ID: either the decode is a misread or the label is wrong (a lattice
/// slip under distortion, a ring of a second board labeled into this one), and
/// nothing here tells which. It is returned unlabeled, for the join (its own
/// decode, if a homography agrees) or completion (a refit at its cell) to
/// place it. A decode naming no board cell is a misread and is overridden.
fn identify_cells(
    markers: &mut Vec<MarkerRecord>,
    coord_map: CoordMap,
    target: &TargetLayout,
) -> (IdentifyStats, Vec<MarkerRecord>) {
    let mut stats = IdentifyStats::default();
    let mut contradicted = Vec::new();
    for mut m in std::mem::take(markers) {
        let Some(c) = m.grid_coord else {
            markers.push(m);
            continue;
        };
        let board = coord_map.apply(Coord::new(c[0], c[1]));
        let Some(xy) = target.cell_xy_mm(board) else {
            stats.n_off_board += 1;
            continue;
        };
        let cell_id = target.id_of(board);
        if cell_id.is_some() && decode_contradicts_cell(&m, cell_id, target) {
            stats.n_id_contradicted += 1;
            m.grid_coord = None;
            m.board_xy_mm = None;
            contradicted.push(m);
            continue;
        }
        if m.id.is_none() && cell_id.is_some() {
            stats.n_id_from_cell += 1;
        }
        m.grid_coord = Some([board.u, board.v]);
        m.board_xy_mm = Some([f64::from(xy[0]), f64::from(xy[1])]);
        if cell_id.is_some() {
            m.id = cell_id;
        }
        markers.push(m);
    }
    (stats, contradicted)
}

/// Resolve the board origin and, on success, identify every labeled marker
/// with its absolute board cell and replace the frame homography with the
/// anchored board homography. Returns the markers taken out of the labeling
/// (see [`identify_cells`]), or `None` when the origin is unresolved.
fn phase_anchor(
    gray: &GrayImage,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    h_current: &mut nalgebra::Matrix3<f64>,
) -> Option<Vec<MarkerRecord>> {
    let res = crate::pipeline::anchor::resolve_origin(gray, markers, &config.target, mapper)?;
    let (stats, contradicted) = identify_cells(markers, res.coord_map, &config.target);
    if config.target.is_coded() {
        tracing::info!(?stats, "grid-first cell identification");
    }
    *h_current = res.h;
    Some(contradicted)
}

/// Let decoded rings the labeler dropped join the anchored grid on the same
/// terms the decode-anchored global filter accepts a decode: a RANSAC
/// homography over every identified marker plus every dropped ring whose
/// decoded cell is still free, at the configured inlier threshold. Dropped
/// rings that are inliers join with their decoded cell, and the RANSAC
/// homography (fit over the whole view, not just the labeled patch, whose
/// homography extrapolates poorly) replaces `h` — unless most identified
/// markers are outliers to it. Identified markers stay regardless — geometric
/// verify judges them. Returns the number joined.
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
    // Accept the fit only when the candidates it joins are corroboration in
    // their own right — `min_inliers` of them, as any decode-anchored
    // homography needs — and most identified markers stay inliers. A few
    // misread candidates far from a small anchored patch otherwise bend the
    // fit (precision harness: 15 identified rings plus 2 joined misreads out of
    // 150 candidates on an Extended board bent the homography a lattice row,
    // and completion along it reported 365 wrong IDs).
    let n_identified = src.len() - candidates.len();
    let (identified_mask, candidate_mask) = fit.inlier_mask.split_at(n_identified);
    let n_identified_inliers = identified_mask.iter().filter(|&&inlier| inlier).count();
    let n_candidate_inliers = candidate_mask.iter().filter(|&&inlier| inlier).count();
    if n_candidate_inliers < config.advanced.ransac_homography.min_inliers
        || 2 * n_identified_inliers <= n_identified
    {
        tracing::info!(
            n_identified,
            n_identified_inliers,
            n_candidate_inliers,
            "grid-first: join homography not corroborated; nothing joined"
        );
        return 0;
    }
    *h = fit.h;
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

/// Plain targets: anchor an unresolved labeling once more after its relative
/// completion. The labeler returns one connected component, which can stop
/// short of the origin dots, and dots are verified only next to labeled rings
/// (`pipeline::anchor`); completion grows the patch over the visible board.
/// Returns whether the origin is now resolved (always `false` for coded
/// targets, which anchor by decoded IDs before completion or fall back).
fn reanchor_plain(
    gray: &GrayImage,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    h_current: &mut nalgebra::Matrix3<f64>,
) -> bool {
    !config.target.is_coded() && phase_anchor(gray, markers, config, mapper, h_current).is_some()
}

/// Coordinate-keyed completion plus projective-center correction for the newly
/// refitted markers only (adopted rings were corrected before labeling).
///
/// Coded targets grow the identified patch outward first, adopting the rings
/// the labeler dropped before refitting, then sweep the rest of the board once
/// (a growth stall — a lattice ring with nothing fittable — must not hide the
/// cells beyond it). Plain targets keep their pre-A/B behavior: one scope, no
/// pool.
///
/// A labeled ring the output contract would drop ([`meets_output_contract`]:
/// its projective centre correction failed, its confidence is below
/// `min_confidence`, or its outer ring is far smaller than its neighbours')
/// has its cell offered to completion's refit like a missing one, under the
/// same gates, and is put back unchanged when the refit does not take the
/// cell.
fn phase_grid_completion(
    gray: &GrayImage,
    h_current: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<MarkerRecord>,
    mut unlabeled: Vec<MarkerRecord>,
    config: &DetectConfig,
    anchored: bool,
    mapper: Option<&dyn PixelMapper>,
) -> usize {
    // Offered only once anchored: an unresolved plain labeling re-anchors on
    // the completed patch, which a reshuffled patch can cost (precision
    // harness, plain rect at 60°: anchored before, relative after).
    let unreportable = if anchored {
        take_unreportable_labeled(markers, config)
    } else {
        Vec::new()
    };
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
    let n_added = added.len();
    restore_unreportable(markers, unreportable, &config.target);
    n_added
}

/// Remove and return the labeled markers the output contract would drop (see
/// [`phase_grid_completion`]).
fn take_unreportable_labeled(
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
) -> Vec<MarkerRecord> {
    let (unreportable, kept): (Vec<_>, Vec<_>) = std::mem::take(markers)
        .into_iter()
        .partition(|m| m.grid_coord.is_some() && !meets_output_contract(m, config));
    *markers = kept;
    unreportable
}

/// Put back the unreportable markers whose cell completion left free.
fn restore_unreportable(
    markers: &mut Vec<MarkerRecord>,
    unreportable: Vec<MarkerRecord>,
    target: &TargetLayout,
) {
    let present: HashSet<Coord> = markers
        .iter()
        .filter_map(|m| marker_cell_coord(m, target))
        .collect();
    markers.extend(
        unreportable
            .into_iter()
            .filter(|m| marker_cell_coord(m, target).is_none_or(|c| !present.contains(&c))),
    );
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
        finalize_decode_anchored_fallback
    } else {
        finalize_unlabeled_result
    };
    finalize(gray, markers, config, mapper, homography_frame, image_size)
}

/// Grid-first finalize: grid assignment → origin anchor (+ cell
/// identification) → completion → (plain, unresolved: origin anchor on the
/// completed patch) → marker fixup → final H refit → geometric verify.
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
    let contradicted = phase_anchor(gray, &mut final_markers, config, mapper, &mut h_current);
    let anchored = contradicted.is_some();
    let anchor_elapsed = anchor_start.elapsed();
    if let (false, Some(input)) = (anchored, fallback_input) {
        tracing::info!("grid-first: origin unresolved; decode-anchored fallback");
        return finalize_decode_anchored_fallback(
            gray,
            input,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    }

    let mut unlabeled = assignment.unlabeled;
    unlabeled.extend(contradicted.into_iter().flatten());
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
    let retry_start = Instant::now();
    let anchored =
        anchored || reanchor_plain(gray, &mut final_markers, config, mapper, &mut h_current);
    let anchor_elapsed = anchor_elapsed + retry_start.elapsed();
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
        n_completed,
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
