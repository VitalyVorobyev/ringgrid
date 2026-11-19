//! Joining the adaptive candidates' anchored results.
//!
//! Adaptive detection runs several scale-tier candidates over one image and
//! keeps the best-scoring result. Each candidate's tiers fit a different
//! subset of the rings, so on a large oblique plate a cell the kept result
//! lacks is often one another candidate labeled — in the same board frame.
//! [`join_anchored_candidates`] adds those markers, then refits the final
//! homography and re-runs geometric verification on the union: the same
//! final-H and verification steps finalize ran, on more correspondences.

use std::collections::HashSet;

use projective_grid::Coord;

use super::common::{phase_final_h, phase_geometric_verify};
use crate::homography::homography_project;
use crate::pipeline::{
    BoardFrame, DetectConfig, DetectionFrame, PipelineResult, marker_cell_coord, matrix3_to_array,
};

/// Largest median distance (px) between another candidate's labeled centres
/// and the kept homography's projection of their cells for that candidate to
/// count as the same board placement. Two fits of one placement agree to a
/// fraction of a pixel; a placement off by one cell, or mirrored, puts the
/// median a whole pitch away. The same 2 px as the verification floors.
const SAME_PLACEMENT_TOL_PX: f64 = 2.0;

/// Add to `best` the markers of `others` at cells `best` lacks, from every
/// candidate whose labeling is the same board placement as `best`'s; then
/// refit the final homography over the union and verify it.
///
/// Only absolute board frames in the image frame join (adaptive detection
/// has no mapper): a relative frame's cells are not comparable across
/// candidates. A cell present in `best` keeps `best`'s marker. The joined
/// markers are not yet photometrically refined or held to the output
/// contract — both run after, once, on the joined result.
pub(in crate::pipeline) fn join_anchored_candidates(
    best: &mut PipelineResult,
    others: Vec<PipelineResult>,
    config: &DetectConfig,
) {
    let target = &config.target;
    let joinable = |r: &PipelineResult| {
        r.board_frame == Some(BoardFrame::Absolute)
            && r.homography_frame == DetectionFrame::Image
            && r.homography.is_some()
    };
    if !joinable(best) {
        return;
    }
    let Some(h) = best
        .homography
        .map(|h| nalgebra::Matrix3::from_fn(|r, c| h[r][c]))
    else {
        return;
    };
    let mut cells: HashSet<Coord> = best
        .markers
        .iter()
        .filter_map(|m| marker_cell_coord(m, target))
        .collect();
    let n_before = best.markers.len();
    for other in others.into_iter().filter(joinable) {
        let labeled: Vec<_> = other
            .markers
            .into_iter()
            .filter(|m| m.board_xy_mm.is_some() && marker_cell_coord(m, target).is_some())
            .collect();
        let offsets: Vec<f64> = labeled
            .iter()
            .filter_map(|m| {
                let [x, y] = m.board_xy_mm?;
                let p = homography_project(&h, x, y);
                Some((m.center[0] - p[0]).hypot(m.center[1] - p[1]))
            })
            .collect();
        let same_placement =
            crate::pipeline::stats::median_f64(offsets).is_some_and(|d| d <= SAME_PLACEMENT_TOL_PX);
        if !same_placement {
            continue;
        }
        for m in labeled {
            if marker_cell_coord(&m, target).is_some_and(|c| cells.insert(c)) {
                best.markers.push(m);
            }
        }
    }
    let n_joined = best.markers.len() - n_before;
    if n_joined == 0 {
        return;
    }
    let (h_final, ransac) = phase_final_h(&best.markers, Some(h), best.ransac.take(), config);
    phase_geometric_verify(&mut best.markers, h_final.as_ref(), config);
    best.homography = h_final.as_ref().map(matrix3_to_array);
    best.ransac = ransac;
    tracing::debug!(
        n_joined,
        n_markers = best.markers.len(),
        "joined other adaptive candidates' anchored markers"
    );
}
