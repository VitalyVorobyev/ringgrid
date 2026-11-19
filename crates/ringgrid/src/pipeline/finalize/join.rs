//! Joining the adaptive candidates' anchored results.
//!
//! Adaptive detection runs several scale-tier candidates over one image and
//! keeps the best-scoring result. Each candidate's tiers fit a different
//! subset of the rings, so on a large oblique plate a cell the kept result
//! lacks is often one another candidate labeled — in the same board frame.
//! [`join_anchored_candidates`] adds those markers, then refits the final
//! homography and re-runs geometric verification on the union: the same
//! final-H and verification steps finalize ran, on more correspondences.

use std::collections::HashMap;

use projective_grid::Coord;

use super::common::{phase_final_h, phase_geometric_verify};
use super::meets_output_contract;
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
/// candidates. A cell present in `best` keeps `best`'s marker unless the
/// output contract would drop it ([`meets_output_contract`]: no corrected
/// centre, or below `min_confidence`) and another candidate's marker there
/// meets it — the cell is then that marker's; likewise between the other
/// candidates, in order. The joined markers are not yet photometrically
/// refined or held to the output contract — both run after, once, on the
/// joined result.
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
    // Cell -> (index in `best.markers`, whether that marker meets the
    // output contract).
    let mut cells: HashMap<Coord, (usize, bool)> = HashMap::new();
    for (i, m) in best.markers.iter().enumerate() {
        if let Some(c) = marker_cell_coord(m, target) {
            cells
                .entry(c)
                .or_insert((i, meets_output_contract(m, config)));
        }
    }
    let mut n_joined = 0usize;
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
            let Some(c) = marker_cell_coord(&m, target) else {
                continue;
            };
            let reportable = meets_output_contract(&m, config);
            match cells.get_mut(&c) {
                None => {
                    cells.insert(c, (best.markers.len(), reportable));
                    best.markers.push(m);
                }
                Some((i, held_reportable)) if reportable && !*held_reportable => {
                    best.markers[*i] = m;
                    *held_reportable = true;
                }
                Some(_) => continue,
            }
            n_joined += 1;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{CenterSource, MarkerRecord};
    use crate::target::TargetLayout;

    /// Board mm → image px of the synthetic view: 2 px/mm, offset (40, 30).
    const H: [[f64; 3]; 3] = [[2.0, 0.0, 40.0], [0.0, 2.0, 30.0], [0.0, 0.0, 1.0]];

    fn config() -> DetectConfig {
        DetectConfig::from_target(TargetLayout::default_hex())
    }

    /// A labeled marker of cell `id`, exactly at its projected position.
    fn marker(config: &DetectConfig, id: usize, centre: CenterSource) -> MarkerRecord {
        let target = &config.target;
        let xy = target.xy_mm_of_id(id).expect("id on board");
        let (x, y) = (f64::from(xy[0]), f64::from(xy[1]));
        let coord = target.coord_of_id(id).expect("id on board");
        MarkerRecord {
            id: Some(id),
            grid_coord: Some([coord.u, coord.v]),
            board_xy_mm: Some([x, y]),
            center: [2.0 * x + 40.0, 2.0 * y + 30.0],
            center_source: centre,
            confidence: 0.5,
            ..MarkerRecord::default()
        }
    }

    fn result(markers: Vec<MarkerRecord>) -> PipelineResult {
        PipelineResult {
            markers,
            center_frame: DetectionFrame::Image,
            homography_frame: DetectionFrame::Image,
            homography: Some(H),
            board_frame: Some(BoardFrame::Absolute),
            ..PipelineResult::default()
        }
    }

    fn at_cell(result: &PipelineResult, id: usize) -> Vec<&MarkerRecord> {
        result.markers.iter().filter(|m| m.id == Some(id)).collect()
    }

    fn kept_candidate(config: &DetectConfig) -> PipelineResult {
        result(
            (0..16)
                .map(|id| marker(config, id, CenterSource::ProjectiveCenter))
                .collect(),
        )
    }

    #[test]
    fn reportable_marker_takes_a_cell_held_by_an_unreportable_one() {
        let config = config();
        let mut best = kept_candidate(&config);
        // Cell 7 of the kept candidate: no corrected centre.
        best.markers[7].center_source = CenterSource::OuterEllipse;
        let other = result(vec![marker(&config, 7, CenterSource::ProjectiveCenter)]);

        join_anchored_candidates(&mut best, vec![other], &config);

        let at_7 = at_cell(&best, 7);
        assert_eq!(at_7.len(), 1);
        assert_eq!(at_7[0].center_source, CenterSource::ProjectiveCenter);
        assert_eq!(best.markers.len(), 16);
    }

    #[test]
    fn reportable_marker_keeps_its_cell() {
        let config = config();
        let mut best = kept_candidate(&config);
        let mut rival = marker(&config, 7, CenterSource::ProjectiveCenter);
        rival.confidence = 0.9;
        rival.center[0] += 0.25;

        join_anchored_candidates(&mut best, vec![result(vec![rival])], &config);

        let at_7 = at_cell(&best, 7);
        assert_eq!(at_7.len(), 1);
        assert_eq!(at_7[0].confidence, 0.5, "the kept candidate's marker stays");
    }

    #[test]
    fn unreportable_marker_does_not_displace_a_reportable_one() {
        let config = config();
        let mut best = kept_candidate(&config);
        let mut weak = marker(&config, 7, CenterSource::ProjectiveCenter);
        weak.confidence = 0.5 * config.min_confidence;
        let fresh = marker(&config, 40, CenterSource::ProjectiveCenter);

        join_anchored_candidates(&mut best, vec![result(vec![weak, fresh])], &config);

        let at_7 = at_cell(&best, 7);
        assert_eq!(at_7.len(), 1);
        assert_eq!(at_7[0].confidence, 0.5);
        // A cell the kept candidate lacks still joins.
        assert_eq!(at_cell(&best, 40).len(), 1);
    }
}
