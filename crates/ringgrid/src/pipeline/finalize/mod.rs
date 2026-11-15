//! Finalize stage: projective centers → grid labeling → origin anchor → cell
//! identification → completion → final H refit → geometric verify, with a
//! decode-anchored fallback (ID correction + global filter) for views where no
//! grid can be labeled or anchored.

mod common;
mod fallback;
mod grid;

use super::time_compat::{Instant, duration_ms};

use image::GrayImage;

use super::axis_ratio_filter::remove_axis_ratio_outliers;
use super::geometric_verify::annotate_h_reproj_err_px;
use super::{
    DetectConfig, annotate_neighbor_radius_ratios, apply_projective_centers,
    try_recover_inner_as_outer, warn_center_correction_without_intrinsics,
};
use crate::detector::MarkerRecord;
use crate::pipeline::{BoardFrame, DetectionFrame, PipelineResult, frame_xy_mm, matrix3_to_array};
use crate::pixelmap::PixelMapper;
use crate::target::TargetLayout;
use projective_grid::Coord;

use common::sync_marker_board_correspondence;
use fallback::finalize_unlabeled_result;
use grid::finalize_grid_result;

/// Inner-as-outer recovery (when enabled) followed by the axis-ratio
/// consistency filter, on working-frame markers (centers not yet mapped to
/// the image). Both coded paths (grid-first and the decode-anchored fallback)
/// run it before the final H refit and geometric verify, so recovered markers
/// are verified and feed the final homography.
fn apply_marker_fixup(
    gray: &GrayImage,
    markers: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) {
    // Both stages key on IDs (recovery re-anchors through one, the filter
    // gates only id'd markers); plain markers carry none.
    if !config.target.is_coded() {
        return;
    }
    let recovery = &config.advanced.inner_as_outer_recovery;
    if recovery.enable {
        annotate_radius_ratios(markers, config);
        try_recover_inner_as_outer(gray, markers, config, mapper);
        sync_marker_board_correspondence(markers, &config.target);
    }
    let removed = remove_axis_ratio_outliers(markers);
    if removed > 0 {
        tracing::info!(removed, "axis-ratio consistency filter removed markers");
    }
}

/// Annotate every marker's `neighbor_radius_ratio`: the recovery trigger, and
/// a diagnostic every finalize path refreshes last on its output markers.
fn annotate_radius_ratios(markers: &mut [MarkerRecord], config: &DetectConfig) {
    let recovery = &config.advanced.inner_as_outer_recovery;
    annotate_neighbor_radius_ratios(markers, recovery.k_neighbors, recovery.ratio_threshold);
}

/// Apply projective-center correction to `fit_markers`.
///
/// Returns the corrected marker list **without** labeling, completion, or
/// final H refit. These post-merge stages are deferred to
/// [`finalize_postmerge`] after all tier outputs have been merged by
/// `merge_multiscale_markers`.
///
/// Called once per scale tier by `detect_multiscale`.
pub(super) fn finalize_premerge(
    fit_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
) -> Vec<MarkerRecord> {
    let mut corrected_markers = fit_markers;
    let projective_center_start = Instant::now();
    apply_projective_centers(&mut corrected_markers, config);
    tracing::debug!(
        markers = corrected_markers.len(),
        projective_center_ms = duration_ms(projective_center_start.elapsed()),
        "finalize premerge timing summary"
    );
    corrected_markers
}

/// Grid-first labeling, anchoring, completion, final H refit and geometric
/// verify on the merged marker pool (see `grid.rs`; fallbacks in
/// `fallback.rs`).
///
/// Called once after all tier outputs have been merged by
/// `merge_multiscale_markers`.
pub(super) fn finalize_postmerge(
    gray: &GrayImage,
    merged_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> PipelineResult {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    let homography_frame = if mapper.is_some() {
        DetectionFrame::Working
    } else {
        DetectionFrame::Image
    };

    if !config.advanced.use_global_filter {
        return finalize_unlabeled_result(
            gray,
            merged_markers,
            config,
            mapper,
            homography_frame,
            image_size,
        );
    }

    finalize_grid_result(
        gray,
        merged_markers,
        config,
        mapper,
        homography_frame,
        image_size,
    )
}

/// Refit a finished result's homography and RANSAC statistics from the
/// centres it now carries — for a stage that moves centres after finalize
/// (photometric refinement) — and refresh each marker's `h_reproj_err_px`
/// against the kept homography.
///
/// The refit is finalize's own final-H step ([`common::phase_final_h`]):
/// refit over every labeled marker and keep whichever of the refit and the
/// current homography reprojects the centres better. It runs where finalize
/// ran it: in the homography's frame (working-frame `center_mapped` when that
/// is `Working`) and source plane (a plain target's relative frame, whose
/// assignment-frame positions the public contract cleared from
/// `board_xy_mm`). A result without a homography is left as it is.
pub(super) fn refit_final_homography(result: &mut PipelineResult, config: &DetectConfig) {
    let Some(h) = result.homography else {
        return;
    };
    let mut view = homography_frame_view(result, &config.target);
    let h_current = nalgebra::Matrix3::from_fn(|r, c| h[r][c]);
    let (h_final, ransac) =
        common::phase_final_h(&view, Some(h_current), result.ransac.take(), config);
    annotate_h_reproj_err_px(&mut view, h_final.as_ref());
    for (marker, v) in result.markers.iter_mut().zip(&view) {
        marker.fit.h_reproj_err_px = v.fit.h_reproj_err_px;
    }
    result.homography = h_final.as_ref().map(matrix3_to_array);
    result.ransac = ransac;
}

/// Output contract: when the configuration asks for a centre correction
/// (projective or photometric), every reported marker carries a corrected
/// centre. A marker whose correction failed (no inner ellipse, or every
/// correction rejected by its gates) keeps the outer-ellipse centre, which
/// perspective and blur bias (precision harness, operating range: median
/// error 0.25–1.0 px against 0.12–0.21 px corrected, about half the centres
/// beyond the loose tolerance); it is not reported. Such rings still served
/// as lattice evidence (labeling, anchor votes, homography support) — their
/// position is good to a pixel, their centre is not good to calibration
/// accuracy. Runs on the finished result, after photometric refinement can
/// still correct them; the homography is refit from the markers kept.
pub(super) fn drop_uncorrected_centres(result: &mut PipelineResult, config: &DetectConfig) {
    if !config.circle_refinement.uses_projective_center() {
        return;
    }
    let before = result.markers.len();
    result
        .markers
        .retain(|m| m.center_source != crate::detector::CenterSource::OuterEllipse);
    if result.markers.len() == before {
        return;
    }
    tracing::info!(
        n_dropped = before - result.markers.len(),
        "markers without a corrected centre not reported"
    );
    if result
        .markers
        .iter()
        .any(|m| m.grid_coord.is_some() || m.id.is_some())
    {
        refit_final_homography(result, config);
    } else {
        result.homography = None;
        result.ransac = None;
        result.board_frame = None;
    }
}

/// Correspondence-only copies of a result's markers, index-aligned, as
/// finalize saw them when it fitted the homography: centre in the
/// homography's frame, position in its source plane. A marker without a
/// centre in that frame carries no label and so takes no part.
fn homography_frame_view(result: &PipelineResult, target: &TargetLayout) -> Vec<MarkerRecord> {
    let working = result.homography_frame == DetectionFrame::Working;
    let relative = result.board_frame == Some(BoardFrame::RelativeCanonical);
    result
        .markers
        .iter()
        .map(|m| {
            let center = if working {
                m.center_mapped
            } else {
                Some(m.center)
            };
            let board_xy_mm = match (relative, m.grid_coord) {
                (true, Some(c)) => Some(frame_xy_mm(target, Coord::new(c[0], c[1]))),
                _ => m.board_xy_mm,
            };
            MarkerRecord {
                id: center.and(m.id),
                grid_coord: center.and(m.grid_coord),
                center: center.unwrap_or(m.center),
                board_xy_mm,
                confidence: m.confidence,
                ..MarkerRecord::default()
            }
        })
        .collect()
}

/// Single-scale finalize: [`finalize_premerge`] followed directly by
/// [`finalize_postmerge`] (there is no tier merge in between).
pub(super) fn run(
    gray: &GrayImage,
    fit_markers: Vec<MarkerRecord>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> PipelineResult {
    let total_start = Instant::now();
    let markers_in = fit_markers.len();

    warn_center_correction_without_intrinsics(config, mapper.is_some());

    let premerge_start = Instant::now();
    let corrected_markers = finalize_premerge(fit_markers, config);
    let premerge_elapsed = premerge_start.elapsed();

    let tail_start = Instant::now();
    let result = finalize_postmerge(gray, corrected_markers, config, mapper);
    tracing::info!(
        markers_in,
        markers_out = result.markers.len(),
        premerge_ms = duration_ms(premerge_elapsed),
        tail_ms = duration_ms(tail_start.elapsed()),
        total_ms = duration_ms(total_start.elapsed()),
        "finalize stage timing summary"
    );
    result
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

    fn marker(center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            confidence: 1.0,
            center,
            ..MarkerRecord::default()
        }
    }

    fn marker_with_id(id: usize, center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            id: Some(id),
            confidence: 1.0,
            center,
            ..MarkerRecord::default()
        }
    }

    fn no_global_filter_config() -> DetectConfig {
        let mut config = DetectConfig {
            circle_refinement: crate::CircleRefinementMethod::None,
            ..DetectConfig::default()
        };
        config.advanced.use_global_filter = false;
        config
    }

    #[test]
    fn no_mapper_keeps_image_space_centers() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([12.0, 22.0])];
        let (result, _) = run(&img, markers, &no_global_filter_config(), None).split();

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
        let expected = config.target.xy_mm_of_id(0).expect("board marker 0");
        let markers = vec![marker_with_id(0, [12.0, 22.0])];
        let (result, _) = run(&img, markers, &config, None).split();

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
        let expected = config.target.xy_mm_of_id(1).expect("board marker 1");
        let markers = vec![marker_with_id(1, [20.0, 30.0])];
        let mapper = OffsetMapper;
        let (result, _) = run(&img, markers, &config, Some(&mapper)).split();

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
        let (result, _) = run(&img, markers, &no_global_filter_config(), Some(&mapper)).split();

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [10.0, 10.0]);
        assert_eq!(result.detected_markers[0].center_mapped, Some([10.0, 10.0]));
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn invalid_id_is_cleared_and_board_xy_is_none() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let invalid_id = config.target.max_marker_id() + 1;
        let markers = vec![marker_with_id(invalid_id, [12.0, 22.0])];
        let (result, _) = run(&img, markers, &config, None).split();

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].id, None);
        assert!(result.detected_markers[0].board_xy_mm.is_none());
    }

    #[test]
    fn serialization_omits_none_board_xy_and_includes_finite_when_present() {
        let img = GrayImage::new(100, 80);
        let config = no_global_filter_config();
        let markers = vec![marker([12.0, 22.0]), marker_with_id(0, [20.0, 30.0])];
        let (result, _) = run(&img, markers, &config, None).split();
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
