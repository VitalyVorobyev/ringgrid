//! Full ring detection pipeline: proposal → edge sampling → fit → decode → global filter.

use image::GrayImage;

use crate::board_layout::BoardLayout;
use crate::camera::PixelMapper;
use crate::debug_dump as dbg;
use crate::homography::{self, RansacHomographyConfig};
use crate::{DetectedMarker, DetectionResult, RansacStats};

use self::post::dedup::dedup_with_debug as dedup_with_debug_impl;
use self::post::dedup::{dedup_by_id as dedup_by_id_impl, dedup_markers as dedup_markers_impl};
use self::post::global_filter::global_filter as global_filter_impl;
use self::post::global_filter::global_filter_with_debug as global_filter_with_debug_impl;
mod center_correction;
mod completion;
mod homography_utils;
mod inner_fit;
mod marker_build;
mod outer_fit;
mod post;
mod refine_h;
mod stages;
mod two_pass;
use center_correction::{apply_projective_centers, warn_center_correction_without_intrinsics};
use completion::CompletionDebugOptions;
pub(crate) use completion::{CompletionAttemptRecord, CompletionStats};
use outer_fit::{
    compute_center, fit_outer_ellipse_robust_with_reason, marker_outer_radius_expected_px,
    mean_axis_px_from_marker, median_outer_radius_from_neighbors_px, OuterFitCandidate,
};

mod config;
use config::config_mapper;
pub use config::{
    CircleRefinementMethod, CompletionParams, DebugCollectConfig, DetectConfig, MarkerScalePrior,
    ProjectiveCenterParams, SeedProposalParams, TwoPassParams,
};
pub use two_pass::{
    detect_rings, detect_rings_two_pass_with_mapper, detect_rings_with_mapper,
    detect_rings_with_self_undistort,
};

pub(super) fn find_proposals_with_seeds(
    gray: &GrayImage,
    proposal_cfg: &super::proposal::ProposalConfig,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> Vec<super::proposal::Proposal> {
    two_pass::find_proposals_with_seeds(gray, proposal_cfg, seed_centers_image, seed_cfg)
}

/// Run the full ring detection pipeline and collect a versioned debug dump.
///
/// Debug collection currently uses single-pass execution.
pub fn detect_rings_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
) -> (DetectionResult, dbg::DebugDump) {
    detect_rings_with_debug_and_mapper(gray, config, debug_cfg, config_mapper(config))
}

/// Run the full ring detection pipeline with debug collection and optional custom mapper.
///
/// Debug collection currently uses single-pass execution.
pub fn detect_rings_with_debug_and_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (DetectionResult, dbg::DebugDump) {
    let (result, dump) = stages::run(
        gray,
        config,
        mapper,
        &[],
        &SeedProposalParams::default(),
        Some(debug_cfg),
    );
    (
        result,
        dump.expect("debug dump present when debug_cfg is provided"),
    )
}

fn dedup_with_debug(
    markers: Vec<DetectedMarker>,
    cand_idx: Vec<usize>,
    radius: f64,
) -> (Vec<DetectedMarker>, Vec<usize>, dbg::DedupDebug) {
    dedup_with_debug_impl(markers, cand_idx, radius)
}

fn global_filter_with_debug(
    markers: &[DetectedMarker],
    cand_idx: &[usize],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
    dbg::RansacDebug,
) {
    global_filter_with_debug_impl(markers, cand_idx, config, board)
}

fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> (Vec<DetectedMarker>, dbg::RefineDebug) {
    refine_h::refine_with_homography_with_debug(gray, markers, h, config, board, mapper)
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    dedup_by_id_impl(markers);
}

/// Apply global homography RANSAC filter.
///
/// Returns (filtered markers, RANSAC result, stats).
fn global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    global_filter_impl(markers, config, board)
}

/// Refine marker centers using H: project board coords through H as priors,
/// re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    let (refined, _debug) =
        refine_h::refine_with_homography_with_debug(gray, markers, h, config, board, mapper);
    refined
}

/// Try to complete missing IDs using a fitted homography.
///
/// This is intentionally conservative: it only runs when H exists and rejects
/// any fit that deviates from the H-projected center by more than a tight gate.
fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<DetectedMarker>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
    debug: CompletionDebugOptions,
) -> (CompletionStats, Option<Vec<CompletionAttemptRecord>>) {
    completion::complete_with_h(gray, h, markers, config, board, mapper, debug)
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    homography_utils::matrix3_to_array(m)
}

fn mean_reproj_error_px(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    board: &BoardLayout,
) -> f64 {
    homography_utils::mean_reproj_error_px(h, markers, board)
}

fn compute_h_stats(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    thresh_px: f64,
    board: &BoardLayout,
) -> Option<RansacStats> {
    homography_utils::compute_h_stats(h, markers, thresh_px, board)
}

fn refit_homography_matrix(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> Option<(nalgebra::Matrix3<f64>, RansacStats)> {
    homography_utils::refit_homography_matrix(markers, config, board)
}

/// Remove duplicate detections: keep the highest-confidence marker within dedup_radius.
fn dedup_markers(markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    dedup_markers_impl(markers, radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PixelMapper;
    use crate::conic::ConicCoeffs;
    use crate::FitMetrics;
    use image::GrayImage;
    use image::Luma;
    use nalgebra::Matrix3;
    use nalgebra::Vector3;

    #[test]
    fn marker_scale_prior_rederives_scale_coupled_params() {
        let mut cfg = DetectConfig::default();
        cfg.set_marker_scale_prior(MarkerScalePrior::new(24.0, 40.0));

        assert_eq!(cfg.marker_scale.diameter_range_px(), [24.0, 40.0]);
        assert!((cfg.proposal.r_min - 4.8).abs() < 1e-6);
        assert!((cfg.proposal.r_max - 34.0).abs() < 1e-6);
        assert!((cfg.edge_sample.r_max - 40.0).abs() < 1e-6);
        assert_eq!(cfg.projective_center.max_center_shift_px, Some(32.0));
    }

    #[test]
    fn debug_dump_does_not_panic_when_stages_skipped() {
        let img = GrayImage::new(64, 64);
        let cfg = DetectConfig {
            use_global_filter: false,
            refine_with_h: false,
            ..DetectConfig::default()
        };

        let dbg_cfg = DebugCollectConfig {
            image_path: Some("dummy.png".to_string()),
            marker_diameter_px: 32.0,
            max_candidates: 10,
            store_points: false,
        };

        let (res, dump) = detect_rings_with_debug(&img, &cfg, &dbg_cfg);
        assert_eq!(res.image_size, [64, 64]);
        assert_eq!(dump.schema_version, crate::debug_dump::DEBUG_SCHEMA_V6);
        assert_eq!(dump.stages.stage0_proposals.n_total, 0);
        assert!(!dump.stages.stage3_ransac.enabled);
    }

    #[test]
    fn seeded_proposals_include_injected_centers() {
        let img = GrayImage::new(32, 32);
        let seeds = vec![[10.0f32, 12.0f32], [20.0f32, 22.0f32]];
        let props = find_proposals_with_seeds(
            &img,
            &super::super::proposal::ProposalConfig::default(),
            &seeds,
            &SeedProposalParams::default(),
        );
        assert_eq!(props.len(), seeds.len());
    }

    #[test]
    fn map_marker_image_to_working_maps_primary_and_projective_centers() {
        struct ShiftMapper;
        impl PixelMapper for ShiftMapper {
            fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some([image_xy[0] - 5.0, image_xy[1] + 2.0])
            }
            fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some([working_xy[0] + 5.0, working_xy[1] - 2.0])
            }
        }

        let marker = DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center: [1.0, 2.0],
            center_projective: Some([3.0, 4.0]),
            center_projective_residual: None,
            ellipse_outer: None,
            ellipse_inner: None,
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        };

        let mapped = two_pass::map_marker_image_to_working(&marker, &ShiftMapper)
            .expect("center mapping should succeed");
        assert_eq!(mapped.center, [-4.0, 4.0]);
        assert_eq!(mapped.center_projective, Some([-2.0, 6.0]));
        assert!(mapped.ellipse_outer.is_none());
        assert!(mapped.ellipse_inner.is_none());
        assert!(mapped.edge_points_outer.is_none());
        assert!(mapped.edge_points_inner.is_none());
    }

    #[test]
    fn detect_accepts_custom_pixel_mapper_adapter() {
        struct IdentityMapper;
        impl PixelMapper for IdentityMapper {
            fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some(image_xy)
            }
            fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some(working_xy)
            }
        }

        let img = GrayImage::new(64, 64);
        let cfg = DetectConfig {
            use_global_filter: false,
            refine_with_h: false,
            ..DetectConfig::default()
        };
        let dbg_cfg = DebugCollectConfig {
            image_path: None,
            marker_diameter_px: 32.0,
            max_candidates: 10,
            store_points: false,
        };

        let mapper = IdentityMapper;
        let (res, _dump) = detect_rings_with_debug_and_mapper(
            &img,
            &cfg,
            &dbg_cfg,
            Some(&mapper as &dyn PixelMapper),
        );
        assert_eq!(res.image_size, [64, 64]);
    }

    #[test]
    fn completion_adds_marker_at_h_projected_center() {
        let w = 128u32;
        let h = 128u32;

        let cfg = DetectConfig::default();

        // Choose an ID that exists on the default board and project it to the
        // image center with an affine homography.
        let id = 0usize;
        let xy = cfg.board.xy_mm(id).expect("board has id=0");
        let tx = 64.0 - xy[0] as f64;
        let ty = 64.0 - xy[1] as f64;
        let h_matrix = Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0);

        // Render a simple concentric ring at the projected center (no code band).
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 64.0;
                let dy = y as f32 - 64.0;
                let r = (dx * dx + dy * dy).sqrt();

                let bg = 0.85f32;
                let dark = 0.12f32;
                let v = if (12.0..=18.0).contains(&r) { dark } else { bg };
                img.put_pixel(x, y, Luma([(v * 255.0).round() as u8]));
            }
        }

        let mut cfg = DetectConfig {
            refine_with_h: false,
            // Make ellipse validation compatible with our synthetic ring radius.
            min_semi_axis: 6.0,
            max_semi_axis: 30.0,
            ..DetectConfig::default()
        };

        // Completion should attempt only this ID and should not be blocked by decoding.
        cfg.completion.enable = true;
        cfg.completion.max_attempts = Some(1);
        cfg.completion.roi_radius_px = 24.0;
        cfg.completion.reproj_gate_px = 3.0;
        cfg.completion.min_arc_coverage = 0.6;
        cfg.completion.min_fit_confidence = 0.6;
        cfg.decode.min_decode_confidence = 1.0; // force decode rejection (avoid mismatch gate)

        let mut markers: Vec<DetectedMarker> = Vec::new();
        let (stats, _attempts) = complete_with_h(
            &img,
            &h_matrix,
            &mut markers,
            &cfg,
            &cfg.board,
            None,
            CompletionDebugOptions::default(),
        );
        assert_eq!(stats.n_added, 1, "expected one completion addition");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].id, Some(id));
    }

    #[test]
    fn apply_projective_centers_promotes_center_field() {
        fn circle_conic(radius: f64) -> Matrix3<f64> {
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -(radius * radius))
        }

        fn project_conic(q_plane: &Matrix3<f64>, h: &Matrix3<f64>) -> Matrix3<f64> {
            let h_inv = h.try_inverse().expect("invertible homography");
            h_inv.transpose() * q_plane * h_inv
        }

        fn conic_matrix_to_params(q: Matrix3<f64>) -> crate::EllipseParams {
            let q_sym = 0.5 * (q + q.transpose());
            let coeffs = ConicCoeffs([
                q_sym[(0, 0)],
                2.0 * q_sym[(0, 1)],
                q_sym[(1, 1)],
                2.0 * q_sym[(0, 2)],
                2.0 * q_sym[(1, 2)],
                q_sym[(2, 2)],
            ]);
            let e = coeffs.to_ellipse().expect("projected circle is an ellipse");
            crate::EllipseParams::from(e)
        }

        let h = Matrix3::new(1.12, 0.21, 321.0, -0.17, 0.94, 245.0, 8.0e-4, -6.0e-4, 1.0);
        let q_inner = project_conic(&circle_conic(4.0), &h);
        let q_outer = project_conic(&circle_conic(7.0), &h);
        let inner = conic_matrix_to_params(q_inner);
        let outer = conic_matrix_to_params(q_outer);

        let center_before = outer.center_xy;
        let mut markers = vec![DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center: center_before,
            center_projective: None,
            center_projective_residual: None,
            ellipse_outer: Some(outer),
            ellipse_inner: Some(inner),
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        }];

        let cfg = DetectConfig::default();
        apply_projective_centers(&mut markers, &cfg);
        let m = &markers[0];

        let gt_h = h * Vector3::new(0.0, 0.0, 1.0);
        let gt_center = [gt_h[0] / gt_h[2], gt_h[1] / gt_h[2]];
        let err =
            ((m.center[0] - gt_center[0]).powi(2) + (m.center[1] - gt_center[1]).powi(2)).sqrt();
        let shift = ((m.center[0] - center_before[0]).powi(2)
            + (m.center[1] - center_before[1]).powi(2))
        .sqrt();

        assert!(
            m.center_projective.is_some(),
            "projective center should be present"
        );
        assert!(
            err < 1e-6,
            "expected near-exact projective center, err={}",
            err
        );
        assert!(
            shift > 1e-3,
            "primary center should be updated from ellipse center, shift={}",
            shift
        );
    }

    #[test]
    fn apply_projective_centers_falls_back_when_shift_gate_rejects() {
        fn circle_conic(radius: f64) -> Matrix3<f64> {
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -(radius * radius))
        }

        fn project_conic(q_plane: &Matrix3<f64>, h: &Matrix3<f64>) -> Matrix3<f64> {
            let h_inv = h.try_inverse().expect("invertible homography");
            h_inv.transpose() * q_plane * h_inv
        }

        fn conic_matrix_to_params(q: Matrix3<f64>) -> crate::EllipseParams {
            let q_sym = 0.5 * (q + q.transpose());
            let coeffs = ConicCoeffs([
                q_sym[(0, 0)],
                2.0 * q_sym[(0, 1)],
                q_sym[(1, 1)],
                2.0 * q_sym[(0, 2)],
                2.0 * q_sym[(1, 2)],
                q_sym[(2, 2)],
            ]);
            let e = coeffs.to_ellipse().expect("projected circle is an ellipse");
            crate::EllipseParams::from(e)
        }

        let h = Matrix3::new(1.12, 0.21, 321.0, -0.17, 0.94, 245.0, 8.0e-4, -6.0e-4, 1.0);
        let q_inner = project_conic(&circle_conic(4.0), &h);
        let q_outer = project_conic(&circle_conic(7.0), &h);
        let inner = conic_matrix_to_params(q_inner);
        let outer = conic_matrix_to_params(q_outer);

        let center_before = outer.center_xy;
        let mut markers = vec![DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center: center_before,
            center_projective: None,
            center_projective_residual: None,
            ellipse_outer: Some(outer),
            ellipse_inner: Some(inner),
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        }];

        let mut cfg = DetectConfig::default();
        cfg.projective_center.max_center_shift_px = Some(1e-6);
        apply_projective_centers(&mut markers, &cfg);
        let m = &markers[0];

        assert_eq!(m.center, center_before);
        assert!(m.center_projective.is_none());
        assert!(m.center_projective_residual.is_none());
    }
}
