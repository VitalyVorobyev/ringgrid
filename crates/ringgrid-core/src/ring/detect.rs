//! Full ring detection pipeline: proposal → edge sampling → fit → decode → global filter.

use image::GrayImage;

use crate::conic::rms_sampson_distance;
use crate::debug_dump as dbg;
use crate::homography::{self, RansacHomographyConfig};
use crate::marker_spec::MarkerSpec;
use crate::refine;
use crate::{
    DecodeMetrics, DetectedMarker, DetectionResult, EllipseParams, FitMetrics, RansacStats,
};

use super::decode::DecodeConfig;
use super::edge_sample::EdgeSampleConfig;
use super::inner_estimate::{estimate_inner_scale_from_outer, InnerStatus};
use super::outer_estimate::OuterEstimationConfig;
use super::pipeline::dedup::{
    dedup_by_id as dedup_by_id_impl, dedup_markers as dedup_markers_impl,
    dedup_with_debug as dedup_with_debug_impl,
};
use super::pipeline::global_filter::{
    global_filter as global_filter_impl, global_filter_with_debug as global_filter_with_debug_impl,
};
use super::proposal::{find_proposals, ProposalConfig};
#[path = "detect/completion.rs"]
mod completion;
#[path = "detect/debug_pipeline.rs"]
mod debug_pipeline;
#[path = "detect/homography_utils.rs"]
mod homography_utils;
#[path = "detect/outer_fit.rs"]
mod outer_fit;
#[path = "detect/refine_h.rs"]
mod refine_h;
use completion::{CompletionAttemptRecord, CompletionAttemptStatus, CompletionStats};
use outer_fit::{
    compute_center, ellipse_to_params, fit_outer_ellipse_robust_with_reason,
    marker_outer_radius_expected_px, mean_axis_px_from_marker,
    median_outer_radius_from_neighbors_px, OuterFitCandidate,
};

/// Debug collection options for `detect_rings_with_debug`.
#[derive(Debug, Clone)]
pub struct DebugCollectConfig {
    pub image_path: Option<String>,
    pub marker_diameter_px: f64,
    pub max_candidates: usize,
    pub store_points: bool,
}

/// Configuration for homography-guided completion: attempt local fits for
/// missing IDs at H-projected board locations.
#[derive(Debug, Clone)]
pub struct CompletionParams {
    /// Enable completion (runs only when a valid homography is available).
    pub enable: bool,
    /// Radial sampling extent (pixels) used for edge sampling around the prior center.
    pub roi_radius_px: f32,
    /// Maximum allowed reprojection error (pixels) between the fitted center and
    /// the H-projected board center.
    pub reproj_gate_px: f32,
    /// Minimum fit confidence in [0, 1].
    pub min_fit_confidence: f32,
    /// Minimum arc coverage (fraction of rays with both edges found).
    pub min_arc_coverage: f32,
    /// Optional cap on how many completion fits to attempt (in ID order).
    pub max_attempts: Option<usize>,
    /// Skip attempts whose projected center is too close to the image boundary.
    pub image_margin_px: f32,
}

impl Default for CompletionParams {
    fn default() -> Self {
        Self {
            enable: true,
            roi_radius_px: 24.0,
            reproj_gate_px: 3.0,
            min_fit_confidence: 0.45,
            min_arc_coverage: 0.35,
            max_attempts: None,
            image_margin_px: 10.0,
        }
    }
}

/// Top-level detection configuration.
#[derive(Debug, Clone)]
pub struct DetectConfig {
    /// Expected marker outer diameter in pixels (for scale-anchored edge extraction).
    pub marker_diameter_px: f32,
    /// Outer edge estimation configuration (anchored on `marker_diameter_px`).
    pub outer_estimation: OuterEstimationConfig,
    pub proposal: ProposalConfig,
    pub edge_sample: EdgeSampleConfig,
    pub decode: DecodeConfig,
    pub marker_spec: MarkerSpec,
    pub completion: CompletionParams,
    /// Minimum semi-axis for a valid outer ellipse.
    pub min_semi_axis: f64,
    /// Maximum semi-axis for a valid outer ellipse.
    pub max_semi_axis: f64,
    /// Maximum aspect ratio (a/b) for a valid ellipse.
    pub max_aspect_ratio: f64,
    /// NMS dedup radius for final markers (pixels).
    pub dedup_radius: f64,
    /// Enable global homography filtering (requires board spec).
    pub use_global_filter: bool,
    /// RANSAC homography configuration.
    pub ransac_homography: RansacHomographyConfig,
    /// Enable one-iteration refinement using H.
    pub refine_with_h: bool,
    /// Non-linear per-marker refinement using board-plane circle fits.
    pub nl_refine: refine::RefineParams,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            marker_diameter_px: 32.0,
            outer_estimation: OuterEstimationConfig::default(),
            proposal: ProposalConfig::default(),
            edge_sample: EdgeSampleConfig::default(),
            decode: DecodeConfig::default(),
            marker_spec: MarkerSpec::default(),
            completion: CompletionParams::default(),
            min_semi_axis: 3.0,
            max_semi_axis: 15.0,
            max_aspect_ratio: 3.0,
            dedup_radius: 6.0,
            use_global_filter: true,
            ransac_homography: RansacHomographyConfig::default(),
            refine_with_h: true,
            nl_refine: refine::RefineParams::default(),
        }
    }
}

/// Run the full ring detection pipeline.
pub fn detect_rings(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let (w, h) = gray.dimensions();

    // Stage 1: Find candidate centers
    let proposals = find_proposals(gray, &config.proposal);
    tracing::info!("{} proposals found", proposals.len());

    // Stages 2-5: For each proposal, sample edges → fit → decode
    let mut markers: Vec<DetectedMarker> = Vec::new();

    for proposal in &proposals {
        // Stage 2-4: Robust outer edge extraction → outer fit → decode
        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            [proposal.x, proposal.y],
            marker_outer_radius_expected_px(config),
            config,
            &config.edge_sample,
            false,
        ) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            decode_result,
            ..
        } = fit;

        // Compute center
        let center = compute_center(&outer);

        // Compute fit metrics
        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: 0,
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: None,
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: None,
        };

        // Stage 4b: Inner edge estimation (only for decoded candidates)
        let inner_params = if decode_result.is_some() {
            let est = estimate_inner_scale_from_outer(gray, &outer, &config.marker_spec, false);
            if est.status == InnerStatus::Ok {
                let s = est
                    .r_inner_found
                    .unwrap_or(config.marker_spec.r_inner_expected) as f64;
                Some(EllipseParams {
                    center_xy: [outer.cx, outer.cy],
                    semi_axes: [outer.a * s, outer.b * s],
                    angle: outer.angle,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Build marker
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);

        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let marker = DetectedMarker {
            id: decode_result.as_ref().map(|d| d.id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params,
            fit,
            decode: decode_metrics,
        };

        markers.push(marker);
    }

    // Stage 5: Dedup by center proximity
    markers = dedup_markers(markers, config.dedup_radius);

    // Stage 5b: Dedup by ID — keep best confidence per decoded ID
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());

    // Stage 6: Global homography filtering
    if !config.use_global_filter {
        return DetectionResult {
            detected_markers: markers,
            image_size: [w, h],
            homography: None,
            ransac: None,
        };
    }

    let (filtered, h_result, ransac_stats) = global_filter(&markers, &config.ransac_homography);

    let h_matrix = h_result.as_ref().map(|r| &r.h);

    // Stage 7: Optional refinement using H
    let mut final_markers = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                refine_with_homography(gray, &filtered, h, config)
            } else {
                filtered
            }
        } else {
            filtered
        }
    } else {
        filtered
    };

    // Stage 8: Homography-guided completion (only when H exists)
    if config.completion.enable {
        if let Some(h) = h_matrix {
            let (_stats, _attempts) =
                complete_with_h(gray, h, &mut final_markers, config, false, false);
        }
    }

    // Stage 9: Non-linear refinement in board plane (optional).
    let mut h_current: Option<nalgebra::Matrix3<f64>> = h_result.as_ref().map(|r| r.h);
    if config.nl_refine.enabled {
        if let Some(h0) = h_current {
            let _ = refine::refine_markers_circle_board(
                gray,
                &h0,
                &mut final_markers,
                &config.nl_refine,
                false,
            );

            if config.nl_refine.enable_h_refit && final_markers.len() >= 10 {
                let max_iters = config.nl_refine.h_refit_iters.clamp(1, 3);
                let mut h_prev = h0;
                let mut mean_prev = mean_reproj_error_px(&h_prev, &final_markers);
                for _ in 0..max_iters {
                    let Some((h_next, _stats1)) =
                        refit_homography_matrix(&final_markers, &config.ransac_homography)
                    else {
                        break;
                    };

                    let mean_next = mean_reproj_error_px(&h_next, &final_markers);
                    if mean_next.is_finite() && (mean_next < mean_prev || !mean_prev.is_finite()) {
                        h_current = Some(h_next);
                        h_prev = h_next;
                        mean_prev = mean_next;

                        let _ = refine::refine_markers_circle_board(
                            gray,
                            &h_prev,
                            &mut final_markers,
                            &config.nl_refine,
                            false,
                        );
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Final H: refit after refinement if enabled (or keep the original RANSAC H).
    let final_h_matrix = if config.refine_with_h && final_markers.len() >= 10 {
        refit_homography_matrix(&final_markers, &config.ransac_homography)
            .map(|(h, _stats)| h)
            .or(h_current)
    } else {
        h_current
    };
    let final_h = final_h_matrix.as_ref().map(matrix3_to_array);
    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| compute_h_stats(h, &final_markers, config.ransac_homography.inlier_threshold))
        .or(ransac_stats);

    tracing::info!(
        "{} markers after global filter{}",
        final_markers.len(),
        if config.refine_with_h {
            " + refinement"
        } else {
            ""
        }
    );

    DetectionResult {
        detected_markers: final_markers,
        image_size: [w, h],
        homography: final_h,
        ransac: final_ransac,
    }
}

/// Run the full ring detection pipeline and collect a versioned debug dump.
pub fn detect_rings_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
) -> (DetectionResult, dbg::DebugDumpV1) {
    debug_pipeline::run(gray, config, debug_cfg)
}

fn dedup_with_debug(
    markers: Vec<DetectedMarker>,
    cand_idx: Vec<usize>,
    radius: f64,
) -> (Vec<DetectedMarker>, Vec<usize>, dbg::DedupDebugV1) {
    dedup_with_debug_impl(markers, cand_idx, radius)
}

fn global_filter_with_debug(
    markers: &[DetectedMarker],
    cand_idx: &[usize],
    config: &RansacHomographyConfig,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
    dbg::RansacDebugV1,
) {
    global_filter_with_debug_impl(markers, cand_idx, config)
}

fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> (Vec<DetectedMarker>, dbg::RefineDebugV1) {
    refine_h::refine_with_homography_with_debug(gray, markers, h, config)
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
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    global_filter_impl(markers, config)
}

/// Refine marker centers using H: project board coords through H as priors,
/// re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> Vec<DetectedMarker> {
    let (refined, _debug) = refine_with_homography_with_debug(gray, markers, h, config);
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
    store_points_in_debug: bool,
    record_debug: bool,
) -> (CompletionStats, Option<Vec<CompletionAttemptRecord>>) {
    completion::complete_with_h(
        gray,
        h,
        markers,
        config,
        store_points_in_debug,
        record_debug,
    )
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    homography_utils::matrix3_to_array(m)
}

fn mean_reproj_error_px(h: &nalgebra::Matrix3<f64>, markers: &[DetectedMarker]) -> f64 {
    homography_utils::mean_reproj_error_px(h, markers)
}

fn compute_h_stats(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    thresh_px: f64,
) -> Option<RansacStats> {
    homography_utils::compute_h_stats(h, markers, thresh_px)
}

fn refit_homography_matrix(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> Option<(nalgebra::Matrix3<f64>, RansacStats)> {
    homography_utils::refit_homography_matrix(markers, config)
}

/// Remove duplicate detections: keep the highest-confidence marker within dedup_radius.
fn dedup_markers(markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    dedup_markers_impl(markers, radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;
    use image::Luma;
    use nalgebra::Matrix3;

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
        assert_eq!(dump.schema_version, crate::debug_dump::DEBUG_SCHEMA_V1);
        assert_eq!(dump.stages.stage0_proposals.n_total, 0);
        assert!(!dump.stages.stage3_ransac.enabled);
    }

    #[test]
    fn completion_adds_marker_at_h_projected_center() {
        use crate::board_spec;

        let w = 128u32;
        let h = 128u32;

        // Choose an ID that exists on the embedded board and project it to the
        // image center with an affine homography.
        let id = 0usize;
        let xy = board_spec::xy_mm(id).expect("board has id=0");
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
        let (stats, _attempts) = complete_with_h(&img, &h_matrix, &mut markers, &cfg, false, false);
        assert_eq!(stats.n_added, 1, "expected one completion addition");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].id, Some(id));
    }
}
