//! Full ring detection pipeline: proposal → edge sampling → fit → decode → global filter.

use image::GrayImage;

use crate::board_spec;
use crate::conic::{
    fit_ellipse_direct, rms_sampson_distance, try_fit_ellipse_ransac, Ellipse, RansacConfig,
};
use crate::debug_dump as dbg;
use crate::homography::{self, project, RansacHomographyConfig};
use crate::marker_spec::MarkerSpec;
use crate::refine;
use crate::{
    DecodeMetrics, DetectedMarker, DetectionResult, EllipseParams, FitMetrics, RansacStats,
};

use super::decode::{decode_marker_with_diagnostics, DecodeConfig};
use super::edge_sample::{bilinear_sample_u8_checked, EdgeSampleConfig, EdgeSampleResult};
use super::inner_estimate::{estimate_inner_scale_from_outer, InnerStatus, Polarity};
use super::outer_estimate::{
    estimate_outer_from_prior, OuterEstimate, OuterEstimationConfig, OuterStatus,
};
use super::proposal::{find_proposals, ProposalConfig};

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

fn fit_outer_ellipse_with_reason(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Result<(Ellipse, Option<crate::conic::RansacResult>), String> {
    // Fit outer ellipse
    let ransac_config = RansacConfig {
        max_iters: 200,
        inlier_threshold: 1.5,
        min_inliers: 6,
        seed: 42,
    };

    let (outer, outer_ransac) = if edge.outer_points.len() >= 8 {
        match try_fit_ellipse_ransac(&edge.outer_points, &ransac_config) {
            Ok(r) => (r.ellipse, Some(r)),
            Err(_) => {
                // Fall back to direct fit
                match fit_ellipse_direct(&edge.outer_points) {
                    Some((_, e)) => (e, None),
                    None => return Err("fit_outer:direct_failed".to_string()),
                }
            }
        }
    } else if edge.outer_points.len() >= 6 {
        match fit_ellipse_direct(&edge.outer_points) {
            Some((_, e)) => (e, None),
            None => return Err("fit_outer:direct_failed".to_string()),
        }
    } else {
        return Err("fit_outer:too_few_points".to_string());
    };

    // Validate outer ellipse
    if outer.a < config.min_semi_axis
        || outer.a > config.max_semi_axis
        || outer.b < config.min_semi_axis
        || outer.b > config.max_semi_axis
        || outer.aspect_ratio() > config.max_aspect_ratio
        || !outer.is_valid()
    {
        return Err("fit_outer:invalid_ellipse".to_string());
    }

    Ok((outer, outer_ransac))
}

/// Marker center used by the detector.
///
/// We use the outer ellipse center as the base estimate. Inner edge estimation
/// is constrained to be concentric with the outer ellipse and is not allowed
/// to bias the center when unreliable.
fn compute_center(outer: &Ellipse) -> [f64; 2] {
    [outer.cx, outer.cy]
}

/// Helper to create EllipseParams from a conic Ellipse.
fn ellipse_to_params(e: &Ellipse) -> EllipseParams {
    EllipseParams {
        center_xy: [e.cx, e.cy],
        semi_axes: [e.a, e.b],
        angle: e.angle,
    }
}

fn marker_outer_radius_expected_px(config: &DetectConfig) -> f32 {
    (config.marker_diameter_px * 0.5).max(2.0)
}

fn mean_axis_px_from_params(params: &EllipseParams) -> f32 {
    ((params.semi_axes[0] + params.semi_axes[1]) * 0.5) as f32
}

fn mean_axis_px_from_marker(marker: &DetectedMarker) -> Option<f32> {
    marker.ellipse_outer.as_ref().map(mean_axis_px_from_params)
}

fn median_outer_radius_from_neighbors_px(
    projected_center: [f64; 2],
    markers: &[DetectedMarker],
    k: usize,
) -> Option<f32> {
    let mut candidates: Vec<(f64, f32)> = Vec::new();
    for m in markers {
        let r = match mean_axis_px_from_marker(m) {
            Some(v) if v.is_finite() && v > 1.0 => v,
            _ => continue,
        };
        let dx = m.center[0] - projected_center[0];
        let dy = m.center[1] - projected_center[1];
        let d2 = dx * dx + dy * dy;
        if d2.is_finite() {
            candidates.push((d2, r));
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let radii: Vec<f32> = candidates
        .iter()
        .take(k.max(1).min(candidates.len()))
        .map(|(_, r)| *r)
        .collect();
    Some(median_f32(&radii))
}

fn sample_outer_edge_points(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r0: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let n_t = edge_cfg.n_rays.max(8);
    let cx = center_prior[0];
    let cy = center_prior[1];

    let refine_hw = refine_halfwidth_px.clamp(0.0, 4.0);
    let refine_step = edge_cfg.r_step.clamp(0.25, 1.0);
    let n_ref = ((refine_hw / refine_step).ceil() as i32).max(1);

    let mut outer_points = Vec::with_capacity(n_t);
    let mut outer_radii = Vec::with_capacity(n_t);

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let dx = theta.cos();
        let dy = theta.sin();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_r = None::<f32>;

        for k in -n_ref..=n_ref {
            let r = r0 + k as f32 * refine_step;
            if r < edge_cfg.r_min || r > edge_cfg.r_max {
                continue;
            }

            // dI/dr at r via small central difference
            let h = 0.25f32;
            if r <= h {
                continue;
            }

            let x1 = cx + dx * (r + h);
            let y1 = cy + dy * (r + h);
            let x0 = cx + dx * (r - h);
            let y0 = cy + dy * (r - h);
            let i1 = match bilinear_sample_u8_checked(gray, x1, y1) {
                Some(v) => v,
                None => continue,
            };
            let i0 = match bilinear_sample_u8_checked(gray, x0, y0) {
                Some(v) => v,
                None => continue,
            };
            let d = (i1 - i0) / (2.0 * h);

            let score = match pol {
                Polarity::Pos => d,
                Polarity::Neg => -d,
            };

            if score > best_score {
                best_score = score;
                best_r = Some(r);
            }
        }

        let r = match best_r {
            Some(r) if best_score.is_finite() && best_score > 0.0 => r,
            _ => continue,
        };

        // Ring depth check across the chosen edge.
        let band = 2.0f32;
        let x_in = cx + dx * (r - band);
        let y_in = cy + dy * (r - band);
        let x_out = cx + dx * (r + band);
        let y_out = cy + dy * (r + band);
        let i_in = match bilinear_sample_u8_checked(gray, x_in, y_in) {
            Some(v) => v,
            None => continue,
        };
        let i_out = match bilinear_sample_u8_checked(gray, x_out, y_out) {
            Some(v) => v,
            None => continue,
        };
        let signed_depth = match pol {
            Polarity::Pos => i_out - i_in,
            Polarity::Neg => i_in - i_out,
        };
        if signed_depth < edge_cfg.min_ring_depth {
            continue;
        }

        let x = cx + dx * r;
        let y = cy + dy * r;
        outer_points.push([x as f64, y as f64]);
        outer_radii.push(r);
    }

    (outer_points, outer_radii)
}

fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

struct OuterFitCandidate {
    edge: EdgeSampleResult,
    outer: Ellipse,
    outer_ransac: Option<crate::conic::RansacResult>,
    outer_estimate: OuterEstimate,
    chosen_hypothesis: usize,
    decode_result: Option<super::decode::DecodeResult>,
    decode_diag: super::decode::DecodeDiagnostics,
    score: f32,
}

fn fit_outer_ellipse_robust_with_reason(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    config: &DetectConfig,
    edge_cfg: &EdgeSampleConfig,
    store_response: bool,
) -> Result<OuterFitCandidate, String> {
    let r_expected = r_outer_expected_px.max(2.0);

    let mut outer_cfg = config.outer_estimation.clone();
    outer_cfg.theta_samples = edge_cfg.n_rays.max(8);

    let outer_estimate =
        estimate_outer_from_prior(gray, center_prior, r_expected, &outer_cfg, store_response);
    if outer_estimate.status != OuterStatus::Ok || outer_estimate.hypotheses.is_empty() {
        return Err(format!(
            "outer_estimate:{}",
            outer_estimate
                .reason
                .as_deref()
                .unwrap_or("unknown_failure")
        ));
    }

    let pol = outer_estimate
        .polarity
        .ok_or_else(|| "outer_estimate:no_polarity".to_string())?;

    let mut best: Option<OuterFitCandidate> = None;

    for (hi, hyp) in outer_estimate.hypotheses.iter().enumerate() {
        let (outer_points, outer_radii) = sample_outer_edge_points(
            gray,
            center_prior,
            hyp.r_outer_px,
            pol,
            edge_cfg,
            outer_cfg.refine_halfwidth_px,
        );

        if outer_points.len() < edge_cfg.min_rays_with_ring {
            continue;
        }

        let outer_radius = median_f32(&outer_radii);
        let edge = EdgeSampleResult {
            center: center_prior,
            outer_points,
            inner_points: Vec::new(),
            outer_radius,
            inner_radius: 0.0,
            outer_radii,
            inner_radii: Vec::new(),
            n_good_rays: 0,
            n_total_rays: edge_cfg.n_rays.max(8),
        };
        // The outer edge sampler only records rays where an outer edge is found.
        let mut edge = edge;
        edge.n_good_rays = edge.outer_points.len();

        let (outer, outer_ransac) = match fit_outer_ellipse_with_reason(&edge, config) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let (decode_result, decode_diag) =
            decode_marker_with_diagnostics(gray, &outer, &config.decode);

        let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
        let inlier_ratio = outer_ransac
            .as_ref()
            .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
            .unwrap_or(1.0);
        let residual = rms_sampson_distance(&outer, &edge.outer_points) as f32;

        let mean_axis = ((outer.a + outer.b) * 0.5) as f32;
        let size_err = ((mean_axis - r_expected).abs() / r_expected.max(1.0)).min(2.0);
        let size_score = (1.0 - size_err).clamp(0.0, 1.0);

        let decode_score = decode_diag.decode_confidence;

        let score = 2.0 * decode_score + 0.7 * (arc_cov * inlier_ratio) + 0.3 * size_score
            - 0.05 * residual;

        let cand = OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            outer_estimate: outer_estimate.clone(),
            chosen_hypothesis: hi,
            decode_result,
            decode_diag,
            score,
        };

        match &best {
            Some(b) if b.score >= cand.score => {}
            _ => {
                best = Some(cand);
            }
        }
    }

    best.ok_or_else(|| "outer_fit:no_valid_hypothesis".to_string())
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
    use crate::board_spec::{BOARD_N, BOARD_PITCH_MM, BOARD_SIZE_MM};
    use crate::codebook::{CODEBOOK_BITS, CODEBOOK_N};

    let (w, h) = gray.dimensions();

    // Stage 0: proposals
    let proposals = find_proposals(gray, &config.proposal);

    let n_rec = proposals.len().min(debug_cfg.max_candidates);
    let mut stage0 = dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    };
    for (i, p) in proposals.iter().take(n_rec).enumerate() {
        stage0.candidates.push(dbg::CandidateDebugV1 {
            cand_idx: i,
            proposal: dbg::ProposalDebugV1 {
                center_xy: [p.x, p.y],
                score: p.score,
            },
            ring_fit: None,
            decode: None,
            decision: dbg::DecisionDebugV1 {
                status: dbg::DecisionStatusV1::Accepted,
                reason: "proposal".to_string(),
            },
            derived: dbg::DerivedDebugV1 {
                id: None,
                confidence: None,
                center_xy: None,
            },
        });
    }

    // Stage 1: per-proposal fit + decode
    let mut stage1 = dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    };

    let mut markers: Vec<DetectedMarker> = Vec::new();
    let mut marker_cand_idx: Vec<usize> = Vec::new(); // parallel to markers

    for (i, proposal) in proposals.iter().enumerate() {
        let mut cand_debug = if i < n_rec {
            Some(dbg::CandidateDebugV1 {
                cand_idx: i,
                proposal: dbg::ProposalDebugV1 {
                    center_xy: [proposal.x, proposal.y],
                    score: proposal.score,
                },
                ring_fit: None,
                decode: None,
                decision: dbg::DecisionDebugV1 {
                    status: dbg::DecisionStatusV1::Rejected,
                    reason: "unprocessed".to_string(),
                },
                derived: dbg::DerivedDebugV1 {
                    id: None,
                    confidence: None,
                    center_xy: None,
                },
            })
        } else {
            None
        };

        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            [proposal.x, proposal.y],
            marker_outer_radius_expected_px(config),
            config,
            &config.edge_sample,
            debug_cfg.store_points,
        ) {
            Ok(v) => v,
            Err(reason) => {
                if let Some(cd) = cand_debug.as_mut() {
                    cd.decision = dbg::DecisionDebugV1 {
                        status: dbg::DecisionStatusV1::Rejected,
                        reason,
                    };
                }
                if let Some(cd) = cand_debug {
                    stage1.candidates.push(cd);
                }
                continue;
            }
        };

        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            outer_estimate,
            chosen_hypothesis,
            decode_result,
            decode_diag,
            ..
        } = fit;

        let center = compute_center(&outer);

        let fit_metrics = FitMetrics {
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

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let derived_id = decode_result.as_ref().map(|d| d.id);

        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let inner_est = if decode_result.is_some() {
            Some(estimate_inner_scale_from_outer(
                gray,
                &outer,
                &config.marker_spec,
                debug_cfg.store_points,
            ))
        } else {
            None
        };
        let inner_params = inner_est.as_ref().and_then(|est| {
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
        });

        let marker = DetectedMarker {
            id: derived_id,
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params.clone(),
            fit: fit_metrics.clone(),
            decode: decode_metrics,
        };

        markers.push(marker);
        marker_cand_idx.push(i);

        if let Some(cd) = cand_debug.as_mut() {
            let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
            cd.ring_fit = Some(dbg::RingFitDebugV1 {
                center_xy_fit: [center[0] as f32, center[1] as f32],
                edges: dbg::RingEdgesDebugV1 {
                    n_angles_total: edge.n_total_rays,
                    n_angles_with_both: edge.n_good_rays,
                    inner_peak_r: if edge.inner_radii.is_empty() {
                        None
                    } else {
                        Some(edge.inner_radii.clone())
                    },
                    outer_peak_r: Some(edge.outer_radii.clone()),
                },
                outer_estimation: Some({
                    let chosen = outer_estimate.hypotheses.get(chosen_hypothesis);
                    dbg::OuterEstimationDebugV1 {
                        r_outer_expected_px: outer_estimate.r_outer_expected_px,
                        search_window_px: outer_estimate.search_window_px,
                        r_outer_found_px: chosen.map(|h| h.r_outer_px),
                        polarity: outer_estimate.polarity.map(|p| match p {
                            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                        }),
                        peak_strength: chosen.map(|h| h.peak_strength),
                        theta_consistency: chosen.map(|h| h.theta_consistency),
                        status: match outer_estimate.status {
                            OuterStatus::Ok => dbg::OuterEstimationStatusDebugV1::Ok,
                            OuterStatus::Rejected => dbg::OuterEstimationStatusDebugV1::Rejected,
                            OuterStatus::Failed => dbg::OuterEstimationStatusDebugV1::Failed,
                        },
                        reason: outer_estimate.reason.clone(),
                        hypotheses: outer_estimate
                            .hypotheses
                            .iter()
                            .map(|h| dbg::OuterHypothesisDebugV1 {
                                r_outer_px: h.r_outer_px,
                                peak_strength: h.peak_strength,
                                theta_consistency: h.theta_consistency,
                            })
                            .collect(),
                        chosen_hypothesis: Some(chosen_hypothesis),
                        radial_response_agg: outer_estimate.radial_response_agg.clone(),
                        r_samples: outer_estimate.r_samples.clone(),
                    }
                }),
                ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                    center_xy: [outer.cx as f32, outer.cy as f32],
                    semi_axes: [outer.a as f32, outer.b as f32],
                    angle: outer.angle as f32,
                }),
                ellipse_inner: inner_params.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                    center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                    semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                    angle: p.angle as f32,
                }),
                inner_estimation: inner_est.as_ref().map(|est| dbg::InnerEstimationDebugV1 {
                    r_inner_expected: est.r_inner_expected,
                    search_window: est.search_window,
                    r_inner_found: est.r_inner_found,
                    polarity: est.polarity.map(|p| match p {
                        Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                        Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                    }),
                    peak_strength: est.peak_strength,
                    theta_consistency: est.theta_consistency,
                    status: match est.status {
                        InnerStatus::Ok => dbg::InnerEstimationStatusDebugV1::Ok,
                        InnerStatus::Rejected => dbg::InnerEstimationStatusDebugV1::Rejected,
                        InnerStatus::Failed => dbg::InnerEstimationStatusDebugV1::Failed,
                    },
                    reason: est.reason.clone(),
                    radial_response_agg: est.radial_response_agg.clone(),
                    r_samples: est.r_samples.clone(),
                }),
                metrics: dbg::RingFitMetricsDebugV1 {
                    inlier_ratio_inner: fit_metrics.ransac_inlier_ratio_inner,
                    inlier_ratio_outer: fit_metrics.ransac_inlier_ratio_outer,
                    mean_resid_inner: fit_metrics.rms_residual_inner.map(|v| v as f32),
                    mean_resid_outer: fit_metrics.rms_residual_outer.map(|v| v as f32),
                    arc_coverage: arc_cov,
                    valid_inner: inner_params.is_some(),
                    valid_outer: true,
                },
                points_outer: if debug_cfg.store_points {
                    Some(
                        edge.outer_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                points_inner: if debug_cfg.store_points {
                    Some(
                        edge.inner_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
            });

            cd.decode = Some(dbg::DecodeDebugV1 {
                sector_means: decode_diag.sector_intensities,
                threshold: decode_diag.threshold,
                observed_word_hex: format!("0x{:04X}", decode_diag.used_word),
                inverted_used: decode_diag.inverted_used,
                r#match: dbg::DecodeMatchDebugV1 {
                    best_id: decode_diag.best_id,
                    best_rotation: decode_diag.best_rotation,
                    best_dist: decode_diag.best_dist,
                    margin: decode_diag.margin,
                    decode_confidence: decode_diag.decode_confidence,
                },
                accepted: Some(decode_result.is_some()),
                reject_reason: decode_diag.reject_reason.clone(),
            });

            cd.decision = dbg::DecisionDebugV1 {
                status: dbg::DecisionStatusV1::Accepted,
                reason: if let Some(r) = decode_diag.reject_reason {
                    format!("ok_with_decode_reject:{}", r)
                } else {
                    "ok".to_string()
                },
            };
            cd.derived = dbg::DerivedDebugV1 {
                id: derived_id,
                confidence: Some(confidence),
                center_xy: Some([center[0] as f32, center[1] as f32]),
            };
        }

        if let Some(cd) = cand_debug {
            stage1.candidates.push(cd);
        }
    }

    // Stage 2: dedup (proximity + id)
    let (markers_dedup, cand_idx_dedup, dedup_debug) =
        dedup_with_debug(markers, marker_cand_idx, config.dedup_radius);

    // Stage 3: global filter
    let (filtered, h_result, ransac_stats, ransac_debug) = if !config.use_global_filter {
        (
            markers_dedup,
            None,
            None,
            dbg::RansacDebugV1 {
                enabled: false,
                h_best: None,
                correspondences_used: 0,
                inlier_ids: Vec::new(),
                outlier_ids: Vec::new(),
                per_id_error_px: None,
                stats: dbg::RansacStatsDebugV1 {
                    iters: 0,
                    thresh_px: config.ransac_homography.inlier_threshold,
                    n_corr: 0,
                    n_inliers: 0,
                    mean_err_inliers: 0.0,
                    p95_err_inliers: 0.0,
                },
                notes: vec!["global_filter_disabled".to_string()],
            },
        )
    } else {
        global_filter_with_debug(&markers_dedup, &cand_idx_dedup, &config.ransac_homography)
    };

    let h_matrix = h_result.as_ref().map(|r| &r.h);

    // Stage 4: refine (optional)
    let (mut final_markers, mut refine_debug) = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                let (refined, refine_dbg) =
                    refine_with_homography_with_debug(gray, &filtered, h, config);
                (refined, Some(refine_dbg))
            } else {
                (filtered, None)
            }
        } else {
            (filtered, None)
        }
    } else {
        (filtered, None)
    };

    // Stage 5: completion (optional, only when H exists)
    let (completion_stats, completion_attempts) = if config.completion.enable {
        if let Some(h) = h_matrix {
            complete_with_h(
                gray,
                h,
                &mut final_markers,
                config,
                debug_cfg.store_points,
                true,
            )
        } else {
            (CompletionStats::default(), Some(Vec::new()))
        }
    } else {
        (CompletionStats::default(), Some(Vec::new()))
    };

    let completion_debug = dbg::CompletionDebugV1 {
        enabled: config.completion.enable && h_matrix.is_some(),
        params: dbg::CompletionParamsDebugV1 {
            roi_radius_px: config.completion.roi_radius_px,
            reproj_gate_px: config.completion.reproj_gate_px,
            min_fit_confidence: config.completion.min_fit_confidence,
            min_arc_coverage: config.completion.min_arc_coverage,
            max_attempts: config.completion.max_attempts,
            image_margin_px: config.completion.image_margin_px,
        },
        attempted: completion_attempts
            .unwrap_or_default()
            .into_iter()
            .map(|a| dbg::CompletionAttemptDebugV1 {
                id: a.id,
                projected_center_xy: a.projected_center_xy,
                status: match a.status {
                    CompletionAttemptStatus::Added => dbg::CompletionAttemptStatusDebugV1::Added,
                    CompletionAttemptStatus::SkippedPresent => {
                        dbg::CompletionAttemptStatusDebugV1::SkippedPresent
                    }
                    CompletionAttemptStatus::SkippedOob => {
                        dbg::CompletionAttemptStatusDebugV1::SkippedOob
                    }
                    CompletionAttemptStatus::FailedFit => {
                        dbg::CompletionAttemptStatusDebugV1::FailedFit
                    }
                    CompletionAttemptStatus::FailedGate => {
                        dbg::CompletionAttemptStatusDebugV1::FailedGate
                    }
                },
                reason: a.reason,
                reproj_err_px: a.reproj_err_px,
                fit_confidence: a.fit_confidence,
                fit: a.fit,
            })
            .collect(),
        stats: dbg::CompletionStatsDebugV1 {
            n_candidates_total: completion_stats.n_candidates_total,
            n_in_image: completion_stats.n_in_image,
            n_attempted: completion_stats.n_attempted,
            n_added: completion_stats.n_added,
            n_failed_fit: completion_stats.n_failed_fit,
            n_failed_gate: completion_stats.n_failed_gate,
        },
        notes: Vec::new(),
    };

    // Stage 6: Non-linear refinement in board plane (optional).
    let mut h_current: Option<nalgebra::Matrix3<f64>> = h_result.as_ref().map(|r| r.h);
    let mut nl_refine_debug = dbg::NlRefineDebugV1 {
        enabled: config.nl_refine.enabled && h_current.is_some(),
        params: dbg::NlRefineParamsV1 {
            enabled: config.nl_refine.enabled,
            max_iters: config.nl_refine.max_iters,
            huber_delta_mm: config.nl_refine.huber_delta_mm,
            min_points: config.nl_refine.min_points,
            reject_shift_mm: config.nl_refine.reject_thresh_mm,
            enable_h_refit: config.nl_refine.enable_h_refit,
            h_refit_iters: config.nl_refine.h_refit_iters,
            marker_outer_radius_mm: crate::board_spec::marker_outer_radius_mm() as f64,
        },
        h_used: h_current.as_ref().map(matrix3_to_array),
        h_refit: None,
        stats: dbg::NlRefineStatsDebugV1 {
            n_inliers: 0,
            n_refined: 0,
            n_failed: 0,
            mean_before_mm: 0.0,
            mean_after_mm: 0.0,
            p95_before_mm: 0.0,
            p95_after_mm: 0.0,
        },
        refined_markers: Vec::new(),
        notes: Vec::new(),
    };

    if config.nl_refine.enabled {
        if let Some(h0) = h_current {
            let (stats0, records0) = refine::refine_markers_circle_board(
                gray,
                &h0,
                &mut final_markers,
                &config.nl_refine,
                debug_cfg.store_points,
            );

            nl_refine_debug.stats = dbg::NlRefineStatsDebugV1 {
                n_inliers: stats0.n_inliers,
                n_refined: stats0.n_refined,
                n_failed: stats0.n_failed,
                mean_before_mm: stats0.mean_before_mm,
                mean_after_mm: stats0.mean_after_mm,
                p95_before_mm: stats0.p95_before_mm,
                p95_after_mm: stats0.p95_after_mm,
            };
            nl_refine_debug.refined_markers = records0
                .into_iter()
                .map(|r| dbg::NlRefinedMarkerDebugV1 {
                    id: r.id,
                    n_points: r.n_points,
                    init_center_board_mm: r.init_center_board_mm,
                    refined_center_board_mm: r.refined_center_board_mm,
                    center_img_before: r.center_img_before,
                    center_img_after: r.center_img_after,
                    before_rms_mm: r.before_rms_mm,
                    after_rms_mm: r.after_rms_mm,
                    delta_center_mm: r.delta_center_mm,
                    edge_points_img: r.edge_points_img,
                    edge_points_board_mm: r.edge_points_board_mm,
                    status: match r.status {
                        refine::MarkerRefineStatus::Ok => dbg::NlRefineStatusDebugV1::Ok,
                        refine::MarkerRefineStatus::Rejected => {
                            dbg::NlRefineStatusDebugV1::Rejected
                        }
                        refine::MarkerRefineStatus::Failed => dbg::NlRefineStatusDebugV1::Failed,
                        refine::MarkerRefineStatus::Skipped => dbg::NlRefineStatusDebugV1::Skipped,
                    },
                    reason: r.reason,
                })
                .collect();

            if config.nl_refine.enable_h_refit && final_markers.len() >= 10 {
                let max_iters = config.nl_refine.h_refit_iters.clamp(1, 3);
                let mut h_prev = h0;
                let mut mean_prev = mean_reproj_error_px(&h_prev, &final_markers);
                for iter in 0..max_iters {
                    let Some((h_next, _stats1)) =
                        refit_homography_matrix(&final_markers, &config.ransac_homography)
                    else {
                        nl_refine_debug
                            .notes
                            .push(format!("h_refit_iter{}:refit_failed", iter));
                        break;
                    };

                    let mean_next = mean_reproj_error_px(&h_next, &final_markers);
                    if mean_next.is_finite() && (mean_next < mean_prev || !mean_prev.is_finite()) {
                        nl_refine_debug.h_refit = Some(matrix3_to_array(&h_next));
                        nl_refine_debug.notes.push(format!(
                            "h_refit_iter{}:accepted mean_err_px {:.3} -> {:.3}",
                            iter, mean_prev, mean_next
                        ));

                        h_current = Some(h_next);
                        h_prev = h_next;
                        mean_prev = mean_next;

                        let (stats_i, records_i) = refine::refine_markers_circle_board(
                            gray,
                            &h_prev,
                            &mut final_markers,
                            &config.nl_refine,
                            debug_cfg.store_points,
                        );
                        nl_refine_debug.stats = dbg::NlRefineStatsDebugV1 {
                            n_inliers: stats_i.n_inliers,
                            n_refined: stats_i.n_refined,
                            n_failed: stats_i.n_failed,
                            mean_before_mm: stats_i.mean_before_mm,
                            mean_after_mm: stats_i.mean_after_mm,
                            p95_before_mm: stats_i.p95_before_mm,
                            p95_after_mm: stats_i.p95_after_mm,
                        };
                        nl_refine_debug.refined_markers = records_i
                            .into_iter()
                            .map(|r| dbg::NlRefinedMarkerDebugV1 {
                                id: r.id,
                                n_points: r.n_points,
                                init_center_board_mm: r.init_center_board_mm,
                                refined_center_board_mm: r.refined_center_board_mm,
                                center_img_before: r.center_img_before,
                                center_img_after: r.center_img_after,
                                before_rms_mm: r.before_rms_mm,
                                after_rms_mm: r.after_rms_mm,
                                delta_center_mm: r.delta_center_mm,
                                edge_points_img: r.edge_points_img,
                                edge_points_board_mm: r.edge_points_board_mm,
                                status: match r.status {
                                    refine::MarkerRefineStatus::Ok => {
                                        dbg::NlRefineStatusDebugV1::Ok
                                    }
                                    refine::MarkerRefineStatus::Rejected => {
                                        dbg::NlRefineStatusDebugV1::Rejected
                                    }
                                    refine::MarkerRefineStatus::Failed => {
                                        dbg::NlRefineStatusDebugV1::Failed
                                    }
                                    refine::MarkerRefineStatus::Skipped => {
                                        dbg::NlRefineStatusDebugV1::Skipped
                                    }
                                },
                                reason: r.reason,
                            })
                            .collect();
                    } else {
                        nl_refine_debug.notes.push(format!(
                            "h_refit_iter{}:rejected mean_err_px {:.3} -> {:.3}",
                            iter, mean_prev, mean_next
                        ));
                        break;
                    }
                }
            }
        } else {
            nl_refine_debug
                .notes
                .push("skipped_no_homography".to_string());
        }
    }

    // Final H: refit after refinement if we have enough markers.
    let did_refit = config.refine_with_h && final_markers.len() >= 10;
    let final_h_matrix = if did_refit {
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

    if did_refit {
        if let Some(ref mut rd) = refine_debug {
            rd.h_refit = final_h;
        }
    }

    let result = DetectionResult {
        detected_markers: final_markers.clone(),
        image_size: [w, h],
        homography: final_h,
        ransac: final_ransac,
    };

    let dump = dbg::DebugDumpV1 {
        schema_version: dbg::DEBUG_SCHEMA_V1.to_string(),
        image: dbg::ImageDebugV1 {
            path: debug_cfg.image_path.clone(),
            width: w,
            height: h,
        },
        board: dbg::BoardDebugV1 {
            pitch_mm: BOARD_PITCH_MM,
            board_mm: BOARD_SIZE_MM[0],
            board_size_mm: BOARD_SIZE_MM,
            marker_count: BOARD_N,
            codebook_bits: CODEBOOK_BITS,
            codebook_n: CODEBOOK_N,
        },
        params: dbg::ParamsDebugV1 {
            marker_diameter_px: config.marker_diameter_px as f64,
            proposal: dbg::ProposalParamsV1 {
                r_min: config.proposal.r_min,
                r_max: config.proposal.r_max,
                grad_threshold: config.proposal.grad_threshold,
                nms_radius: config.proposal.nms_radius,
                min_vote_frac: config.proposal.min_vote_frac,
                accum_sigma: config.proposal.accum_sigma,
            },
            edge_sample: dbg::EdgeSampleParamsV1 {
                n_rays: config.edge_sample.n_rays,
                r_max: config.edge_sample.r_max,
                r_min: config.edge_sample.r_min,
                r_step: config.edge_sample.r_step,
                min_ring_depth: config.edge_sample.min_ring_depth,
                min_rays_with_ring: config.edge_sample.min_rays_with_ring,
            },
            outer_estimation: Some(dbg::OuterEstimationParamsV1 {
                search_halfwidth_px: config.outer_estimation.search_halfwidth_px,
                radial_samples: config.outer_estimation.radial_samples,
                theta_samples: config.outer_estimation.theta_samples,
                aggregator: config.outer_estimation.aggregator,
                grad_polarity: match config.outer_estimation.grad_polarity {
                    super::outer_estimate::OuterGradPolarity::DarkToLight => {
                        dbg::OuterGradPolarityParamsV1::DarkToLight
                    }
                    super::outer_estimate::OuterGradPolarity::LightToDark => {
                        dbg::OuterGradPolarityParamsV1::LightToDark
                    }
                    super::outer_estimate::OuterGradPolarity::Auto => {
                        dbg::OuterGradPolarityParamsV1::Auto
                    }
                },
                min_theta_coverage: config.outer_estimation.min_theta_coverage,
                min_theta_consistency: config.outer_estimation.min_theta_consistency,
                allow_two_hypotheses: config.outer_estimation.allow_two_hypotheses,
                second_peak_min_rel: config.outer_estimation.second_peak_min_rel,
                refine_halfwidth_px: config.outer_estimation.refine_halfwidth_px,
            }),
            decode: dbg::DecodeParamsV1 {
                code_band_ratio: config.decode.code_band_ratio,
                samples_per_sector: config.decode.samples_per_sector,
                n_radial_rings: config.decode.n_radial_rings,
                max_decode_dist: config.decode.max_decode_dist,
                min_decode_confidence: config.decode.min_decode_confidence,
            },
            marker_spec: config.marker_spec.clone(),
            nl_refine: Some(nl_refine_debug.params.clone()),
            min_semi_axis: config.min_semi_axis,
            max_semi_axis: config.max_semi_axis,
            max_aspect_ratio: config.max_aspect_ratio,
            dedup_radius: config.dedup_radius,
            use_global_filter: config.use_global_filter,
            ransac_homography: dbg::RansacHomographyParamsV1 {
                max_iters: config.ransac_homography.max_iters,
                inlier_threshold: config.ransac_homography.inlier_threshold,
                min_inliers: config.ransac_homography.min_inliers,
                seed: config.ransac_homography.seed,
            },
            refine_with_h: config.refine_with_h,
            debug: dbg::DebugOptionsV1 {
                max_candidates: debug_cfg.max_candidates,
                store_points: debug_cfg.store_points,
            },
        },
        stages: dbg::StagesDebugV1 {
            stage0_proposals: stage0,
            stage1_fit_decode: stage1,
            stage2_dedup: dedup_debug,
            stage3_ransac: ransac_debug,
            stage4_refine: refine_debug,
            stage5_completion: Some(completion_debug),
            stage6_nl_refine: Some(nl_refine_debug),
            final_: dbg::FinalDebugV1 {
                h_final: result.homography,
                detections: result.detected_markers.clone(),
                notes: if completion_stats.n_added > 0 {
                    vec![format!("completion_added={}", completion_stats.n_added)]
                } else {
                    Vec::new()
                },
            },
        },
    };

    (result, dump)
}

fn dedup_with_debug(
    mut markers: Vec<DetectedMarker>,
    mut cand_idx: Vec<usize>,
    radius: f64,
) -> (Vec<DetectedMarker>, Vec<usize>, dbg::DedupDebugV1) {
    // Sort by confidence descending (keep cand_idx in sync)
    let mut order: Vec<usize> = (0..markers.len()).collect();
    order.sort_by(|&a, &b| {
        markers[b]
            .confidence
            .partial_cmp(&markers[a].confidence)
            .unwrap()
    });

    markers = order.iter().map(|&i| markers[i].clone()).collect();
    cand_idx = order.iter().map(|&i| cand_idx[i]).collect();

    let mut keep = vec![true; markers.len()];
    let r2 = radius * radius;
    let mut kept_by_proximity: Vec<dbg::KeptByProximityDebugV1> = Vec::new();

    for i in 0..markers.len() {
        if !keep[i] {
            continue;
        }
        let mut dropped: Vec<usize> = Vec::new();
        for j in (i + 1)..markers.len() {
            if !keep[j] {
                continue;
            }
            let dx = markers[i].center[0] - markers[j].center[0];
            let dy = markers[i].center[1] - markers[j].center[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
                dropped.push(cand_idx[j]);
            }
        }
        if !dropped.is_empty() {
            kept_by_proximity.push(dbg::KeptByProximityDebugV1 {
                kept_cand_idx: cand_idx[i],
                dropped_cand_indices: dropped,
                reasons: vec!["within_dedup_radius".to_string()],
            });
        }
    }

    let mut markers2: Vec<DetectedMarker> = Vec::new();
    let mut cand2: Vec<usize> = Vec::new();
    for ((m, k), ci) in markers
        .into_iter()
        .zip(keep.into_iter())
        .zip(cand_idx.into_iter())
    {
        if k {
            markers2.push(m);
            cand2.push(ci);
        }
    }

    // Dedup by ID (keep best confidence)
    use std::collections::{HashMap, HashSet};
    let mut best_idx: HashMap<usize, usize> = HashMap::new();
    for (i, m) in markers2.iter().enumerate() {
        if let Some(id) = m.id {
            match best_idx.get(&id) {
                Some(&prev) if markers2[prev].confidence >= m.confidence => {}
                _ => {
                    best_idx.insert(id, i);
                }
            }
        }
    }

    let keep_set: HashSet<usize> = best_idx.values().copied().collect();
    let mut kept_by_id: Vec<dbg::KeptByIdDebugV1> = Vec::new();
    for (&id, &kept_i) in best_idx.iter() {
        let mut dropped: Vec<usize> = Vec::new();
        for (i, m) in markers2.iter().enumerate() {
            if i == kept_i {
                continue;
            }
            if m.id == Some(id) {
                dropped.push(cand2[i]);
            }
        }
        if !dropped.is_empty() {
            kept_by_id.push(dbg::KeptByIdDebugV1 {
                id,
                kept_cand_idx: cand2[kept_i],
                dropped_cand_indices: dropped,
                reasons: vec!["lower_confidence".to_string()],
            });
        }
    }
    kept_by_id.sort_by_key(|e| e.id);

    let mut markers3: Vec<DetectedMarker> = Vec::new();
    let mut cand3: Vec<usize> = Vec::new();
    for (i, m) in markers2.into_iter().enumerate() {
        let keep_it = m.id.is_none() || keep_set.contains(&i);
        if keep_it {
            markers3.push(m);
            cand3.push(cand2[i]);
        }
    }

    (
        markers3,
        cand3,
        dbg::DedupDebugV1 {
            kept_by_proximity,
            kept_by_id,
            notes: Vec::new(),
        },
    )
}

fn global_filter_with_debug(
    markers: &[DetectedMarker],
    _cand_idx: &[usize],
    config: &RansacHomographyConfig,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
    dbg::RansacDebugV1,
) {
    // Build correspondences from decoded markers
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut corr_ids: Vec<usize> = Vec::new();

    for m in markers {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                corr_ids.push(id);
            }
        }
    }

    if src_pts.len() < 4 {
        let dbg = dbg::RansacDebugV1 {
            enabled: true,
            h_best: None,
            correspondences_used: src_pts.len(),
            inlier_ids: Vec::new(),
            outlier_ids: Vec::new(),
            per_id_error_px: None,
            stats: dbg::RansacStatsDebugV1 {
                iters: config.max_iters,
                thresh_px: config.inlier_threshold,
                n_corr: src_pts.len(),
                n_inliers: 0,
                mean_err_inliers: 0.0,
                p95_err_inliers: 0.0,
            },
            notes: vec![format!("too_few_correspondences({}<4)", src_pts.len())],
        };
        return (markers.to_vec(), None, None, dbg);
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            let dbg = dbg::RansacDebugV1 {
                enabled: true,
                h_best: None,
                correspondences_used: src_pts.len(),
                inlier_ids: Vec::new(),
                outlier_ids: Vec::new(),
                per_id_error_px: None,
                stats: dbg::RansacStatsDebugV1 {
                    iters: config.max_iters,
                    thresh_px: config.inlier_threshold,
                    n_corr: src_pts.len(),
                    n_inliers: 0,
                    mean_err_inliers: 0.0,
                    p95_err_inliers: 0.0,
                },
                notes: vec![format!("ransac_failed:{}", e)],
            };
            return (markers.to_vec(), None, None, dbg);
        }
    };

    // Collect inliers/outliers and per-id errors
    let mut filtered: Vec<DetectedMarker> = Vec::new();
    let mut inlier_errors: Vec<f64> = Vec::new();
    let mut inlier_ids: Vec<usize> = Vec::new();
    let mut outlier_ids: Vec<usize> = Vec::new();
    let mut per_id_error: Vec<dbg::PerIdErrorDebugV1> = Vec::new();

    for (j, &id) in corr_ids.iter().enumerate() {
        let err = result.errors[j];
        per_id_error.push(dbg::PerIdErrorDebugV1 {
            id,
            reproj_err_px: err,
        });
        if result.inlier_mask[j] {
            inlier_ids.push(id);
            inlier_errors.push(err);
        } else {
            outlier_ids.push(id);
        }
    }

    // Filter markers to inliers only (by matching id list)
    use std::collections::HashSet;
    let inlier_set: HashSet<usize> = inlier_ids.iter().copied().collect();
    for m in markers {
        if let Some(id) = m.id {
            if inlier_set.contains(&id) {
                filtered.push(m.clone());
            }
        }
    }

    // Stats
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    let stats = RansacStats {
        n_candidates: src_pts.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    let dbg = dbg::RansacDebugV1 {
        enabled: true,
        h_best: Some(matrix3_to_array(&result.h)),
        correspondences_used: src_pts.len(),
        inlier_ids,
        outlier_ids,
        per_id_error_px: Some(per_id_error),
        stats: dbg::RansacStatsDebugV1 {
            iters: config.max_iters,
            thresh_px: config.inlier_threshold,
            n_corr: src_pts.len(),
            n_inliers: result.n_inliers,
            mean_err_inliers: mean_err,
            p95_err_inliers: p95_err,
        },
        notes: Vec::new(),
    };

    (filtered, Some(result), Some(stats), dbg)
}

fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> (Vec<DetectedMarker>, dbg::RefineDebugV1) {
    let mut refined = Vec::with_capacity(markers.len());
    let mut refined_dbg = Vec::with_capacity(markers.len());

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board_spec::xy_mm(id) {
            Some(xy) => xy,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let prior = project(h, xy[0] as f64, xy[1] as f64);
        if prior[0].is_nan() || prior[1].is_nan() {
            refined.push(m.clone());
            continue;
        }

        let r_expected =
            mean_axis_px_from_marker(m).unwrap_or(marker_outer_radius_expected_px(config));

        let fit_cand = match fit_outer_ellipse_robust_with_reason(
            gray,
            [prior[0] as f32, prior[1] as f32],
            r_expected,
            config,
            &config.edge_sample,
            false,
        ) {
            Ok(v) => v,
            Err(_) => {
                refined.push(m.clone());
                continue;
            }
        };
        let edge = fit_cand.edge;
        let outer = fit_cand.outer;
        let outer_ransac = fit_cand.outer_ransac;
        let decode_result = fit_cand.decode_result.filter(|d| d.id == id);

        let mean_axis_new = ((outer.a + outer.b) * 0.5) as f32;
        let scale_ok = mean_axis_new.is_finite()
            && mean_axis_new >= (r_expected * 0.75)
            && mean_axis_new <= (r_expected * 1.33);
        if decode_result.is_none() || !scale_ok {
            // Refinement is best-effort and must not degrade decoded detections.
            refined.push(m.clone());
            refined_dbg.push(dbg::RefinedMarkerDebugV1 {
                id,
                prior_center_xy: [prior[0] as f32, prior[1] as f32],
                refined_center_xy: [m.center[0] as f32, m.center[1] as f32],
                ellipse_outer: m.ellipse_outer.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                    center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                    semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                    angle: p.angle as f32,
                }),
                ellipse_inner: m.ellipse_inner.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                    center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                    semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                    angle: p.angle as f32,
                }),
                fit: m.fit.clone(),
            });
            continue;
        }

        let center = compute_center(&outer);

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

        let inner_est = estimate_inner_scale_from_outer(gray, &outer, &config.marker_spec, false);
        let inner_params = if inner_est.status == InnerStatus::Ok {
            let s = inner_est
                .r_inner_found
                .unwrap_or(config.marker_spec.r_inner_expected) as f64;
            Some(EllipseParams {
                center_xy: [outer.cx, outer.cy],
                semi_axes: [outer.a * s, outer.b * s],
                angle: outer.angle,
            })
        } else {
            None
        };

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let updated = DetectedMarker {
            id: Some(id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params.clone(),
            fit: fit.clone(),
            decode: decode_metrics,
        };

        refined_dbg.push(dbg::RefinedMarkerDebugV1 {
            id,
            prior_center_xy: [prior[0] as f32, prior[1] as f32],
            refined_center_xy: [center[0] as f32, center[1] as f32],
            ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                center_xy: [outer.cx as f32, outer.cy as f32],
                semi_axes: [outer.a as f32, outer.b as f32],
                angle: outer.angle as f32,
            }),
            ellipse_inner: inner_params.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                angle: p.angle as f32,
            }),
            fit,
        });

        refined.push(updated);
    }

    (
        refined,
        dbg::RefineDebugV1 {
            h_prior: matrix3_to_array(h),
            refined_markers: refined_dbg,
            h_refit: None,
            notes: Vec::new(),
        },
    )
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    use std::collections::HashMap;
    let mut best_idx: HashMap<usize, usize> = HashMap::new();

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            match best_idx.get(&id) {
                Some(&prev) if markers[prev].confidence >= m.confidence => {}
                _ => {
                    best_idx.insert(id, i);
                }
            }
        }
    }

    let keep_set: std::collections::HashSet<usize> = best_idx.values().copied().collect();
    let mut i = 0;
    markers.retain(|m| {
        let idx = i;
        i += 1;
        // Keep markers without ID (they'll be filtered by RANSAC anyway)
        m.id.is_none() || keep_set.contains(&idx)
    });
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
    // Build correspondences from decoded markers
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut candidate_indices = Vec::new(); // index into markers

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                candidate_indices.push(i);
            }
        }
    }

    tracing::info!(
        "Global filter: {} decoded candidates out of {} total detections",
        candidate_indices.len(),
        markers.len()
    );

    if candidate_indices.len() < 4 {
        tracing::warn!(
            "Too few decoded candidates for homography ({} < 4)",
            candidate_indices.len()
        );
        return (markers.to_vec(), None, None);
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Homography RANSAC failed: {}", e);
            return (markers.to_vec(), None, None);
        }
    };

    // Collect inlier markers
    let mut filtered = Vec::new();
    let mut inlier_errors = Vec::new();

    for (j, &marker_idx) in candidate_indices.iter().enumerate() {
        if result.inlier_mask[j] {
            filtered.push(markers[marker_idx].clone());
            inlier_errors.push(result.errors[j]);
        }
    }

    // Compute stats
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    let stats = RansacStats {
        n_candidates: candidate_indices.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    tracing::info!(
        "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
        result.n_inliers,
        candidate_indices.len(),
        mean_err,
        p95_err,
    );

    (filtered, Some(result), Some(stats))
}

/// Refine marker centers using H: project board coords through H as priors,
/// re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> Vec<DetectedMarker> {
    let mut refined = Vec::with_capacity(markers.len());

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board_spec::xy_mm(id) {
            Some(xy) => xy,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        // Project board coords through H to get refined center prior
        let prior = project(h, xy[0] as f64, xy[1] as f64);
        if prior[0].is_nan() || prior[1].is_nan() {
            refined.push(m.clone());
            continue;
        }

        let r_expected =
            mean_axis_px_from_marker(m).unwrap_or(marker_outer_radius_expected_px(config));

        let fit_cand = match fit_outer_ellipse_robust_with_reason(
            gray,
            [prior[0] as f32, prior[1] as f32],
            r_expected,
            config,
            &config.edge_sample,
            false,
        ) {
            Ok(v) => v,
            Err(_) => {
                refined.push(m.clone());
                continue;
            }
        };
        let edge = fit_cand.edge;
        let outer = fit_cand.outer;
        let outer_ransac = fit_cand.outer_ransac;
        let decode_result = fit_cand.decode_result.filter(|d| d.id == id);

        let mean_axis_new = ((outer.a + outer.b) * 0.5) as f32;
        let scale_ok = mean_axis_new.is_finite()
            && mean_axis_new >= (r_expected * 0.75)
            && mean_axis_new <= (r_expected * 1.33);
        if decode_result.is_none() || !scale_ok {
            refined.push(m.clone());
            continue;
        }

        let center = compute_center(&outer);

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

        let inner_est = estimate_inner_scale_from_outer(gray, &outer, &config.marker_spec, false);
        let inner_params = if inner_est.status == InnerStatus::Ok {
            let s = inner_est
                .r_inner_found
                .unwrap_or(config.marker_spec.r_inner_expected) as f64;
            Some(EllipseParams {
                center_xy: [outer.cx, outer.cy],
                semi_axes: [outer.a * s, outer.b * s],
                angle: outer.angle,
            })
        } else {
            None
        };

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        // Keep original ID (validated by RANSAC), but update geometry
        refined.push(DetectedMarker {
            id: Some(id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params,
            fit,
            decode: decode_metrics,
        });
    }

    refined
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionAttemptStatus {
    Added,
    SkippedPresent,
    SkippedOob,
    FailedFit,
    FailedGate,
}

#[derive(Debug, Clone)]
struct CompletionAttemptRecord {
    id: usize,
    projected_center_xy: [f32; 2],
    status: CompletionAttemptStatus,
    reason: Option<String>,
    reproj_err_px: Option<f32>,
    fit_confidence: Option<f32>,
    fit: Option<dbg::RingFitDebugV1>,
}

#[derive(Debug, Clone, Default)]
struct CompletionStats {
    n_candidates_total: usize,
    n_in_image: usize,
    n_attempted: usize,
    n_added: usize,
    n_failed_fit: usize,
    n_failed_gate: usize,
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
    use crate::board_spec;
    use std::collections::HashSet;

    let params = &config.completion;
    if !params.enable {
        return (
            CompletionStats::default(),
            if record_debug { Some(Vec::new()) } else { None },
        );
    }

    let (w, h_img) = gray.dimensions();
    let w_f = w as f64;
    let h_f = h_img as f64;

    let roi_radius = params.roi_radius_px.clamp(8.0, 200.0) as f64;
    let safe_margin = roi_radius + params.image_margin_px.max(0.0) as f64;

    // Build fast lookup for already-present IDs.
    let present_ids: HashSet<usize> = markers.iter().filter_map(|m| m.id).collect();

    // Completion uses a slightly relaxed edge sampler threshold to allow partial arcs.
    let mut edge_cfg = config.edge_sample.clone();
    edge_cfg.r_max = roi_radius as f32;
    edge_cfg.min_rays_with_ring = ((edge_cfg.n_rays as f32) * params.min_arc_coverage)
        .ceil()
        .max(6.0) as usize;
    edge_cfg.min_rays_with_ring = edge_cfg.min_rays_with_ring.min(edge_cfg.n_rays);

    let mut stats = CompletionStats {
        n_candidates_total: board_spec::n_markers(),
        ..Default::default()
    };

    let mut attempts: Option<Vec<CompletionAttemptRecord>> = if record_debug {
        Some(Vec::with_capacity(board_spec::n_markers()))
    } else {
        None
    };

    let mut attempted_fits = 0usize;

    for id in 0..board_spec::n_markers() {
        let projected_center = if let Some(xy) = board_spec::xy_mm(id) {
            project(h, xy[0] as f64, xy[1] as f64)
        } else {
            [f64::NAN, f64::NAN]
        };

        let proj_xy_f32 = [projected_center[0] as f32, projected_center[1] as f32];

        if present_ids.contains(&id) {
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::SkippedPresent,
                    reason: None,
                    reproj_err_px: None,
                    fit_confidence: None,
                    fit: None,
                });
            }
            continue;
        }

        if !projected_center[0].is_finite() || !projected_center[1].is_finite() {
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::SkippedOob,
                    reason: Some("projected_center_nan".to_string()),
                    reproj_err_px: None,
                    fit_confidence: None,
                    fit: None,
                });
            }
            continue;
        }

        // Keep away from boundaries so `sample_edges` cannot read out-of-bounds.
        if projected_center[0] < safe_margin
            || projected_center[0] >= (w_f - safe_margin)
            || projected_center[1] < safe_margin
            || projected_center[1] >= (h_f - safe_margin)
        {
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::SkippedOob,
                    reason: Some("projected_center_outside_safe_bounds".to_string()),
                    reproj_err_px: None,
                    fit_confidence: None,
                    fit: None,
                });
            }
            continue;
        }
        stats.n_in_image += 1;

        if let Some(max) = params.max_attempts {
            if attempted_fits >= max {
                break;
            }
        }
        attempted_fits += 1;
        stats.n_attempted += 1;

        let r_expected = median_outer_radius_from_neighbors_px(projected_center, markers, 12)
            .unwrap_or(marker_outer_radius_expected_px(config));

        // Robust local ring fit at the H-projected center.
        let fit_cand = match fit_outer_ellipse_robust_with_reason(
            gray,
            [projected_center[0] as f32, projected_center[1] as f32],
            r_expected,
            config,
            &edge_cfg,
            store_points_in_debug,
        ) {
            Ok(v) => v,
            Err(reason) => {
                stats.n_failed_fit += 1;
                if let Some(a) = attempts.as_mut() {
                    a.push(CompletionAttemptRecord {
                        id,
                        projected_center_xy: proj_xy_f32,
                        status: CompletionAttemptStatus::FailedFit,
                        reason: Some(reason),
                        reproj_err_px: None,
                        fit_confidence: None,
                        fit: None,
                    });
                }
                continue;
            }
        };
        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            outer_estimate,
            chosen_hypothesis,
            decode_result,
            ..
        } = fit_cand;

        let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
        let center = compute_center(&outer);
        let inlier_ratio = outer_ransac
            .as_ref()
            .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
            .unwrap_or(1.0);
        let fit_confidence = (arc_cov * inlier_ratio).clamp(0.0, 1.0);
        let mean_axis_new = ((outer.a + outer.b) * 0.5) as f32;
        let scale_ok = mean_axis_new.is_finite()
            && mean_axis_new >= (r_expected * 0.75)
            && mean_axis_new <= (r_expected * 1.33);

        let fit_dbg_pre_gates = if record_debug {
            Some(dbg::RingFitDebugV1 {
                center_xy_fit: [center[0] as f32, center[1] as f32],
                edges: dbg::RingEdgesDebugV1 {
                    n_angles_total: edge.n_total_rays,
                    n_angles_with_both: edge.n_good_rays,
                    inner_peak_r: if edge.inner_radii.is_empty() {
                        None
                    } else {
                        Some(edge.inner_radii.clone())
                    },
                    outer_peak_r: Some(edge.outer_radii.clone()),
                },
                outer_estimation: Some({
                    let chosen = outer_estimate.hypotheses.get(chosen_hypothesis);
                    dbg::OuterEstimationDebugV1 {
                        r_outer_expected_px: outer_estimate.r_outer_expected_px,
                        search_window_px: outer_estimate.search_window_px,
                        r_outer_found_px: chosen.map(|h| h.r_outer_px),
                        polarity: outer_estimate.polarity.map(|p| match p {
                            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                        }),
                        peak_strength: chosen.map(|h| h.peak_strength),
                        theta_consistency: chosen.map(|h| h.theta_consistency),
                        status: match outer_estimate.status {
                            OuterStatus::Ok => dbg::OuterEstimationStatusDebugV1::Ok,
                            OuterStatus::Rejected => dbg::OuterEstimationStatusDebugV1::Rejected,
                            OuterStatus::Failed => dbg::OuterEstimationStatusDebugV1::Failed,
                        },
                        reason: outer_estimate.reason.clone(),
                        hypotheses: outer_estimate
                            .hypotheses
                            .iter()
                            .map(|h| dbg::OuterHypothesisDebugV1 {
                                r_outer_px: h.r_outer_px,
                                peak_strength: h.peak_strength,
                                theta_consistency: h.theta_consistency,
                            })
                            .collect(),
                        chosen_hypothesis: Some(chosen_hypothesis),
                        radial_response_agg: outer_estimate.radial_response_agg.clone(),
                        r_samples: outer_estimate.r_samples.clone(),
                    }
                }),
                ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                    center_xy: [outer.cx as f32, outer.cy as f32],
                    semi_axes: [outer.a as f32, outer.b as f32],
                    angle: outer.angle as f32,
                }),
                ellipse_inner: None,
                inner_estimation: None,
                metrics: dbg::RingFitMetricsDebugV1 {
                    inlier_ratio_inner: None,
                    inlier_ratio_outer: Some(inlier_ratio),
                    mean_resid_inner: None,
                    mean_resid_outer: Some(rms_sampson_distance(&outer, &edge.outer_points) as f32),
                    arc_coverage: arc_cov,
                    valid_inner: false,
                    valid_outer: true,
                },
                points_outer: if store_points_in_debug {
                    Some(
                        edge.outer_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                points_inner: if store_points_in_debug {
                    Some(
                        edge.inner_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
            })
        } else {
            None
        };

        let reproj_err = {
            let dx = center[0] - projected_center[0];
            let dy = center[1] - projected_center[1];
            (dx * dx + dy * dy).sqrt() as f32
        };

        let mut added_reason: Option<String> = None;

        // Optional decode check: if decoding succeeds but disagrees with the expected ID,
        // it's likely we snapped to a neighboring marker.
        if let Some(ref d) = decode_result {
            if d.id != id {
                // If we are extremely consistent with H and the ring fit is strong, accept
                // by H only (do not attach mismatched decode fields).
                let accept_by_h = reproj_err <= (params.reproj_gate_px * 0.35).max(0.75)
                    && fit_confidence >= params.min_fit_confidence.max(0.60)
                    && arc_cov >= params.min_arc_coverage.max(0.45)
                    && scale_ok;
                if !accept_by_h {
                    stats.n_failed_gate += 1;
                    if let Some(a) = attempts.as_mut() {
                        a.push(CompletionAttemptRecord {
                            id,
                            projected_center_xy: proj_xy_f32,
                            status: CompletionAttemptStatus::FailedGate,
                            reason: Some(format!("decode_mismatch(expected={}, got={})", id, d.id)),
                            reproj_err_px: Some(reproj_err),
                            fit_confidence: Some(fit_confidence),
                            fit: fit_dbg_pre_gates.clone(),
                        });
                    }
                    continue;
                }
                added_reason = Some(format!(
                    "decode_mismatch_accepted(expected={}, got={})",
                    id, d.id
                ));
            }
        }

        // Gates: arc coverage, fit confidence, reprojection error.
        if arc_cov < params.min_arc_coverage {
            stats.n_failed_gate += 1;
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::FailedGate,
                    reason: Some(format!(
                        "arc_coverage({:.2}<{:.2})",
                        arc_cov, params.min_arc_coverage
                    )),
                    reproj_err_px: Some(reproj_err),
                    fit_confidence: Some(fit_confidence),
                    fit: fit_dbg_pre_gates.clone(),
                });
            }
            continue;
        }
        if fit_confidence < params.min_fit_confidence {
            stats.n_failed_gate += 1;
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::FailedGate,
                    reason: Some(format!(
                        "fit_confidence({:.2}<{:.2})",
                        fit_confidence, params.min_fit_confidence
                    )),
                    reproj_err_px: Some(reproj_err),
                    fit_confidence: Some(fit_confidence),
                    fit: fit_dbg_pre_gates.clone(),
                });
            }
            continue;
        }
        if (reproj_err as f64) > (params.reproj_gate_px as f64) {
            stats.n_failed_gate += 1;
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::FailedGate,
                    reason: Some(format!(
                        "reproj_err({:.2}>{:.2})",
                        reproj_err, params.reproj_gate_px
                    )),
                    reproj_err_px: Some(reproj_err),
                    fit_confidence: Some(fit_confidence),
                    fit: fit_dbg_pre_gates.clone(),
                });
            }
            continue;
        }
        if !scale_ok {
            stats.n_failed_gate += 1;
            if let Some(a) = attempts.as_mut() {
                a.push(CompletionAttemptRecord {
                    id,
                    projected_center_xy: proj_xy_f32,
                    status: CompletionAttemptStatus::FailedGate,
                    reason: Some(format!(
                        "scale_gate(mean_axis={:.2}, expected={:.2})",
                        mean_axis_new, r_expected
                    )),
                    reproj_err_px: Some(reproj_err),
                    fit_confidence: Some(fit_confidence),
                    fit: fit_dbg_pre_gates.clone(),
                });
            }
            continue;
        }

        // Inner estimation (only for accepted completions, unless we're recording debug).
        let inner_est = if record_debug || store_points_in_debug {
            Some(estimate_inner_scale_from_outer(
                gray,
                &outer,
                &config.marker_spec,
                store_points_in_debug,
            ))
        } else {
            Some(estimate_inner_scale_from_outer(
                gray,
                &outer,
                &config.marker_spec,
                false,
            ))
        };
        let inner_params = inner_est.as_ref().and_then(|est| {
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
        });

        // Build fit metrics and marker.
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

        let decode_metrics = decode_result
            .as_ref()
            .filter(|d| d.id == id)
            .map(|d| DecodeMetrics {
                observed_word: d.raw_word,
                best_id: d.id,
                best_rotation: d.rotation,
                best_dist: d.dist,
                margin: d.margin,
                decode_confidence: d.confidence,
            });

        let confidence = decode_metrics
            .as_ref()
            .map(|d| d.decode_confidence)
            .unwrap_or(fit_confidence);

        markers.push(DetectedMarker {
            id: Some(id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params.clone(),
            fit: fit.clone(),
            decode: decode_metrics,
        });

        stats.n_added += 1;
        tracing::debug!("Completion added id={} reproj_err={:.2}px", id, reproj_err);

        if let Some(a) = attempts.as_mut() {
            // Best-effort ring fit debug for manual inspection.
            let fit_dbg = if record_debug {
                let arc_cov_dbg = arc_cov;
                Some(dbg::RingFitDebugV1 {
                    center_xy_fit: [center[0] as f32, center[1] as f32],
                    edges: dbg::RingEdgesDebugV1 {
                        n_angles_total: edge.n_total_rays,
                        n_angles_with_both: edge.n_good_rays,
                        inner_peak_r: if edge.inner_radii.is_empty() {
                            None
                        } else {
                            Some(edge.inner_radii.clone())
                        },
                        outer_peak_r: Some(edge.outer_radii.clone()),
                    },
                    outer_estimation: Some({
                        let chosen = outer_estimate.hypotheses.get(chosen_hypothesis);
                        dbg::OuterEstimationDebugV1 {
                            r_outer_expected_px: outer_estimate.r_outer_expected_px,
                            search_window_px: outer_estimate.search_window_px,
                            r_outer_found_px: chosen.map(|h| h.r_outer_px),
                            polarity: outer_estimate.polarity.map(|p| match p {
                                Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                                Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                            }),
                            peak_strength: chosen.map(|h| h.peak_strength),
                            theta_consistency: chosen.map(|h| h.theta_consistency),
                            status: match outer_estimate.status {
                                OuterStatus::Ok => dbg::OuterEstimationStatusDebugV1::Ok,
                                OuterStatus::Rejected => {
                                    dbg::OuterEstimationStatusDebugV1::Rejected
                                }
                                OuterStatus::Failed => dbg::OuterEstimationStatusDebugV1::Failed,
                            },
                            reason: outer_estimate.reason.clone(),
                            hypotheses: outer_estimate
                                .hypotheses
                                .iter()
                                .map(|h| dbg::OuterHypothesisDebugV1 {
                                    r_outer_px: h.r_outer_px,
                                    peak_strength: h.peak_strength,
                                    theta_consistency: h.theta_consistency,
                                })
                                .collect(),
                            chosen_hypothesis: Some(chosen_hypothesis),
                            radial_response_agg: outer_estimate.radial_response_agg.clone(),
                            r_samples: outer_estimate.r_samples.clone(),
                        }
                    }),
                    ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                        center_xy: [outer.cx as f32, outer.cy as f32],
                        semi_axes: [outer.a as f32, outer.b as f32],
                        angle: outer.angle as f32,
                    }),
                    ellipse_inner: inner_params.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                        center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                        semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                        angle: p.angle as f32,
                    }),
                    inner_estimation: inner_est.as_ref().map(|est| dbg::InnerEstimationDebugV1 {
                        r_inner_expected: est.r_inner_expected,
                        search_window: est.search_window,
                        r_inner_found: est.r_inner_found,
                        polarity: est.polarity.map(|p| match p {
                            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                        }),
                        peak_strength: est.peak_strength,
                        theta_consistency: est.theta_consistency,
                        status: match est.status {
                            InnerStatus::Ok => dbg::InnerEstimationStatusDebugV1::Ok,
                            InnerStatus::Rejected => dbg::InnerEstimationStatusDebugV1::Rejected,
                            InnerStatus::Failed => dbg::InnerEstimationStatusDebugV1::Failed,
                        },
                        reason: est.reason.clone(),
                        radial_response_agg: est.radial_response_agg.clone(),
                        r_samples: est.r_samples.clone(),
                    }),
                    metrics: dbg::RingFitMetricsDebugV1 {
                        inlier_ratio_inner: None,
                        inlier_ratio_outer: fit.ransac_inlier_ratio_outer,
                        mean_resid_inner: None,
                        mean_resid_outer: fit.rms_residual_outer.map(|v| v as f32),
                        arc_coverage: arc_cov_dbg,
                        valid_inner: inner_params.is_some(),
                        valid_outer: true,
                    },
                    points_outer: if store_points_in_debug {
                        Some(
                            edge.outer_points
                                .iter()
                                .map(|p| [p[0] as f32, p[1] as f32])
                                .collect(),
                        )
                    } else {
                        None
                    },
                    points_inner: if store_points_in_debug {
                        Some(
                            edge.inner_points
                                .iter()
                                .map(|p| [p[0] as f32, p[1] as f32])
                                .collect(),
                        )
                    } else {
                        None
                    },
                })
            } else {
                None
            };

            a.push(CompletionAttemptRecord {
                id,
                projected_center_xy: proj_xy_f32,
                status: CompletionAttemptStatus::Added,
                reason: added_reason,
                reproj_err_px: Some(reproj_err),
                fit_confidence: Some(fit_confidence),
                fit: fit_dbg,
            });
        }
    }

    if stats.n_added > 0 {
        tracing::info!(
            "Completion: added {} markers (attempted {}, in_image {})",
            stats.n_added,
            stats.n_attempted,
            stats.n_in_image
        );
    } else {
        tracing::info!(
            "Completion: added 0 markers (attempted {}, in_image {})",
            stats.n_attempted,
            stats.n_in_image
        );
    }

    (stats, attempts)
}

/// Refit homography using refined marker centers, return (H_array, stats).
fn refit_homography(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> (Option<[[f64; 3]; 3]>, Option<RansacStats>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for m in markers {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src.push([xy[0] as f64, xy[1] as f64]);
                dst.push(m.center);
            }
        }
    }

    if src.len() < 4 {
        return (None, None);
    }

    // Use a light RANSAC (most outliers already removed)
    let light_config = RansacHomographyConfig {
        max_iters: 500,
        inlier_threshold: config.inlier_threshold,
        min_inliers: config.min_inliers,
        seed: config.seed + 1,
    };

    match homography::fit_homography_ransac(&src, &dst, &light_config) {
        Ok(result) => {
            let mut errors: Vec<f64> = result
                .inlier_mask
                .iter()
                .zip(&result.errors)
                .filter(|(&m, _)| m)
                .map(|(_, &e)| e)
                .collect();
            errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean_err = if errors.is_empty() {
                0.0
            } else {
                errors.iter().sum::<f64>() / errors.len() as f64
            };
            let p95_err = if errors.is_empty() {
                0.0
            } else {
                let idx = ((errors.len() as f64 * 0.95) as usize).min(errors.len() - 1);
                errors[idx]
            };

            let stats = RansacStats {
                n_candidates: src.len(),
                n_inliers: result.n_inliers,
                threshold_px: light_config.inlier_threshold,
                mean_err_px: mean_err,
                p95_err_px: p95_err,
            };

            (Some(matrix3_to_array(&result.h)), Some(stats))
        }
        Err(_) => (None, None),
    }
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

fn array_to_matrix3(m: &[[f64; 3]; 3]) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    )
}

fn mean_reproj_error_px(h: &nalgebra::Matrix3<f64>, markers: &[DetectedMarker]) -> f64 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for m in markers {
        let Some(id) = m.id else {
            continue;
        };
        let Some(xy) = board_spec::xy_mm(id) else {
            continue;
        };
        let err = homography::reprojection_error(
            h,
            &[xy[0] as f64, xy[1] as f64],
            &[m.center[0], m.center[1]],
        );
        if err.is_finite() {
            sum += err;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

fn compute_h_stats(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    thresh_px: f64,
) -> Option<RansacStats> {
    let mut errors: Vec<f64> = Vec::new();
    for m in markers {
        let Some(id) = m.id else {
            continue;
        };
        let Some(xy) = board_spec::xy_mm(id) else {
            continue;
        };
        let err = homography::reprojection_error(
            h,
            &[xy[0] as f64, xy[1] as f64],
            &[m.center[0], m.center[1]],
        );
        if err.is_finite() {
            errors.push(err);
        }
    }
    if errors.len() < 4 {
        return None;
    }

    let mut inlier_errors: Vec<f64> = errors.iter().copied().filter(|&e| e <= thresh_px).collect();
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    Some(RansacStats {
        n_candidates: errors.len(),
        n_inliers: inlier_errors.len(),
        threshold_px: thresh_px,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    })
}

fn refit_homography_matrix(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> Option<(nalgebra::Matrix3<f64>, RansacStats)> {
    let (h_arr, stats) = refit_homography(markers, config);
    match (h_arr, stats) {
        (Some(h_arr), Some(stats)) => Some((array_to_matrix3(&h_arr), stats)),
        _ => None,
    }
}

/// Remove duplicate detections: keep the highest-confidence marker within dedup_radius.
fn dedup_markers(mut markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    // Sort by confidence descending
    markers.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; markers.len()];
    let r2 = radius * radius;

    for i in 0..markers.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..markers.len() {
            if !keep[j] {
                continue;
            }
            let dx = markers[i].center[0] - markers[j].center[0];
            let dy = markers[i].center[1] - markers[j].center[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
            }
        }
    }

    markers
        .into_iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(m, _)| m)
        .collect()
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
