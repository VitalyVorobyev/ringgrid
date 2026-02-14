use image::GrayImage;

use crate::conic::rms_sampson_distance;
use crate::conic::Ellipse;
use crate::debug_dump as dbg;
use crate::homography::homography_project as project;
use crate::ring::edge_sample::EdgeSampleResult;
use crate::ring::outer_estimate::OuterEstimate;
use crate::{DetectedMarker, FitMetrics};

use super::inner_fit::InnerFitResult;
use super::{
    compute_center, fit_outer_ellipse_robust_with_reason,
    marker_build::{
        decode_metrics_from_result, fit_metrics_with_inner, inner_ellipse_params,
        marker_with_defaults,
    },
    marker_outer_radius_expected_px, median_outer_radius_from_neighbors_px, CompletionParams,
    DetectConfig, OuterFitCandidate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionAttemptStatus {
    Added,
    SkippedPresent,
    SkippedOob,
    FailedFit,
    FailedGate,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompletionAttemptRecord {
    pub id: usize,
    pub projected_center_xy: [f32; 2],
    pub status: CompletionAttemptStatus,
    pub reason: Option<String>,
    pub reproj_err_px: Option<f32>,
    pub fit_confidence: Option<f32>,
    pub fit: Option<dbg::RingFitDebug>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CompletionStats {
    pub n_candidates_total: usize,
    pub n_in_image: usize,
    pub n_attempted: usize,
    pub n_added: usize,
    pub n_failed_fit: usize,
    pub n_failed_gate: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CompletionDebugOptions {
    pub(crate) store_points: bool,
    pub(crate) record: bool,
}

// ---------------------------------------------------------------------------
// Helper: quality metrics computed from a single outer-fit candidate
// ---------------------------------------------------------------------------

struct CandidateQuality {
    center: [f64; 2],
    arc_cov: f32,
    inlier_ratio: f32,
    fit_confidence: f32,
    mean_axis: f32,
    scale_ok: bool,
    reproj_err: f32,
}

fn compute_candidate_quality(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    projected_center: [f64; 2],
    r_expected: f32,
) -> CandidateQuality {
    let center = compute_center(outer);
    let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
    let inlier_ratio = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0);
    let fit_confidence = (arc_cov * inlier_ratio).clamp(0.0, 1.0);
    let mean_axis = ((outer.a + outer.b) * 0.5) as f32;
    let scale_ok = mean_axis.is_finite()
        && mean_axis >= (r_expected * 0.75)
        && mean_axis <= (r_expected * 1.33);
    let reproj_err = {
        let dx = center[0] - projected_center[0];
        let dy = center[1] - projected_center[1];
        (dx * dx + dy * dy).sqrt() as f32
    };
    CandidateQuality {
        center,
        arc_cov,
        inlier_ratio,
        fit_confidence,
        mean_axis,
        scale_ok,
        reproj_err,
    }
}

// ---------------------------------------------------------------------------
// Helper: decode-mismatch gate
// ---------------------------------------------------------------------------

/// Returns `Some(reason)` when decode disagrees with expected ID.
fn check_decode_gate(
    decode_result: Option<&crate::marker::decode::DecodeResult>,
    expected_id: usize,
) -> Option<String> {
    if let Some(d) = decode_result {
        if d.id != expected_id {
            return Some(format!(
                "decode_mismatch_accepted(expected={}, got={})",
                expected_id, d.id
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helper: quality gates (arc coverage, fit confidence, reproj, scale)
// ---------------------------------------------------------------------------

fn check_quality_gates(
    quality: &CandidateQuality,
    params: &CompletionParams,
    r_expected: f32,
) -> Result<(), String> {
    if quality.arc_cov < params.min_arc_coverage {
        return Err(format!(
            "arc_coverage({:.2}<{:.2})",
            quality.arc_cov, params.min_arc_coverage
        ));
    }
    if quality.fit_confidence < params.min_fit_confidence {
        return Err(format!(
            "fit_confidence({:.2}<{:.2})",
            quality.fit_confidence, params.min_fit_confidence
        ));
    }
    if (quality.reproj_err as f64) > (params.reproj_gate_px as f64) {
        return Err(format!(
            "reproj_err({:.2}>{:.2})",
            quality.reproj_err, params.reproj_gate_px
        ));
    }
    if !quality.scale_ok {
        return Err(format!(
            "scale_gate(mean_axis={:.2}, expected={:.2})",
            quality.mean_axis, r_expected
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: debug record constructors
// ---------------------------------------------------------------------------

fn make_attempt_record(
    id: usize,
    proj_xy: [f32; 2],
    status: CompletionAttemptStatus,
    reason: Option<String>,
    quality: Option<&CandidateQuality>,
    fit: Option<dbg::RingFitDebug>,
) -> CompletionAttemptRecord {
    CompletionAttemptRecord {
        id,
        projected_center_xy: proj_xy,
        status,
        reason,
        reproj_err_px: quality.map(|q| q.reproj_err),
        fit_confidence: quality.map(|q| q.fit_confidence),
        fit,
    }
}

fn edge_for_debug(edge: &EdgeSampleResult, store_points: bool) -> EdgeSampleResult {
    if store_points {
        return edge.clone();
    }
    let mut out = edge.clone();
    out.outer_points.clear();
    out.inner_points.clear();
    out
}

struct FitDebugContext<'a> {
    edge: &'a EdgeSampleResult,
    outer: &'a Ellipse,
    outer_estimate: &'a OuterEstimate,
    chosen_hypothesis: usize,
    quality: &'a CandidateQuality,
    store_points: bool,
}

fn build_pre_gate_fit_debug(ctx: &FitDebugContext<'_>) -> dbg::RingFitDebug {
    dbg::RingFitDebug {
        center_xy_fit: [ctx.quality.center[0], ctx.quality.center[1]],
        edge: edge_for_debug(ctx.edge, ctx.store_points),
        outer_estimation: Some(ctx.outer_estimate.clone()),
        chosen_outer_hypothesis: Some(ctx.chosen_hypothesis),
        ellipse_outer: Some(*ctx.outer),
        ellipse_inner: None,
        inner_estimation: None,
        fit: FitMetrics {
            n_angles_total: ctx.edge.n_total_rays,
            n_angles_with_both_edges: ctx.edge.n_good_rays,
            n_points_outer: ctx.edge.outer_points.len(),
            n_points_inner: ctx.edge.inner_points.len(),
            ransac_inlier_ratio_outer: Some(ctx.quality.inlier_ratio),
            ransac_inlier_ratio_inner: None,
            rms_residual_outer: Some(rms_sampson_distance(ctx.outer, &ctx.edge.outer_points)),
            rms_residual_inner: None,
        },
        inner_points_fit: None,
    }
}

fn build_success_fit_debug(
    ctx: &FitDebugContext<'_>,
    inner_fit: &InnerFitResult,
    inner_params: Option<&Ellipse>,
    fit: &crate::FitMetrics,
) -> dbg::RingFitDebug {
    dbg::RingFitDebug {
        center_xy_fit: [ctx.quality.center[0], ctx.quality.center[1]],
        edge: edge_for_debug(ctx.edge, ctx.store_points),
        outer_estimation: Some(ctx.outer_estimate.clone()),
        chosen_outer_hypothesis: Some(ctx.chosen_hypothesis),
        ellipse_outer: Some(*ctx.outer),
        ellipse_inner: inner_params.cloned(),
        inner_estimation: Some(inner_fit.estimate.clone()),
        fit: fit.clone(),
        inner_points_fit: if ctx.store_points {
            Some(inner_fit.points_inner.clone())
        } else {
            None
        },
    }
}

// ---------------------------------------------------------------------------
// Main completion entry point
// ---------------------------------------------------------------------------

/// Try to complete missing IDs using a fitted homography.
///
/// This is intentionally conservative: it only runs when H exists and rejects
/// any fit that deviates from the H-projected center by more than a tight gate.
pub(crate) fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<DetectedMarker>,
    config: &DetectConfig,
    board: &crate::board_layout::BoardLayout,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
    debug: CompletionDebugOptions,
) -> (CompletionStats, Option<Vec<CompletionAttemptRecord>>) {
    use std::collections::HashSet;

    let params = &config.completion;
    let inner_fit_cfg = super::inner_fit::InnerFitConfig::default();
    if !params.enable {
        return (
            CompletionStats::default(),
            if debug.record { Some(Vec::new()) } else { None },
        );
    }

    let (w, h_img) = gray.dimensions();
    let w_f = w as f64;
    let h_f = h_img as f64;

    let roi_radius = params.roi_radius_px.clamp(8.0, 200.0) as f64;
    let safe_margin = roi_radius + params.image_margin_px.max(0.0) as f64;

    let present_ids: HashSet<usize> = markers.iter().filter_map(|m| m.id).collect();

    let mut edge_cfg = config.edge_sample.clone();
    edge_cfg.r_max = roi_radius as f32;
    edge_cfg.min_rays_with_ring = ((edge_cfg.n_rays as f32) * params.min_arc_coverage)
        .ceil()
        .max(6.0) as usize;
    edge_cfg.min_rays_with_ring = edge_cfg.min_rays_with_ring.min(edge_cfg.n_rays);

    let mut stats = CompletionStats {
        n_candidates_total: board.n_markers(),
        ..Default::default()
    };
    let mut attempts: Option<Vec<CompletionAttemptRecord>> = if debug.record {
        Some(Vec::with_capacity(board.n_markers()))
    } else {
        None
    };
    let mut attempted_fits = 0usize;

    for id in board.marker_ids() {
        let projected_center = match board.xy_mm(id) {
            Some(xy) => project(h, xy[0] as f64, xy[1] as f64),
            None => continue,
        };
        let proj_xy_f32 = [projected_center[0] as f32, projected_center[1] as f32];

        // Skip already-detected IDs.
        if present_ids.contains(&id) {
            if let Some(a) = attempts.as_mut() {
                a.push(make_attempt_record(
                    id,
                    proj_xy_f32,
                    CompletionAttemptStatus::SkippedPresent,
                    None,
                    None,
                    None,
                ));
            }
            continue;
        }

        // Skip non-finite or out-of-bounds projections.
        if !projected_center[0].is_finite() || !projected_center[1].is_finite() {
            if let Some(a) = attempts.as_mut() {
                a.push(make_attempt_record(
                    id,
                    proj_xy_f32,
                    CompletionAttemptStatus::SkippedOob,
                    Some("projected_center_nan".to_string()),
                    None,
                    None,
                ));
            }
            continue;
        }
        if projected_center[0] < safe_margin
            || projected_center[0] >= (w_f - safe_margin)
            || projected_center[1] < safe_margin
            || projected_center[1] >= (h_f - safe_margin)
        {
            if let Some(a) = attempts.as_mut() {
                a.push(make_attempt_record(
                    id,
                    proj_xy_f32,
                    CompletionAttemptStatus::SkippedOob,
                    Some("projected_center_outside_safe_bounds".to_string()),
                    None,
                    None,
                ));
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
            mapper,
            &edge_cfg,
            debug.store_points,
        ) {
            Ok(v) => v,
            Err(reason) => {
                stats.n_failed_fit += 1;
                if let Some(a) = attempts.as_mut() {
                    a.push(make_attempt_record(
                        id,
                        proj_xy_f32,
                        CompletionAttemptStatus::FailedFit,
                        Some(reason),
                        None,
                        None,
                    ));
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

        let quality = compute_candidate_quality(
            &edge,
            &outer,
            outer_ransac.as_ref(),
            projected_center,
            r_expected,
        );
        let fit_dbg_ctx = FitDebugContext {
            edge: &edge,
            outer: &outer,
            outer_estimate: &outer_estimate,
            chosen_hypothesis,
            quality: &quality,
            store_points: debug.store_points,
        };
        let fit_dbg_pre = if debug.record {
            Some(build_pre_gate_fit_debug(&fit_dbg_ctx))
        } else {
            None
        };

        // Quality gates.
        if let Err(reason) = check_quality_gates(&quality, params, r_expected) {
            stats.n_failed_gate += 1;
            if let Some(a) = attempts.as_mut() {
                a.push(make_attempt_record(
                    id,
                    proj_xy_f32,
                    CompletionAttemptStatus::FailedGate,
                    Some(reason),
                    Some(&quality),
                    fit_dbg_pre.clone(),
                ));
            }
            continue;
        }

        // Decode-mismatch note (quality-gate order is now uniform).
        let added_reason = check_decode_gate(decode_result.as_ref(), id);

        // Inner fit + marker construction.
        let inner_fit = super::inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            mapper,
            &inner_fit_cfg,
            debug.record || debug.store_points,
        );
        let inner_params = inner_ellipse_params(&inner_fit);
        let fit = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner_fit);
        let decode_metrics =
            decode_metrics_from_result(decode_result.as_ref().filter(|d| d.id == id));
        let confidence = decode_metrics
            .as_ref()
            .map(|d| d.decode_confidence)
            .unwrap_or(quality.fit_confidence);

        markers.push(marker_with_defaults(
            Some(id),
            confidence,
            quality.center,
            Some(outer),
            inner_params,
            Some(edge.outer_points.clone()),
            Some(inner_fit.points_inner.clone()),
            fit.clone(),
            decode_metrics,
        ));
        stats.n_added += 1;
        tracing::debug!(
            "Completion added id={} reproj_err={:.2}px",
            id,
            quality.reproj_err
        );

        if let Some(a) = attempts.as_mut() {
            let fit_dbg = if debug.record {
                Some(build_success_fit_debug(
                    &fit_dbg_ctx,
                    &inner_fit,
                    inner_params.as_ref(),
                    &fit,
                ))
            } else {
                None
            };
            a.push(make_attempt_record(
                id,
                proj_xy_f32,
                CompletionAttemptStatus::Added,
                added_reason,
                Some(&quality),
                fit_dbg,
            ));
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
