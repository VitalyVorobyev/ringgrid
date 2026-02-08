use image::GrayImage;

use crate::conic::rms_sampson_distance;
use crate::debug_dump as dbg;
use crate::homography::project;
use crate::ring::inner_estimate::{InnerStatus, Polarity};
use crate::ring::outer_estimate::OuterStatus;
use crate::DetectedMarker;

use super::{
    compute_center, debug_conv, fit_outer_ellipse_robust_with_reason,
    marker_build::{
        decode_metrics_from_result, fit_metrics_with_inner, inner_ellipse_params,
        marker_with_defaults,
    },
    marker_outer_radius_expected_px, median_outer_radius_from_neighbors_px, DetectConfig,
    OuterFitCandidate,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CompletionAttemptStatus {
    Added,
    SkippedPresent,
    SkippedOob,
    FailedFit,
    FailedGate,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct CompletionAttemptRecord {
    pub(super) id: usize,
    pub(super) projected_center_xy: [f32; 2],
    pub(super) status: CompletionAttemptStatus,
    pub(super) reason: Option<String>,
    pub(super) reproj_err_px: Option<f32>,
    pub(super) fit_confidence: Option<f32>,
    pub(super) fit: Option<dbg::RingFitDebugV1>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(super) struct CompletionStats {
    pub(super) n_candidates_total: usize,
    pub(super) n_in_image: usize,
    pub(super) n_attempted: usize,
    pub(super) n_added: usize,
    pub(super) n_failed_fit: usize,
    pub(super) n_failed_gate: usize,
}

/// Try to complete missing IDs using a fitted homography.
///
/// This is intentionally conservative: it only runs when H exists and rejects
/// any fit that deviates from the H-projected center by more than a tight gate.
pub(super) fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<DetectedMarker>,
    config: &DetectConfig,
    board: &crate::board_layout::BoardLayout,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    store_points_in_debug: bool,
    record_debug: bool,
) -> (CompletionStats, Option<Vec<CompletionAttemptRecord>>) {
    use std::collections::HashSet;

    let params = &config.completion;
    let inner_fit_cfg = super::inner_fit::InnerFitConfig::default();
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
        n_candidates_total: board.n_markers(),
        ..Default::default()
    };

    let mut attempts: Option<Vec<CompletionAttemptRecord>> = if record_debug {
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
            mapper,
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
                ellipse_outer: Some(debug_conv::ellipse_from_conic(&outer)),
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

        let inner_fit = super::inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            mapper,
            &inner_fit_cfg,
            record_debug || store_points_in_debug,
        );
        let inner_params = inner_ellipse_params(&inner_fit);

        // Build fit metrics and marker.
        let fit = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner_fit);

        let decode_metrics =
            decode_metrics_from_result(decode_result.as_ref().filter(|d| d.id == id));

        let confidence = decode_metrics
            .as_ref()
            .map(|d| d.decode_confidence)
            .unwrap_or(fit_confidence);

        markers.push(marker_with_defaults(
            Some(id),
            confidence,
            center,
            Some(crate::EllipseParams::from(&outer)),
            inner_params.clone(),
            fit.clone(),
            decode_metrics,
        ));

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
                    ellipse_outer: Some(debug_conv::ellipse_from_conic(&outer)),
                    ellipse_inner: inner_params.as_ref().map(debug_conv::ellipse_from_params),
                    inner_estimation: Some(dbg::InnerEstimationDebugV1 {
                        r_inner_expected: inner_fit.estimate.r_inner_expected,
                        search_window: inner_fit.estimate.search_window,
                        r_inner_found: inner_fit.estimate.r_inner_found,
                        polarity: inner_fit.estimate.polarity.map(|p| match p {
                            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                        }),
                        peak_strength: inner_fit.estimate.peak_strength,
                        theta_consistency: inner_fit.estimate.theta_consistency,
                        status: match inner_fit.estimate.status {
                            InnerStatus::Ok => dbg::InnerEstimationStatusDebugV1::Ok,
                            InnerStatus::Rejected => dbg::InnerEstimationStatusDebugV1::Rejected,
                            InnerStatus::Failed => dbg::InnerEstimationStatusDebugV1::Failed,
                        },
                        reason: inner_fit.estimate.reason.clone(),
                        radial_response_agg: inner_fit.estimate.radial_response_agg.clone(),
                        r_samples: inner_fit.estimate.r_samples.clone(),
                    }),
                    metrics: debug_conv::ring_fit_metrics(
                        &fit,
                        arc_cov_dbg,
                        inner_params.is_some(),
                        true,
                    ),
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
                            inner_fit
                                .points_inner
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
