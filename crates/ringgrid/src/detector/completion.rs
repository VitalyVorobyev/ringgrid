use image::GrayImage;

use crate::conic::Ellipse;
use crate::homography::homography_project as project;
use crate::marker::codebook::CODEBOOK_MIN_CYCLIC_DIST;
use crate::ring::edge_sample::EdgeSampleResult;
use crate::DetectedMarker;

use super::{
    fit_outer_candidate_from_prior_for_completion,
    marker_build::{decode_metrics_from_result, fit_metrics_with_inner},
    median_outer_radius_from_neighbors_px, CompletionParams, DetectConfig, OuterFitCandidate,
};

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CompletionStats {
    pub n_candidates_total: usize,
    pub n_in_image: usize,
    pub n_attempted: usize,
    pub n_added: usize,
    pub n_failed_fit: usize,
    pub n_failed_gate: usize,
}

struct CandidateQuality {
    center: [f64; 2],
    arc_cov: f32,
    fit_confidence: f32,
    mean_axis: f32,
    scale_ok: bool,
    reproj_err: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompletionGateRejectReason {
    ArcCoverageLow,
    FitConfidenceLow,
    ReprojectionTooHigh,
    ScaleOutOfRange,
    PerfectDecodeRequired,
}

impl CompletionGateRejectReason {
    const fn code(self) -> &'static str {
        match self {
            Self::ArcCoverageLow => "arc_coverage_low",
            Self::FitConfidenceLow => "fit_confidence_low",
            Self::ReprojectionTooHigh => "reprojection_too_high",
            Self::ScaleOutOfRange => "scale_out_of_range",
            Self::PerfectDecodeRequired => "perfect_decode_required",
        }
    }
}

impl std::fmt::Display for CompletionGateRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CompletionGateRejectContext {
    ArcCoverageLow {
        observed_arc_coverage: f32,
        min_required_arc_coverage: f32,
    },
    FitConfidenceLow {
        observed_fit_confidence: f32,
        min_required_fit_confidence: f32,
    },
    ReprojectionTooHigh {
        observed_reproj_error_px: f32,
        max_allowed_reproj_error_px: f32,
    },
    ScaleOutOfRange {
        observed_mean_axis_px: f32,
        expected_radius_px: f32,
        min_allowed_axis_px: f32,
        max_allowed_axis_px: f32,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CompletionGateReject {
    reason: CompletionGateRejectReason,
    context: CompletionGateRejectContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompletionDecodeNoticeReason {
    DecodeMismatchAccepted,
}

impl CompletionDecodeNoticeReason {
    const fn code(self) -> &'static str {
        match self {
            Self::DecodeMismatchAccepted => "decode_mismatch_accepted",
        }
    }
}

impl std::fmt::Display for CompletionDecodeNoticeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CompletionDecodeNotice {
    reason: CompletionDecodeNoticeReason,
    expected_id: usize,
    observed_id: usize,
}

fn compute_candidate_quality(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    projected_center: [f64; 2],
    r_expected: f32,
) -> CandidateQuality {
    let center = outer.center();
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
        fit_confidence,
        mean_axis,
        scale_ok,
        reproj_err,
    }
}

fn check_decode_gate(
    decode_result: Option<&crate::marker::decode::DecodeResult>,
    expected_id: usize,
) -> Option<CompletionDecodeNotice> {
    if let Some(d) = decode_result {
        if d.id != expected_id {
            return Some(CompletionDecodeNotice {
                reason: CompletionDecodeNoticeReason::DecodeMismatchAccepted,
                expected_id,
                observed_id: d.id,
            });
        }
    }
    None
}

fn check_quality_gates(
    quality: &CandidateQuality,
    params: &CompletionParams,
    r_expected: f32,
) -> Result<(), CompletionGateReject> {
    if quality.arc_cov < params.min_arc_coverage {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::ArcCoverageLow,
            context: CompletionGateRejectContext::ArcCoverageLow {
                observed_arc_coverage: quality.arc_cov,
                min_required_arc_coverage: params.min_arc_coverage,
            },
        });
    }
    if quality.fit_confidence < params.min_fit_confidence {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::FitConfidenceLow,
            context: CompletionGateRejectContext::FitConfidenceLow {
                observed_fit_confidence: quality.fit_confidence,
                min_required_fit_confidence: params.min_fit_confidence,
            },
        });
    }
    if (quality.reproj_err as f64) > (params.reproj_gate_px as f64) {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::ReprojectionTooHigh,
            context: CompletionGateRejectContext::ReprojectionTooHigh {
                observed_reproj_error_px: quality.reproj_err,
                max_allowed_reproj_error_px: params.reproj_gate_px,
            },
        });
    }
    if !quality.scale_ok {
        let min_allowed_axis_px = r_expected * 0.75;
        let max_allowed_axis_px = r_expected * 1.33;
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::ScaleOutOfRange,
            context: CompletionGateRejectContext::ScaleOutOfRange {
                observed_mean_axis_px: quality.mean_axis,
                expected_radius_px: r_expected,
                min_allowed_axis_px,
                max_allowed_axis_px,
            },
        });
    }
    Ok(())
}

pub(crate) fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<DetectedMarker>,
    config: &DetectConfig,
    board: &crate::board_layout::BoardLayout,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
) -> CompletionStats {
    use std::collections::HashSet;

    let params = &config.completion;
    if !params.enable {
        return CompletionStats::default();
    }

    let (w, h_img) = gray.dimensions();
    let w_f = w as f64;
    let h_f = h_img as f64;

    let roi_radius = params.roi_radius_px.clamp(8.0, 200.0) as f64;
    let safe_margin = roi_radius + params.image_margin_px.max(0.0) as f64;

    let present_ids: HashSet<usize> = markers.iter().filter_map(|m| m.id).collect();

    let mut stats = CompletionStats {
        n_candidates_total: board.n_markers(),
        ..Default::default()
    };
    let mut attempted_fits = 0usize;

    for id in board.marker_ids() {
        let projected_center = match board.xy_mm(id) {
            Some(xy) => project(h, xy[0] as f64, xy[1] as f64),
            None => continue,
        };

        if present_ids.contains(&id) {
            continue;
        }

        if !projected_center[0].is_finite() || !projected_center[1].is_finite() {
            continue;
        }
        if projected_center[0] < safe_margin
            || projected_center[0] >= (w_f - safe_margin)
            || projected_center[1] < safe_margin
            || projected_center[1] >= (h_f - safe_margin)
        {
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
            .unwrap_or(config.marker_scale.nominal_outer_radius_px());

        let fit_cand = match fit_outer_candidate_from_prior_for_completion(
            gray,
            [projected_center[0] as f32, projected_center[1] as f32],
            r_expected,
            config,
            mapper,
        ) {
            Ok(v) => v,
            Err(_) => {
                stats.n_failed_fit += 1;
                continue;
            }
        };
        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
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

        if let Err(reject) = check_quality_gates(&quality, params, r_expected) {
            tracing::trace!(
                "Completion id={} gate_reject={} context={:?}",
                id,
                reject.reason,
                reject.context
            );
            stats.n_failed_gate += 1;
            continue;
        }

        // Optional gate: require dist=0 and margin >= codebook min distance.
        // Recommended when homography accuracy is low (no calibrated camera model).
        if params.require_perfect_decode {
            let is_perfect = decode_result.as_ref().is_some_and(|d| {
                d.dist == 0 && d.margin >= CODEBOOK_MIN_CYCLIC_DIST as u8
            });
            if !is_perfect {
                tracing::trace!(
                    "Completion id={} gate_reject={} (dist={:?}, margin={:?})",
                    id,
                    CompletionGateRejectReason::PerfectDecodeRequired.code(),
                    decode_result.as_ref().map(|d| d.dist),
                    decode_result.as_ref().map(|d| d.margin),
                );
                stats.n_failed_gate += 1;
                continue;
            }
        }

        if let Some(notice) = check_decode_gate(decode_result.as_ref(), id) {
            tracing::debug!(
                "Completion id={} {} expected={} observed={}",
                id,
                notice.reason,
                notice.expected_id,
                notice.observed_id
            );
        }

        let inner_fit = super::inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            mapper,
            &config.inner_fit,
            false,
        );
        let fit = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner_fit);
        let decode_metrics =
            decode_metrics_from_result(decode_result.as_ref().filter(|d| d.id == id));
        let confidence = decode_metrics
            .as_ref()
            .map(|d| d.decode_confidence)
            .unwrap_or(quality.fit_confidence);

        markers.push(DetectedMarker {
            id: Some(id),
            confidence,
            center: quality.center,
            ellipse_outer: Some(outer),
            ellipse_inner: inner_fit.ellipse_inner,
            edge_points_outer: Some(edge.outer_points.clone()),
            edge_points_inner: Some(inner_fit.points_inner.clone()),
            fit,
            decode: decode_metrics,
            ..DetectedMarker::default()
        });
        stats.n_added += 1;
        tracing::debug!(
            "Completion added id={} reproj_err={:.2}px",
            id,
            quality.reproj_err
        );
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

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_gate_reason_serialization_is_stable() {
        let reason = CompletionGateRejectReason::ScaleOutOfRange;
        assert_eq!(reason.to_string(), "scale_out_of_range");
        let json = serde_json::to_string(&reason).expect("serialize completion gate reason");
        assert_eq!(json, "\"scale_out_of_range\"");
    }

    #[test]
    fn completion_quality_gate_reports_typed_arc_coverage_context() {
        let quality = CandidateQuality {
            center: [0.0, 0.0],
            arc_cov: 0.2,
            fit_confidence: 0.9,
            mean_axis: 20.0,
            scale_ok: true,
            reproj_err: 0.5,
        };
        let params = CompletionParams::default();
        let reject = check_quality_gates(&quality, &params, 20.0).expect_err("expected gate fail");
        assert_eq!(reject.reason, CompletionGateRejectReason::ArcCoverageLow);
        match reject.context {
            CompletionGateRejectContext::ArcCoverageLow {
                observed_arc_coverage,
                min_required_arc_coverage,
            } => {
                assert!(observed_arc_coverage < min_required_arc_coverage);
            }
            other => panic!("unexpected completion gate context: {other:?}"),
        }
    }
}
