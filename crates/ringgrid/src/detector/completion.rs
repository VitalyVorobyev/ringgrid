use image::GrayImage;

use crate::conic::Ellipse;
use crate::homography::homography_project as project;
use crate::ring::edge_sample::EdgeSampleResult;
use crate::DetectedMarker;

use super::{
    fit_outer_candidate_from_prior_for_completion,
    marker_build::{decode_metrics_from_result, fit_metrics_with_inner},
    median_outer_radius_from_neighbors_px, CompletionParams,
    DetectConfig, OuterFitCandidate,
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

        if check_quality_gates(&quality, params, r_expected).is_err() {
            stats.n_failed_gate += 1;
            continue;
        }

        if let Some(reason) = check_decode_gate(decode_result.as_ref(), id) {
            tracing::debug!("Completion id={} {}", id, reason);
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
