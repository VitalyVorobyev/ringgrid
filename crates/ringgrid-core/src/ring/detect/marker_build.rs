use crate::conic::{self, Ellipse};
use crate::ring::decode::DecodeResult;
use crate::ring::edge_sample::EdgeSampleResult;
use crate::ring::inner_estimate::InnerStatus;
use crate::{DecodeMetrics, DetectedMarker, EllipseParams, FitMetrics};

pub(crate) fn fit_metrics_from_outer(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&conic::RansacResult>,
) -> FitMetrics {
    FitMetrics {
        n_angles_total: edge.n_total_rays,
        n_angles_with_both_edges: edge.n_good_rays,
        n_points_outer: edge.outer_points.len(),
        n_points_inner: 0,
        ransac_inlier_ratio_outer: outer_ransac
            .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
        ransac_inlier_ratio_inner: None,
        rms_residual_outer: Some(conic::rms_sampson_distance(outer, &edge.outer_points)),
        rms_residual_inner: None,
    }
}

pub(crate) fn decode_metrics_from_result(
    decode_result: Option<&DecodeResult>,
) -> Option<DecodeMetrics> {
    decode_result.map(|d| DecodeMetrics {
        observed_word: d.raw_word,
        best_id: d.id,
        best_rotation: d.rotation,
        best_dist: d.dist,
        margin: d.margin,
        decode_confidence: d.confidence,
    })
}

pub(crate) fn inner_params_from_estimate(
    outer: &Ellipse,
    status: InnerStatus,
    r_inner_found: Option<f32>,
    r_inner_expected: f32,
) -> Option<EllipseParams> {
    if status != InnerStatus::Ok {
        return None;
    }
    let s = r_inner_found.unwrap_or(r_inner_expected) as f64;
    Some(EllipseParams {
        center_xy: [outer.cx, outer.cy],
        semi_axes: [outer.a * s, outer.b * s],
        angle: outer.angle,
    })
}

pub(crate) fn marker_with_defaults(
    id: Option<usize>,
    confidence: f32,
    center: [f64; 2],
    ellipse_outer: Option<EllipseParams>,
    ellipse_inner: Option<EllipseParams>,
    fit: FitMetrics,
    decode: Option<DecodeMetrics>,
) -> DetectedMarker {
    DetectedMarker {
        id,
        confidence,
        center,
        center_projective: None,
        vanishing_line: None,
        center_projective_residual: None,
        ellipse_outer,
        ellipse_inner,
        fit,
        decode,
    }
}
