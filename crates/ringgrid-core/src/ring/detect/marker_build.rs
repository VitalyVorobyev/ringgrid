use crate::conic::{self, Ellipse};
use crate::ring::decode::DecodeResult;
use crate::ring::edge_sample::EdgeSampleResult;
use crate::{DecodeMetrics, DetectedMarker, EllipseParams, FitMetrics};

use super::inner_fit::InnerFitResult;

fn fit_metrics_from_outer(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&conic::RansacResult>,
    n_points_inner: usize,
    ransac_inlier_ratio_inner: Option<f32>,
    rms_residual_inner: Option<f64>,
) -> FitMetrics {
    FitMetrics {
        n_angles_total: edge.n_total_rays,
        n_angles_with_both_edges: edge.n_good_rays,
        n_points_outer: edge.outer_points.len(),
        n_points_inner,
        ransac_inlier_ratio_outer: outer_ransac
            .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
        ransac_inlier_ratio_inner,
        rms_residual_outer: Some(conic::rms_sampson_distance(outer, &edge.outer_points)),
        rms_residual_inner,
    }
}

/// Build fit metrics from outer fit + inner fit result, avoiding repeated
/// field extraction at each call site.
pub(crate) fn fit_metrics_with_inner(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&conic::RansacResult>,
    inner: &InnerFitResult,
) -> FitMetrics {
    fit_metrics_from_outer(
        edge,
        outer,
        outer_ransac,
        inner.points_inner.len(),
        inner.ransac_inlier_ratio_inner,
        inner.rms_residual_inner,
    )
}

/// Extract inner ellipse params from an inner fit result.
pub(crate) fn inner_ellipse_params(inner: &InnerFitResult) -> Option<EllipseParams> {
    inner.ellipse_inner.as_ref().map(EllipseParams::from)
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
