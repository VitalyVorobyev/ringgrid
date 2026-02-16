use crate::conic::{self, Ellipse};
use crate::marker::decode::DecodeResult;
use crate::marker::DecodeMetrics;
use crate::ring::edge_sample::EdgeSampleResult;

use super::inner_fit::InnerFitResult;

/// Fit quality metrics for a detected marker.
///
/// Reports the edge sampling and ellipse fit quality. High RANSAC inlier
/// ratios (> 0.8) and low RMS Sampson residuals (< 0.5 px) indicate a
/// precise ellipse fit.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FitMetrics {
    /// Total number of radial rays cast.
    pub n_angles_total: usize,
    /// Number of rays where both inner and outer ring edges were found.
    pub n_angles_with_both_edges: usize,
    /// Number of outer edge points used for ellipse fit.
    pub n_points_outer: usize,
    /// Number of inner edge points used for ellipse fit.
    pub n_points_inner: usize,
    /// RANSAC inlier ratio for outer ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac_inlier_ratio_outer: Option<f32>,
    /// RANSAC inlier ratio for inner ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac_inlier_ratio_inner: Option<f32>,
    /// RMS Sampson residual for outer ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_residual_outer: Option<f64>,
    /// RMS Sampson residual for inner ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_residual_inner: Option<f64>,
}

/// A detected marker with its refined center and optional ID.
///
/// The `center` field is always in image-pixel coordinates, regardless of
/// whether a [`PixelMapper`](crate::PixelMapper) was used. When a mapper is
/// active, `center_mapped` provides the working-frame (undistorted)
/// coordinates. Ellipses are in the working frame when a mapper is active.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DetectedMarker {
    /// Decoded marker ID (codebook index), or None if decoding was rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    /// Combined detection + decode confidence in [0, 1].
    pub confidence: f32,
    /// Marker center in raw image pixel coordinates.
    ///
    /// This field is always image-space, independent of mapper usage.
    pub center: [f64; 2],
    /// Marker center in mapper working coordinates, when a mapper is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_mapped: Option<[f64; 2]>,
    /// Outer ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<Ellipse>,
    /// Inner ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<Ellipse>,
    /// Raw sub-pixel outer edge inlier points used for ellipse fitting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_points_outer: Option<Vec<[f64; 2]>>,
    /// Raw sub-pixel inner edge inlier points used for ellipse fitting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_points_inner: Option<Vec<[f64; 2]>>,
    /// Fit quality metrics.
    pub fit: FitMetrics,
    /// Decode metrics (present if decoding was attempted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeMetrics>,
}

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
