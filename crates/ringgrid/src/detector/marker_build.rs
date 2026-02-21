use crate::conic::{self, Ellipse};
use crate::marker::decode::DecodeResult;
use crate::marker::DecodeMetrics;
use crate::ring::edge_sample::EdgeSampleResult;

use super::config::InnerFitConfig;
use super::inner_fit::{InnerFitReason, InnerFitResult, InnerFitStatus};

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
    /// Maximum angular gap (radians) between consecutive outer edge points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angular_gap_outer: Option<f64>,
    /// Maximum angular gap (radians) between consecutive inner edge points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angular_gap_inner: Option<f64>,
    /// Inner fit outcome: `"ok"`, `"rejected"`, or `"failed"`. Absent when fit
    /// succeeded without issue.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_fit_status: Option<InnerFitStatus>,
    /// Inner fit rejection reason code. Present only when `inner_fit_status` is
    /// `"rejected"` or `"failed"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_fit_reason: Option<InnerFitReason>,
    /// Ratio of this marker's outer radius to the median outer radius of its
    /// k nearest decoded neighbors. Values well below 1.0 (< 0.75) indicate a
    /// potential inner-as-outer substitution. Populated in the finalization stage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neighbor_radius_ratio: Option<f32>,
}

/// A detected marker with its refined center and optional ID.
///
/// The `center` field is always in image-pixel coordinates, regardless of
/// whether a [`PixelMapper`](crate::PixelMapper) was used. When a mapper is
/// active, `center_mapped` provides the working-frame (undistorted)
/// coordinates. `board_xy_mm` provides board-space marker coordinates in
/// millimeters when the decoded `id` is valid for the active [`BoardLayout`](crate::BoardLayout).
/// Ellipses are in the working frame when a mapper is active.
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
    /// Marker center on the physical board in millimeters `[x_mm, y_mm]`.
    ///
    /// Populated when `id` is present and valid for the active board layout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub board_xy_mm: Option<[f64; 2]>,
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

#[allow(clippy::too_many_arguments)]
fn fit_metrics_from_outer(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&conic::RansacResult>,
    n_points_inner: usize,
    ransac_inlier_ratio_inner: Option<f32>,
    rms_residual_inner: Option<f64>,
    max_angular_gap_inner: Option<f64>,
    inner_fit_status: Option<InnerFitStatus>,
    inner_fit_reason: Option<InnerFitReason>,
) -> FitMetrics {
    use super::outer_fit::max_angular_gap;
    let gap_outer = if edge.outer_points.is_empty() {
        None
    } else {
        Some(max_angular_gap(outer.center(), &edge.outer_points))
    };
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
        max_angular_gap_outer: gap_outer,
        max_angular_gap_inner,
        inner_fit_status,
        inner_fit_reason,
        neighbor_radius_ratio: None,
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
    let inner_fit_status = Some(inner.status);
    let inner_fit_reason = if inner.status == InnerFitStatus::Ok {
        None
    } else {
        inner.reason
    };
    fit_metrics_from_outer(
        edge,
        outer,
        outer_ransac,
        inner.points_inner.len(),
        inner.ransac_inlier_ratio_inner,
        inner.rms_residual_inner,
        inner.max_angular_gap,
        inner_fit_status,
        inner_fit_reason,
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

fn fallback_fit_confidence(
    edge: &EdgeSampleResult,
    outer_ransac: Option<&conic::RansacResult>,
) -> f32 {
    let arc_cov = edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32;
    let inlier_ratio = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0);
    (arc_cov * inlier_ratio).clamp(0.0, 1.0)
}

/// Composite confidence score incorporating decode quality, angular coverage,
/// RANSAC inlier ratio, inner fit quality, and RMS residual. Each factor is
/// in [0, 1]; multiplicative composition ensures any single failing dimension
/// pulls confidence toward zero.
pub(crate) fn compute_marker_confidence(
    decode_result: Option<&DecodeResult>,
    edge: &EdgeSampleResult,
    outer_ransac: Option<&conic::RansacResult>,
    inner_fit: &InnerFitResult,
    fit_metrics: &FitMetrics,
    inner_fit_config: &InnerFitConfig,
) -> f32 {
    // 1. Decode signal (base)
    let decode_conf = decode_result
        .map(|d| d.confidence)
        .unwrap_or_else(|| fallback_fit_confidence(edge, outer_ransac));

    // 2. Outer angular coverage: linear map gap -> [0, 1]
    let outer_gap = fit_metrics
        .max_angular_gap_outer
        .unwrap_or(std::f64::consts::TAU);
    let angular_outer = (1.0 - outer_gap / std::f64::consts::TAU).clamp(0.0, 1.0) as f32;

    // 3. RANSAC inlier ratio
    let inlier_factor = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    // 4. Inner fit quality: angular coverage when present, miss penalty otherwise
    let inner_factor = if inner_fit.ellipse_inner.is_some() {
        let inner_gap = fit_metrics.max_angular_gap_inner.unwrap_or(0.0);
        (1.0 - inner_gap / std::f64::consts::TAU).clamp(0.5, 1.0) as f32
    } else {
        inner_fit_config.miss_confidence_factor
    };

    // 5. RMS residual penalty: 1/(1+rms)
    let rms_factor = match fit_metrics.rms_residual_outer {
        Some(rms) if rms > 0.0 && rms.is_finite() => 1.0 / (1.0 + rms as f32),
        _ => 1.0,
    };

    (decode_conf * angular_outer * inlier_factor * inner_factor * rms_factor).clamp(0.0, 1.0)
}
