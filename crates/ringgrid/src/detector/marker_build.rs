use image::GrayImage;

use crate::conic::{self, Ellipse};
use crate::marker::DecodeMetrics;
use crate::marker::decode::DecodeResult;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_quality::{border_distance_radii, outer_edge_width_px};
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleResult};

use super::config::{DetectConfig, InnerFitConfig};
use super::inner_fit::{
    InnerFitReason, InnerFitResult, InnerFitStatus, fit_inner_ellipse_from_outer_hint,
};
use super::outer_fit::OuterFitCandidate;

/// Indicates which pipeline stage produced a marker.
///
/// Exposed via [`MarkerDiagnostics::source`](crate::diagnostics::MarkerDiagnostics).
/// Used to separate false-positive sources in post-processing and analysis.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DetectionSource {
    /// Marker was produced by the primary gradient-proposal → fit → decode path.
    #[default]
    FitDecoded,
    /// Marker was produced by the H-guided completion stage at a missing board ID.
    Completion,
    /// Marker was produced by the seeded two-pass path (`detect_with_mapper`).
    /// The proposal seed came from pass-1 results; the fit ran with an active
    /// distortion mapper.
    SeededPass,
}

/// Which estimator produced a marker's reported centre.
///
/// Exposed as [`DetectedMarker::center_source`](crate::DetectedMarker::center_source).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CenterSource {
    /// Centre of the fitted outer ellipse: no correction was requested, the
    /// marker has no inner ellipse, or every correction was rejected. Under
    /// perspective this is not the image of the ring centre, and under defocus
    /// blur it inherits the code-dependent bias of the edge points.
    #[default]
    OuterEllipse,
    /// Projective centre recovered from the inner and outer ellipses.
    ProjectiveCenter,
    /// Photometric fit of the blurred marker pattern; the marker carries a
    /// centre covariance.
    Photometric,
}

/// Outcome of the photometric fit of one marker.
///
/// Exposed via [`MarkerDiagnostics::photometric`](crate::diagnostics::MarkerDiagnostics::photometric).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PhotometricStatus {
    /// The fit was accepted and produced the marker's centre.
    Accepted,
    /// Too few usable (in-image, unsaturated) pixels.
    InsufficientPixels,
    /// The local board-to-image map or the normal equations were singular.
    Degenerate,
    /// The optimizer did not converge within the iteration budget.
    NotConverged,
    /// The fitted centre moved further than the configured gate.
    ShiftTooLarge,
    /// The fitted blur or contrast is implausible (no marker signal).
    Implausible,
    /// The pattern model does not explain the pixels: the residual is well
    /// above the image's typical fit residual (glare halo, occlusion, print
    /// defect), so the centre is not trusted.
    PoorFit,
    /// The marker was not fitted because its printed pattern cannot be placed
    /// on the board: a coded marker without a decoded id, a plain marker
    /// without a lattice label, or a result without a board homography.
    NoBoardPosition,
}

/// Photometric fit summary for one marker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PhotometricMetrics {
    /// Whether the fit produced the centre, or why not.
    pub status: PhotometricStatus,
    /// Fitted Gaussian PSF standard deviation (pixels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blur_sigma_px: Option<f64>,
    /// Fitted paper-minus-ink contrast at the marker centre (grey levels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f64>,
    /// Robust standard deviation of the pixel residuals (grey levels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_sigma: Option<f64>,
    /// Pixels in the fit.
    pub n_pixels: usize,
    /// Fraction of the patch excluded as saturated.
    pub masked_fraction: f64,
    /// Distance from the starting centre to the fitted centre (pixels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_shift_px: Option<f64>,
}

/// Fit quality metrics for a detected marker.
///
/// Reports the edge sampling and ellipse fit quality. High RANSAC inlier
/// ratios (> 0.8) and low RMS Sampson residuals (< 0.5 px) indicate a
/// precise ellipse fit.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
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
    /// Theta consistency score from the inner estimate stage. Fraction of theta
    /// samples that agree on the inner edge location. Present when estimation ran,
    /// including when it failed the quality gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_theta_consistency: Option<f32>,
    /// Standard deviation of per-ray outer radii (pixels). High values (> 30% of the
    /// mean outer radius) indicate inner/outer edge contamination — some rays landed
    /// on the inner ring instead of the outer one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radii_std_outer_px: Option<f32>,
    /// Reprojection error (pixels) between this marker's center and the board position
    /// projected through the final estimated homography. Populated in the finalization
    /// stage when a global homography is available. Absent for markers detected in
    /// no-H mode or when `board_xy_mm` is not available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_reproj_err_px: Option<f32>,
    /// Width (pixels) of the outer edge's intensity ramp — the median
    /// paper-to-ink step across the rim divided by the median peak radial
    /// gradient; about 2.5 σ for a Gaussian blur of σ. A factor of the marker
    /// [`confidence`](crate::DetectedMarker::confidence). Absent when the rim
    /// is mostly outside the image or has no step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_edge_width_px: Option<f32>,
}

/// Internal rich working marker used while the detection pipeline runs.
///
/// This is the mutable per-marker state threaded through every fit / decode /
/// id-correction / completion / finalization stage. It carries algorithm
/// internals (fit metrics, decode metrics, raw edge sample points, stage
/// provenance) that several stages both read and mutate.
///
/// It is **not** part of the public API. At the [`Detector::detect`](crate::Detector::detect)
/// boundary each `MarkerRecord` is split into the public slim
/// [`DetectedMarker`](crate::DetectedMarker) plus its
/// [`MarkerDiagnostics`](crate::diagnostics::MarkerDiagnostics) counterpart.
///
/// The `center` field is always in image-pixel coordinates, regardless of
/// whether a [`PixelMapper`](crate::PixelMapper) was used. When a mapper is
/// active, `center_mapped` provides the working-frame (undistorted)
/// coordinates. `board_xy_mm` provides board-space marker coordinates in
/// millimeters when the decoded `id` is valid for the active [`TargetLayout`](crate::TargetLayout).
/// Ellipses are in the working frame when a mapper is active.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct MarkerRecord {
    /// Decoded marker ID (codebook index), or None if decoding was rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    /// Lattice cell coordinate `[u, v]` assigned to this marker.
    ///
    /// For coded targets this is the board-frame cell coordinate of the decoded
    /// ID. For plain targets it is assigned by grid labeling and is
    /// board-frame only when the origin was resolved (see
    /// [`BoardFrame`](crate::BoardFrame)); otherwise it lives in the canonical
    /// relative frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_coord: Option<[i32; 2]>,
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
    /// For plain targets, the plain pipeline uses this field internally for the
    /// assignment-frame position and clears it before output when the origin is
    /// unresolved.
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
    /// Pipeline stage that produced this marker.
    pub source: DetectionSource,
    /// Estimator that produced `center`.
    #[serde(default)]
    pub center_source: CenterSource,
    /// Covariance of `center` (px²), when the estimator provides one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_cov: Option<[[f64; 2]; 2]>,
    /// Photometric fit summary, when the photometric stage ran on this marker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photometric: Option<PhotometricMetrics>,
}

#[derive(Debug, Clone, Copy)]
struct InnerFitSummary {
    n_points_inner: usize,
    ransac_inlier_ratio_inner: Option<f32>,
    rms_residual_inner: Option<f64>,
    max_angular_gap_inner: Option<f64>,
    inner_fit_status: Option<InnerFitStatus>,
    inner_fit_reason: Option<InnerFitReason>,
    inner_theta_consistency: Option<f32>,
}

impl InnerFitSummary {
    fn from_result(inner: &InnerFitResult) -> Self {
        Self {
            n_points_inner: inner.points_inner.len(),
            ransac_inlier_ratio_inner: inner.ransac_inlier_ratio_inner,
            rms_residual_inner: inner.rms_residual_inner,
            max_angular_gap_inner: inner.max_angular_gap,
            inner_fit_status: Some(inner.status),
            inner_fit_reason: (inner.status != InnerFitStatus::Ok)
                .then_some(inner.reason)
                .flatten(),
            inner_theta_consistency: inner.theta_consistency,
        }
    }
}

fn radii_std(radii: &[f32]) -> Option<f32> {
    if radii.len() < 2 {
        return None;
    }
    let mean = radii.iter().sum::<f32>() / radii.len() as f32;
    let variance = radii.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / radii.len() as f32;
    Some(variance.sqrt())
}

fn fit_metrics_from_outer(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&conic::RansacResult>,
    inner_summary: &InnerFitSummary,
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
        n_points_inner: inner_summary.n_points_inner,
        ransac_inlier_ratio_outer: outer_ransac
            .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
        ransac_inlier_ratio_inner: inner_summary.ransac_inlier_ratio_inner,
        rms_residual_outer: Some(conic::rms_sampson_distance(outer, &edge.outer_points)),
        rms_residual_inner: inner_summary.rms_residual_inner,
        max_angular_gap_outer: gap_outer,
        max_angular_gap_inner: inner_summary.max_angular_gap_inner,
        inner_fit_status: inner_summary.inner_fit_status,
        inner_fit_reason: inner_summary.inner_fit_reason,
        neighbor_radius_ratio: None,
        inner_theta_consistency: inner_summary.inner_theta_consistency,
        radii_std_outer_px: radii_std(&edge.outer_radii),
        h_reproj_err_px: None,
        outer_edge_width_px: None,
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
    let inner_summary = InnerFitSummary::from_result(inner);
    fit_metrics_from_outer(edge, outer, outer_ransac, &inner_summary)
}

pub(crate) fn decode_metrics_from_result(
    decode_result: Option<&DecodeResult>,
) -> Option<DecodeMetrics> {
    decode_result.map(|d| DecodeMetrics {
        observed_word: d.raw_word,
        best_id: d.id,
        rotation: d.rotation,
    })
}

/// Composite fit-quality score: arc coverage × RANSAC inlier ratio, clamped to [0, 1].
///
/// When no RANSAC result is available (direct-fit path), the inlier ratio defaults to 1.0
/// so the score degrades gracefully to pure arc coverage.
pub(crate) fn fit_support_score(
    edge: &EdgeSampleResult,
    outer_ransac: Option<&conic::RansacResult>,
) -> f32 {
    let arc_cov = edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32;
    let inlier_ratio = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0);
    (arc_cov * inlier_ratio).clamp(0.0, 1.0)
}

/// Angular gap of the outer edge points (radians) over which the coverage
/// factor falls by `e`: a quarter of the rim unsupported costs ~86 %.
const COVERAGE_GAP_SCALE_RAD: f64 = std::f64::consts::FRAC_PI_4;
/// Outer edge width (px) at which the sharpness factor is one half.
const EDGE_WIDTH_SCALE_PX: f32 = 3.0;
/// For a photometric centre, the half-sharpness edge width grows with the
/// ring beyond a 16 px outer radius (in outer radii): the same 3 px at 16 px,
/// 8.4 px at 45 px (see [`confidence_of_photometric_centre`]).
const PHOTOMETRIC_EDGE_WIDTH_SCALE_RADII: f32 = 3.0 / 16.0;
/// Distance to the image border (outer radii) below which the border factor
/// falls linearly to zero.
const BORDER_SCALE_RADII: f32 = 2.0;

/// Per-marker confidence in `[0, 1]`: how much the marker's centre can be
/// trusted. A product of factors, each in `[0, 1]`:
///
/// - **evidence** — the code distance of its decode (`1 − dist/6`), or, for
///   an undecoded ring, its outer-edge arc coverage × RANSAC inlier ratio;
/// - **outer fit** — RANSAC inlier ratio × `1/(1 + rms/2)` (Sampson RMS, px);
/// - **rim coverage** — `exp(−gap / (π/4))`, `gap` the largest angular gap
///   between outer edge points (radians);
/// - **inner edge** — the fraction of rays whose inner-edge peak agrees with
///   the inner ring's radius, times `miss_confidence_factor` when no inner
///   ellipse was fitted;
/// - **sharpness** — `1/(1 + (w / 3 px)²)`, `w` the outer edge width
///   ([`FitMetrics::outer_edge_width_px`]; 0 when unmeasurable); measured
///   against the ring's size once a photometric fit has replaced the centre
///   ([`confidence_of_photometric_centre`]);
/// - **border** — `min(1, d / 2)`, `d` the centre's distance to the image
///   border in outer radii.
///
/// Rim coverage, inner-edge agreement, sharpness and border are the
/// predictors of a wrong centre the precision harness measured (centre error
/// beyond 1 px: ROC AUC 0.95 for this product vs 0.89 for the previous
/// confidence); the evidence and outer-fit factors keep ranking duplicate
/// fits and ID votes.
pub(crate) fn compute_marker_confidence(
    decode_result: Option<&DecodeResult>,
    edge: &EdgeSampleResult,
    outer_ransac: Option<&conic::RansacResult>,
    inner_fit: &InnerFitResult,
    fit_metrics: &FitMetrics,
    border_radii: f32,
    inner_fit_config: &InnerFitConfig,
) -> f32 {
    // A decode is an exact codeword match: full evidence.
    let evidence = decode_result
        .map(|_| 1.0)
        .unwrap_or_else(|| fit_support_score(edge, outer_ransac));
    let inlier = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let rms = match fit_metrics.rms_residual_outer {
        Some(rms) if rms > 0.0 && rms.is_finite() => 1.0 / (1.0 + rms as f32 / 2.0),
        _ => 1.0,
    };
    let gap = fit_metrics
        .max_angular_gap_outer
        .unwrap_or(std::f64::consts::TAU);
    let coverage = (-gap / COVERAGE_GAP_SCALE_RAD).exp() as f32;
    let agreement = fit_metrics.inner_theta_consistency.unwrap_or(0.0);
    let inner = if inner_fit.ellipse_inner.is_some() {
        agreement
    } else {
        agreement * inner_fit_config.miss_confidence_factor
    };
    let sharpness = sharpness_factor(fit_metrics.outer_edge_width_px, EDGE_WIDTH_SCALE_PX);
    let border = (border_radii / BORDER_SCALE_RADII).min(1.0);
    (evidence * inlier * rms * coverage * inner * sharpness * border).clamp(0.0, 1.0)
}

/// `1/(1 + (w / scale)²)` for an outer edge width `w`; 0 when unmeasurable.
fn sharpness_factor(edge_width_px: Option<f32>, scale_px: f32) -> f32 {
    edge_width_px.map_or(0.0, |w| 1.0 / (1.0 + (w / scale_px).powi(2)))
}

/// The confidence of a marker whose centre the photometric fit just
/// replaced: [`compute_marker_confidence`] with the sharpness factor measured
/// against the ring's size, `s = max(3 px, 0.1875 r)` (`r` the outer
/// ellipse's mean semi-axis), instead of the absolute 3 px.
///
/// The absolute scale predicts the error of an edge-fitted centre under blur
/// (precision harness: ROC AUC for centre errors beyond 1 px drops from 0.90
/// to 0.84 when edge-fitted centres are judged ring-relative). The
/// photometric fit models the blur (its PSF width is a fit parameter), so
/// what remains is blur relative to the ring the fit integrates over: over
/// the harness in photometric mode, ring-relative sharpness reports 360 more
/// centres, none off by more than 1 px. Rings up to 32 px across are
/// unaffected.
pub(crate) fn confidence_of_photometric_centre(marker: &MarkerRecord) -> f32 {
    let w = marker.fit.outer_edge_width_px;
    let absolute = sharpness_factor(w, EDGE_WIDTH_SCALE_PX);
    let Some(outer) = marker.ellipse_outer.filter(|_| absolute > 0.0) else {
        return marker.confidence;
    };
    let scale =
        EDGE_WIDTH_SCALE_PX.max(PHOTOMETRIC_EDGE_WIDTH_SCALE_RADII * outer.mean_axis() as f32);
    let relative = sharpness_factor(w, scale);
    (marker.confidence / absolute * relative).clamp(0.0, 1.0)
}

/// Turn an accepted outer-fit candidate into a marker: fit the inner ring from
/// the outer hint, derive fit / decode metrics and the confidence, and
/// assemble the record. Confidence is [`compute_marker_confidence`] for every
/// source (fit-decode, inner-as-outer recovery and completion).
///
/// `cand.decode_result` is the decode attributed to the marker — callers
/// filter or replace it before the call. The record's `id` is that decode's
/// ID and its `center` is the outer-ellipse centre in the working frame;
/// provenance (`source`, `grid_coord`, image-frame centre, a non-decoded
/// `id`) is the caller's to set.
pub(crate) fn marker_from_outer_fit(
    gray: &GrayImage,
    cand: OuterFitCandidate,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> MarkerRecord {
    let OuterFitCandidate {
        edge,
        outer,
        outer_ransac,
        decode_result,
        ..
    } = cand;
    let center = outer.center();

    let inner = fit_inner_ellipse_from_outer_hint(
        gray,
        &outer,
        &config.advanced.marker_spec,
        mapper,
        &config.advanced.inner_fit,
        false,
    );
    if inner.status != InnerFitStatus::Ok {
        tracing::trace!(
            "inner fit rejected/failed at ({:.1},{:.1}): status={:?}, reason={:?}, context={:?}",
            center[0],
            center[1],
            inner.status,
            inner.reason.map(|reason| reason.code()),
            inner.reason_context
        );
    }

    let sampler = DistortionAwareSampler::new(gray, mapper);
    let mut fit = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner);
    fit.outer_edge_width_px = outer_edge_width_px(sampler, &outer);
    let decode = decode_metrics_from_result(decode_result.as_ref());
    let confidence = compute_marker_confidence(
        decode_result.as_ref(),
        &edge,
        outer_ransac.as_ref(),
        &inner,
        &fit,
        border_distance_radii(sampler, center, outer.a),
        &config.advanced.inner_fit,
    );

    MarkerRecord {
        id: decode_result.as_ref().map(|d| d.id),
        confidence,
        center,
        ellipse_outer: Some(outer),
        ellipse_inner: inner.ellipse_inner,
        edge_points_outer: Some(edge.outer_points),
        edge_points_inner: Some(inner.points_inner),
        fit,
        decode,
        ..MarkerRecord::default()
    }
}
