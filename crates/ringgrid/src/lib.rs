//! ringgrid — pure-Rust detector for coded ring calibration targets.
//!
//! Designed for Scheimpflug cameras with strong anisotropic defocus blur.
//! The pipeline stages are:
//!
//! 1. **Preprocess** – illumination normalization, band-pass filtering.
//! 2. **Edges** – sub-pixel edge detection (Canny-like + gradient interpolation).
//! 3. **Conic** – robust ellipse fitting via direct conic least-squares + RANSAC.
//! 4. **Lattice** – neighbor graph construction, vanishing line estimation,
//!    affine-rectification homography for center-bias correction.
//! 5. **Codec** – marker ID decoding from ring sector pattern.
//! 7. **Ring** – end-to-end ring detection pipeline: proposal → edge sampling
//!    → fit → decode.
//!
//! # Public API
//! The stable v1 API is intentionally small:
//! - [`Detector`] and [`TargetSpec`] as primary entry points
//! - [`DetectConfig`] for advanced tuning
//! - camera/mapper traits and result structures
//!
//! Low-level math and pipeline internals are not part of the public v1 surface.

mod board_layout;
mod camera;
#[cfg(feature = "cli-internal")]
pub mod codebook;
#[cfg(not(feature = "cli-internal"))]
mod codebook;
#[cfg(feature = "cli-internal")]
pub mod codec;
#[cfg(not(feature = "cli-internal"))]
mod codec;
mod conic;
#[cfg(feature = "cli-internal")]
pub mod debug_dump;
#[cfg(not(feature = "cli-internal"))]
mod debug_dump;
mod detector;
mod homography;
mod marker_spec;
mod projective_center;
mod ring;
mod self_undistort;

pub use board_layout::{BoardLayout, BoardMarker};
pub use camera::{CameraIntrinsics, CameraModel, PixelMapper, RadialTangentialDistortion};
pub use detector::{Detector, TargetSpec};
pub use homography::RansacHomographyConfig;
pub use marker_spec::{AngularAggregator, GradPolarity, MarkerSpec};
pub use ring::decode::DecodeConfig;
pub use ring::detect::{
    CircleRefinementMethod, CompletionParams, DetectConfig, MarkerScalePrior,
    ProjectiveCenterParams,
};
pub use ring::edge_sample::EdgeSampleConfig;
pub use ring::outer_estimate::OuterEstimationConfig;
pub use ring::proposal::ProposalConfig;
pub use self_undistort::{DivisionModel, SelfUndistortConfig, SelfUndistortResult};

#[cfg(feature = "cli-internal")]
pub use debug_dump::DebugDump;
#[cfg(feature = "cli-internal")]
pub use ring::detect::DebugCollectConfig;

/// Ellipse parameters for serialization (center + geometry).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EllipseParams {
    /// Center (x, y) in detector working pixel coordinates.
    ///
    /// This is the undistorted pixel frame when camera intrinsics are provided,
    /// otherwise it is the raw image pixel frame.
    pub center_xy: [f64; 2],
    /// Semi-axes [a, b] in working-frame pixels.
    pub semi_axes: [f64; 2],
    /// Rotation angle in radians.
    pub angle: f64,
}

impl From<conic::Ellipse> for EllipseParams {
    fn from(e: conic::Ellipse) -> Self {
        Self {
            center_xy: [e.cx, e.cy],
            semi_axes: [e.a, e.b],
            angle: e.angle,
        }
    }
}

impl From<&conic::Ellipse> for EllipseParams {
    fn from(e: &conic::Ellipse) -> Self {
        Self {
            center_xy: [e.cx, e.cy],
            semi_axes: [e.a, e.b],
            angle: e.angle,
        }
    }
}

impl From<EllipseParams> for conic::Ellipse {
    fn from(p: EllipseParams) -> Self {
        Self {
            cx: p.center_xy[0],
            cy: p.center_xy[1],
            a: p.semi_axes[0].abs(),
            b: p.semi_axes[1].abs(),
            angle: p.angle,
        }
    }
}

impl From<&EllipseParams> for conic::Ellipse {
    fn from(p: &EllipseParams) -> Self {
        Self {
            cx: p.center_xy[0],
            cy: p.center_xy[1],
            a: p.semi_axes[0].abs(),
            b: p.semi_axes[1].abs(),
            angle: p.angle,
        }
    }
}

/// Fit quality metrics for a detected marker.
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

/// Decode quality metrics for a detected marker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeMetrics {
    /// Raw 16-bit word sampled from the code band.
    pub observed_word: u16,
    /// Best-matching codebook entry index.
    pub best_id: usize,
    /// Cyclic rotation that produced the best match.
    pub best_rotation: u8,
    /// Hamming distance to the best-matching codeword.
    pub best_dist: u8,
    /// Margin: second_best_dist - best_dist.
    pub margin: u8,
    /// Confidence heuristic in [0, 1].
    pub decode_confidence: f32,
}

/// A detected marker with its refined center and optional ID.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectedMarker {
    /// Decoded marker ID (codebook index), or None if decoding was rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    /// Combined detection + decode confidence in [0, 1].
    pub confidence: f32,
    /// Marker center in detector working pixel coordinates.
    ///
    /// This is the undistorted pixel frame when camera intrinsics are provided,
    /// otherwise it is the raw image pixel frame.
    pub center: [f64; 2],
    /// Projective unbiased center estimated from inner+outer ring conics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_projective: Option<[f64; 2]>,
    /// Vanishing line estimate associated with the unbiased center (homogeneous line ax+by+c=0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vanishing_line: Option<[f64; 3]>,
    /// Selection residual used by the projective-center eigenpair chooser.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_projective_residual: Option<f64>,
    /// Outer ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<EllipseParams>,
    /// Inner ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<EllipseParams>,
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

/// RANSAC statistics for homography fitting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RansacStats {
    /// Number of decoded candidates fed to RANSAC.
    pub n_candidates: usize,
    /// Number of inliers after RANSAC.
    pub n_inliers: usize,
    /// Inlier threshold in working-frame pixels.
    pub threshold_px: f64,
    /// Mean reprojection error of inliers (working-frame pixels).
    pub mean_err_px: f64,
    /// 95th percentile reprojection error (working-frame pixels).
    pub p95_err_px: f64,
}

/// Full detection result for a single image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionResult {
    /// Detected markers in detector working pixel coordinates.
    pub detected_markers: Vec<DetectedMarker>,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
    /// Fitted board-to-working-frame homography (3x3, row-major), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homography: Option<[[f64; 3]; 3]>,
    /// RANSAC statistics, if homography was fitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac: Option<RansacStats>,
    /// Camera model used for distortion-aware processing, if configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<camera::CameraModel>,
    /// Estimated self-undistort division model, if self-undistort was run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_undistort: Option<self_undistort::SelfUndistortResult>,
}

impl DetectionResult {
    /// Construct an empty result for an image with the provided dimensions.
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            detected_markers: Vec::new(),
            image_size: [width, height],
            homography: None,
            ransac: None,
            camera: None,
            self_undistort: None,
        }
    }
}
