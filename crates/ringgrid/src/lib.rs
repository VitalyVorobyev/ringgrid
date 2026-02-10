//! ringgrid — pure-Rust detector for coded ring calibration targets.
//!
//! The high-level stage order is owned by `pipeline` (internal):
//! proposal -> local fit/decode -> dedup -> global filter -> refinement -> completion.
//! Low-level fitting/sampling primitives are implemented in `detector` and `ring`.

mod api;
mod board_layout;
mod conic;
mod debug_dump;
mod detector;
mod homography;
mod marker;
mod pipeline;
mod pixelmap;
mod ring;

pub use api::{Detector, TargetSpec};
pub use board_layout::{BoardLayout, BoardMarker};
pub use conic::Ellipse;
pub use detector::{
    CircleRefinementMethod, CompletionParams, DetectConfig, MarkerScalePrior,
    ProjectiveCenterParams, ProposalConfig,
};
pub use homography::RansacHomographyConfig;
pub use marker::{AngularAggregator, DecodeConfig, GradPolarity, MarkerSpec};
pub use pipeline::{
    detect_rings, detect_rings_two_pass_with_mapper, detect_rings_with_mapper,
    detect_rings_with_self_undistort,
};
pub use pixelmap::{
    CameraIntrinsics, CameraModel, DivisionModel, PixelMapper, RadialTangentialDistortion,
    SelfUndistortConfig, SelfUndistortResult, UndistortConfig,
};
pub use ring::{EdgeSampleConfig, OuterEstimationConfig};

#[cfg(feature = "cli-internal")]
pub use debug_dump::DebugDump;
#[cfg(feature = "cli-internal")]
pub use detector::DebugCollectConfig;
#[cfg(feature = "cli-internal")]
pub use marker::codebook;
#[cfg(feature = "cli-internal")]
pub use marker::codec;

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
    /// Selection residual used by the projective-center eigenpair chooser.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_projective_residual: Option<f64>,
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
    pub camera: Option<pixelmap::CameraModel>,
    /// Estimated self-undistort division model, if self-undistort was run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_undistort: Option<pixelmap::SelfUndistortResult>,
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
