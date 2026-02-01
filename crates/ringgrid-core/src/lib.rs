//! ringgrid-core — algorithms for circle/ring calibration target detection.
//!
//! Designed for Scheimpflug cameras with strong anisotropic defocus blur.
//! The pipeline stages are:
//!
//! 1. **Preprocess** – illumination normalization, band-pass filtering.
//! 2. **Edges** – sub-pixel edge detection (Canny-like + gradient interpolation).
//! 3. **Conic** – robust ellipse fitting via direct conic least-squares + RANSAC.
//! 4. **Lattice** – neighbor graph construction, vanishing line estimation,
//!    affine-rectification homography for center-bias correction.
//! 5. **Refine** – per-marker shared-center dual-ring Levenberg–Marquardt refinement.
//! 6. **Codec** – marker ID decoding from ring sector pattern.
//! 7. **Ring** – end-to-end ring detection pipeline: proposal → edge sampling → fit → decode.

pub mod board_spec;
pub mod codebook;
pub mod codec;
pub mod conic;
pub mod debug_dump;
pub mod edges;
pub mod homography;
pub mod lattice;
pub mod preprocess;
pub mod refine;
pub mod ring;

/// Ellipse parameters for serialization (center + geometry).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EllipseParams {
    /// Center (x, y) in image pixels.
    pub center_xy: [f64; 2],
    /// Semi-axes [a, b] in pixels.
    pub semi_axes: [f64; 2],
    /// Rotation angle in radians.
    pub angle: f64,
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
    /// Marker center in image coordinates (pixels).
    pub center: [f64; 2],
    /// Outer ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<EllipseParams>,
    /// Inner ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<EllipseParams>,
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
    /// Inlier threshold in pixels.
    pub threshold_px: f64,
    /// Mean reprojection error of inliers (pixels).
    pub mean_err_px: f64,
    /// 95th percentile reprojection error (pixels).
    pub p95_err_px: f64,
}

/// Full detection result for a single image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionResult {
    pub detected_markers: Vec<DetectedMarker>,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
    /// Fitted board-to-image homography (3x3, row-major), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homography: Option<[[f64; 3]; 3]>,
    /// RANSAC statistics, if homography was fitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac: Option<RansacStats>,
}

impl DetectionResult {
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            detected_markers: Vec::new(),
            image_size: [width, height],
            homography: None,
            ransac: None,
        }
    }
}
