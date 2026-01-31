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

pub mod preprocess;
pub mod edges;
pub mod conic;
pub mod lattice;
pub mod refine;
pub mod codebook;
pub mod codec;

/// Ellipse parameters for serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EllipseParams {
    /// Semi-axes [a, b] in pixels.
    pub semi_axes: [f64; 2],
    /// Rotation angle in radians.
    pub angle: f64,
}

/// Per-marker debug/diagnostic info (optional, included when `--debug` is used).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MarkerDebug {
    /// Number of edge points used for outer ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_edge_count: Option<usize>,
    /// Number of edge points used for inner ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_edge_count: Option<usize>,
    /// RMS Sampson distance of the outer ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_fit_rms: Option<f64>,
    /// RMS Sampson distance of the inner ellipse fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_fit_rms: Option<f64>,
    /// Raw 16-bit word read from the code band.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_word: Option<u16>,
    /// Hamming distance of the codec match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_dist: Option<u8>,
    /// Margin of the codec match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_margin: Option<u8>,
}

/// A detected marker with its refined center and optional ID.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectedMarker {
    /// Marker center in image coordinates (pixels), after center-bias correction.
    pub center_xy: [f64; 2],
    /// Outer ellipse parameters (semi-axes and angle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_params: Option<EllipseParams>,
    /// Decoded marker ID, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    /// Codec confidence in [0, 1], if decoded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Per-marker debug/diagnostic info.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<MarkerDebug>,
}

/// Full detection result for a single image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionResult {
    pub detected_markers: Vec<DetectedMarker>,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
}

impl DetectionResult {
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            detected_markers: Vec::new(),
            image_size: [width, height],
        }
    }
}
