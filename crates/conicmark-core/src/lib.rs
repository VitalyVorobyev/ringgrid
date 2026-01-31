//! ConicMark Core — algorithms for circle/ring calibration target detection.
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
pub mod codec;

/// A detected marker with its refined center and optional ID.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectedMarker {
    /// Marker center in image coordinates (pixels), after center-bias correction.
    pub center: [f64; 2],
    /// Semi-axes of the outer ellipse [a, b] in pixels.
    pub semi_axes: [f64; 2],
    /// Rotation angle of the outer ellipse in radians.
    pub angle: f64,
    /// Decoded marker ID, if available.
    pub id: Option<u32>,
    /// Residual of the dual-ring fit (RMS pixel error).
    pub fit_residual: f64,
}

/// Full detection result for a single image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionResult {
    pub markers: Vec<DetectedMarker>,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
}

impl DetectionResult {
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            markers: Vec::new(),
            image_size: [width, height],
        }
    }
}
