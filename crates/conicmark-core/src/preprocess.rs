//! Image preprocessing: illumination normalization and band-pass filtering.
//!
//! TODO Milestone 2:
//! - Local mean / morphological background subtraction for uneven illumination.
//! - Band-pass (DoG or Laplacian-of-Gaussian) tuned for ring marker spatial frequency.
//! - Optional downscaling for coarse detection pass.

use image::GrayImage;

/// Normalize illumination by subtracting a large-kernel local mean.
///
/// Placeholder — returns input unchanged for now.
pub fn normalize_illumination(img: &GrayImage) -> GrayImage {
    // TODO Milestone 2: implement local-mean subtraction
    img.clone()
}

/// Band-pass filter to enhance ring-shaped features.
///
/// Placeholder — returns input unchanged for now.
pub fn bandpass_filter(img: &GrayImage, _marker_diameter_px: f64) -> GrayImage {
    // TODO Milestone 2: implement DoG or LoG filter
    img.clone()
}
