//! Sub-pixel edge detection for ring marker boundaries.
//!
//! TODO Milestone 2:
//! - Gradient computation (Sobel / Scharr).
//! - Non-maximum suppression along gradient direction.
//! - Hysteresis thresholding.
//! - Sub-pixel refinement via parabolic interpolation of gradient magnitude.
//! - Edge grouping into candidate arcs.

use image::GrayImage;

/// A sub-pixel edge point with gradient direction.
#[derive(Debug, Clone, Copy)]
pub struct EdgePoint {
    pub x: f64,
    pub y: f64,
    /// Gradient direction in radians.
    pub grad_angle: f64,
    /// Gradient magnitude.
    pub grad_mag: f64,
}

/// Detect sub-pixel edges in a grayscale image.
///
/// Placeholder — returns empty list for now.
pub fn detect_edges(_img: &GrayImage, _low_thresh: f64, _high_thresh: f64) -> Vec<EdgePoint> {
    // TODO Milestone 2: implement Canny-like sub-pixel edge detection
    Vec::new()
}

/// Group edge points into candidate arcs that may belong to ellipses.
///
/// Placeholder — returns empty list for now.
pub fn group_arcs(_edges: &[EdgePoint], _max_gap: f64) -> Vec<Vec<usize>> {
    // TODO Milestone 2: implement arc grouping
    Vec::new()
}
