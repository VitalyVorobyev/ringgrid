use super::PixelMapper;
use serde::{Deserialize, Serialize};

/// Single-parameter division distortion model.
///
/// Negative lambda corresponds to barrel distortion (most common),
/// positive to pincushion distortion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DivisionModel {
    /// Distortion parameter.
    pub lambda: f64,
    /// Distortion center x (pixels).
    pub cx: f64,
    /// Distortion center y (pixels).
    pub cy: f64,
}

impl DivisionModel {
    /// Create a division model with explicit parameters.
    pub fn new(lambda: f64, cx: f64, cy: f64) -> Self {
        Self { lambda, cx, cy }
    }

    /// Create a division model centered on the image.
    pub fn centered(lambda: f64, width: u32, height: u32) -> Self {
        Self {
            lambda,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
        }
    }

    /// Identity model (zero distortion) centered on the image.
    pub fn identity(width: u32, height: u32) -> Self {
        Self::centered(0.0, width, height)
    }

    /// Undistort a single point.
    pub fn undistort_point(&self, distorted_xy: [f64; 2]) -> [f64; 2] {
        let dx = distorted_xy[0] - self.cx;
        let dy = distorted_xy[1] - self.cy;
        let r2 = dx * dx + dy * dy;
        let denom = 1.0 + self.lambda * r2;
        if denom.abs() < 1e-12 || !denom.is_finite() {
            return distorted_xy;
        }
        let scale = 1.0 / denom;
        [self.cx + dx * scale, self.cy + dy * scale]
    }

    /// Distort a point (inverse mapping: undistorted → distorted).
    ///
    /// Uses iterative fixed-point method since the inverse is not closed-form.
    ///
    /// TODO: make it analytically
    pub fn distort_point(&self, undistorted_xy: [f64; 2]) -> Option<[f64; 2]> {
        if self.lambda.abs() < 1e-18 {
            return Some(undistorted_xy);
        }
        let ux = undistorted_xy[0] - self.cx;
        let uy = undistorted_xy[1] - self.cy;
        let mut dx = ux;
        let mut dy = uy;
        for _ in 0..20 {
            let r2 = dx * dx + dy * dy;
            let factor = 1.0 + self.lambda * r2;
            if factor.abs() < 1e-12 || !factor.is_finite() {
                return None;
            }
            let dx_next = ux * factor;
            let dy_next = uy * factor;
            if !dx_next.is_finite() || !dy_next.is_finite() {
                return None;
            }
            let delta = (dx_next - dx).powi(2) + (dy_next - dy).powi(2);
            dx = dx_next;
            dy = dy_next;
            if delta.sqrt() < 1e-12 {
                break;
            }
        }
        let out = [self.cx + dx, self.cy + dy];
        if out[0].is_finite() && out[1].is_finite() {
            Some(out)
        } else {
            None
        }
    }

    /// Undistort a batch of points.
    pub fn undistort_points(&self, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        points.iter().map(|p| self.undistort_point(*p)).collect()
    }
}

impl PixelMapper for DivisionModel {
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
        Some(self.undistort_point(image_xy))
    }

    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
        self.distort_point(working_xy)
    }
}

/// Brown-Conrady radial-tangential distortion coefficients.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RadialTangentialDistortion {
    /// Radial coefficient k1.
    pub k1: f64,
    /// Radial coefficient k2.
    pub k2: f64,
    /// Tangential coefficient p1.
    pub p1: f64,
    /// Tangential coefficient p2.
    pub p2: f64,
    /// Radial coefficient k3.
    pub k3: f64,
}

impl Default for RadialTangentialDistortion {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }
}

impl RadialTangentialDistortion {
    /// Apply distortion to normalized coordinates.
    pub fn distort_normalized(self, normalized_xy: [f64; 2]) -> [f64; 2] {
        let x = normalized_xy[0];
        let y = normalized_xy[1];
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let x_tan = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let y_tan = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        [x * radial + x_tan, y * radial + y_tan]
    }
}

/// Distortion inversion settings used by iterative undistortion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct UndistortConfig {
    /// Maximum fixed-point iterations.
    pub max_iters: usize,
    /// Stop when coordinate update norm is below this threshold.
    pub eps: f64,
}

impl Default for UndistortConfig {
    fn default() -> Self {
        Self {
            max_iters: 15,
            eps: 1e-12,
        }
    }
}
