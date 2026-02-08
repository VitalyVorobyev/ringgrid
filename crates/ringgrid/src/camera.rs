//! Camera intrinsics and radial-tangential distortion model.
//!
//! This module provides small, reusable primitives for distortion-aware sampling.
//! The current integration keeps camera usage optional and non-breaking.

use serde::{Deserialize, Serialize};

/// Mapping between raw image pixels and detector working-frame pixels.
///
/// The working frame is the coordinate system used by sampling/fitting stages.
/// For distortion-aware processing this is typically an undistorted pixel frame.
pub trait PixelMapper {
    /// Map from image (distorted) pixel coordinates to working coordinates.
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]>;
    /// Map from working coordinates back to image (distorted) pixel coordinates.
    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]>;
}

/// Pinhole camera intrinsics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Returns `true` when focal lengths are finite and non-zero.
    pub fn is_valid(self) -> bool {
        self.fx.is_finite()
            && self.fy.is_finite()
            && self.cx.is_finite()
            && self.cy.is_finite()
            && self.fx.abs() > 1e-12
            && self.fy.abs() > 1e-12
    }

    /// Convert pixel coordinates to normalized pinhole coordinates.
    pub fn pixel_to_normalized(self, pixel_xy: [f64; 2]) -> Option<[f64; 2]> {
        if !self.is_valid() {
            return None;
        }
        let x = (pixel_xy[0] - self.cx) / self.fx;
        let y = (pixel_xy[1] - self.cy) / self.fy;
        if x.is_finite() && y.is_finite() {
            Some([x, y])
        } else {
            None
        }
    }

    /// Convert normalized pinhole coordinates to pixel coordinates.
    pub fn normalized_to_pixel(self, normalized_xy: [f64; 2]) -> [f64; 2] {
        [
            self.fx * normalized_xy[0] + self.cx,
            self.fy * normalized_xy[1] + self.cy,
        ]
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

/// Complete camera model (intrinsics + radial-tangential distortion).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CameraModel {
    /// Camera intrinsics.
    pub intrinsics: CameraIntrinsics,
    /// Distortion coefficients.
    pub distortion: RadialTangentialDistortion,
}

impl CameraModel {
    /// Distort an undistorted pixel point into image pixel coordinates.
    pub fn distort_pixel(self, undistorted_pixel_xy: [f64; 2]) -> Option<[f64; 2]> {
        let xn = self.intrinsics.pixel_to_normalized(undistorted_pixel_xy)?;
        let xd = self.distortion.distort_normalized(xn);
        let pix = self.intrinsics.normalized_to_pixel(xd);
        if pix[0].is_finite() && pix[1].is_finite() {
            Some(pix)
        } else {
            None
        }
    }

    /// Undistort a pixel point with default iterative settings.
    pub fn undistort_pixel(self, distorted_pixel_xy: [f64; 2]) -> Option<[f64; 2]> {
        self.undistort_pixel_with(distorted_pixel_xy, UndistortConfig::default())
    }

    /// Undistort a pixel point with custom iterative settings.
    pub fn undistort_pixel_with(
        self,
        distorted_pixel_xy: [f64; 2],
        cfg: UndistortConfig,
    ) -> Option<[f64; 2]> {
        let xd = self.intrinsics.pixel_to_normalized(distorted_pixel_xy)?;
        let mut x = xd[0];
        let mut y = xd[1];

        for _ in 0..cfg.max_iters.max(1) {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial =
                1.0 + self.distortion.k1 * r2 + self.distortion.k2 * r4 + self.distortion.k3 * r6;
            if !radial.is_finite() || radial.abs() < 1e-12 {
                return None;
            }

            let dx_tan = 2.0 * self.distortion.p1 * x * y + self.distortion.p2 * (r2 + 2.0 * x * x);
            let dy_tan = self.distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * self.distortion.p2 * x * y;
            let x_next = (xd[0] - dx_tan) / radial;
            let y_next = (xd[1] - dy_tan) / radial;

            if !x_next.is_finite() || !y_next.is_finite() {
                return None;
            }

            let dx = x_next - x;
            let dy = y_next - y;
            x = x_next;
            y = y_next;

            if (dx * dx + dy * dy).sqrt() <= cfg.eps.max(0.0) {
                break;
            }
        }

        let out = self.intrinsics.normalized_to_pixel([x, y]);
        if out[0].is_finite() && out[1].is_finite() {
            Some(out)
        } else {
            None
        }
    }
}

impl PixelMapper for CameraModel {
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
        self.undistort_pixel(image_xy)
    }

    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
        self.distort_pixel(working_xy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_camera() -> CameraModel {
        CameraModel {
            intrinsics: CameraIntrinsics {
                fx: 900.0,
                fy: 920.0,
                cx: 640.0,
                cy: 480.0,
            },
            distortion: RadialTangentialDistortion {
                k1: -0.12,
                k2: 0.03,
                p1: 0.001,
                p2: -0.0008,
                k3: 0.0,
            },
        }
    }

    #[test]
    fn intrinsics_validation_rejects_zero_focal() {
        let k = CameraIntrinsics {
            fx: 0.0,
            fy: 500.0,
            cx: 0.0,
            cy: 0.0,
        };
        assert!(!k.is_valid());
        assert!(k.pixel_to_normalized([100.0, 100.0]).is_none());
    }

    #[test]
    fn zero_distortion_roundtrip_is_exact() {
        let cam = CameraModel {
            intrinsics: CameraIntrinsics {
                fx: 800.0,
                fy: 820.0,
                cx: 640.0,
                cy: 480.0,
            },
            distortion: RadialTangentialDistortion::default(),
        };
        let p = [300.25, 210.75];
        let d = cam.distort_pixel(p).unwrap();
        let u = cam.undistort_pixel(d).unwrap();
        assert!((u[0] - p[0]).abs() < 1e-12);
        assert!((u[1] - p[1]).abs() < 1e-12);
    }

    #[test]
    fn roundtrip_with_distortion_is_stable() {
        let cam = sample_camera();
        let p = [250.0, 180.0];
        let d = cam.distort_pixel(p).unwrap();
        let u = cam.undistort_pixel(d).unwrap();
        assert!((u[0] - p[0]).abs() < 1e-5, "x={}, p={}", u[0], p[0]);
        assert!((u[1] - p[1]).abs() < 1e-5, "y={}, p={}", u[1], p[1]);
    }
}
