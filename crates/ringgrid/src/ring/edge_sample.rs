//! Edge sampling primitives shared by ring detection stages.

use image::GrayImage;

use crate::pixelmap::PixelMapper;

/// Configuration for radial edge sampling.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EdgeSampleConfig {
    /// Number of radial rays to cast.
    pub n_rays: usize,
    /// Maximum sampling radius (pixels).
    pub r_max: f32,
    /// Minimum sampling radius (pixels).
    pub r_min: f32,
    /// Step size along each ray (pixels).
    pub r_step: f32,
    /// Minimum intensity drop (0-1) for a valid ring.
    pub min_ring_depth: f32,
    /// Minimum number of rays with detected ring for a valid candidate.
    pub min_rays_with_ring: usize,
}

impl Default for EdgeSampleConfig {
    fn default() -> Self {
        Self {
            n_rays: 48,
            r_max: 14.0,
            r_min: 1.5,
            r_step: 0.5,
            min_ring_depth: 0.08,
            min_rays_with_ring: 16,
        }
    }
}

/// Result of edge sampling for one candidate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeSampleResult {
    /// Outer edge points (sub-pixel positions in image coords).
    pub outer_points: Vec<[f64; 2]>,
    /// Inner edge points (sub-pixel positions in image coords).
    pub inner_points: Vec<[f64; 2]>,
    /// Per-ray outer radii for rays where both edges were found.
    pub outer_radii: Vec<f32>,
    /// Per-ray inner radii for rays where both edges were found.
    pub inner_radii: Vec<f32>,
    /// Number of rays with both edges found.
    pub n_good_rays: usize,
    /// Total rays attempted.
    pub n_total_rays: usize,
}

/// Distortion-aware point sampler working in a "working" coordinate frame.
///
/// Working-frame coordinates are:
/// - image pixel coordinates when `camera` is `None`,
/// - undistorted pixel coordinates when `camera` is provided.
#[derive(Clone, Copy)]
pub struct DistortionAwareSampler<'a> {
    img: &'a GrayImage,
    mapper: Option<&'a dyn PixelMapper>,
}

impl<'a> DistortionAwareSampler<'a> {
    /// Create a sampler for one image and optional camera model.
    pub fn new(img: &'a GrayImage, mapper: Option<&'a dyn PixelMapper>) -> Self {
        Self { img, mapper }
    }

    /// Convert image-space (distorted) pixel coordinates into working-frame coordinates.
    pub fn image_to_working_xy(self, img_xy: [f32; 2]) -> Option<[f32; 2]> {
        if let Some(mapper) = self.mapper {
            let u = mapper.image_to_working_pixel([img_xy[0] as f64, img_xy[1] as f64])?;
            let out = [u[0] as f32, u[1] as f32];
            if out[0].is_finite() && out[1].is_finite() {
                Some(out)
            } else {
                None
            }
        } else {
            Some(img_xy)
        }
    }

    /// Convert working-frame coordinates into image-space (distorted) pixel coordinates.
    pub fn working_to_image_xy(self, working_xy: [f32; 2]) -> Option<[f32; 2]> {
        if let Some(mapper) = self.mapper {
            let d = mapper.working_to_image_pixel([working_xy[0] as f64, working_xy[1] as f64])?;
            let out = [d[0] as f32, d[1] as f32];
            if out[0].is_finite() && out[1].is_finite() {
                Some(out)
            } else {
                None
            }
        } else {
            Some(working_xy)
        }
    }

    /// Sample at a working-frame coordinate.
    ///
    /// Returns intensity in [0, 1] or `None` if mapped position is out of image bounds.
    #[inline]
    pub fn sample_checked(self, x_working: f32, y_working: f32) -> Option<f32> {
        let img_xy = self.working_to_image_xy([x_working, y_working])?;
        bilinear_sample_u8_checked(self.img, img_xy[0], img_xy[1])
    }
}

/// Sample a grayscale image at sub-pixel position using bilinear interpolation.
/// Returns intensity in [0, 1].
#[inline]
#[allow(dead_code)]
pub fn bilinear_sample_u8(img: &GrayImage, x: f32, y: f32) -> f32 {
    bilinear_sample_u8_checked(img, x, y).unwrap_or(0.0)
}

/// Sample a grayscale image at sub-pixel position using bilinear interpolation.
/// Returns intensity in [0, 1] or `None` if sampling is out of bounds.
#[inline]
pub fn bilinear_sample_u8_checked(img: &GrayImage, x: f32, y: f32) -> Option<f32> {
    let (w, h) = img.dimensions();
    if w < 2 || h < 2 || x < 0.0 || y < 0.0 {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    if x0 >= w - 1 || y0 >= h - 1 {
        return None;
    }

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let stride = w as usize;
    let x0 = x0 as usize;
    let y0 = y0 as usize;
    let idx00 = y0 * stride + x0;
    let idx10 = idx00 + 1;
    let idx01 = idx00 + stride;
    let idx11 = idx01 + 1;

    let raw = img.as_raw();
    const INV_255: f32 = 1.0 / 255.0;
    let p00 = raw[idx00] as f32 * INV_255;
    let p10 = raw[idx10] as f32 * INV_255;
    let p01 = raw[idx01] as f32 * INV_255;
    let p11 = raw[idx11] as f32 * INV_255;

    Some(
        (1.0 - fx) * (1.0 - fy) * p00
            + fx * (1.0 - fy) * p10
            + (1.0 - fx) * fy * p01
            + fx * fy * p11,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixelmap::{CameraIntrinsics, CameraModel, PixelMapper, RadialTangentialDistortion};

    #[test]
    fn test_bilinear_sample() {
        let mut img = GrayImage::new(4, 4);
        img.put_pixel(1, 1, image::Luma([100]));
        img.put_pixel(2, 1, image::Luma([200]));
        img.put_pixel(1, 2, image::Luma([100]));
        img.put_pixel(2, 2, image::Luma([200]));
        let val = bilinear_sample_u8(&img, 1.5, 1.5);
        let expected = 150.0 / 255.0;
        assert!(
            (val - expected).abs() < 0.01,
            "bilinear at midpoint should be ~{:.3}, got {:.3}",
            expected,
            val
        );
    }

    #[test]
    fn distortion_aware_sampler_no_camera_is_identity() {
        let mut img = GrayImage::new(8, 8);
        img.put_pixel(3, 4, image::Luma([255]));
        let s = DistortionAwareSampler::new(&img, None);
        let v = s.sample_checked(3.0, 4.0).unwrap();
        assert!(v > 0.99);
    }

    #[test]
    fn distortion_aware_sampler_maps_working_to_image() {
        let mut img = GrayImage::new(16, 16);
        img.put_pixel(8, 8, image::Luma([255]));
        let cam = CameraModel {
            intrinsics: CameraIntrinsics {
                fx: 1000.0,
                fy: 1000.0,
                cx: 8.0,
                cy: 8.0,
            },
            distortion: RadialTangentialDistortion {
                k1: -0.2,
                k2: 0.02,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            },
        };
        let s = DistortionAwareSampler::new(&img, Some(&cam as &dyn PixelMapper));
        let work = s.image_to_working_xy([8.0, 8.0]).unwrap();
        let back = s.working_to_image_xy(work).unwrap();
        assert!((back[0] - 8.0).abs() < 1e-4);
        assert!((back[1] - 8.0).abs() < 1e-4);
        let v = s.sample_checked(work[0], work[1]).unwrap();
        assert!(v > 0.99);
    }
}
