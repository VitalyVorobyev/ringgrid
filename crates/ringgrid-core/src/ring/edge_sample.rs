//! Edge sampling primitives shared by ring detection stages.

use image::GrayImage;

/// Configuration for radial edge sampling.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct EdgeSampleResult {
    /// Candidate center (unchanged from input).
    pub center: [f32; 2],
    /// Outer edge points (sub-pixel positions in image coords).
    pub outer_points: Vec<[f64; 2]>,
    /// Inner edge points (sub-pixel positions in image coords).
    pub inner_points: Vec<[f64; 2]>,
    /// Median outer radius from ray measurements.
    pub outer_radius: f32,
    /// Median inner radius from ray measurements.
    pub inner_radius: f32,
    /// Per-ray outer radii for rays where both edges were found.
    pub outer_radii: Vec<f32>,
    /// Per-ray inner radii for rays where both edges were found.
    pub inner_radii: Vec<f32>,
    /// Number of rays with both edges found.
    pub n_good_rays: usize,
    /// Total rays attempted.
    pub n_total_rays: usize,
}

/// Sample a grayscale image at sub-pixel position using bilinear interpolation.
/// Returns intensity in [0, 1].
#[inline]
pub fn bilinear_sample_u8(img: &GrayImage, x: f32, y: f32) -> f32 {
    let (w, h) = img.dimensions();
    if x < 0.0 || y < 0.0 {
        return 0.0;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    if x0 + 1 >= w || y0 + 1 >= h {
        return 0.0;
    }
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = img.get_pixel(x0, y0)[0] as f32 / 255.0;
    let p10 = img.get_pixel(x0 + 1, y0)[0] as f32 / 255.0;
    let p01 = img.get_pixel(x0, y0 + 1)[0] as f32 / 255.0;
    let p11 = img.get_pixel(x0 + 1, y0 + 1)[0] as f32 / 255.0;
    (1.0 - fx) * (1.0 - fy) * p00 + fx * (1.0 - fy) * p10 + (1.0 - fx) * fy * p01 + fx * fy * p11
}

/// Sample a grayscale image at sub-pixel position using bilinear interpolation.
/// Returns intensity in [0, 1] or `None` if sampling is out of bounds.
#[inline]
pub fn bilinear_sample_u8_checked(img: &GrayImage, x: f32, y: f32) -> Option<f32> {
    let (w, h) = img.dimensions();
    if x < 0.0 || y < 0.0 {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    if x0 + 1 >= w || y0 + 1 >= h {
        return None;
    }
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = img.get_pixel(x0, y0)[0] as f32 / 255.0;
    let p10 = img.get_pixel(x0 + 1, y0)[0] as f32 / 255.0;
    let p01 = img.get_pixel(x0, y0 + 1)[0] as f32 / 255.0;
    let p11 = img.get_pixel(x0 + 1, y0 + 1)[0] as f32 / 255.0;
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
}
