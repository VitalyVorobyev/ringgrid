//! Per-candidate radial edge sampling.
//!
//! For each candidate center, casts radial rays and locates inner/outer ring
//! edges from the intensity profile. Under heavy blur the two rings merge into
//! one broad dark band; the strategy is to find the intensity minimum and then
//! locate the steepest inward and outward transitions.

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

/// Sample edges along radial rays from a candidate center.
///
/// Returns `None` if insufficient rays have valid ring detections.
pub fn sample_edges(
    gray: &GrayImage,
    center: [f32; 2],
    config: &EdgeSampleConfig,
) -> Option<EdgeSampleResult> {
    let (w, h) = gray.dimensions();
    let cx = center[0];
    let cy = center[1];

    let mut outer_points = Vec::with_capacity(config.n_rays);
    let mut inner_points = Vec::with_capacity(config.n_rays);
    let mut outer_radii = Vec::with_capacity(config.n_rays);
    let mut inner_radii = Vec::with_capacity(config.n_rays);

    let n_steps = ((config.r_max - config.r_min) / config.r_step).ceil() as usize + 1;

    for i in 0..config.n_rays {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / config.n_rays as f32;
        let dx = angle.cos();
        let dy = angle.sin();

        // Sample intensity profile along the ray
        let mut profile = Vec::with_capacity(n_steps);
        let mut radii = Vec::with_capacity(n_steps);
        let mut r = config.r_min;
        while r <= config.r_max {
            let px = cx + dx * r;
            let py = cy + dy * r;
            if px < 0.0 || py < 0.0 || px >= (w - 1) as f32 || py >= (h - 1) as f32 {
                break;
            }
            profile.push(bilinear_sample_u8(gray, px, py));
            radii.push(r);
            r += config.r_step;
        }

        if profile.len() < 5 {
            continue;
        }

        // Find the intensity minimum along this ray (the dark ring band)
        let mut min_val = f32::MAX;
        let mut min_idx = 0;
        for (j, &v) in profile.iter().enumerate() {
            if v < min_val {
                min_val = v;
                min_idx = j;
            }
        }

        // Background intensity: use the endpoints
        let bg_inner = profile[0];
        let bg_outer = *profile.last().unwrap();
        let bg = bg_inner.max(bg_outer);
        let depth = bg - min_val;

        if depth < config.min_ring_depth {
            continue;
        }

        // Find inner edge: steepest negative gradient going inward from minimum
        // (i.e., the point where intensity drops most sharply going outward
        //  on the inner side of the ring)
        let mut best_inner_idx = None;
        let mut best_inner_grad = 0.0f32;
        if min_idx > 0 {
            for j in 0..min_idx {
                // gradient: profile[j+1] - profile[j] (negative means dropping)
                let grad = profile[j + 1] - profile[j];
                if grad < best_inner_grad {
                    best_inner_grad = grad;
                    best_inner_idx = Some(j);
                }
            }
        }

        // Find outer edge: steepest positive gradient going outward from minimum
        let mut best_outer_idx = None;
        let mut best_outer_grad = 0.0f32;
        if min_idx + 1 < profile.len() {
            for j in min_idx..(profile.len() - 1) {
                let grad = profile[j + 1] - profile[j];
                if grad > best_outer_grad {
                    best_outer_grad = grad;
                    best_outer_idx = Some(j);
                }
            }
        }

        // Require both edges found
        if let (Some(ii), Some(oi)) = (best_inner_idx, best_outer_idx) {
            // Sub-pixel refinement: the edge is between profile[idx] and profile[idx+1],
            // interpolate to where the gradient peak is
            let inner_r = refine_edge_position(&radii, &profile, ii, true);
            let outer_r = refine_edge_position(&radii, &profile, oi, false);

            if outer_r > inner_r && (outer_r - inner_r) > 0.5 {
                let inner_x = cx + dx * inner_r;
                let inner_y = cy + dy * inner_r;
                let outer_x = cx + dx * outer_r;
                let outer_y = cy + dy * outer_r;

                inner_points.push([inner_x as f64, inner_y as f64]);
                outer_points.push([outer_x as f64, outer_y as f64]);
                inner_radii.push(inner_r);
                outer_radii.push(outer_r);
            }
        }
    }

    let n_good = outer_points.len();
    if n_good < config.min_rays_with_ring {
        return None;
    }

    // Median radii (use sorted copies; keep original order for debug)
    let mut outer_sorted = outer_radii.clone();
    let mut inner_sorted = inner_radii.clone();
    outer_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    inner_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_outer = outer_sorted[outer_sorted.len() / 2];
    let median_inner = inner_sorted[inner_sorted.len() / 2];

    Some(EdgeSampleResult {
        center,
        outer_points,
        inner_points,
        outer_radius: median_outer,
        inner_radius: median_inner,
        outer_radii,
        inner_radii,
        n_good_rays: n_good,
        n_total_rays: config.n_rays,
    })
}

/// Refine edge position using parabolic interpolation on the gradient profile.
fn refine_edge_position(radii: &[f32], profile: &[f32], idx: usize, _is_inner: bool) -> f32 {
    // Compute gradients
    if idx == 0 || idx + 2 >= profile.len() {
        // Can't do parabolic fit, return midpoint
        return (radii[idx] + radii[idx + 1]) / 2.0;
    }

    let g_prev = profile[idx] - profile[idx - 1];
    let g_curr = profile[idx + 1] - profile[idx];
    let g_next = profile[idx + 2] - profile[idx + 1];

    // Parabolic interpolation on the gradient magnitude
    let g_prev_abs = g_prev.abs();
    let g_curr_abs = g_curr.abs();
    let g_next_abs = g_next.abs();

    let denom = g_prev_abs - 2.0 * g_curr_abs + g_next_abs;
    if denom.abs() < 1e-6 {
        return (radii[idx] + radii[idx + 1]) / 2.0;
    }

    let offset = 0.5 * (g_prev_abs - g_next_abs) / denom;
    let r_base = (radii[idx] + radii[idx + 1]) / 2.0;
    let step = radii[1] - radii[0];
    (r_base + offset * step).clamp(radii[idx], radii[idx + 1])
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

    #[test]
    fn test_edge_sample_on_ring() {
        // Create a ring image: bright background, dark ring at radius 10
        let (w, h) = (64, 64);
        let (cx, cy) = (32.0f32, 32.0f32);
        let inner_r = 7.0f32;
        let outer_r = 10.0f32;
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = if dist >= inner_r && dist <= outer_r {
                    30u8
                } else {
                    200u8
                };
                img.put_pixel(x, y, image::Luma([val]));
            }
        }

        let config = EdgeSampleConfig {
            n_rays: 32,
            r_max: 16.0,
            r_min: 2.0,
            r_step: 0.5,
            min_ring_depth: 0.1,
            min_rays_with_ring: 8,
        };

        let result = sample_edges(&img, [cx, cy], &config);
        assert!(result.is_some(), "should detect ring edges");
        let result = result.unwrap();
        assert!(result.n_good_rays >= 8, "should have enough good rays");

        // Median radii should be roughly correct
        assert!(
            (result.inner_radius - inner_r).abs() < 3.0,
            "inner radius {:.1} should be near {:.1}",
            result.inner_radius,
            inner_r
        );
        assert!(
            (result.outer_radius - outer_r).abs() < 3.0,
            "outer radius {:.1} should be near {:.1}",
            result.outer_radius,
            outer_r
        );
    }
}
