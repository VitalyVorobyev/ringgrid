//! Shared radial-estimator core used by inner/outer ring estimators.
//!
//! The core is policy-free: it samples radial intensity profiles over theta,
//! converts them to smoothed radial derivatives, and provides common
//! aggregation/coverage helpers. Stage-specific selection and gating remain in
//! `inner_estimate` and `outer_estimate`.

use crate::marker::AngularAggregator;

use super::radial_profile::{self, Polarity};

/// Uniform radial sampling grid over a search window.
#[derive(Debug, Clone)]
pub(crate) struct RadialSampleGrid {
    pub(crate) r_samples: Vec<f32>,
    pub(crate) r_step: f32,
}

impl RadialSampleGrid {
    /// Build a uniform radial grid over `[window_min, window_max]`.
    ///
    /// Returns `None` when the window is invalid or `n_r < 2`.
    pub(crate) fn from_window(search_window: [f32; 2], n_r: usize) -> Option<Self> {
        if n_r < 2 || search_window[1] <= search_window[0] + 1e-6 {
            return None;
        }
        let r_step = (search_window[1] - search_window[0]) / (n_r as f32 - 1.0);
        if !r_step.is_finite() || r_step <= 0.0 {
            return None;
        }
        let r_samples: Vec<f32> = (0..n_r)
            .map(|i| search_window[0] + i as f32 * r_step)
            .collect();
        Some(Self { r_samples, r_step })
    }
}

/// Common radial scan output for one search window.
///
/// Derivative curves are stored as `[theta][radius]` for valid theta samples.
#[derive(Debug, Clone)]
pub(crate) struct RadialScanResult {
    pub(crate) grid: RadialSampleGrid,
    pub(crate) n_total_theta: usize,
    pub(crate) n_valid_theta: usize,
    curves_flat: Vec<f32>,
    per_theta_peak_pos: Vec<f32>,
    per_theta_peak_neg: Vec<f32>,
}

impl RadialScanResult {
    /// Fraction of attempted theta samples with valid in-bounds profiles.
    #[inline]
    pub(crate) fn coverage(&self) -> f32 {
        self.n_valid_theta as f32 / self.n_total_theta.max(1) as f32
    }

    /// Per-theta peak radii for the requested derivative polarity.
    #[inline]
    pub(crate) fn per_theta_peaks(&self, pol: Polarity) -> &[f32] {
        let peaks = match pol {
            Polarity::Pos => &self.per_theta_peak_pos,
            Polarity::Neg => &self.per_theta_peak_neg,
        };
        debug_assert_eq!(
            peaks.len(),
            self.n_valid_theta,
            "requested per-theta peaks were not tracked in this scan"
        );
        peaks
    }

    /// Aggregate valid theta curves into one radial response profile.
    pub(crate) fn aggregate_response(&self, agg: &AngularAggregator) -> Vec<f32> {
        let n_r = self.grid.r_samples.len();
        let mut agg_resp = vec![0.0f32; n_r];
        if self.n_valid_theta == 0 {
            return agg_resp;
        }

        let mut scratch = Vec::<f32>::with_capacity(self.n_valid_theta);
        for (ri, out) in agg_resp.iter_mut().enumerate() {
            scratch.clear();
            for ti in 0..self.n_valid_theta {
                scratch.push(self.curves_flat[ti * n_r + ri]);
            }
            *out = radial_profile::aggregate(&mut scratch, agg);
        }
        agg_resp
    }
}

/// Run a shared radial derivative sweep over theta.
///
/// `sample_theta_profile` must fill `i_vals` with intensity samples at the
/// provided normalized/canonical `r_samples` for one direction `(ct, st)`.
/// It returns `true` when all samples were valid (in-bounds), `false` to skip
/// this theta sample.
pub(crate) fn scan_radial_derivatives<F>(
    grid: RadialSampleGrid,
    n_theta: usize,
    track_pos_peaks: bool,
    track_neg_peaks: bool,
    mut sample_theta_profile: F,
) -> RadialScanResult
where
    F: FnMut(f32, f32, &[f32], &mut [f32]) -> bool,
{
    let n_r = grid.r_samples.len();
    let n_total_theta = n_theta.max(1);

    let mut curves_flat = Vec::<f32>::with_capacity(n_total_theta * n_r);
    let mut per_theta_peak_pos =
        Vec::<f32>::with_capacity(if track_pos_peaks { n_total_theta } else { 0 });
    let mut per_theta_peak_neg =
        Vec::<f32>::with_capacity(if track_neg_peaks { n_total_theta } else { 0 });

    let mut i_vals = vec![0.0f32; n_r];
    let mut d_vals = vec![0.0f32; n_r];
    let mut n_valid_theta = 0usize;

    let d_theta = 2.0 * std::f32::consts::PI / n_total_theta as f32;
    let c_step = d_theta.cos();
    let s_step = d_theta.sin();
    let mut ct = 1.0f32;
    let mut st = 0.0f32;
    for _ in 0..n_total_theta {
        if sample_theta_profile(ct, st, &grid.r_samples, &mut i_vals) {
            radial_profile::radial_derivative_into(&i_vals, grid.r_step, &mut d_vals);
            radial_profile::smooth_3point(&mut d_vals);
            curves_flat.extend_from_slice(&d_vals);
            if track_pos_peaks {
                per_theta_peak_pos
                    .push(grid.r_samples[radial_profile::peak_idx(&d_vals, Polarity::Pos)]);
            }
            if track_neg_peaks {
                per_theta_peak_neg
                    .push(grid.r_samples[radial_profile::peak_idx(&d_vals, Polarity::Neg)]);
            }
            n_valid_theta += 1;
        }

        let next_ct = ct * c_step - st * s_step;
        let next_st = st * c_step + ct * s_step;
        ct = next_ct;
        st = next_st;
    }

    RadialScanResult {
        grid,
        n_total_theta,
        n_valid_theta,
        curves_flat,
        per_theta_peak_pos,
        per_theta_peak_neg,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use image::{GrayImage, Luma};

    use super::*;
    use crate::marker::AngularAggregator;
    use crate::ring::edge_sample::DistortionAwareSampler;

    fn blur_gray(img: &GrayImage, sigma: f32) -> GrayImage {
        let (w, h) = img.dimensions();
        let mut f = image::ImageBuffer::<Luma<f32>, Vec<f32>>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                f.put_pixel(x, y, Luma([img.get_pixel(x, y)[0] as f32 / 255.0]));
            }
        }
        let blurred = imageproc::filter::gaussian_blur_f32(&f, sigma);
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = blurred.get_pixel(x, y)[0].clamp(0.0, 1.0);
                out.put_pixel(x, y, Luma([(v * 255.0).round() as u8]));
            }
        }
        out
    }

    fn draw_circular_dark_ring(
        w: u32,
        h: u32,
        center: [f32; 2],
        inner_radius: f32,
        outer_radius: f32,
    ) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let r = (dx * dx + dy * dy).sqrt();
                let val = if r >= inner_radius && r <= outer_radius {
                    0.10f32
                } else {
                    0.90f32
                };
                img.put_pixel(x, y, Luma([(val * 255.0).round() as u8]));
            }
        }
        img
    }

    fn draw_elliptic_dark_ring(
        w: u32,
        h: u32,
        center: [f32; 2],
        axes: [f32; 2],
        angle: f32,
        inner_ratio: f32,
    ) -> GrayImage {
        let ca = angle.cos();
        let sa = angle.sin();
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let xr = ca * dx + sa * dy;
                let yr = -sa * dx + ca * dy;
                let rho = ((xr / axes[0]).powi(2) + (yr / axes[1]).powi(2)).sqrt();
                let val = if rho >= inner_ratio && rho <= 1.0 {
                    0.12f32
                } else {
                    0.88f32
                };
                img.put_pixel(x, y, Luma([(val * 255.0).round() as u8]));
            }
        }
        img
    }

    #[test]
    fn shared_scan_circular_path_recovers_outer_edge_with_unit_tolerance() {
        let center = [72.0f32, 66.0f32];
        let r_outer = 27.35f32;
        let img = blur_gray(
            &draw_circular_dark_ring(160, 152, center, 16.0, r_outer),
            1.1,
        );

        let grid = RadialSampleGrid::from_window([23.0, 31.0], 97).expect("valid grid");
        let sampler = DistortionAwareSampler::new(&img, None);
        let scan = scan_radial_derivatives(grid, 72, true, false, |ct, st, r_samples, i_vals| {
            for (ri, &r) in r_samples.iter().enumerate() {
                let x = center[0] + ct * r;
                let y = center[1] + st * r;
                let Some(v) = sampler.sample_checked(x, y) else {
                    return false;
                };
                i_vals[ri] = v;
            }
            true
        });

        assert_abs_diff_eq!(scan.coverage(), 1.0, epsilon = 0.1);

        let agg = scan.aggregate_response(&AngularAggregator::Median);
        let idx = radial_profile::peak_idx(&agg, Polarity::Pos);
        let r_found = scan.grid.r_samples[idx];
        assert_abs_diff_eq!(r_found, r_outer, epsilon = 0.1);
    }

    #[test]
    fn shared_scan_ellipse_normalized_path_recovers_inner_edge_with_precision_tolerance() {
        let center = [84.0f32, 70.0f32];
        let axes = [34.0f32, 25.0f32];
        let angle = 0.31f32;
        let r_inner = 0.52f32;

        let img = blur_gray(
            &draw_elliptic_dark_ring(176, 154, center, axes, angle, r_inner),
            1.0,
        );

        let grid = RadialSampleGrid::from_window([0.38, 0.66], 101).expect("valid grid");
        let sampler = DistortionAwareSampler::new(&img, None);
        let ca = angle.cos();
        let sa = angle.sin();
        let scan = scan_radial_derivatives(grid, 96, false, true, |ct, st, r_samples, i_vals| {
            for (ri, &r) in r_samples.iter().enumerate() {
                let vx = axes[0] * r * ct;
                let vy = axes[1] * r * st;
                let x = center[0] + ca * vx - sa * vy;
                let y = center[1] + sa * vx + ca * vy;
                let Some(v) = sampler.sample_checked(x, y) else {
                    return false;
                };
                i_vals[ri] = v;
            }
            true
        });

        assert_abs_diff_eq!(scan.coverage(), 1.0, epsilon = 0.1);

        let agg = scan.aggregate_response(&AngularAggregator::Median);
        let idx = radial_profile::peak_idx(&agg, Polarity::Neg);
        let r_found = scan.grid.r_samples[idx];
        assert_abs_diff_eq!(r_found, r_inner, epsilon = 0.05);
    }
}
