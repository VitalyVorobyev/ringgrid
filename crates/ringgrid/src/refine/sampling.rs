use image::GrayImage;

use crate::camera::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;
use crate::ring::inner_estimate::Polarity;
use crate::EllipseParams;

#[derive(Debug, Clone)]
pub(super) struct SampleOutcome {
    pub(super) points: Vec<[f64; 2]>,
    pub(super) score_sum: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OuterSampleConfig {
    pub(super) theta_samples: usize,
    pub(super) search_halfwidth_px: f32,
    pub(super) r_step_px: f32,
    pub(super) min_ring_depth: f32,
}

fn ellipse_direction_radius_px(e: &EllipseParams, dir: [f32; 2]) -> Option<f32> {
    let a = e.semi_axes[0] as f32;
    let b = e.semi_axes[1] as f32;
    if !a.is_finite() || !b.is_finite() || a <= 1e-3 || b <= 1e-3 {
        return None;
    }
    let phi = e.angle as f32;
    let (c, s) = (phi.cos(), phi.sin());
    // Rotate direction into ellipse frame: d' = R(-phi) * d
    let dx = dir[0];
    let dy = dir[1];
    let dxp = c * dx + s * dy;
    let dyp = -s * dx + c * dy;
    let denom = (dxp * dxp) / (a * a) + (dyp * dyp) / (b * b);
    if !denom.is_finite() || denom <= 1e-9 {
        return None;
    }
    Some(1.0 / denom.sqrt())
}

pub(super) fn sample_outer_points_around_ellipse(
    gray: &GrayImage,
    ellipse: &EllipseParams,
    mapper: Option<&dyn PixelMapper>,
    cfg: OuterSampleConfig,
    polarity: Polarity,
) -> SampleOutcome {
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let n_t = cfg.theta_samples.max(8);
    let cx = ellipse.center_xy[0] as f32;
    let cy = ellipse.center_xy[1] as f32;

    let hw = cfg.search_halfwidth_px.max(0.0);
    let step = cfg.r_step_px.clamp(0.25, 1.0);
    let n_ref = ((hw / step).ceil() as i32).max(1);

    let mut points = Vec::with_capacity(n_t);
    let mut score_sum = 0.0f32;

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let dir = [theta.cos(), theta.sin()];

        let r_pred = match ellipse_direction_radius_px(ellipse, dir) {
            Some(r) => r,
            None => continue,
        };

        let mut best_score = f32::NEG_INFINITY;
        let mut best_r = None::<f32>;

        for k in -n_ref..=n_ref {
            let r = r_pred + k as f32 * step;
            if r <= 0.5 {
                continue;
            }

            // dI/dr at r via small central difference
            let h = 0.25f32;
            if r <= h {
                continue;
            }

            let x1 = cx + dir[0] * (r + h);
            let y1 = cy + dir[1] * (r + h);
            let x0 = cx + dir[0] * (r - h);
            let y0 = cy + dir[1] * (r - h);
            let i1 = match sampler.sample_checked(x1, y1) {
                Some(v) => v,
                None => continue,
            };
            let i0 = match sampler.sample_checked(x0, y0) {
                Some(v) => v,
                None => continue,
            };
            let d = (i1 - i0) / (2.0 * h);

            let score = match polarity {
                Polarity::Pos => d,
                Polarity::Neg => -d,
            };

            if score > best_score {
                best_score = score;
                best_r = Some(r);
            }
        }

        let r = match best_r {
            Some(r) if best_score.is_finite() && best_score > 0.0 => r,
            _ => continue,
        };

        // Ring depth check across the chosen edge.
        let band = 2.0f32;
        let x_in = cx + dir[0] * (r - band);
        let y_in = cy + dir[1] * (r - band);
        let x_out = cx + dir[0] * (r + band);
        let y_out = cy + dir[1] * (r + band);
        let i_in = match sampler.sample_checked(x_in, y_in) {
            Some(v) => v,
            None => continue,
        };
        let i_out = match sampler.sample_checked(x_out, y_out) {
            Some(v) => v,
            None => continue,
        };
        let signed_depth = match polarity {
            Polarity::Pos => i_out - i_in,
            Polarity::Neg => i_in - i_out,
        };
        if signed_depth < cfg.min_ring_depth {
            continue;
        }

        let x = cx + dir[0] * r;
        let y = cy + dir[1] * r;
        points.push([x as f64, y as f64]);
        score_sum += best_score.max(0.0);
    }

    SampleOutcome { points, score_sum }
}
