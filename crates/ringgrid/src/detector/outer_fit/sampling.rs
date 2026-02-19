use crate::detector::DetectedMarker;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleConfig};
use crate::ring::inner_estimate::Polarity;

pub(crate) fn median_outer_radius_from_neighbors_px(
    projected_center: [f64; 2],
    markers: &[DetectedMarker],
    k: usize,
) -> Option<f32> {
    let mut candidates: Vec<(f64, f32)> = Vec::new();
    for m in markers {
        let r = match m.ellipse_outer {
            Some(v) => v.mean_axis(),
            None => continue,
        };
        let dx = m.center[0] - projected_center[0];
        let dy = m.center[1] - projected_center[1];
        let d2 = dx * dx + dy * dy;
        if d2.is_finite() {
            candidates.push((d2, r as f32));
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let radii: Vec<f32> = candidates
        .iter()
        .take(k.max(1).min(candidates.len()))
        .map(|(_, r)| *r)
        .collect();
    Some(median_f32(&radii))
}

fn sample_signed_radial_derivative(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r: f32,
) -> Option<f32> {
    let h = 0.25f32;
    if r <= h {
        return None;
    }

    let x1 = ray_origin[0] + ray_dir[0] * (r + h);
    let y1 = ray_origin[1] + ray_dir[1] * (r + h);
    let x0 = ray_origin[0] + ray_dir[0] * (r - h);
    let y0 = ray_origin[1] + ray_dir[1] * (r - h);
    let i1 = sampler.sample_checked(x1, y1)?;
    let i0 = sampler.sample_checked(x0, y0)?;
    Some((i1 - i0) / (2.0 * h))
}

fn passes_ring_depth_gate(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r: f32,
    pol: Polarity,
    min_ring_depth: f32,
) -> bool {
    let band = 2.0f32;
    let x_in = ray_origin[0] + ray_dir[0] * (r - band);
    let y_in = ray_origin[1] + ray_dir[1] * (r - band);
    let x_out = ray_origin[0] + ray_dir[0] * (r + band);
    let y_out = ray_origin[1] + ray_dir[1] * (r + band);
    let Some(i_in) = sampler.sample_checked(x_in, y_in) else {
        return false;
    };
    let Some(i_out) = sampler.sample_checked(x_out, y_out) else {
        return false;
    };
    let signed_depth = match pol {
        Polarity::Pos => i_out - i_in,
        Polarity::Neg => i_in - i_out,
    };
    signed_depth >= min_ring_depth
}

fn pick_best_radius_on_ray(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r0: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> Option<f32> {
    let refine_hw = refine_halfwidth_px.clamp(0.0, 4.0);
    let refine_step = edge_cfg.r_step.clamp(0.25, 1.0);
    let n_ref = ((refine_hw / refine_step).ceil() as i32).max(1);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_r = None::<f32>;

    for k in -n_ref..=n_ref {
        let r = r0 + k as f32 * refine_step;
        if r < edge_cfg.r_min || r > edge_cfg.r_max {
            continue;
        }

        let Some(d) = sample_signed_radial_derivative(sampler, ray_origin, ray_dir, r) else {
            continue;
        };
        let score = match pol {
            Polarity::Pos => d,
            Polarity::Neg => -d,
        };

        if score > best_score {
            best_score = score;
            best_r = Some(r);
        }
    }

    match best_r {
        Some(r) if best_score.is_finite() && best_score > 0.0 => Some(r),
        _ => None,
    }
}

pub(super) fn collect_outer_edge_points_near_radius(
    sampler: DistortionAwareSampler<'_>,
    center_prior: [f32; 2],
    r0: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let n_t = edge_cfg.n_rays.max(8);

    let mut outer_points = Vec::with_capacity(n_t);
    let mut outer_radii = Vec::with_capacity(n_t);

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let ray_dir = [theta.cos(), theta.sin()];

        let Some(r) = pick_best_radius_on_ray(
            sampler,
            center_prior,
            ray_dir,
            r0,
            pol,
            edge_cfg,
            refine_halfwidth_px,
        ) else {
            continue;
        };

        if !passes_ring_depth_gate(
            sampler,
            center_prior,
            ray_dir,
            r,
            pol,
            edge_cfg.min_ring_depth,
        ) {
            continue;
        }

        let x = center_prior[0] + ray_dir[0] * r;
        let y = center_prior[1] + ray_dir[1] * r;
        outer_points.push([x as f64, y as f64]);
        outer_radii.push(r);
    }

    (outer_points, outer_radii)
}

fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn draw_ring_image(
        w: u32,
        h: u32,
        center: [f32; 2],
        outer_radius: f32,
        inner_radius: f32,
    ) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let d = (dx * dx + dy * dy).sqrt();
                let pix = if d >= inner_radius && d <= outer_radius {
                    24u8
                } else {
                    230u8
                };
                img.put_pixel(x, y, Luma([pix]));
            }
        }
        img
    }

    #[test]
    fn pick_best_radius_on_ray_hits_synthetic_outer_edge() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);
        let sampler = DistortionAwareSampler::new(&img, None);

        let edge_cfg = EdgeSampleConfig {
            r_min: 1.5,
            r_max: 40.0,
            r_step: 0.5,
            ..EdgeSampleConfig::default()
        };

        let r = pick_best_radius_on_ray(
            sampler,
            center,
            [1.0, 0.0],
            outer_radius,
            Polarity::Pos,
            &edge_cfg,
            2.0,
        )
        .expect("expected radius pick on synthetic ring");

        assert!((r - outer_radius).abs() <= 1.5);
        assert!(passes_ring_depth_gate(
            sampler,
            center,
            [1.0, 0.0],
            r,
            Polarity::Pos,
            edge_cfg.min_ring_depth,
        ));
    }

    #[test]
    fn ring_depth_gate_rejects_low_depth_ray() {
        let img = GrayImage::from_pixel(64, 64, Luma([128u8]));
        let sampler = DistortionAwareSampler::new(&img, None);
        assert!(!passes_ring_depth_gate(
            sampler,
            [32.0, 32.0],
            [1.0, 0.0],
            16.0,
            Polarity::Pos,
            0.01,
        ));
    }
}
