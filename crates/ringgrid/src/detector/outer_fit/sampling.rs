use crate::conic::Ellipse;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleConfig};
use crate::ring::inner_estimate::Polarity;

const MAX_EXPECTED_RADIUS_DEVIATION_FRAC: f32 = 0.4;

/// Largest radius (px) at which an outer edge point is accepted on a ring
/// expected at `r_expected`: the upper bound of the per-ray expected-radius
/// gate (`MAX_EXPECTED_RADIUS_DEVIATION_FRAC`), i.e. the sampler's reach.
pub(crate) fn max_outer_edge_radius_px(r_expected: f32) -> f32 {
    r_expected + r_expected.max(1.0) * MAX_EXPECTED_RADIUS_DEVIATION_FRAC
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

/// How far past an outer edge point (px, as a fraction of its radius, at
/// least [`MIN_EDGE_REACH_PX`]) the depth gate follows the edge's rise and
/// measures the ray's swing: wide enough for the rise of a heavily blurred
/// edge (σ = 6 px on a 50 px ring) to complete.
const EDGE_REACH_FRAC: f32 = 0.5;
const MIN_EDGE_REACH_PX: f32 = 4.0;

/// Fraction of its ray's own ink-to-paper swing an outer edge point's step
/// must carry.
///
/// The step is the edge's full monotone rise through the point (ink plateau
/// to paper plateau, however wide blur makes it); the swing is max - min of
/// the ray from the marker centre to the reach past the edge. Both scale
/// with the image gain and offset, and both widen together under blur, so the
/// gate passes the same edges in a dark, low-contrast or defocused image as
/// in a sharp bright one. What fails it is a weak edge inside a marker of
/// stronger contrast — a code-band or inner-ring transition that the refine
/// window picked instead of the ring's ink-to-paper edge.
const MIN_EDGE_SWING_FRACTION: f32 = 0.4;

/// Whether the edge at radius `r` on a ray is a real ring edge of polarity
/// `pol`: its rise reaches both the absolute noise floor `min_ring_depth` and
/// [`MIN_EDGE_SWING_FRACTION`] of the ray's swing.
///
/// The rise is followed outward (inward) from `r` at 1 px spacing while the
/// polarity-signed intensity stays within a small tolerance of its running
/// maximum (minimum): a reversal larger than that ends it.
fn passes_ring_depth_gate(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r: f32,
    pol: Polarity,
    min_ring_depth: f32,
) -> bool {
    // Polarity-signed intensity: the edge always rises outward.
    let at = |dr: f32| {
        let v = sampler.sample_checked(
            ray_origin[0] + ray_dir[0] * (r + dr),
            ray_origin[1] + ray_dir[1] * (r + dr),
        )?;
        Some(match pol {
            Polarity::Pos => v,
            Polarity::Neg => -v,
        })
    };
    let n_out = (EDGE_REACH_FRAC * r).max(MIN_EDGE_REACH_PX).ceil() as usize;
    let n_in = r.floor() as usize;
    // Samples from the reach past the edge (index 0) inward to the centre
    // (index n_out + n_in); `None` off the image.
    let profile: Vec<Option<f32>> = (0..=n_out + n_in)
        .map(|i| at(n_out as f32 - i as f32))
        .collect();
    let Some(s0) = profile[n_out] else {
        return false;
    };
    let (lo, hi) = profile
        .iter()
        .flatten()
        .fold((s0, s0), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    let swing = hi - lo;
    // A reversal ends the rise once it exceeds the noise floor or a tenth of
    // the swing, whichever is smaller (the latter keeps the walk
    // gain-invariant on a dark, clean image).
    let tol = min_ring_depth.min(MAX_RISE_REVERSAL_FRAC * swing);
    // Outward from the edge: the rise's top; inward: its bottom.
    let top = rise_extreme(
        profile[..n_out].iter().rev(),
        s0,
        |s, top| s >= top - tol,
        f32::max,
    );
    let bottom = rise_extreme(
        profile[n_out + 1..].iter(),
        s0,
        |s, b| s <= b + tol,
        f32::min,
    );
    let rise = top - bottom;
    // The point must sit on the rise, not on a plateau beside it.
    let on_rise = (s0 - bottom).min(top - s0) >= MIN_RISE_POSITION_FRAC * rise;
    on_rise && rise >= min_ring_depth && rise >= MIN_EDGE_SWING_FRACTION * swing
}

/// Least share of the rise an edge point's own level must lie above the
/// rise's bottom and below its top: the point is on the edge's transition,
/// not on the ink or paper plateau next to it.
const MIN_RISE_POSITION_FRAC: f32 = 0.2;

/// Largest reversal, as a fraction of the ray's swing, that the depth gate's
/// rise walk tolerates (it also never tolerates more than the noise floor).
const MAX_RISE_REVERSAL_FRAC: f32 = 0.1;

/// Walk `samples` from `start` while each continues the run (`cont(sample,
/// extreme)`), returning the run's extreme (`pick` of the samples so far).
fn rise_extreme<'a>(
    samples: impl Iterator<Item = &'a Option<f32>>,
    start: f32,
    cont: impl Fn(f32, f32) -> bool,
    pick: impl Fn(f32, f32) -> f32,
) -> f32 {
    let mut extreme = start;
    for s in samples {
        match s {
            Some(s) if cont(*s, extreme) => extreme = pick(extreme, *s),
            _ => break,
        }
    }
    extreme
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
    hyp: &crate::ring::outer_estimate::OuterHypothesis,
    r_expected: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let collect = |ray_r0: &dyn Fn(f32) -> f32| {
        collect_outer_edge_points(
            sampler,
            center_prior,
            |theta| Some(ray_r0(theta)),
            Some(r_expected),
            pol,
            edge_cfg,
            refine_halfwidth_px,
        )
    };
    let r_star = hyp.r_outer_px;
    let Some(model) = hyp.radius_model else {
        return collect(&|_| r_star);
    };
    // Eccentric markers: recenter each ray's local search on the
    // eccentricity model, so the narrow refine window tracks the elliptical
    // edge instead of losing the rays near the major/minor axes. Anchored at
    // the aggregated peak `r_star` (the model's differential offset added) or
    // at the model's own mean radius `c0`: the two agree on a mildly
    // eccentric ring, but the attach gate lets them differ by up to a quarter
    // search window, and which one sits on the edge depends on the ring (the
    // aggregate of a strongly eccentric ring smears over its radius range).
    // When they differ by more than the window, both are sampled and the
    // centring that finds more edge points seeds the fit.
    let anchored = collect(&|theta| r_star + (model.radius_at(theta) - model.c0));
    if (r_star - model.c0).abs() <= refine_halfwidth_px {
        return anchored;
    }
    let own = collect(&|theta| model.radius_at(theta));
    if own.0.len() > anchored.0.len() {
        own
    } else {
        anchored
    }
}

/// Re-sample the outer edge along every ray at the radius where the ray
/// meets a fitted `ellipse` — the exact ring model, where the hypothesis's
/// second-harmonic radius model is only a first-order one (its error grows
/// past the refine window for axis ratios below ~0.7, and a sharp edge then
/// falls outside the window on most rays). Points keep the same per-ray gate
/// against the scale prior's expected radius as the hypothesis pass.
pub(super) fn collect_outer_edge_points_near_ellipse(
    sampler: DistortionAwareSampler<'_>,
    center_prior: [f32; 2],
    ellipse: &Ellipse,
    r_expected: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let ray_r0 = |theta: f32| ray_ellipse_radius(ellipse, center_prior, [theta.cos(), theta.sin()]);
    collect_outer_edge_points(
        sampler,
        center_prior,
        ray_r0,
        Some(r_expected),
        pol,
        edge_cfg,
        refine_halfwidth_px,
    )
}

/// Distance from `origin` along the unit direction `dir` to the boundary of
/// `ellipse`, or `None` when the ray misses it.
pub(super) fn ray_ellipse_radius(
    ellipse: &Ellipse,
    origin: [f32; 2],
    dir: [f32; 2],
) -> Option<f32> {
    let (s, c) = ellipse.angle.sin_cos();
    let ox = f64::from(origin[0]) - ellipse.cx;
    let oy = f64::from(origin[1]) - ellipse.cy;
    let (dx, dy) = (f64::from(dir[0]), f64::from(dir[1]));
    // Origin and direction in the ellipse's axis frame, scaled to the unit circle.
    let (qx, qy) = (
        (c * ox + s * oy) / ellipse.a,
        (-s * ox + c * oy) / ellipse.b,
    );
    let (ux, uy) = (
        (c * dx + s * dy) / ellipse.a,
        (-s * dx + c * dy) / ellipse.b,
    );
    let qa = ux * ux + uy * uy;
    let qb = 2.0 * (qx * ux + qy * uy);
    let qc = qx * qx + qy * qy - 1.0;
    let disc = qb * qb - 4.0 * qa * qc;
    if qa <= 0.0 || disc < 0.0 {
        return None;
    }
    let r = (-qb + disc.sqrt()) / (2.0 * qa);
    (r.is_finite() && r > 0.0).then_some(r as f32)
}

/// Outer edge points on `edge_cfg.n_rays` rays from `center_prior`, each
/// searched within `refine_halfwidth_px` of `ray_r0(theta)` and, when
/// `r_expected` is given, kept only within `MAX_EXPECTED_RADIUS_DEVIATION_FRAC`
/// of it.
fn collect_outer_edge_points(
    sampler: DistortionAwareSampler<'_>,
    center_prior: [f32; 2],
    ray_r0: impl Fn(f32) -> Option<f32>,
    r_expected: Option<f32>,
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
        let Some(ray_r0) = ray_r0(theta) else {
            continue;
        };

        let Some(r) = pick_best_radius_on_ray(
            sampler,
            center_prior,
            ray_dir,
            ray_r0,
            pol,
            edge_cfg,
            refine_halfwidth_px,
        ) else {
            continue;
        };

        if r_expected
            .is_some_and(|re| (r - re).abs() > re.max(1.0) * MAX_EXPECTED_RADIUS_DEVIATION_FRAC)
        {
            continue;
        }

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

/// Compute the largest angular gap (radians) between consecutive edge points
/// around a center. Returns `2*PI` if fewer than 2 points are provided.
pub(crate) fn max_angular_gap(center: [f64; 2], points: &[[f64; 2]]) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    if points.len() < 2 {
        return tau;
    }
    let mut angles: Vec<f64> = points
        .iter()
        .map(|p| (p[1] - center[1]).atan2(p[0] - center[0]))
        .collect();
    angles.sort_by(|a, b| a.total_cmp(b));
    let mut gap = 0.0f64;
    for i in 1..angles.len() {
        gap = gap.max(angles[i] - angles[i - 1]);
    }
    // Wrap-around gap
    let wrap = tau - (angles.last().unwrap() - angles.first().unwrap());
    gap.max(wrap)
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
        crate::test_utils::draw_ring_image(w, h, center, outer_radius, inner_radius, 24, 230)
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

    /// Concentric bands around `center`: `levels[i]` from `radii[i - 1]` (0
    /// for the first) out to `radii[i]`, the last level beyond.
    fn banded_image(center: [f32; 2], radii: &[f32], levels: &[u8]) -> GrayImage {
        GrayImage::from_fn(96, 96, |x, y| {
            let r = (x as f32 - center[0]).hypot(y as f32 - center[1]);
            let band = radii.iter().take_while(|&&b| r >= b).count();
            Luma([levels[band]])
        })
    }

    #[test]
    fn ring_depth_gate_is_gain_invariant() {
        // Paper / ink / paper: the same ring at full contrast and at 18 grey
        // levels on a dark background (a step the old absolute 0.08 floor,
        // 20 levels, rejected).
        let center = [48.0f32, 48.0];
        let default_floor = EdgeSampleConfig::default().min_ring_depth;
        for (paper, ink) in [(230u8, 24u8), (48, 30)] {
            let img = banded_image(center, &[10.0, 24.0], &[paper, ink, paper]);
            let sampler = DistortionAwareSampler::new(&img, None);
            assert!(
                passes_ring_depth_gate(
                    sampler,
                    center,
                    [1.0, 0.0],
                    23.5,
                    Polarity::Pos,
                    default_floor
                ),
                "paper {paper} / ink {ink}"
            );
        }
    }

    #[test]
    fn ring_depth_gate_rejects_a_weak_edge_inside_a_strong_marker() {
        // Paper centre, ink, a slightly lighter band from r = 14, paper from
        // r = 24: the r = 14 step (15 levels, above the noise floor) carries a
        // tenth of the ray's 160-level swing; the true outer edge carries all.
        let center = [48.0f32, 48.0];
        let img = banded_image(center, &[6.0, 14.0, 24.0], &[200, 40, 55, 200]);
        let sampler = DistortionAwareSampler::new(&img, None);
        let floor = EdgeSampleConfig::default().min_ring_depth;
        let gate = |r| passes_ring_depth_gate(sampler, center, [1.0, 0.0], r, Polarity::Pos, floor);
        assert!(!gate(13.5));
        assert!(gate(23.5));
        // On the paper plateau 3 px past the edge: not an edge point.
        assert!(!gate(26.5));
    }

    #[test]
    fn ring_depth_gate_passes_a_heavily_blurred_edge() {
        // Ink to paper over a 16 px linear ramp (blur sigma ~ 5 px): the 4 px
        // step at its centre is 0.25 of the contrast, the full rise all of it.
        let center = [48.0f32, 48.0];
        let img = GrayImage::from_fn(96, 96, |x, y| {
            let r = (x as f32 - center[0]).hypot(y as f32 - center[1]);
            let t = ((r - 22.0) / 16.0).clamp(0.0, 1.0);
            Luma([(40.0 + 160.0 * t).round() as u8])
        });
        let sampler = DistortionAwareSampler::new(&img, None);
        let floor = EdgeSampleConfig::default().min_ring_depth;
        assert!(passes_ring_depth_gate(
            sampler,
            center,
            [1.0, 0.0],
            30.0,
            Polarity::Pos,
            floor
        ));
    }

    #[test]
    fn collect_outer_edge_points_rejects_rays_far_from_expected_radius() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);
        let sampler = DistortionAwareSampler::new(&img, None);
        let edge_cfg = EdgeSampleConfig {
            r_min: 1.5,
            r_max: 40.0,
            r_step: 0.5,
            n_rays: 24,
            ..EdgeSampleConfig::default()
        };

        let hyp = crate::ring::outer_estimate::OuterHypothesis {
            r_outer_px: inner_radius,
            peak_strength: 1.0,
            theta_consistency: 1.0,
            radius_model: None,
        };
        let (points, radii) = collect_outer_edge_points_near_radius(
            sampler,
            center,
            &hyp,
            outer_radius,
            Polarity::Pos,
            &edge_cfg,
            2.0,
        );

        assert!(
            points.is_empty(),
            "contaminated inner-ring rays should be rejected"
        );
        assert!(
            radii.is_empty(),
            "no radii should survive the expected-radius gate"
        );
    }

    #[test]
    fn max_angular_gap_full_circle() {
        let center = [0.0, 0.0];
        let n = 48;
        let points: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let theta = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
                [theta.cos() * 10.0, theta.sin() * 10.0]
            })
            .collect();
        let gap = max_angular_gap(center, &points);
        let expected = 2.0 * std::f64::consts::PI / n as f64;
        assert!(
            (gap - expected).abs() < 1e-10,
            "full circle gap={gap:.6} expected={expected:.6}"
        );
    }

    #[test]
    fn max_angular_gap_half_circle() {
        let center = [0.0, 0.0];
        let n = 24;
        let points: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let theta = i as f64 * std::f64::consts::PI / (n - 1) as f64;
                [theta.cos() * 10.0, theta.sin() * 10.0]
            })
            .collect();
        let gap = max_angular_gap(center, &points);
        assert!(
            gap > std::f64::consts::PI - 0.2,
            "half circle gap={gap:.4} should be ~PI"
        );
    }

    #[test]
    fn max_angular_gap_single_point() {
        let gap = max_angular_gap([0.0, 0.0], &[[1.0, 0.0]]);
        assert!(
            (gap - 2.0 * std::f64::consts::PI).abs() < 1e-10,
            "single point gap={gap:.6}"
        );
    }

    #[test]
    fn max_angular_gap_empty() {
        let gap = max_angular_gap([0.0, 0.0], &[]);
        assert!(
            (gap - 2.0 * std::f64::consts::PI).abs() < 1e-10,
            "empty gap={gap:.6}"
        );
    }
}
