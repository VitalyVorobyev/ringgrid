//! Photometric quality of a fitted outer edge: how sharp it is and how far
//! the marker sits from the image border — the two image conditions under
//! which a fitted ring's centre is least reliable (precision harness and the
//! owner's review of real images: blurred markers and markers near the
//! border hold most of the wrong centres).

use crate::conic::Ellipse;

use super::edge_sample::DistortionAwareSampler;

/// Rays cast through the outer rim.
const N_RAYS: usize = 32;
/// Paper sample distance outside the rim, in semi-major axes.
const OUTSIDE: f32 = 0.25;
/// Ink sample distance inside the rim, in semi-major axes (inside the outer
/// dark band of every supported marker).
const INSIDE: f32 = 0.12;
/// Radial step of the gradient search, px.
const STEP_PX: f32 = 0.25;
/// Fewest valid rays for a width.
const MIN_RAYS: usize = 8;

fn median(v: &mut [f32]) -> Option<f32> {
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied()
}

/// Width (px) of the outer edge's intensity ramp: the median paper-to-ink
/// step across the rim divided by the median peak radial gradient — for a
/// Gaussian-blurred step, `σ·√(2π) ≈ 2.5 σ`. Rays run from the ellipse centre
/// through `N_RAYS` rim points in the working frame. `None` when fewer than
/// `MIN_RAYS` rays are in the image or the edge has no step.
pub(crate) fn outer_edge_width_px(
    sampler: DistortionAwareSampler<'_>,
    outer: &Ellipse,
) -> Option<f32> {
    let (a, b) = (outer.a as f32, outer.b as f32);
    let (cx, cy) = (outer.cx as f32, outer.cy as f32);
    let (sa, ca) = (outer.angle as f32).sin_cos();
    let mut steps = Vec::with_capacity(N_RAYS);
    let mut grads = Vec::with_capacity(N_RAYS);
    for k in 0..N_RAYS {
        let t = std::f32::consts::TAU * k as f32 / N_RAYS as f32;
        let rim = [
            cx + ca * a * t.cos() - sa * b * t.sin(),
            cy + sa * a * t.cos() + ca * b * t.sin(),
        ];
        let (dx, dy) = (rim[0] - cx, rim[1] - cy);
        let len = dx.hypot(dy).max(1e-6);
        let at = |d: f32| sampler.sample_checked(rim[0] + dx / len * d, rim[1] + dy / len * d);
        let (Some(outside), Some(inside)) = (at(OUTSIDE * a), at(-INSIDE * a)) else {
            continue;
        };
        let mut peak = 0.0f32;
        let mut d = -INSIDE * a;
        while d <= OUTSIDE * a {
            if let (Some(i1), Some(i0)) = (at(d + 0.5 * STEP_PX), at(d - 0.5 * STEP_PX)) {
                peak = peak.max((i1 - i0).abs() / STEP_PX);
            }
            d += STEP_PX;
        }
        steps.push(outside - inside);
        grads.push(peak);
    }
    if steps.len() < MIN_RAYS {
        return None;
    }
    let step = median(&mut steps)?.abs();
    let grad = median(&mut grads)?;
    (step > 0.0 && grad > 0.0).then(|| step / grad)
}

/// Distance from a working-frame point to the nearest image border, in the
/// image frame, in units of `radius_px`. `0` when the point maps outside.
pub(crate) fn border_distance_radii(
    sampler: DistortionAwareSampler<'_>,
    center: [f64; 2],
    radius_px: f64,
) -> f32 {
    let Some(p) = sampler.working_to_image_xy([center[0] as f32, center[1] as f32]) else {
        return 0.0;
    };
    let (w, h) = sampler.image_size();
    let d = p[0]
        .min(p[1])
        .min(w as f32 - 1.0 - p[0])
        .min(h as f32 - 1.0 - p[1]);
    (d.max(0.0) / radius_px.max(1e-6) as f32).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn disk(blur: f32) -> GrayImage {
        let img = crate::test_utils::draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0, 30, 220);
        crate::test_utils::blur_gray(&img, blur)
    }

    fn outer() -> Ellipse {
        Ellipse {
            cx: 64.0,
            cy: 64.0,
            a: 24.0,
            b: 24.0,
            angle: 0.0,
        }
    }

    #[test]
    fn edge_width_grows_with_blur_and_is_about_2_5_sigma() {
        let sharp = disk(1.0);
        let blurred = disk(2.5);
        let w1 = outer_edge_width_px(DistortionAwareSampler::new(&sharp, None), &outer())
            .expect("width");
        let w2 = outer_edge_width_px(DistortionAwareSampler::new(&blurred, None), &outer())
            .expect("width");
        assert!(w2 > w1 * 1.8, "{w1} -> {w2}");
        assert!((w2 / 2.5 - 2.5).abs() < 0.6, "width/sigma = {}", w2 / 2.5);
    }

    #[test]
    fn border_distance_is_in_radii() {
        let img = GrayImage::new(100, 80);
        let s = DistortionAwareSampler::new(&img, None);
        assert!((border_distance_radii(s, [10.0, 40.0], 5.0) - 2.0).abs() < 1e-6);
        assert_eq!(border_distance_radii(s, [-3.0, 40.0], 5.0), 0.0);
    }
}
