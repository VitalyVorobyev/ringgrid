use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::GrayImage;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod marker {
    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum AngularAggregator {
        Median,
        TrimmedMean { trim_fraction: f32 },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum GradPolarity {
        DarkToLight,
        LightToDark,
        Auto,
    }
}

#[allow(dead_code)]
mod pixelmap {
    pub trait PixelMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]>;
        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]>;
    }

    #[derive(Debug, Clone, Copy)]
    pub struct CameraIntrinsics {
        pub fx: f64,
        pub fy: f64,
        pub cx: f64,
        pub cy: f64,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RadialTangentialDistortion {
        pub k1: f64,
        pub k2: f64,
        pub p1: f64,
        pub p2: f64,
        pub k3: f64,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct CameraModel {
        pub intrinsics: CameraIntrinsics,
        pub distortion: RadialTangentialDistortion,
    }

    impl PixelMapper for CameraModel {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            let _ = self.intrinsics;
            let _ = self.distortion;
            Some(image_xy)
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            let _ = self.intrinsics;
            let _ = self.distortion;
            Some(working_xy)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RadialMapper {
        pub cx: f64,
        pub cy: f64,
        pub k1: f64,
    }

    impl PixelMapper for RadialMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            let mut x = image_xy[0] - self.cx;
            let mut y = image_xy[1] - self.cy;
            for _ in 0..4 {
                let r2 = x * x + y * y;
                let radial = 1.0 + self.k1 * r2;
                if !radial.is_finite() || radial.abs() < 1e-12 {
                    return None;
                }
                x = (image_xy[0] - self.cx) / radial;
                y = (image_xy[1] - self.cy) / radial;
            }
            Some([x + self.cx, y + self.cy])
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            let x = working_xy[0] - self.cx;
            let y = working_xy[1] - self.cy;
            let r2 = x * x + y * y;
            let radial = 1.0 + self.k1 * r2;
            if !radial.is_finite() {
                return None;
            }
            Some([self.cx + x * radial, self.cy + y * radial])
        }
    }
}

#[allow(dead_code, unused_imports)]
#[path = "../src/conic/mod.rs"]
mod conic_impl;
#[allow(dead_code, unused_imports)]
#[path = "../src/ring/edge_sample.rs"]
mod edge_sample;
#[allow(dead_code, unused_imports)]
#[path = "../src/ring/outer_estimate.rs"]
mod outer_estimate;
#[allow(dead_code, unused_imports)]
#[path = "../src/detector/proposal.rs"]
mod proposal_impl;
#[allow(dead_code, unused_imports)]
#[path = "../src/ring/radial_profile.rs"]
mod radial_profile;

fn make_proposal_fixture(width: u32, height: u32, seed: u64) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    let buf = img.as_mut();

    // Deterministic background texture with gentle gradients to emulate camera noise.
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let v = 128.0
                + 35.0 * ((x as f32 * 0.007).sin() + (y as f32 * 0.011).cos())
                + 10.0 * ((x as f32 * 0.021 + y as f32 * 0.017).sin());
            buf[idx] = v.clamp(0.0, 255.0) as u8;
        }
    }

    let cols = 14usize;
    let rows = 10usize;
    let pitch_x = width as f32 / (cols as f32 + 1.0);
    let pitch_y = height as f32 / (rows as f32 + 1.0);
    let outer_r = pitch_x.min(pitch_y) * 0.28;
    let band_half = outer_r * 0.10;

    let mut rng = StdRng::seed_from_u64(seed);

    for r in 0..rows {
        for c in 0..cols {
            let cx = pitch_x * (c as f32 + 1.0) + if r % 2 == 0 { 0.0 } else { 0.5 * pitch_x };
            let cy = pitch_y * (r as f32 + 1.0);
            if cx < outer_r + 2.0 || cx >= width as f32 - outer_r - 2.0 {
                continue;
            }
            if cy < outer_r + 2.0 || cy >= height as f32 - outer_r - 2.0 {
                continue;
            }

            let jitter = rng.gen_range(-0.8f32..0.8f32);
            let rr = outer_r + jitter;
            let x0 = (cx - rr - 2.0).floor().max(0.0) as u32;
            let x1 = (cx + rr + 2.0).ceil().min((width - 1) as f32) as u32;
            let y0 = (cy - rr - 2.0).floor().max(0.0) as u32;
            let y1 = (cy + rr + 2.0).ceil().min((height - 1) as f32) as u32;

            for y in y0..=y1 {
                for x in x0..=x1 {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let d = (dx * dx + dy * dy).sqrt();
                    if (d - rr).abs() <= band_half {
                        let idx = (y * width + x) as usize;
                        buf[idx] = 24;
                    }
                }
            }
        }
    }

    img
}

fn proposal_config() -> proposal_impl::ProposalConfig {
    proposal_impl::ProposalConfig {
        r_min: 4.0,
        r_max: 18.0,
        grad_threshold: 0.04,
        nms_radius: 5.0,
        min_vote_frac: 0.06,
        accum_sigma: 1.5,
        max_candidates: Some(600),
    }
}

fn bench_proposal(c: &mut Criterion) {
    let cfg = proposal_config();
    let img_1280 = make_proposal_fixture(1280, 1024, 7);
    let img_1920 = make_proposal_fixture(1920, 1080, 9);

    c.bench_function("proposal_1280x1024", |b| {
        b.iter(|| {
            let proposals = proposal_impl::find_proposals(black_box(&img_1280), black_box(&cfg));
            black_box(proposals.len())
        })
    });

    c.bench_function("proposal_1920x1080", |b| {
        b.iter(|| {
            let proposals = proposal_impl::find_proposals(black_box(&img_1920), black_box(&cfg));
            black_box(proposals.len())
        })
    });
}

fn make_radial_fixture(theta_samples: usize, radial_samples: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut curves = vec![vec![0.0f32; radial_samples]; theta_samples];
    let r_samples: Vec<f32> = (0..radial_samples).map(|i| i as f32 * 0.5).collect();
    let mut rng = StdRng::seed_from_u64(99);

    for (ti, curve) in curves.iter_mut().enumerate() {
        let phase = 2.0 * PI_F32 * ti as f32 / theta_samples as f32;
        let edge_r = 6.5 + 0.25 * phase.sin();
        for (ri, &r) in r_samples.iter().enumerate() {
            let base = 185.0 - 120.0 / (1.0 + (-(r - edge_r) * 3.0).exp());
            let band = 8.0 * (3.0 * phase + r * 0.3).sin();
            let noise = rng.gen_range(-1.0f32..1.0f32);
            curve[ri] = base + band + noise;
        }
    }

    (curves, r_samples)
}

fn bench_radial_profile(c: &mut Criterion) {
    let theta_samples = 180usize;
    let radial_samples = 32usize;
    let r_step = 0.5f32;
    let (curves, r_samples) = make_radial_fixture(theta_samples, radial_samples);
    let agg = marker::AngularAggregator::Median;

    c.bench_function("radial_profile_32r_180a", |b| {
        b.iter(|| {
            let mut d_curves = Vec::with_capacity(theta_samples);
            for curve in &curves {
                let mut d = radial_profile::radial_derivative(curve, r_step);
                radial_profile::smooth_3point(&mut d);
                d_curves.push(d);
            }

            let mut per_theta = radial_profile::per_theta_peak_r(
                &d_curves,
                &r_samples,
                radial_profile::Polarity::Neg,
            );
            let r_star = radial_profile::aggregate(&mut per_theta, &agg);
            black_box(radial_profile::theta_consistency(
                &per_theta, r_star, r_step, 0.25,
            ))
        })
    });
}

fn make_outer_fixture(width: u32, height: u32, seed: u64) -> (GrayImage, [f32; 2], f32) {
    let mut img = GrayImage::new(width, height);
    let center = [width as f32 * 0.5, height as f32 * 0.5];
    let outer_r = width.min(height) as f32 * 0.22;
    let inner_r = outer_r * 0.62;
    let band_half = outer_r * 0.09;
    let mut rng = StdRng::seed_from_u64(seed);

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center[0];
            let dy = y as f32 - center[1];
            let d = (dx * dx + dy * dy).sqrt();
            let mut v = 220.0f32;

            if (d - outer_r).abs() <= band_half {
                v = 34.0;
            } else if (d - inner_r).abs() <= band_half {
                v = 28.0;
            } else if d < inner_r - band_half {
                v = 175.0;
            }

            v += 8.0 * ((x as f32 * 0.01).sin() + (y as f32 * 0.015).cos());
            v += rng.gen_range(-1.5f32..1.5f32);
            img.as_mut()[(y * width + x) as usize] = v.clamp(0.0, 255.0) as u8;
        }
    }

    (img, center, outer_r)
}

fn bench_outer_estimate(c: &mut Criterion) {
    let (img, center, r_expected) = make_outer_fixture(512, 512, 123);
    let cfg = outer_estimate::OuterEstimationConfig {
        search_halfwidth_px: 4.0,
        radial_samples: 64,
        theta_samples: 48,
        aggregator: marker::AngularAggregator::Median,
        grad_polarity: marker::GradPolarity::DarkToLight,
        min_theta_coverage: 0.6,
        min_theta_consistency: 0.35,
        allow_two_hypotheses: true,
        second_peak_min_rel: 0.85,
        refine_halfwidth_px: 1.0,
    };
    let mapper = pixelmap::RadialMapper {
        cx: 256.0,
        cy: 256.0,
        k1: -2.0e-7,
    };

    c.bench_function("outer_estimate_64r_48t_nomapper", |b| {
        b.iter(|| {
            let est = outer_estimate::estimate_outer_from_prior_with_mapper(
                black_box(&img),
                black_box(center),
                black_box(r_expected),
                black_box(&cfg),
                None,
                false,
            );
            black_box(est.hypotheses.len())
        })
    });

    c.bench_function("outer_estimate_64r_48t_mapper", |b| {
        b.iter(|| {
            let est = outer_estimate::estimate_outer_from_prior_with_mapper(
                black_box(&img),
                black_box(center),
                black_box(r_expected),
                black_box(&cfg),
                Some(black_box(&mapper) as &dyn pixelmap::PixelMapper),
                false,
            );
            black_box(est.hypotheses.len())
        })
    });
}

fn make_ellipse_points(n: usize) -> Vec<[f64; 2]> {
    let cx = 640.0f64;
    let cy = 512.0f64;
    let a = 34.0f64;
    let b = 17.5f64;
    let angle = 0.31f64;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let mut rng = StdRng::seed_from_u64(12345);

    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = 2.0 * PI_F64 * (i as f64) / (n as f64);
        let ex = a * t.cos();
        let ey = b * t.sin();
        let x = cx + cos_a * ex - sin_a * ey + rng.gen_range(-0.35f64..0.35f64);
        let y = cy + sin_a * ex + cos_a * ey + rng.gen_range(-0.35f64..0.35f64);
        pts.push([x, y]);
    }
    pts
}

fn bench_ellipse_fit(c: &mut Criterion) {
    let points = make_ellipse_points(50);
    c.bench_function("ellipse_fit_50pts", |b| {
        b.iter(|| {
            let fit = conic_impl::fit_ellipse_direct(black_box(&points))
                .expect("deterministic fixture should always fit");
            black_box(fit)
        })
    });
}

criterion_group!(
    hotpaths,
    bench_proposal,
    bench_radial_profile,
    bench_outer_estimate,
    bench_ellipse_fit
);
criterion_main!(hotpaths);
