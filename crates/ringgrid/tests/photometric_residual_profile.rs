//! Sanity of the photometric residual profiles on synthetic scenes.
//!
//! On a scene that follows the intensity model's own assumptions (Gaussian
//! PSF, affine illumination, white noise) the residual profiles are flat; with
//! a Gaussian-plus-halo PSF the signed edge profile (table (b) of the
//! `photometric_residual_profile` example) shows the structure the model
//! cannot fit.

mod support;

use ringgrid::diagnostics::PhotometricStatus;
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};
use support::blur_scene::{BoardRaster, Psf, SceneSpec, render};
use support::residual_profile::{Acc, EDGE_BIN, EDGE_RANGE, MarkerData, N_EDGE, Report, analyze};

const OUTER_RADIUS_PX: f64 = 16.5;
const BLUR: f64 = 1.5;
const NOISE_SIGMA: f64 = 2.0;
/// Frames rendered per scene (each its own tilt axis and noise).
const FRAMES: u64 = 1;
/// A small frame keeps the unoptimised test run short.
const IMAGE_SIZE: [u32; 2] = [400, 300];

/// The Gaussian-PSF profile, shared by the tests.
fn gaussian_profile() -> &'static Report {
    static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
    REPORT.get_or_init(|| profile(Psf::Gaussian { sigma: BLUR }))
}

fn profile(psf: Psf) -> Report {
    let target = TargetLayout::default_hex();
    let board = BoardRaster::new(target.clone());
    let mut config = DetectConfig::from_target_and_marker_diameter(
        target.clone(),
        (2.0 * OUTER_RADIUS_PX) as f32,
    );
    config.circle_refinement = CircleRefinementMethod::Photometric;
    let detector = Detector::with_config(config);
    let mut markers = Vec::new();
    for frame in 0..FRAMES {
        let mut spec = SceneSpec::for_target(&target, OUTER_RADIUS_PX, BLUR, 15.0, 60.0, frame + 1);
        spec.image_size = IMAGE_SIZE;
        spec.supersample = 2;
        spec.noise_sigma = NOISE_SIGMA;
        spec.tilt_axis_deg += 50.0 * frame as f64;
        spec.psf = psf;
        let scene = render(&board, &spec);
        let (_, diag, residuals) = detector
            .detect_with_photometric_residuals(&scene.image)
            .expect("detect");
        for r in residuals {
            let accepted = diag.markers[r.marker_index]
                .photometric
                .as_ref()
                .is_some_and(|p| p.status == PhotometricStatus::Accepted);
            if accepted {
                markers.push(MarkerData {
                    center: r.center,
                    blur_sigma_px: r.blur_sigma_px,
                    samples: r.samples,
                });
            }
        }
    }
    assert!(markers.len() >= 30, "only {} accepted fits", markers.len());
    analyze(&markers)
}

/// All terciles of one edge class pooled into one accumulator per bin.
fn pooled_edge(report: &Report, class: usize) -> Vec<Acc> {
    (0..N_EDGE)
        .map(|b| {
            let mut a = Acc::default();
            for t in 0..3 {
                let x = &report.edge[class][t][b];
                a.n += x.n;
                a.sum += x.sum;
                a.sumsq += x.sumsq;
            }
            a
        })
        .collect()
}

/// Mean residual over the bins whose centre distance lies in `[lo, hi)`.
fn band_mean(bins: &[Acc], lo: f64, hi: f64) -> (f64, u64) {
    let mut a = Acc::default();
    for (b, x) in bins.iter().enumerate() {
        let c = -EDGE_RANGE + (b as f64 + 0.5) * EDGE_BIN;
        if c >= lo && c < hi {
            a.n += x.n;
            a.sum += x.sum;
        }
    }
    (a.sum / a.n.max(1) as f64, a.n)
}

#[test]
fn profiles_are_flat_under_the_model_assumptions() {
    let report = gaussian_profile();
    let mut worst = 0.0f64;
    for class in 0..2 {
        for (b, a) in pooled_edge(report, class).iter().enumerate() {
            if a.n < 400 {
                continue;
            }
            let tol = 5.0 * NOISE_SIGMA / (a.n as f64).sqrt() + 0.4;
            worst = worst.max(a.mean().abs());
            assert!(
                a.mean().abs() < tol,
                "class {class} bin {b}: mean {:.3} (n {}) exceeds {tol:.3}",
                a.mean(),
                a.n
            );
        }
    }
    // Radius profile, pooled over terciles.
    for b in 0..report.radius[0].len() {
        let mut a = Acc::default();
        for t in 0..3 {
            let x = &report.radius[t][b];
            a.n += x.n;
            a.sum += x.sum;
        }
        if a.n >= 400 {
            let tol = 5.0 * NOISE_SIGMA / (a.n as f64).sqrt() + 0.4;
            assert!(a.mean().abs() < tol, "radius bin {b}: mean {:.3}", a.mean());
        }
    }
    let all = &report.split[3];
    println!(
        "gaussian: worst edge-bin |mean| {worst:.3}, split {all:?}, flat {:?}",
        report.flat
    );
    // The flat-region noise is the injected noise.
    assert!(
        (report.flat.sigma - NOISE_SIGMA).abs() < 0.3,
        "flat sigma {}",
        report.flat.sigma
    );
    assert!(
        report.flat.acf[2].abs() < 0.1,
        "flat acf {}",
        report.flat.acf[2]
    );
}

#[test]
fn a_defocus_disc_psf_leaves_a_signed_edge_profile() {
    // A uniform disc is not a Gaussian: the fitted Gaussian is too soft at the
    // edge core and too sharp in its tails, so the residual changes sign
    // across the edge. (A 15 % wide halo is absorbed by the affine paper level
    // and the blur and leaves under 0.3 grey levels.)
    let gauss = gaussian_profile();
    let disc = profile(Psf::Disc { radius: 2.0 * BLUR });
    let bands = |r: &Report| {
        let bins = pooled_edge(r, 1);
        let (core, n1) = band_mean(&bins, -1.5, 0.5);
        let (ink, n2) = band_mean(&bins, 0.5, 2.0);
        let (paper, n3) = band_mean(&bins, -3.2, -2.0);
        assert!(n1 > 5000 && n2 > 5000 && n3 > 5000, "too few pixels");
        let range = bins
            .iter()
            .filter(|a| a.n >= 2000)
            .map(Acc::mean)
            .fold((f64::MAX, f64::MIN), |(lo, hi), m| (lo.min(m), hi.max(m)));
        (core, ink, paper, range.1 - range.0)
    };
    let (gc, gi, gp, gr) = bands(gauss);
    let (dc, di, dp, dr) = bands(&disc);
    println!(
        "outer-ring edge profile (core, ink side, paper side, range): \
         gaussian ({gc:.2}, {gi:.2}, {gp:.2}, {gr:.2}) disc ({dc:.2}, {di:.2}, {dp:.2}, {dr:.2})"
    );
    println!("disc split {:?}", disc.split[3]);
    assert!(
        dc < -0.3 && di > 0.3 && dp > 0.3,
        "sign pattern: core {dc:.2}, ink {di:.2}, paper {dp:.2}"
    );
    assert!(dr > 1.5 && dr > 5.0 * gr, "range {dr:.2} vs {gr:.2}");
    // A common, deterministic misfit: the mean profile explains a visible
    // share of the disc residual and none of the Gaussian one.
    let share = |r: &Report| r.split[3].explained_ms / r.split[3].total_ms;
    assert!(
        share(&disc) > 0.05 && share(gauss) < 0.01,
        "explained {} vs {}",
        share(&disc),
        share(gauss)
    );
}
