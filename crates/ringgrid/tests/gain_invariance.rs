//! Gain invariance: the same view exposed at full gain and at a fraction of
//! it (plus a black-level offset) must give the same markers — the same IDs,
//! and centres as accurate as at full gain.
//!
//! The outer-edge depth gate used to require an absolute intensity step (8 %
//! of full scale across ±2 px), so a dark exposure of a sharp view lost most
//! of its markers while the same view at full gain kept them. The gate is
//! contrast-relative now: an edge's step is compared with its own ray's
//! ink-to-paper swing, above a small absolute noise floor.
//!
//! `cargo test --release -p ringgrid --test gain_invariance -- --nocapture`
//! prints the per-exposure counts and centre errors.

mod support;

use std::collections::BTreeMap;

use image::{GrayImage, Luma};
use ringgrid::{DetectedMarker, Detector, MarkerScalePrior, TargetLayout};
use support::blur_scene::{BoardRaster, CenterStats, Radiance, Scene, SceneSpec, radiance, score};

/// Outer ring radius (px) at the image centre.
const OUTER_RADIUS_PX: f64 = 16.0;
/// Paper-minus-ink contrast of the full-gain exposure, grey levels.
const FULL_CONTRAST: f64 = 150.0;
/// Gaussian blur sigma (px): a sharp view, so the ±2 px edge step carries
/// nearly the whole contrast.
const BLUR_PX: f64 = 1.2;
/// The dark exposure: a fraction of the full gain plus a black-level offset
/// (22 grey levels of paper-minus-ink contrast, a dim but clean capture).
const LOW_GAIN: f64 = 0.15;
const LOW_OFFSET: f64 = 12.0;
/// Distance (px) from the image border inside which markers are compared.
/// Rings at the border lose rays and code-band samples to the frame, so their
/// detection is marginal at any gain (two low-confidence rings within a few
/// px of the border differ between these exposures).
const BORDER_PX: f64 = 2.5 * OUTER_RADIUS_PX;
/// Allowed growth (px) of the centre error statistics at low gain: the dark
/// exposure quantizes the same radiance into ~7x fewer grey levels.
const CENTRE_SLACK_PX: f64 = 0.05;

/// Quantize `gain * radiance + offset` into an 8-bit image (no noise: the two
/// exposures differ only in gain and quantization).
fn expose(rad: &Radiance, size: [u32; 2], gain: f64, offset: f64) -> Scene {
    let image = GrayImage::from_fn(size[0], size[1], |x, y| {
        let v = rad.buf[(y * size[0] + x) as usize];
        Luma([(gain * v + offset).round().clamp(0.0, 255.0) as u8])
    });
    Scene {
        image,
        h: rad.h,
        distortion: None,
    }
}

/// Decoded markers away from the image border, by ID.
fn detect(target: &TargetLayout, scene: &Scene) -> BTreeMap<usize, DetectedMarker> {
    let d = (2.0 * OUTER_RADIUS_PX) as f32;
    let detector =
        Detector::with_marker_scale(target.clone(), MarkerScalePrior::new(0.5 * d, 1.6 * d));
    let (w, h) = (
        f64::from(scene.image.width()),
        f64::from(scene.image.height()),
    );
    let inside = |c: [f64; 2]| {
        c[0] >= BORDER_PX && c[1] >= BORDER_PX && c[0] <= w - BORDER_PX && c[1] <= h - BORDER_PX
    };
    detector
        .detect(&scene.image)
        .expect("detect")
        .detected_markers
        .into_iter()
        .filter(|m| inside(m.center))
        .filter_map(|m| Some((m.id?, m)))
        .collect()
}

fn centre_errors(
    target: &TargetLayout,
    scene: &Scene,
    markers: &BTreeMap<usize, DetectedMarker>,
) -> CenterStats {
    let markers: Vec<DetectedMarker> = markers.values().cloned().collect();
    let stats = score(scene, target, &markers, false);
    assert_eq!(stats.n_wrong_id, 0, "a marker is reported at a wrong cell");
    stats
}

#[test]
fn dark_exposure_finds_the_same_markers_as_a_bright_one() {
    let target = TargetLayout::default_hex();
    let spec = SceneSpec {
        noise_sigma: 0.0,
        ..SceneSpec::for_target(&target, OUTER_RADIUS_PX, BLUR_PX, 20.0, FULL_CONTRAST, 5)
    };
    let rad = radiance(&BoardRaster::new(target.clone()), &spec);
    let bright = expose(&rad, spec.image_size, 1.0, 0.0);
    let dark = expose(&rad, spec.image_size, LOW_GAIN, LOW_OFFSET);
    let full = detect(&target, &bright);
    let low = detect(&target, &dark);

    let (e_full, e_low) = (
        centre_errors(&target, &bright, &full),
        centre_errors(&target, &dark, &low),
    );
    for (name, n, e) in [
        ("full gain", full.len(), &e_full),
        ("dark", low.len(), &e_low),
    ] {
        println!(
            "{name}: {n} IDs, centre error rms {:.3} p95 {:.3} max {:.3} px",
            e.rms(),
            e.quantile(0.95),
            e.quantile(1.0)
        );
    }

    assert!(
        full.len() >= 150,
        "the full-gain view must be well detected: {} IDs",
        full.len()
    );
    let only_full: Vec<_> = full.keys().filter(|id| !low.contains_key(id)).collect();
    let only_low: Vec<_> = low.keys().filter(|id| !full.contains_key(id)).collect();
    assert!(
        only_full.is_empty() && only_low.is_empty(),
        "IDs differ between exposures: only at full gain {only_full:?}, only dark {only_low:?}"
    );
    for (what, a, b) in [
        ("rms", e_full.rms(), e_low.rms()),
        ("p95", e_full.quantile(0.95), e_low.quantile(0.95)),
    ] {
        assert!(
            b <= a + CENTRE_SLACK_PX,
            "dark-exposure centre error {what} {b:.3} px vs {a:.3} px at full gain"
        );
    }
}
