//! Inner-edge recovery under a focus gradient: a coded view whose blur grows
//! from sharp at the top to strongly defocused at the bottom (a tilted,
//! Scheimpflug-like plane of focus, as in the rtv3d captures), detected
//! through `detect_adaptive` with the default config.
//!
//! Blur merges each ring stroke's two edges, so the inner edge's derivative
//! peak moves towards the centre (and the outer edge's outwards): the
//! measured inner/outer ratio falls below its sharp value by up to
//! `(1 + r_inner) σ / R`. The inner-edge search used a fixed window around the
//! sharp ratio, so in the defocused rows its peak sat on the window's inner
//! bound (`PeakAtSearchWindowEdge`), the ring got no inner ellipse, hence no
//! projective centre, and the output contract dropped it. The search now
//! moves that bound in by the blur reach of the outer edge's measured width
//! when the nominal search ends there.
//!
//! On 0.16 this view identifies 137 of 203 rings; now 181, none wrong, with
//! the centre error p95 0.59 → 0.64 px (the recovered rings are the most
//! defocused ones).

mod support;

use ringgrid::{Detector, TargetLayout};
use support::blur_scene::{BlurForm, BoardRaster, Psf, SceneSpec, render, score};

/// Outer ring radius (px) at the image centre.
const OUTER_RADIUS_PX: f64 = 12.0;
/// Blur sigma (px) at the top and bottom rows.
const SIGMA_TOP: f64 = 1.0;
const SIGMA_BOTTOM: f64 = 3.0;
/// Board tilt (degrees) and paper-minus-ink contrast (grey levels).
const TILT_DEG: f64 = 30.0;
const CONTRAST: f64 = 40.0;

#[test]
fn defocused_rings_keep_their_inner_edge() {
    let target = TargetLayout::default_hex();
    let base = SceneSpec::for_target(&target, OUTER_RADIUS_PX, 1.0, TILT_DEG, CONTRAST, 5);
    let height = f64::from(base.image_size[1]);
    let spec = SceneSpec {
        psf: Psf::VaryingGaussian {
            sigma0: SIGMA_TOP,
            gradient: (SIGMA_BOTTOM - SIGMA_TOP) / height,
            y0: 0.0,
            sigma_range: [SIGMA_TOP, SIGMA_BOTTOM],
            form: BlurForm::Scatter,
        },
        ..base
    };
    let scene = render(&BoardRaster::new(target.clone()), &spec);

    let result = Detector::new(target.clone())
        .detect_adaptive(&scene.image)
        .expect("detect");
    let stats = score(&scene, &target, &result.detected_markers, false);
    println!(
        "visible {} identified {} wrong {}; centre error rms {:.3} p95 {:.3} max {:.3} px",
        stats.n_visible,
        stats.n_decoded,
        stats.n_wrong_id,
        stats.rms(),
        stats.quantile(0.95),
        stats.quantile(1.0)
    );

    assert_eq!(stats.n_wrong_id, 0);
    assert!(
        stats.n_decoded * 100 >= stats.n_visible * 85,
        "identified {} of {} visible",
        stats.n_decoded,
        stats.n_visible
    );
    assert!(
        stats.quantile(0.95) < 0.8,
        "p95 centre error {:.3} px",
        stats.quantile(0.95)
    );
}
