//! Completion of large rings: a coded view whose rings are larger than the
//! default scale prior's nominal size, detected through `detect_adaptive`
//! with the default config (the published CLI without `--marker-diameter`).
//!
//! Adaptive detection fits each scale tier with its own prior but finalizes
//! — and completes missing cells — once, with the caller's config. The
//! completion refit's reach used to be a fixed ROI derived from that config's
//! prior (`0.75 x` its nominal diameter: 30 px for the default prior), so it
//! could not refit a ring of radius > 30 px, however well its neighbours
//! predicted it. It is sized from the ring's expected radius at its cell now.
//!
//! The proposal stage is capped so fit-decode leaves cells to completion: on
//! 0.16 completion refits none of them (32 of 76 visible rings identified),
//! now 21 (54 of 76), with no wrong ID.

mod support;

use ringgrid::diagnostics::DetectionSource;
use ringgrid::{Detector, TargetLayout};
use support::blur_scene::{BoardRaster, SceneSpec, render, score};

/// Outer ring radius (px) at the image centre: rings ~74 px across.
const OUTER_RADIUS_PX: f64 = 37.0;
/// Proposals kept per scale tier: fewer than the rings in view.
const MAX_PROPOSALS: usize = 30;

#[test]
fn completion_refits_rings_larger_than_the_default_prior() {
    let target = TargetLayout::default_hex();
    let spec = SceneSpec {
        image_size: [960, 720],
        ..SceneSpec::for_target(&target, OUTER_RADIUS_PX, 1.0, 15.0, 60.0, 11)
    };
    let scene = render(&BoardRaster::new(target.clone()), &spec);

    let mut detector = Detector::new(target.clone());
    detector.config_mut().advanced.proposal.max_candidates = Some(MAX_PROPOSALS);
    let (result, diag) = detector
        .detect_adaptive_with_diagnostics(&scene.image)
        .expect("detect");
    let n_completed = diag
        .markers
        .iter()
        .filter(|m| m.source == DetectionSource::Completion)
        .count();
    let stats = score(&scene, &target, &result.detected_markers, false);
    println!(
        "visible {} identified {} (completed {n_completed}) wrong {}; centre error rms {:.3} p95 {:.3} px",
        stats.n_visible,
        stats.n_decoded,
        stats.n_wrong_id,
        stats.rms(),
        stats.quantile(0.95)
    );

    assert_eq!(stats.n_wrong_id, 0);
    assert!(
        n_completed >= 18,
        "completion refits only {n_completed} large rings"
    );
    assert!(
        stats.n_decoded * 100 >= stats.n_visible * 65,
        "identified {} of {} visible",
        stats.n_decoded,
        stats.n_visible
    );
    assert!(
        stats.quantile(0.95) < 0.5,
        "p95 centre error {:.3} px",
        stats.quantile(0.95)
    );
}
