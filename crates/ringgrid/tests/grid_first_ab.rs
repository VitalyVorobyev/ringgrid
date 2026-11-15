//! Grid-first adoption study — A/B stress tests (TEMPORARY marking).
//!
//! Hard views for the coded back half: strong tilt, partial views, an
//! occluder that splits the board, a lone marker. Each test asserts what the
//! better finalize path achieves; run both paths with
//!
//! ```text
//! cargo test --release -p ringgrid --test grid_first_ab -- --nocapture
//! RINGGRID_GRID_FIRST=1 cargo test --release -p ringgrid --test grid_first_ab -- --nocapture
//! ```
//!
//! Tests the current default path cannot pass are `#[ignore]`d with the reason
//! (they pass with `RINGGRID_GRID_FIRST=1 ... -- --ignored`). At the adoption
//! decision the marking goes away and these stay as permanent tests of the
//! winner.

mod support;

use image::Luma;
use ringgrid::{DetectedMarker, Detector, MarkerScalePrior, TargetLayout};
use support::blur_scene::{BoardRaster, CenterStats, Scene, SceneSpec, render, score};

fn coded_rect() -> TargetLayout {
    TargetLayout::coded_rect(14.0, 20, 20, 4.8, 3.2, 1.152).expect("valid coded rect")
}

/// A clean scene of `target`: `outer_radius_px` at the image centre, mild
/// blur, no glare.
fn scene_spec(target: &TargetLayout, outer_radius_px: f64, tilt_deg: f64, seed: u64) -> SceneSpec {
    SceneSpec {
        image_size: [1280, 960],
        tilt_axis_deg: 25.0,
        roll_deg: 17.0,
        ..SceneSpec::for_target(target, outer_radius_px, 0.8, tilt_deg, 60.0, seed)
    }
}

fn detect(target: &TargetLayout, scene: &Scene, outer_radius_px: f64) -> Vec<DetectedMarker> {
    let d = 2.0 * outer_radius_px as f32;
    let detector =
        Detector::with_marker_scale(target.clone(), MarkerScalePrior::new(0.5 * d, 1.6 * d));
    detector
        .detect(&scene.image)
        .expect("detect")
        .detected_markers
}

fn report(name: &str, stats: &CenterStats) {
    println!(
        "{name}: visible {} identified {} wrong {}",
        stats.n_visible, stats.n_decoded, stats.n_wrong_id
    );
}

#[test]
#[ignore = "A/B: passes on grid-first only (old path 312/400); RINGGRID_GRID_FIRST=1 -- --ignored"]
fn ab_coded_rect_tilted_55() {
    // Strong perspective: the decode-anchored homography of the old path
    // identifies 312 of 400 visible markers (the foreshortened far side fails
    // to decode and completion seeds from a far-off H); grid-first labels the
    // undecoded rings directly and identifies 394.
    let target = coded_rect();
    let spec = scene_spec(&target, 11.0, 55.0, 3);
    let scene = render(&BoardRaster::new(target.clone()), &spec);
    let stats = score(&scene, &target, &detect(&target, &scene, 11.0), false);
    report("coded rect, tilt 55", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    assert!(
        stats.n_decoded * 100 >= stats.n_visible * 95,
        "identified {} of {} visible",
        stats.n_decoded,
        stats.n_visible
    );
}

#[test]
fn ab_coded_hex_split_by_occluder() {
    // A dark band through the middle splits the visible markers into two
    // disconnected groups; both must come back with correct IDs.
    let target = TargetLayout::default_hex();
    let spec = scene_spec(&target, 18.0, 20.0, 5);
    let mut scene = render(&BoardRaster::new(target.clone()), &spec);
    let (w, h) = scene.image.dimensions();
    let band = |x: f64, y: f64| (x - 0.5 * f64::from(w)) * 0.92 + (y - 0.5 * f64::from(h)) * 0.39;
    for y in 0..h {
        for x in 0..w {
            if band(f64::from(x), f64::from(y)).abs() < 70.0 {
                scene.image.put_pixel(x, y, Luma([60]));
            }
        }
    }
    let markers = detect(&target, &scene, 18.0);
    let stats = score(&scene, &target, &markers, false);
    report("coded hex, split by occluder", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    // Both groups come back: count identified markers on each side.
    let identified: Vec<&DetectedMarker> = markers.iter().filter(|m| m.id.is_some()).collect();
    let n_pos = identified
        .iter()
        .filter(|m| band(m.center[0], m.center[1]) > 0.0)
        .count();
    let n_neg = identified.len() - n_pos;
    println!("  sides: {n_pos} / {n_neg}");
    assert!(
        n_pos >= 60 && n_neg >= 60,
        "both groups identified: {n_pos} / {n_neg}"
    );
}

#[test]
fn ab_coded_rect_corner_view() {
    // Zoomed on a board corner: ~100 markers, the board edge in view.
    let target = coded_rect();
    let spec = SceneSpec {
        board_center_mm: [-95.0, -95.0],
        ..scene_spec(&target, 26.0, 20.0, 7)
    };
    let scene = render(&BoardRaster::new(target.clone()), &spec);
    let stats = score(&scene, &target, &detect(&target, &scene, 26.0), false);
    report("coded rect, corner view", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    assert!(
        stats.n_decoded * 100 >= stats.n_visible * 90,
        "identified {} of {} visible",
        stats.n_decoded,
        stats.n_visible
    );
}

#[test]
fn ab_single_marker_keeps_its_id() {
    // Fallback ladder floor: one decodable marker in view is returned with
    // its ID (no grid can be labeled, no homography can be fit).
    let target = TargetLayout::default_hex();
    let cell = target.cells()[100];
    let xy = [f64::from(cell.xy_mm[0]), f64::from(cell.xy_mm[1])];
    let board = BoardRaster::new(target.clone()).isolated(xy, 6.0);
    let spec = SceneSpec {
        board_center_mm: {
            let (lo, hi) = target.marker_bounds_mm().expect("bounds");
            let mid = [
                0.5 * f64::from(lo[0] + hi[0]),
                0.5 * f64::from(lo[1] + hi[1]),
            ];
            [xy[0] - mid[0], xy[1] - mid[1]]
        },
        ..scene_spec(&target, 30.0, 10.0, 11)
    };
    let scene = render(&board, &spec);
    let markers = detect(&target, &scene, 30.0);
    println!(
        "single marker: {:?}",
        markers.iter().map(|m| m.id).collect::<Vec<_>>()
    );
    assert_eq!(markers.len(), 1, "exactly the one printed marker");
    assert_eq!(markers[0].id, cell.id, "the marker keeps its decoded ID");
}
