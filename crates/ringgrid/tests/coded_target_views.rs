//! Coded targets in hard views, end to end: the grid-first back half (grid
//! labeling → decoded-ID anchor → cell identification) and its fallback
//! ladder (no grid → decode-anchored homography corroborated by the rings'
//! own decodes → otherwise markers without IDs or homography).
//!
//! `cargo test --release -p ringgrid --test coded_target_views -- --nocapture`
//! prints the per-scene counts.

mod support;

use image::Luma;
use ringgrid::diagnostics::{DetectionDiagnostics, DetectionSource};
use ringgrid::{DetectionResult, Detector, MarkerScalePrior, TargetLayout};
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

fn detect(
    target: &TargetLayout,
    scene: &Scene,
    outer_radius_px: f64,
) -> (DetectionResult, DetectionDiagnostics) {
    let d = 2.0 * outer_radius_px as f32;
    let detector =
        Detector::with_marker_scale(target.clone(), MarkerScalePrior::new(0.5 * d, 1.6 * d));
    detector
        .detect_with_diagnostics(&scene.image)
        .expect("detect")
}

fn report(name: &str, stats: &CenterStats) {
    println!(
        "{name}: visible {} identified {} wrong {}",
        stats.n_visible, stats.n_decoded, stats.n_wrong_id
    );
}

/// Default decode acceptance gates (`DecodeConfig` defaults): a ring whose own
/// code band decoded to an accepted ID.
fn decode_accepted(diag: &ringgrid::diagnostics::MarkerDiagnostics) -> Option<usize> {
    let d = diag.decode.as_ref()?;
    (d.best_dist <= 3 && d.margin >= 1 && d.decode_confidence >= 0.3).then_some(d.best_id)
}

#[test]
fn coded_rect_tilted_55_identifies_undecoded_rings_from_the_grid() {
    // Strong perspective: the far side's code bands do not decode. The
    // decode-anchored homography identified 312 of 400 visible markers here;
    // grid labeling identifies the undecoded rings directly (394).
    let target = coded_rect();
    let spec = scene_spec(&target, 11.0, 55.0, 3);
    let scene = render(&BoardRaster::new(target.clone()), &spec);
    let (result, diag) = detect(&target, &scene, 11.0);
    let stats = score(&scene, &target, &result.detected_markers, false);
    report("coded rect, tilt 55", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    assert!(
        stats.n_decoded * 100 >= stats.n_visible * 95,
        "identified {} of {} visible",
        stats.n_decoded,
        stats.n_visible
    );

    // Fitted (not completed) rings whose own decode did not yield their ID
    // contribute: undecoded rings take their cell's ID, and a decode naming
    // no board cell (a misread) is overridden by the cell's ID. A decode
    // naming another board cell is never overridden — such a ring leaves the
    // labeling (the scorer above already proved every ID correct).
    let (mut undecoded, mut off_board, mut on_board) = (0, 0, 0);
    for (m, d) in result.detected_markers.iter().zip(&diag.markers) {
        if d.source != DetectionSource::FitDecoded || m.id.is_none() {
            continue;
        }
        match decode_accepted(d) {
            None => undecoded += 1,
            Some(id) if Some(id) != m.id && target.coord_of_id(id).is_some() => on_board += 1,
            Some(id) if Some(id) != m.id => off_board += 1,
            Some(_) => {}
        }
    }
    println!(
        "  undecoded fitted rings identified {undecoded}, off-board misreads overridden {off_board}"
    );
    assert!(undecoded >= 20, "undecoded rings identified: {undecoded}");
    assert!(off_board >= 1, "off-board misreads overridden: {off_board}");
    assert_eq!(
        on_board, 0,
        "on-board contradicting decodes are not overridden"
    );
}

#[test]
fn coded_hex_split_by_occluder_recovers_both_groups() {
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
    let (result, _) = detect(&target, &scene, 18.0);
    let stats = score(&scene, &target, &result.detected_markers, false);
    report("coded hex, split by occluder", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    let identified: Vec<_> = result
        .detected_markers
        .iter()
        .filter(|m| m.id.is_some())
        .collect();
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
fn coded_rect_corner_view() {
    // Zoomed on a board corner: ~130 markers, the board edge in view.
    let target = coded_rect();
    let spec = SceneSpec {
        board_center_mm: [-95.0, -95.0],
        ..scene_spec(&target, 26.0, 20.0, 7)
    };
    let scene = render(&BoardRaster::new(target.clone()), &spec);
    let (result, _) = detect(&target, &scene, 26.0);
    let stats = score(&scene, &target, &result.detected_markers, false);
    report("coded rect, corner view", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    // 112 of 126: border-cut rings whose centres sit more than 2 px off the
    // board homography fail geometric verify (117 at the former 10 px floor).
    assert!(
        stats.n_decoded * 100 >= stats.n_visible * 85,
        "identified {} of {} visible",
        stats.n_decoded,
        stats.n_visible
    );
}

/// Frontal hex scene showing only the markers at `cells` (everything else is
/// painted paper), with the board centre at the image centre.
fn sparse_scene(target: &TargetLayout, cells: &[usize], outer_radius_px: f64) -> Scene {
    let spec = scene_spec(target, outer_radius_px, 10.0, 13);
    let mut scene = render(&BoardRaster::new(target.clone()), &spec);
    let keep: Vec<[f64; 2]> = cells
        .iter()
        .map(|&i| {
            let xy = target.cells()[i].xy_mm;
            scene.project([f64::from(xy[0]), f64::from(xy[1])])
        })
        .collect();
    let paper = Luma([spec.white as u8]);
    let r_keep = 1.4 * outer_radius_px;
    let (w, h) = scene.image.dimensions();
    for y in 0..h {
        for x in 0..w {
            let p = [f64::from(x), f64::from(y)];
            if keep
                .iter()
                .all(|k| (p[0] - k[0]).hypot(p[1] - k[1]) > r_keep)
            {
                scene.image.put_pixel(x, y, paper);
            }
        }
    }
    scene
}

/// Board cells near the middle, pairwise at least `min_sep_mm` apart, so the
/// visible rings form no lattice triangle and no grid can be labeled.
fn spread_cells(target: &TargetLayout, n: usize, min_sep_mm: f32) -> Vec<usize> {
    let (lo, hi) = target.marker_bounds_mm().expect("bounds");
    let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let mut order: Vec<usize> = (0..target.cells().len()).collect();
    let dist = |i: usize| {
        let p = target.cells()[i].xy_mm;
        (p[0] - mid[0]).hypot(p[1] - mid[1])
    };
    order.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
    let mut picked: Vec<usize> = Vec::new();
    for i in order {
        let p = target.cells()[i].xy_mm;
        if picked.iter().all(|&j| {
            let q = target.cells()[j].xy_mm;
            (p[0] - q[0]).hypot(p[1] - q[1]) >= min_sep_mm
        }) {
            picked.push(i);
        }
        if picked.len() == n {
            break;
        }
    }
    picked
}

#[test]
fn no_grid_with_enough_decodes_falls_back_to_decode_anchored_homography() {
    // No lattice triangle among the visible rings, so no grid is labeled.
    // The decode-anchored global filter needs >= 4 decoded correspondences to
    // try and `ransac_homography.min_inliers` (6) consistent ones to accept.
    let target = TargetLayout::default_hex();
    let cells = spread_cells(&target, 9, 2.5 * target.pitch_mm());
    let scene = sparse_scene(&target, &cells, 22.0);
    let (result, _) = detect(&target, &scene, 22.0);
    let stats = score(&scene, &target, &result.detected_markers, false);
    report("9 isolated markers", &stats);
    assert_eq!(stats.n_wrong_id, 0);
    assert!(stats.n_decoded >= 6, "identified {}", stats.n_decoded);
    assert!(
        result.homography.is_some(),
        "enough consistent decodes give the decode-anchored homography"
    );
}

#[test]
fn no_grid_with_fewer_than_four_decodes_fails_closed() {
    // Three isolated markers: no grid, no homography to corroborate the
    // decodes — a lone decode can be a misread, so no ID is reported.
    let target = TargetLayout::default_hex();
    let cells = spread_cells(&target, 3, 2.5 * target.pitch_mm());
    let scene = sparse_scene(&target, &cells, 22.0);
    let (result, _) = detect(&target, &scene, 22.0);
    report(
        "3 isolated markers",
        &score(&scene, &target, &result.detected_markers, false),
    );
    assert!(
        !result.detected_markers.is_empty(),
        "the rings are reported"
    );
    assert!(result.detected_markers.iter().all(|m| m.id.is_none()));
    assert!(result.homography.is_none());
    assert!(result.board_frame.is_none());
}

#[test]
fn single_marker_is_reported_without_id() {
    // Fallback ladder floor: one decodable marker in view (no grid can be
    // labeled, no homography can be fit) is returned, but without an ID —
    // nothing corroborates its decode.
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
    let (result, _) = detect(&target, &scene, 30.0);
    let markers = result.detected_markers;
    assert_eq!(markers.len(), 1, "exactly the one printed marker");
    assert_eq!(
        markers[0].id, None,
        "an uncorroborated decode is not reported"
    );
    assert!(result.homography.is_none());
}
