//! Large plain rings seen obliquely under a focus gradient (ADR-026): the
//! view anchors, every reported board position is true, and nearly every
//! ring whose whole outer ellipse lies in the image is reported, through the
//! adaptive entry point a calibration wrapper uses (`detect_adaptive`,
//! photometric centres).
//!
//! The board is the 24×24 plain rect preset: its ring/pitch ratios (outer
//! radius 0.4 pitch, inner 0.2 pitch) are a precision plate's, and at ~100 px
//! ring diameters only a window of it is in view. One scene per cause of lost
//! rings on real precision-plate captures:
//!
//! - `sharp_oblique` — sharp rings at an axis ratio of ~0.6: the outer fit's
//!   second-harmonic edge model missed the sharp edge on most rays, and the
//!   grid labeller walked a sheared lattice basis, so the view never anchored;
//! - `focus_gradient` — large rings blurred from σ ≈ 1 to 6 px: the
//!   confidence's sharpness term, an absolute edge width, dropped every ring
//!   beyond a few px of blur, whatever its size.
//!
//! On 0.16 the first scene does not anchor at all and the second reports 31
//! of 79 rings; now 108 of 110 and 79 of 79. The first scene's floor is not
//! 1: two rings ~10–20 px from the image border are not fitted (ADR-028,
//! "Remaining losses").
//!
//! Each runs a full adaptive detection on a 1280×960 image (seconds in
//! release); an unoptimized build takes minutes, so plain `cargo test` skips
//! them with a note, as the golden snapshots do. `RINGGRID_SLOW_TESTS=1`
//! forces a run in any profile.

mod support;

use std::sync::OnceLock;

use ringgrid::{BoardFrame, CircleRefinementMethod, DetectConfig, Detector, TargetLayout};
use support::blur_scene::{BlurForm, BoardRaster, Psf, Scene, SceneSpec, render};

const IMAGE_SIZE: [u32; 2] = [1280, 960];
/// Margin (px) by which an expected ring's outer ellipse lies inside the image.
const MARGIN_PX: f64 = 2.0;

fn board() -> &'static BoardRaster {
    static BOARD: OnceLock<BoardRaster> = OnceLock::new();
    BOARD.get_or_init(|| BoardRaster::new(TargetLayout::rect_24x24()))
}

fn run_slow() -> bool {
    !cfg!(debug_assertions) || std::env::var_os("RINGGRID_SLOW_TESTS").is_some()
}

struct View {
    /// Outer ring diameter (px) at the image centre for a frontal view.
    ring_diameter_px: f64,
    tilt_deg: f64,
    tilt_axis_deg: f64,
    roll_deg: f64,
    /// Offset (mm) of the imaged board point from the board centre.
    look_mm: [f64; 2],
    psf: Psf,
}

fn scene(view: &View, seed: u64) -> Scene {
    let target = &board().target;
    let outer_mm = f64::from(target.outer_draw_radius_mm());
    let spec = SceneSpec {
        image_size: IMAGE_SIZE,
        focal_px: 8000.0,
        px_per_mm: 0.5 * view.ring_diameter_px / outer_mm,
        tilt_deg: view.tilt_deg,
        tilt_axis_deg: view.tilt_axis_deg,
        roll_deg: view.roll_deg,
        board_center_mm: view.look_mm,
        psf: view.psf,
        illum_gradient: 0.15,
        noise_sigma: 1.5,
        glare: None,
        ..SceneSpec::with_contrast(1.0, 0.0, seed, 140.0)
    };
    render(board(), &spec)
}

/// Board cells whose whole outer ring projects inside the image with
/// [`MARGIN_PX`] to spare.
fn expected_cells(scene: &Scene) -> Vec<[i32; 2]> {
    let target = &board().target;
    let r = f64::from(target.ring().outer_radius_mm);
    let [w, h] = [f64::from(IMAGE_SIZE[0]), f64::from(IMAGE_SIZE[1])];
    target
        .cells()
        .iter()
        .filter(|c| {
            let xy = [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])];
            (0..64).all(|k| {
                let a = f64::from(k) * std::f64::consts::TAU / 64.0;
                let p = scene.project([xy[0] + r * a.cos(), xy[1] + r * a.sin()]);
                p[0] >= MARGIN_PX
                    && p[1] >= MARGIN_PX
                    && p[0] <= w - 1.0 - MARGIN_PX
                    && p[1] <= h - 1.0 - MARGIN_PX
            })
        })
        .map(|c| [c.coord.u, c.coord.v])
        .collect()
}

/// Detect with the calibration wrapper's configuration and check: anchored,
/// every board position true, at least `min_recall` of the expected rings
/// reported, centres tight.
fn assert_complete(name: &str, view: &View, min_recall: f64) {
    if !run_slow() {
        eprintln!("{name}: skipped in an unoptimized build (set RINGGRID_SLOW_TESTS=1)");
        return;
    }
    let target = board().target.clone();
    let s = scene(view, 7);
    let mut config = DetectConfig::from_target(target.clone());
    config.circle_refinement = CircleRefinementMethod::Photometric;
    let result = Detector::with_config(config)
        .detect_adaptive(&s.image)
        .expect("detect");
    assert_eq!(
        result.board_frame,
        Some(BoardFrame::Absolute),
        "{name}: view not anchored ({} markers)",
        result.detected_markers.len()
    );
    let spacing_mm = f64::from(target.lattice().min_center_spacing_mm());
    let mut errors = Vec::new();
    let mut reported = std::collections::HashSet::new();
    for m in &result.detected_markers {
        let Some(xy) = m.board_xy_mm else { continue };
        let p = s.project(xy);
        let err = (p[0] - m.center[0]).hypot(p[1] - m.center[1]);
        let pitch_px = {
            let q = s.project([xy[0] + spacing_mm, xy[1]]);
            (q[0] - p[0]).hypot(q[1] - p[1])
        };
        assert!(
            err < 0.25 * pitch_px,
            "{name}: marker {:?} is {err:.1} px from its board position's projection",
            m.grid_coord
        );
        errors.push(err);
        if let Some(c) = m.grid_coord {
            reported.insert(c);
        }
    }
    let expected = expected_cells(&s);
    let missing: Vec<_> = expected.iter().filter(|c| !reported.contains(*c)).collect();
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    eprintln!(
        "{name}: expected {} reported {} missing {} median centre error {median:.3} px",
        expected.len(),
        reported.len(),
        missing.len()
    );
    assert!(expected.len() >= 40, "{name}: scene shows too few rings");
    let recall = 1.0 - missing.len() as f64 / expected.len() as f64;
    assert!(
        recall >= min_recall,
        "{name}: {} of {} fully visible rings missing: {missing:?}",
        missing.len(),
        expected.len()
    );
    assert!(median < 0.15, "{name}: median centre error {median:.3} px");
}

#[test]
fn sharp_oblique() {
    assert_complete(
        "sharp_oblique",
        &View {
            ring_diameter_px: 100.0,
            tilt_deg: 52.0,
            tilt_axis_deg: 0.0,
            roll_deg: 18.0,
            look_mm: [9.0, 5.0],
            psf: Psf::Gaussian { sigma: 0.9 },
        },
        // 108 of 110: two border rings are not fitted.
        0.98,
    );
}

#[test]
fn focus_gradient() {
    assert_complete(
        "focus_gradient",
        &View {
            ring_diameter_px: 110.0,
            tilt_deg: 45.0,
            tilt_axis_deg: 0.0,
            roll_deg: -12.0,
            look_mm: [9.0, 5.0],
            psf: Psf::VaryingGaussian {
                sigma0: 0.8,
                gradient: 5.2 / f64::from(IMAGE_SIZE[1]),
                y0: 0.0,
                sigma_range: [0.8, 6.0],
                form: BlurForm::Scatter,
            },
        },
        1.0,
    );
}
