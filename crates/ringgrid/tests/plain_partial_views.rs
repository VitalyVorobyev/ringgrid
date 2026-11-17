//! Partial views of large plain boards: a camera window showing a small part
//! of a big plain board anchors to the absolute board frame whenever the
//! origin-dot triad is in view — through blur, perspective and any in-plane
//! rotation — and reports no board position (rather than a wrong one) when the
//! triad is out of view, cut by the image border, or made ambiguous by a
//! dark blob in a neighbouring gap.

mod support;

use std::sync::OnceLock;

use image::GrayImage;
use ringgrid::{BoardFrame, DetectionResult, Detector, OriginDots, TargetLayout};
use support::blur_scene::{BlurForm, BoardRaster, Psf, Scene, SceneSpec, render};

/// Raster density of the boards the camera images are resampled from: two or
/// more raster pixels per camera pixel at the densest view below.
const BOARD_DPI: f32 = 1300.0;

/// A 58×58 plain rect board at 2 mm pitch (a precision plate).
fn big_rect() -> &'static BoardRaster {
    static BOARD: OnceLock<BoardRaster> = OnceLock::new();
    BOARD.get_or_init(|| {
        let target =
            TargetLayout::plain_rect(2.0, 58, 58, 0.8, 0.4, OriginDots::Auto).expect("plain rect");
        BoardRaster::with_dpi(target, BOARD_DPI)
    })
}

/// A 1,640-cell plain hex board (nearest-neighbour spacing 2√3 ≈ 3.46 mm).
fn big_hex() -> &'static BoardRaster {
    static BOARD: OnceLock<BoardRaster> = OnceLock::new();
    BOARD.get_or_init(|| {
        let target =
            TargetLayout::plain_hex(2.0, 41, 40, 1.2, 0.6, OriginDots::Auto).expect("plain hex");
        BoardRaster::with_dpi(target, 900.0)
    })
}

/// Board-mm centre of the marker field.
fn board_mid_mm(target: &TargetLayout) -> [f64; 2] {
    let (lo, hi) = target.marker_bounds_mm().expect("non-empty");
    [
        0.5 * f64::from(lo[0] + hi[0]),
        0.5 * f64::from(lo[1] + hi[1]),
    ]
}

/// Board-mm position of the origin cell `(0, 0)`, which the triad surrounds.
fn origin_mm(target: &TargetLayout) -> [f64; 2] {
    let xy = target
        .cells()
        .iter()
        .find(|c| c.coord.u == 0 && c.coord.v == 0)
        .expect("origin cell")
        .xy_mm;
    [f64::from(xy[0]), f64::from(xy[1])]
}

#[derive(Clone, Copy)]
struct View {
    /// Image size in pixels.
    size: [u32; 2],
    /// Board scale at the image centre, px / mm.
    px_per_mm: f64,
    /// Offset (mm) from the origin cell to the board point imaged at the
    /// principal point.
    look_offset_mm: [f64; 2],
    roll_deg: f64,
    tilt_deg: f64,
    psf: Psf,
}

impl View {
    /// A `cols × rows`-marker window of a board of `pitch_mm`, in a
    /// 720-pixel-wide image.
    fn window(pitch_mm: f64, cols: f64, rows: f64) -> Self {
        let px_per_mm = 720.0 / (cols * pitch_mm);
        let height = (rows * pitch_mm * px_per_mm).round() as u32;
        Self {
            size: [720, height],
            px_per_mm,
            look_offset_mm: [0.0, 0.0],
            roll_deg: 0.0,
            tilt_deg: 0.0,
            psf: Psf::Gaussian { sigma: 1.5 },
        }
    }

    fn look(self, dx_mm: f64, dy_mm: f64) -> Self {
        Self {
            look_offset_mm: [dx_mm, dy_mm],
            ..self
        }
    }

    fn roll(self, deg: f64) -> Self {
        Self {
            roll_deg: deg,
            ..self
        }
    }

    fn tilt(self, deg: f64) -> Self {
        Self {
            tilt_deg: deg,
            ..self
        }
    }

    fn psf(self, psf: Psf) -> Self {
        Self { psf, ..self }
    }
}

/// A focus gradient across the image height: sharp at the top, σ ≈ 2.5 px at
/// the bottom.
fn focus_gradient(height: u32) -> Psf {
    Psf::VaryingGaussian {
        sigma0: 0.6,
        gradient: 1.9 / f64::from(height),
        y0: 0.0,
        sigma_range: [0.5, 3.0],
        form: BlurForm::Scatter,
    }
}

fn scene(board: &BoardRaster, view: &View, seed: u64) -> Scene {
    let mid = board_mid_mm(&board.target);
    let origin = origin_mm(&board.target);
    let look_at = [
        origin[0] + view.look_offset_mm[0],
        origin[1] + view.look_offset_mm[1],
    ];
    let spec = SceneSpec {
        image_size: view.size,
        px_per_mm: view.px_per_mm,
        roll_deg: view.roll_deg,
        tilt_deg: view.tilt_deg,
        board_center_mm: [look_at[0] - mid[0], look_at[1] - mid[1]],
        psf: view.psf,
        illum_gradient: 0.2,
        noise_sigma: 1.5,
        ..SceneSpec::with_contrast(1.0, 0.0, seed, 120.0)
    };
    render(board, &spec)
}

fn detect(target: &TargetLayout, image: &GrayImage, view: &View) -> DetectionResult {
    let d_px = 2.0 * f64::from(target.outer_draw_radius_mm()) * view.px_per_mm;
    let detector = Detector::with_marker_diameter_hint(target.clone(), d_px as f32);
    detector.detect(image).expect("detect")
}

/// Every reported board position is the true one: the marker's centre lies
/// within a quarter spacing of the projection of its board position. Returns
/// the number of markers with a board position.
fn n_true_absolute_labels(result: &DetectionResult, scene: &Scene, spacing_px: f64) -> usize {
    let mut n = 0;
    for m in &result.detected_markers {
        let Some(xy) = m.board_xy_mm else { continue };
        let p = scene.project(xy);
        let err = (p[0] - m.center[0]).hypot(p[1] - m.center[1]);
        assert!(
            err < 0.25 * spacing_px,
            "marker {:?} at {:?} is {err:.1} px from its board position's projection",
            m.grid_coord,
            m.center
        );
        n += 1;
    }
    n
}

/// Board cells whose whole ring projects inside the image.
fn n_visible_cells(target: &TargetLayout, scene: &Scene, ring_radius_px: f64) -> usize {
    let [w, h] = [
        f64::from(scene.image.width()),
        f64::from(scene.image.height()),
    ];
    target
        .cells()
        .iter()
        .filter(|c| {
            let p = scene.project([f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])]);
            p[0] >= ring_radius_px
                && p[1] >= ring_radius_px
                && p[0] <= w - 1.0 - ring_radius_px
                && p[1] <= h - 1.0 - ring_radius_px
        })
        .count()
}

/// The view anchors, every board position is true, and most visible rings
/// carry one.
fn assert_anchored(board: &BoardRaster, view: View, seed: u64) {
    let target = &board.target;
    let s = scene(board, &view, seed);
    let result = detect(target, &s.image, &view);
    let spacing_px = f64::from(target.lattice().min_center_spacing_mm()) * view.px_per_mm;
    let n_visible = n_visible_cells(
        target,
        &s,
        f64::from(target.outer_draw_radius_mm()) * view.px_per_mm,
    );
    assert_eq!(
        result.board_frame,
        Some(BoardFrame::Absolute),
        "view must anchor ({} markers of {n_visible} visible)",
        result.detected_markers.len()
    );
    let n = n_true_absolute_labels(&result, &s, spacing_px);
    assert!(
        n * 10 >= n_visible * 6,
        "only {n} of {n_visible} visible rings carry a board position"
    );
}

/// The view yields no board position at all.
fn assert_not_anchored(target: &TargetLayout, image: &GrayImage, view: &View) {
    let result = detect(target, image, view);
    assert_ne!(result.board_frame, Some(BoardFrame::Absolute));
    assert!(
        result
            .detected_markers
            .iter()
            .all(|m| m.board_xy_mm.is_none()),
        "an unanchored view must not report board positions"
    );
    assert!(
        result.detected_markers.len() > 50,
        "the view is still detected ({} markers)",
        result.detected_markers.len()
    );
}

/// Paint a dark, slightly soft disc (a dust speck) into the image.
fn paint_blob(image: &mut GrayImage, center: [f64; 2], radius: f64) {
    let r = radius.ceil() as i64 + 2;
    for dy in -r..=r {
        for dx in -r..=r {
            let (x, y) = (center[0].round() as i64 + dx, center[1].round() as i64 + dy);
            if x < 0 || y < 0 || x >= i64::from(image.width()) || y >= i64::from(image.height()) {
                continue;
            }
            let d = (x as f64 - center[0]).hypot(y as f64 - center[1]);
            let cover = (radius + 0.5 - d).clamp(0.0, 1.0);
            let p = image.get_pixel_mut(x as u32, y as u32);
            p.0[0] = (f64::from(p.0[0]) * (1.0 - cover) + 15.0 * cover).round() as u8;
        }
    }
}

const PITCH_MM: f64 = 2.0;

#[test]
fn rect_window_14x11_anchors_at_offsets_and_rotations() {
    let board = big_rect();
    let base = View::window(PITCH_MM, 14.0, 11.0);
    assert_anchored(board, base.look(3.0, -2.0).roll(4.0), 1);
    assert_anchored(board, base.look(-9.0, 6.0).roll(90.0), 2);
    assert_anchored(board, base.look(8.0, 5.0).roll(180.0), 3);
}

#[test]
fn rect_window_14x11_anchors_under_perspective_and_a_focus_gradient() {
    let board = big_rect();
    let base = View::window(PITCH_MM, 14.0, 11.0);
    assert_anchored(board, base.look(-4.0, 3.0).roll(-23.0).tilt(30.0), 4);
    assert_anchored(
        board,
        base.look(5.0, -4.0)
            .roll(270.0)
            .tilt(20.0)
            .psf(focus_gradient(base.size[1])),
        5,
    );
}

#[test]
fn rect_window_20x16_anchors() {
    let board = big_rect();
    let base = View::window(PITCH_MM, 20.0, 16.0);
    assert_anchored(board, base.look(12.0, 9.0).roll(37.0), 6);
    assert_anchored(board, base.look(-10.0, -8.0).roll(180.0).tilt(20.0), 7);
}

#[test]
fn rect_window_30x30_anchors() {
    let board = big_rect();
    let base = View::window(PITCH_MM, 30.0, 30.0).psf(Psf::Gaussian { sigma: 1.0 });
    assert_anchored(board, base.look(-20.0, 15.0).roll(90.0), 8);
    assert_anchored(board, base.look(18.0, -22.0).roll(-10.0).tilt(25.0), 9);
}

#[test]
fn hex_window_anchors() {
    let board = big_hex();
    let spacing = 2.0 * f64::sqrt(3.0);
    let base = View::window(spacing, 14.0, 11.0);
    assert_anchored(board, base.look(4.0, -3.0).roll(10.0), 10);
    assert_anchored(board, base.look(-12.0, 7.0).roll(180.0).tilt(25.0), 11);
    assert_anchored(board, base.look(9.0, 6.0).roll(-75.0), 12);
}

#[test]
fn window_without_the_triad_is_not_anchored() {
    let board = big_rect();
    let view = View::window(PITCH_MM, 14.0, 11.0)
        .look(40.0, -30.0)
        .roll(12.0);
    let s = scene(board, &view, 13);
    assert_not_anchored(&board.target, &s.image, &view);
}

/// The triad cut by the image border: the left edge runs through the origin
/// cell (the dot at `-½` pitch is out of view, two remain), and, with the top
/// edge one pitch below it, only one dot remains.
#[test]
fn triad_cut_by_the_image_border_is_not_anchored() {
    let board = big_rect();
    let base = View::window(PITCH_MM, 14.0, 11.0);
    let half = [7.0 * PITCH_MM, 5.5 * PITCH_MM];
    // Image centre (look-at) so that the left image edge is at x = 0 of the
    // origin cell.
    let two_dots = base.look(half[0], 0.0);
    let s = scene(board, &two_dots, 14);
    assert_not_anchored(&board.target, &s.image, &two_dots);
    let one_dot = base.look(half[0], half[1] + PITCH_MM);
    let s = scene(board, &one_dot, 15);
    // Check the premise: one dot inside the image, two outside.
    let n_in = board
        .target
        .fiducial_dots_mm()
        .iter()
        .filter(|d| {
            let p = s.project([f64::from(d[0]), f64::from(d[1])]);
            p[0] >= 0.0 && p[1] >= 0.0
        })
        .count();
    assert_eq!(n_in, 1);
    assert_not_anchored(&board.target, &s.image, &one_dot);
}

/// Dark blobs in gaps away from the triad change nothing: the view anchors
/// correctly.
#[test]
fn dust_in_a_gap_never_yields_a_wrong_anchor() {
    let board = big_rect();
    let target = &board.target;
    let view = View::window(PITCH_MM, 14.0, 11.0).look(2.0, 1.0).roll(6.0);
    let s = scene(board, &view, 16);
    let dot_r_px = f64::from(target.fiducials().expect("dots").dot_radius_mm) * view.px_per_mm;
    let origin = origin_mm(target);
    let gap = |du: f64, dv: f64| s.project([origin[0] + du * PITCH_MM, origin[1] + dv * PITCH_MM]);

    let mut far = s.image.clone();
    paint_blob(&mut far, gap(-3.5, -2.5), dot_r_px);
    paint_blob(&mut far, gap(2.5, -1.5), dot_r_px);
    let result = detect(target, &far, &view);
    assert_eq!(result.board_frame, Some(BoardFrame::Absolute));
    assert!(n_true_absolute_labels(&result, &s, PITCH_MM * view.px_per_mm) > 100);
}

/// A dark blob closing the square of four gaps around the triad (the auto
/// triad occupies gaps (½, ½), (−½, ½), (½, 1½) of the origin cell): four
/// equally dark gaps hold four triads, so no placement leads and the view
/// stays relative.
#[test]
fn dark_blob_completing_the_triad_square_is_ambiguous() {
    let board = big_rect();
    let target = &board.target;
    let view = View::window(PITCH_MM, 14.0, 11.0).look(-3.0, 2.0);
    let s = scene(board, &view, 17);
    let dot_r_px = f64::from(target.fiducials().expect("dots").dot_radius_mm) * view.px_per_mm;
    let origin = origin_mm(target);
    let mut image = s.image.clone();
    paint_blob(
        &mut image,
        s.project([origin[0] - 0.5 * PITCH_MM, origin[1] + 1.5 * PITCH_MM]),
        dot_r_px,
    );
    // Paint the real dots too, so all four gaps are equally dark.
    for d in target.fiducial_dots_mm() {
        paint_blob(
            &mut image,
            s.project([f64::from(d[0]), f64::from(d[1])]),
            dot_r_px,
        );
    }
    assert_not_anchored(target, &image, &view);
}
