//! End-to-end detection tests spanning the full valid target matrix:
//! {hex, rect} × {coded, plain} × {origin dots, no dots}, excluding the
//! redundant coded-with-dots pair. Images come from the crate's own PNG target
//! renderer, so these tests exercise generation and detection against each
//! other.

use image::GrayImage;
use ringgrid::{
    BoardFrame, CodebookProfile, CodedRingSpec, DetectConfig, DetectError, DetectionResult,
    Detector, HexGeometry, LatticeGeometry, MarkerCoding, OriginFiducials, PageSpec, RectGeometry,
    RingGeometry, TargetLayout, TargetRenderOptions,
};

/// Count markers assigned a lattice coordinate.
fn labeled_count(result: &DetectionResult) -> usize {
    result
        .detected_markers
        .iter()
        .filter(|m| m.grid_coord.is_some())
        .count()
}

/// Compact plain rect target: 8×8 rings at 14 mm pitch with the
/// L-shaped origin-dot triple near the center.
fn plain_rect_with_dots() -> TargetLayout {
    TargetLayout::new(
        "e2e_rect_dots",
        LatticeGeometry::Rect(RectGeometry {
            rows: 8,
            cols: 8,
            pitch_mm: 14.0,
        }),
        RingGeometry {
            outer_radius_mm: 5.6,
            inner_radius_mm: 2.8,
        },
        MarkerCoding::Plain,
        // Derived triad for an 8x8 board: [49,49], [35,49], [49,63].
        Some(OriginFiducials { dot_radius_mm: 1.4 }),
    )
    .expect("valid plain rect target")
}

/// Compact plain hex target with an auto-placed origin-dot triad.
fn plain_hex_with_dots() -> TargetLayout {
    TargetLayout::with_auto_fiducials(
        "e2e_hex_dots",
        LatticeGeometry::Hex(HexGeometry {
            rows: 7,
            long_row_cols: 7,
            pitch_mm: 8.0,
        }),
        RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 2.4,
        },
        MarkerCoding::Plain,
    )
    .expect("valid plain hex target with auto dots")
}

fn plain_rect_no_dots() -> TargetLayout {
    TargetLayout::new(
        "e2e_rect_nodots",
        LatticeGeometry::Rect(RectGeometry {
            rows: 8,
            cols: 8,
            pitch_mm: 14.0,
        }),
        RingGeometry {
            outer_radius_mm: 5.6,
            inner_radius_mm: 2.8,
        },
        MarkerCoding::Plain,
        None,
    )
    .expect("valid plain rect target")
}

fn render(target: &TargetLayout, dpi: f32) -> GrayImage {
    target
        .render_target_png(
            &TargetRenderOptions::default()
                .with_page(PageSpec::default().with_margin_mm(6.0))
                .with_png_dpi(dpi)
                .with_scale_bar(false),
        )
        .expect("render target png")
}

/// Marker diameter hint for a render at `dpi`.
fn diameter_hint_px(target: &TargetLayout, dpi: f32) -> f32 {
    2.0 * target.ring().outer_radius_mm * dpi / 25.4
}

fn assert_h_consistency(result: &DetectionResult, max_err_px: f64) {
    let h = result.homography.expect("homography must be fitted");
    let project = |x: f64, y: f64| -> [f64; 2] {
        let w = h[0][0] * x + h[0][1] * y + h[0][2];
        let v = h[1][0] * x + h[1][1] * y + h[1][2];
        let z = h[2][0] * x + h[2][1] * y + h[2][2];
        [w / z, v / z]
    };
    for m in &result.detected_markers {
        let Some(xy) = m.board_xy_mm else { continue };
        let p = project(xy[0], xy[1]);
        let err = ((p[0] - m.center[0]).powi(2) + (p[1] - m.center[1]).powi(2)).sqrt();
        assert!(
            err <= max_err_px,
            "marker at {:?} reprojects with error {err:.3}px",
            m.grid_coord
        );
    }
}

#[test]
fn plain_rect_with_dots_detects_anchored() {
    let target = plain_rect_with_dots();
    let dpi = 56.0; // ≈2.2 px/mm → ring outer Ø ≈ 24.7 px
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");

    assert_eq!(
        result.board_frame,
        Some(BoardFrame::Absolute),
        "origin dots must anchor the board frame"
    );
    assert!(
        result.detected_markers.len() >= 60,
        "expected ≥60/64 rings, got {}",
        result.detected_markers.len()
    );
    assert!(
        result
            .detected_markers
            .iter()
            .all(|m| m.id.is_none() && m.grid_coord.is_some() && m.board_xy_mm.is_some()),
        "plain markers carry grid_coord + board_xy_mm, never ids"
    );
    assert_h_consistency(&result, 1.0);

    // Absolute orientation: the render is axis-aligned, so the minimum cell
    // coordinate must be the top-left-most detected marker. Rect coordinates
    // are centered, so an 8x8 board runs -3..=4 and the corner is (-3, -3).
    let corner = result
        .detected_markers
        .iter()
        .find(|m| m.grid_coord == Some([-3, -3]))
        .expect("corner cell (-3,-3) detected");
    for m in &result.detected_markers {
        assert!(
            m.center[0] >= corner.center[0] - 1.0 && m.center[1] >= corner.center[1] - 1.0,
            "cell (-3,-3) must be the top-left corner"
        );
    }
    // Cell (0, 0) is the central one the origin dots surround.
    let center = result
        .detected_markers
        .iter()
        .find(|m| m.grid_coord == Some([0, 0]))
        .expect("cell (0,0) detected");
    assert_eq!(center.board_xy_mm, Some([42.0, 42.0]));
}

#[test]
fn plain_rect_with_dots_anchors_under_image_rotation() {
    let target = plain_rect_with_dots();
    let dpi = 56.0;
    let img = render(&target, dpi);
    // 90° clockwise rotation: orientation-preserving, so the dots must still
    // resolve the origin — and cell (0, 0) now sits top-right.
    let rotated = image::imageops::rotate90(&img);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&rotated).expect("detect");

    assert_eq!(result.board_frame, Some(BoardFrame::Absolute));
    assert!(result.detected_markers.len() >= 60);
    assert_h_consistency(&result, 1.0);

    let corner = result
        .detected_markers
        .iter()
        .find(|m| m.grid_coord == Some([-3, -3]))
        .expect("corner cell (-3,-3) detected");
    for m in &result.detected_markers {
        assert!(
            m.center[0] <= corner.center[0] + 1.0 && m.center[1] >= corner.center[1] - 1.0,
            "after 90° CW rotation cell (-3,-3) must be the top-right corner"
        );
    }
}

#[test]
fn plain_rect_without_dots_stays_in_relative_frame() {
    let target = plain_rect_no_dots();
    let dpi = 56.0;
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");

    assert_eq!(
        result.board_frame,
        Some(BoardFrame::RelativeCanonical),
        "no fiducials ⇒ origin must stay unresolved"
    );
    assert!(result.detected_markers.len() >= 60);
    assert!(result.homography.is_some(), "frame homography still fitted");
    assert!(
        result
            .detected_markers
            .iter()
            .all(|m| m.board_xy_mm.is_none()),
        "unresolved origin ⇒ no millimeter positions"
    );
    assert!(
        result
            .detected_markers
            .iter()
            .all(|m| m.grid_coord.is_some()),
        "labels remain available in the relative frame"
    );
}

#[test]
fn plain_hex_with_dots_detects_anchored() {
    // First end-to-end exercise of hex origin-dot anchoring. Auto dots are
    // ~0.8 mm (0.1×pitch), so render a touch finer than the no-dots hex test
    // to keep the dots resolvable.
    let target = plain_hex_with_dots();
    let dpi = 110.0; // ≈4.3 px/mm → dot radius ≈ 3.5 px, ring outer Ø ≈ 42 px
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");

    assert_eq!(
        result.board_frame,
        Some(BoardFrame::Absolute),
        "hex origin dots must anchor the board frame"
    );
    assert!(
        result
            .detected_markers
            .iter()
            .filter(|m| m.grid_coord.is_some())
            .count()
            >= 27,
        "expected a solid labeled patch, got {}",
        result.detected_markers.len()
    );
    assert!(
        result
            .detected_markers
            .iter()
            .all(|m| m.id.is_none() && m.grid_coord.is_some() && m.board_xy_mm.is_some()),
        "anchored plain markers carry grid_coord + board_xy_mm, never ids"
    );
    assert_h_consistency(&result, 1.5);
}

#[test]
fn plain_hex_detects_in_relative_frame() {
    let target = TargetLayout::new(
        "e2e_hex_plain",
        LatticeGeometry::Hex(HexGeometry {
            rows: 7,
            long_row_cols: 7,
            pitch_mm: 8.0,
        }),
        RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 2.4,
        },
        MarkerCoding::Plain,
        None,
    )
    .expect("valid plain hex target");
    let dpi = 66.0; // ≈2.6 px/mm → ring outer Ø ≈ 25 px
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");
    let n_cells = 45; // rows 7,6,7,6,7,6,7

    assert_eq!(result.board_frame, Some(BoardFrame::RelativeCanonical));
    // Hex labeling drops convex-hull boundary slivers; completion recovers
    // cells inside the labeled patch bbox, so expect solid but not full recall.
    assert!(
        result.detected_markers.len() >= (n_cells * 3) / 5,
        "expected ≥{}/{} hex rings, got {}",
        (n_cells * 3) / 5,
        n_cells,
        result.detected_markers.len()
    );
    assert!(result.homography.is_some());
}

/// Warp `img` through an orientation-preserving homography (rotation +
/// mild perspective about the image center) into a canvas large enough to
/// hold it, on a white background.
fn warp_rotated_perspective(img: &GrayImage, angle_rad: f32, persp: [f32; 2]) -> GrayImage {
    use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (sin, cos) = angle_rad.sin_cos();
    let center = Projection::translate(-0.5 * w, -0.5 * h);
    let rot = Projection::from_matrix([cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0])
        .expect("invertible rotation");
    let tilt = Projection::from_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, persp[0], persp[1], 1.0])
        .expect("invertible perspective");
    let side = 1.6 * w.max(h);
    let out_center = Projection::translate(0.5 * side, 0.5 * side);
    let projection = center.and_then(rot).and_then(tilt).and_then(out_center);
    let mut out = GrayImage::from_pixel(side as u32, side as u32, image::Luma([255]));
    warp_into(
        img,
        projection,
        Interpolation::Bilinear,
        Border::Constant(image::Luma([255])),
        &mut out,
    );
    out
}

/// Assert the reported labels (and homography) are orientation-preserving
/// with respect to the library's own lattice convention.
///
/// The axial step vectors `e_u`, `e_v` are read off `target.cells()` (the
/// single definition of where coordinates sit in board millimeters), so the
/// test does not restate the convention. A printed target seen by a camera
/// images through an orientation-preserving map, so at every labeled cell the
/// image-frame cross product `(p(u+1,v) − p(u,v)) × (p(u,v+1) − p(u,v))` must
/// carry the sign of `e_u × e_v`, and so must the Jacobian determinant of the
/// reported homography at that cell's frame position (`u·e_u + v·e_v`, which
/// the relative frame matches up to a translation).
fn assert_labels_orientation_preserving(result: &DetectionResult, target: &TargetLayout) {
    use std::collections::HashMap;
    let cells: HashMap<(i32, i32), [f64; 2]> = target
        .cells()
        .iter()
        .map(|c| {
            (
                (c.coord.u, c.coord.v),
                [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])],
            )
        })
        .collect();
    let (e_u, e_v) = cells
        .iter()
        .find_map(|(&(u, v), &o)| {
            let pu = cells.get(&(u + 1, v))?;
            let pv = cells.get(&(u, v + 1))?;
            Some(([pu[0] - o[0], pu[1] - o[1]], [pv[0] - o[0], pv[1] - o[1]]))
        })
        .expect("target has a cell with both axial successors");
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let model_sign = cross(e_u, e_v).signum();
    assert!(model_sign != 0.0, "degenerate lattice basis");

    let by_coord: HashMap<(i32, i32), [f64; 2]> = result
        .detected_markers
        .iter()
        .filter_map(|m| m.grid_coord.map(|c| ((c[0], c[1]), m.center)))
        .collect();
    let (mut n_agree, mut n_reversed) = (0usize, 0usize);
    for (&(u, v), &o) in &by_coord {
        let (Some(pu), Some(pv)) = (by_coord.get(&(u + 1, v)), by_coord.get(&(u, v + 1))) else {
            continue;
        };
        let img_cross = cross([pu[0] - o[0], pu[1] - o[1]], [pv[0] - o[0], pv[1] - o[1]]);
        if img_cross.signum() == model_sign {
            n_agree += 1;
        } else {
            n_reversed += 1;
        }
    }
    assert!(
        n_agree + n_reversed > 0,
        "no labeled cell with both axial successors"
    );
    assert_eq!(
        n_reversed, 0,
        "grid_coord→image map is orientation-reversing at {n_reversed} cells \
         ({n_agree} agree): labels are mirrored"
    );

    let h = result.homography.expect("homography must be fitted");
    let apply_h = |p: [f64; 2]| -> [f64; 2] {
        let x = h[0][0] * p[0] + h[0][1] * p[1] + h[0][2];
        let y = h[1][0] * p[0] + h[1][1] * p[1] + h[1][2];
        let z = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
        [x / z, y / z]
    };
    let (&(u, v), _) = by_coord.iter().next().expect("labeled marker");
    let p = [
        f64::from(u) * e_u[0] + f64::from(v) * e_v[0],
        f64::from(u) * e_u[1] + f64::from(v) * e_v[1],
    ];
    let o = apply_h(p);
    let du = apply_h([p[0] + e_u[0], p[1] + e_u[1]]);
    let dv = apply_h([p[0] + e_v[0], p[1] + e_v[1]]);
    let h_cross = cross([du[0] - o[0], du[1] - o[1]], [dv[0] - o[0], dv[1] - o[1]]);
    assert_eq!(
        h_cross.signum(),
        model_sign,
        "reported homography is orientation-reversing in the lattice frame"
    );
}

/// Without origin dots the labels stay in the relative canonical frame, but
/// that frame may only differ from the board by a rotation + translation —
/// never a reflection. Exercised on both lattices under a rotated,
/// perspective view so canonicalization cannot lean on image-axis alignment.
#[test]
fn plain_no_dots_relative_labels_preserve_orientation() {
    let hex = TargetLayout::new(
        "e2e_hex_plain_orient",
        LatticeGeometry::Hex(HexGeometry {
            rows: 7,
            long_row_cols: 7,
            pitch_mm: 8.0,
        }),
        RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 2.4,
        },
        MarkerCoding::Plain,
        None,
    )
    .expect("valid plain hex target");
    for (target, dpi) in [(hex, 66.0f32), (plain_rect_no_dots(), 56.0f32)] {
        for angle_deg in [0.0f32, 25.0, -110.0] {
            let img = warp_rotated_perspective(
                &render(&target, dpi),
                angle_deg.to_radians(),
                [2.0e-4, -1.5e-4],
            );
            let detector =
                Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));
            let result = detector.detect(&img).expect("detect");
            assert_eq!(
                result.board_frame,
                Some(BoardFrame::RelativeCanonical),
                "{} @ {angle_deg}°: no dots ⇒ relative frame",
                target.name()
            );
            assert!(
                labeled_count(&result) >= 20,
                "{} @ {angle_deg}°: only {} labeled",
                target.name(),
                labeled_count(&result)
            );
            assert_labels_orientation_preserving(&result, &target);
        }
    }
}

#[test]
fn coded_rect_detects_absolute_ids() {
    // Rect lattice with the default coded ring geometry (matches the hex
    // preset's marker dimensions, so decode priors hold).
    let target = TargetLayout::new(
        "e2e_rect_coded",
        LatticeGeometry::Rect(RectGeometry {
            rows: 6,
            cols: 6,
            pitch_mm: 14.0,
        }),
        RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 3.2,
        },
        MarkerCoding::Coded16(CodedRingSpec {
            ring_width_mm: 1.152,
            codebook_profile: CodebookProfile::Base,
            id_assignment: None,
        }),
        None,
    )
    .expect("valid coded rect target");
    let dpi = 130.0; // ≈5.1 px/mm → ring outer Ø ≈ 49 px (decode needs pixels)
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");

    assert_eq!(result.board_frame, Some(BoardFrame::Absolute));
    assert!(
        result.detected_markers.len() >= 30,
        "expected ≥30/36 coded rect markers, got {}",
        result.detected_markers.len()
    );
    let n_decoded = result
        .detected_markers
        .iter()
        .filter(|m| m.id.is_some())
        .count();
    assert!(n_decoded >= 30, "expected ≥30 decoded ids, got {n_decoded}");
    assert_h_consistency(&result, 1.0);

    // IDs anchor absolutely: sequential assignment in generation order, with
    // centered coordinates a 6x6 board runs -2..=3, so id == (row+2)*6 + col+2.
    for m in &result.detected_markers {
        let (Some(id), Some(coord)) = (m.id, m.grid_coord) else {
            continue;
        };
        assert_eq!(
            id as i32,
            (coord[1] + 2) * 6 + (coord[0] + 2),
            "decoded id must match its lattice cell"
        );
    }
}

#[test]
fn coded_hex_detects_absolute_ids() {
    // Matrix combo: hex + coded + no dots (the classic board, rendered small).
    let target = TargetLayout::coded_hex(8.0, 5, 5, 4.8, 3.2, 1.152).expect("valid coded hex");
    let dpi = 130.0; // decode needs pixels: ring outer Ø ≈ 49 px
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");

    assert_eq!(result.board_frame, Some(BoardFrame::Absolute));
    let n_decoded = result
        .detected_markers
        .iter()
        .filter(|m| m.id.is_some())
        .count();
    assert!(
        n_decoded >= (target.n_cells() * 4) / 5,
        "expected most of {} hex cells decoded, got {n_decoded}",
        target.n_cells()
    );
    assert_h_consistency(&result, 1.0);
}

#[test]
fn plain_no_dots_reports_board_complete_consistently() {
    // The board_complete signal must be populated for a labeled plain run and
    // equal `labeled >= n_cells` — its definition.
    let target = plain_rect_no_dots();
    let dpi = 56.0;
    let img = render(&target, dpi);
    let detector =
        Detector::with_marker_diameter_hint(target.clone(), diameter_hint_px(&target, dpi));

    let result = detector.detect(&img).expect("detect");

    let complete = labeled_count(&result) >= target.n_cells();
    assert_eq!(
        result.board_complete,
        Some(complete),
        "board_complete must reflect labeled({}) >= n_cells({})",
        labeled_count(&result),
        target.n_cells()
    );
}

#[test]
fn require_complete_board_errors_on_partial_board() {
    // Crop the render so the rightmost markers fall off-image: those cells can
    // never be labeled or completed, so the board is genuinely incomplete and
    // the strict gate must reject it.
    let target = plain_rect_no_dots();
    let dpi = 56.0;
    let full = render(&target, dpi);
    let cropped =
        image::imageops::crop_imm(&full, 0, 0, full.width() * 3 / 4, full.height()).to_image();

    let hint = diameter_hint_px(&target, dpi);
    let mut config = DetectConfig::from_target_and_marker_diameter(target.clone(), hint);
    config.require_complete_board = true;
    let strict = Detector::with_config(config);

    match strict.detect(&cropped) {
        Err(DetectError::IncompleteBoard { found, expected }) => {
            assert!(
                found < expected,
                "found {found} should be < expected {expected}"
            );
            assert_eq!(expected, target.n_cells());
        }
        other => panic!("expected IncompleteBoard error, got {other:?}"),
    }

    // Without the gate, the same detection succeeds with board_complete = false.
    let lenient = Detector::with_marker_diameter_hint(target, hint);
    let result = lenient.detect(&cropped).expect("lenient detect");
    assert_eq!(result.board_complete, Some(false));
}

#[test]
fn require_complete_board_errors_when_grid_assignment_never_runs() {
    // A blank image never reaches grid assignment, so no board frame is
    // produced and `board_complete` is undefined (`None`) in the lenient case.
    // The strict gate must still reject it: the full board was asked for and
    // zero cells are present, so it fails with `found = 0` rather than
    // silently succeeding.
    let target = plain_rect_no_dots();
    let blank = GrayImage::from_pixel(320, 320, image::Luma([255]));

    let mut config = DetectConfig::from_target(target.clone());
    config.require_complete_board = true;
    let strict = Detector::with_config(config);

    match strict.detect(&blank) {
        Err(DetectError::IncompleteBoard { found, expected }) => {
            assert_eq!(found, 0, "a blank image labels no cells");
            assert_eq!(expected, target.n_cells());
        }
        other => panic!("expected IncompleteBoard error, got {other:?}"),
    }

    // Lenient: no error, no false success signal — completeness stays undefined.
    let lenient = Detector::new(target);
    let result = lenient.detect(&blank).expect("lenient detect");
    assert_eq!(result.board_complete, None);
    assert!(result.board_frame.is_none());
}
