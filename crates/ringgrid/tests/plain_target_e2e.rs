//! End-to-end detection tests for the target combinations unlocked by the
//! pipeline back-half generalization: plain rect (with and without origin
//! dots), plain hex, and coded rect. Images come from the crate's own PNG
//! target renderer, so these tests exercise generation and detection against
//! each other.

use image::GrayImage;
use ringgrid::{
    BoardFrame, CodedRingSpec, DetectionResult, Detector, HexGeometry, LatticeGeometry,
    MarkerCoding, OriginFiducials, PngTargetOptions, RectGeometry, RingGeometry, TargetLayout,
};

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
        Some(OriginFiducials {
            dot_radius_mm: 1.4,
            dots_mm: vec![[49.0, 49.0], [35.0, 49.0], [49.0, 63.0]],
        }),
    )
    .expect("valid plain rect target")
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
        .render_target_png(&PngTargetOptions {
            dpi,
            margin_mm: 6.0,
            include_scale_bar: false,
        })
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

    // Absolute orientation: the render is axis-aligned, so cell (0, 0) — the
    // board origin — must be the top-left-most detected marker.
    let origin = result
        .detected_markers
        .iter()
        .find(|m| m.grid_coord == Some([0, 0]))
        .expect("cell (0,0) detected");
    for m in &result.detected_markers {
        assert!(
            m.center[0] >= origin.center[0] - 1.0 && m.center[1] >= origin.center[1] - 1.0,
            "cell (0,0) must be the top-left corner"
        );
    }
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

    let origin = result
        .detected_markers
        .iter()
        .find(|m| m.grid_coord == Some([0, 0]))
        .expect("cell (0,0) detected");
    for m in &result.detected_markers {
        assert!(
            m.center[0] <= origin.center[0] + 1.0 && m.center[1] >= origin.center[1] - 1.0,
            "after 90° CW rotation cell (0,0) must be the top-right corner"
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

    // IDs anchor absolutely: sequential assignment means id == row*6 + col.
    for m in &result.detected_markers {
        let (Some(id), Some(coord)) = (m.id, m.grid_coord) else {
            continue;
        };
        assert_eq!(
            id as i32,
            coord[1] * 6 + coord[0],
            "decoded id must match its lattice cell"
        );
    }
}
