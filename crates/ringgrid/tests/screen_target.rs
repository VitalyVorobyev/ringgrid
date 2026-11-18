use ringgrid::{
    LatticeGeometry, MarkerCoding, OriginDots, ScreenTargetError, ScreenTargetOptions, TargetLayout,
};

const INKPAD_PITCH_MM: f64 = 25.4 / 150.0;

#[test]
fn inkpad_keeps_marker_proportions_and_packs_the_most_rows_and_columns() {
    let template = TargetLayout::default_hex();
    for (diameter, expected) in [(10.752, (16, 10, 152)), (7.168, (25, 15, 363))] {
        let options = ScreenTargetOptions::new(825, 1200, INKPAD_PITCH_MM, diameter);
        let result = template.render_fitted_screen_target(&options).unwrap();
        let m = &result.metadata;
        assert_eq!((m.rows, m.columns, m.marker_count), expected);
        let target = TargetLayout::from_json_str(&result.json_text).unwrap();
        assert_eq!(m.marker_count, target.n_cells());
        assert!((m.drawn_marker_diameter_mm - diameter).abs() < 2e-6);
        assert!((f64::from(target.pitch_mm()) / diameter - 8.0 / 10.752).abs() < 1e-6);
        let band = m.code_band_width_px.unwrap();
        assert!((band * INKPAD_PITCH_MM / diameter - 1.0 / 24.0).abs() < 1e-6);
        let image = image::load_from_memory(&result.png_bytes)
            .unwrap()
            .into_luma8();
        assert_eq!(image.dimensions(), (825, 1200));
        assert!(image.pixels().any(|p| p[0] > 0 && p[0] < 255));
        assert!(
            image.enumerate_pixels().all(|(x, y, p)| {
                p[0] == 255 || ((8..817).contains(&x) && (8..1192).contains(&y))
            })
        );
        let first = target.cells()[0].xy_mm;
        let second = target.cells()[1].xy_mm;
        let center_delta = f64::from(second[0] - first[0]) / INKPAD_PITCH_MM;
        assert!((center_delta - center_delta.round()).abs() > 0.01);
    }
}

#[test]
fn fits_all_lattice_and_coding_combinations_with_canonical_geometry() {
    let opts = ScreenTargetOptions::new(340, 420, 0.31, 11.2);
    let templates = [
        TargetLayout::default_hex(),
        TargetLayout::coded_rect(14.0, 5, 5, 4.8, 3.2, 1.152).unwrap(),
        TargetLayout::plain_hex(8.0, 5, 5, 4.8, 3.2, OriginDots::Auto).unwrap(),
        TargetLayout::rect_24x24(),
    ];
    for template in templates {
        let a = template.render_fitted_screen_target(&opts).unwrap();
        let b = template.render_fitted_screen_target(&opts).unwrap();
        assert_eq!(a, b);
        let fitted = TargetLayout::from_json_str(&a.json_text).unwrap();
        assert_eq!(fitted.lattice_kind(), template.lattice_kind());
        assert_eq!(fitted.is_coded(), template.is_coded());
        assert_eq!(fitted.fiducials().is_some(), template.fiducials().is_some());
        assert_eq!(fitted.n_cells(), a.metadata.marker_count);
        let image = image::load_from_memory(&a.png_bytes).unwrap().into_luma8();
        for dot in fitted.fiducial_dots_mm() {
            let [x, y] = [0, 1].map(|axis| {
                (f64::from(dot[axis]) / opts.pixel_pitch_mm + a.metadata.board_origin_px[axis])
                    .round() as u32
            });
            assert_eq!(image.get_pixel(x, y)[0], 0);
        }
    }
}

#[test]
fn rejects_impossible_fit_and_preserves_explicit_ids_only_on_the_same_grid() {
    let template = TargetLayout::default_hex();
    assert!(matches!(
        template.render_fitted_screen_target(&ScreenTargetOptions::new(
            25,
            25,
            INKPAD_PITCH_MM,
            10.752
        )),
        Err(ScreenTargetError::NoFit)
    ));
    assert!(matches!(
        template.render_fitted_screen_target(&ScreenTargetOptions::new(400, 500, f64::NAN, 10.752)),
        Err(ScreenTargetError::PixelPitch)
    ));
    let mut code = template.coding().clone();
    let MarkerCoding::Coded16(spec) = &mut code else {
        unreachable!()
    };
    spec.id_assignment = Some((0..152).rev().collect());
    let assigned = TargetLayout::new(
        "explicit",
        LatticeGeometry::Hex(ringgrid::HexGeometry {
            rows: 16,
            long_row_cols: 10,
            pitch_mm: 8.0,
        }),
        template.ring(),
        code,
        None,
    )
    .unwrap();
    let opts = ScreenTargetOptions::new(825, 1200, INKPAD_PITCH_MM, 10.752);
    let fitted = assigned.render_fitted_screen_target(&opts).unwrap();
    let fitted = TargetLayout::from_json_str(&fitted.json_text).unwrap();
    assert_eq!(fitted.cells()[0].id, Some(151));
    assert!(matches!(
        assigned.render_fitted_screen_target(&ScreenTargetOptions::new(
            825,
            1200,
            INKPAD_PITCH_MM,
            7.168
        )),
        Err(ScreenTargetError::ExplicitIds)
    ));
}

#[test]
fn overlapping_explicit_origin_dots_render_as_one_ink_region() {
    let target = TargetLayout::new(
        "large_dots",
        LatticeGeometry::Rect(ringgrid::RectGeometry {
            rows: 8,
            cols: 8,
            pitch_mm: 14.0,
        }),
        ringgrid::RingGeometry {
            inner_radius_mm: 0.4,
            outer_radius_mm: 0.8,
        },
        MarkerCoding::Plain,
        Some(ringgrid::OriginFiducials { dot_radius_mm: 8.0 }),
    )
    .unwrap();
    let options = ScreenTargetOptions::new(500, 500, 0.25, 1.6);
    let rendered = target.render_fitted_screen_target(&options).unwrap();
    let image = image::load_from_memory(&rendered.png_bytes)
        .unwrap()
        .into_luma8();
    let fitted = TargetLayout::from_json_str(&rendered.json_text).unwrap();
    let midpoint = [0, 1].map(|axis| {
        let dots = fitted.fiducial_dots_mm();
        let mm = f64::from(dots[0][axis] + dots[1][axis]) * 0.5;
        (mm / options.pixel_pitch_mm + rendered.metadata.board_origin_px[axis]).round() as u32
    });
    assert_eq!(image.get_pixel(midpoint[0], midpoint[1])[0], 0);
}
