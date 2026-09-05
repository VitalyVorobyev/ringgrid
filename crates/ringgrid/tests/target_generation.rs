use image::load_from_memory;
use png::{Decoder as PngDecoder, Unit};
use ringgrid::{
    CodebookProfile, CodedRingSpec, HexGeometry, LatticeGeometry, MarkerCoding, PageOrientation,
    PageSize, PageSpec, RingGeometry, TargetGenerationError, TargetLayout, TargetRenderOptions,
};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const EXPECTED_JSON: &str = include_str!("fixtures/target_generation/fixture_compact_hex.json");
const EXPECTED_SVG: &str = include_str!("fixtures/target_generation/fixture_compact_hex.svg");
const EXPECTED_PNG: &[u8] = include_bytes!("fixtures/target_generation/fixture_compact_hex.png");

fn normalize_text_newlines(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn fixture_target() -> TargetLayout {
    TargetLayout::new(
        "fixture_compact_hex",
        LatticeGeometry::Hex(HexGeometry {
            rows: 3,
            long_row_cols: 4,
            pitch_mm: 8.0,
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
    .expect("fixture target must be valid")
}

fn temp_output_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ringgrid_{prefix}_{}_{}",
        std::process::id(),
        nanos
    ))
}

#[test]
fn json_generation_matches_committed_fixture() {
    let target = fixture_target();
    assert_eq!(
        format!("{}\n", target.to_json_string()),
        normalize_text_newlines(EXPECTED_JSON)
    );
}

#[test]
fn svg_generation_matches_committed_fixture() {
    let target = fixture_target();
    let svg = target
        .render_target_svg(&TargetRenderOptions::default())
        .expect("fixture svg");
    assert_eq!(
        normalize_text_newlines(&svg),
        normalize_text_newlines(EXPECTED_SVG)
    );
}

#[test]
fn png_generation_matches_committed_fixture_pixels() {
    let target = fixture_target();
    let rendered = target
        .render_target_png(&TargetRenderOptions::default().with_png_dpi(96.0))
        .expect("fixture png");
    let expected = load_from_memory(EXPECTED_PNG)
        .expect("decode committed fixture png")
        .into_luma8();

    assert_eq!(rendered.dimensions(), expected.dimensions());
    assert_eq!(rendered.as_raw(), expected.as_raw());
}

#[test]
fn file_writers_create_parent_dirs_and_round_trip() {
    let target = fixture_target();
    let out_dir = temp_output_dir("target_generation");
    let json_path = out_dir.join("nested/fixture.json");
    let svg_path = out_dir.join("nested/fixture.svg");
    let png_path = out_dir.join("nested/fixture.target");

    target
        .write_json_file(&json_path)
        .expect("write fixture json");
    target
        .write_target_svg(&svg_path, &TargetRenderOptions::default())
        .expect("write fixture svg");
    target
        .write_target_png(
            &png_path,
            &TargetRenderOptions::default().with_png_dpi(96.0),
        )
        .expect("write fixture png");

    assert_eq!(
        normalize_text_newlines(&std::fs::read_to_string(&json_path).expect("read written json")),
        normalize_text_newlines(EXPECTED_JSON)
    );
    assert_eq!(
        normalize_text_newlines(&std::fs::read_to_string(&svg_path).expect("read written svg")),
        normalize_text_newlines(EXPECTED_SVG)
    );

    let png_bytes = std::fs::read(&png_path).expect("read written png bytes");
    assert!(png_bytes.starts_with(b"\x89PNG\r\n\x1a\n"));

    let reader = PngDecoder::new(Cursor::new(&png_bytes))
        .read_info()
        .expect("read written png info");
    let pixel_dims = reader.info().pixel_dims.expect("png pHYs metadata");
    let expected_ppm = (96.0_f64 * 1000.0 / 25.4).round() as u32;
    assert_eq!(pixel_dims.xppu, expected_ppm);
    assert_eq!(pixel_dims.yppu, expected_ppm);
    assert_eq!(pixel_dims.unit, Unit::Meter);

    let written_png = load_from_memory(&png_bytes)
        .expect("decode written png")
        .into_luma8();
    let expected_png = load_from_memory(EXPECTED_PNG)
        .expect("decode committed fixture png")
        .into_luma8();
    assert_eq!(written_png.dimensions(), expected_png.dimensions());
    assert_eq!(written_png.as_raw(), expected_png.as_raw());

    let _ = std::fs::remove_dir_all(out_dir);
}

// ── Page placement ──────────────────────────────────────────────────────────

/// Pull `width`/`height` (in mm) out of the generated `<svg>` element.
fn svg_size_mm(svg: &str) -> (f64, f64) {
    let parse = |attr: &str| {
        let start = svg.find(&format!("{attr}=\"")).expect("attribute present") + attr.len() + 2;
        let rest = &svg[start..];
        let end = rest.find("mm").expect("mm unit");
        rest[..end].parse::<f64>().expect("numeric size")
    };
    (parse("width"), parse("height"))
}

#[test]
fn fit_content_page_is_square_and_matches_the_committed_fixture() {
    // The regression lock for the page model: the default page must reproduce
    // the pre-page-model output exactly, asserted here and not only through the
    // CLI's fixture comparison.
    let svg = fixture_target()
        .render_target_svg(&TargetRenderOptions::default())
        .expect("fixture svg");
    assert_eq!(
        normalize_text_newlines(&svg),
        normalize_text_newlines(EXPECTED_SVG)
    );

    let (w, h) = svg_size_mm(&svg);
    assert_eq!(w, h, "fit-content pages are square");
}

#[test]
fn named_page_sizes_drive_the_svg_extent() {
    let target = fixture_target();
    for (size, orientation, expected) in [
        (PageSize::A4, PageOrientation::Portrait, (210.0, 297.0)),
        (PageSize::A4, PageOrientation::Landscape, (297.0, 210.0)),
        (PageSize::Letter, PageOrientation::Portrait, (215.9, 279.4)),
        (
            PageSize::Custom {
                width_mm: 180.0,
                height_mm: 120.0,
            },
            PageOrientation::Portrait,
            (180.0, 120.0),
        ),
    ] {
        let options = TargetRenderOptions::default()
            .with_page(PageSpec::new(size).with_orientation(orientation));
        let svg = target.render_target_svg(&options).expect("render");
        let (w, h) = svg_size_mm(&svg);
        assert!(
            (w - expected.0).abs() < 1e-6,
            "{size:?} {orientation:?} width {w}"
        );
        assert!(
            (h - expected.1).abs() < 1e-6,
            "{size:?} {orientation:?} height {h}"
        );

        // page_size_mm must agree with what the SVG actually carries.
        let [pw, ph] = target.page_size_mm(&options).expect("page size");
        assert!((f64::from(pw) - w).abs() < 1e-3);
        assert!((f64::from(ph) - h).abs() < 1e-3);
    }
}

#[test]
fn content_is_centered_in_the_printable_area() {
    let target = fixture_target();
    let options = TargetRenderOptions::default()
        .with_page(PageSpec::new(PageSize::A4))
        .with_scale_bar(false);
    let svg = target.render_target_svg(&options).expect("render");
    let (page_w, page_h) = svg_size_mm(&svg);

    // The first and last marker centers bracket the drawn content; their
    // midpoint must sit at the page center.
    let centers: Vec<(f64, f64)> = svg
        .lines()
        .filter_map(|l| l.strip_prefix("<circle cx=\""))
        .filter_map(|rest| {
            let (cx, rest) = rest.split_once("\" cy=\"")?;
            let (cy, _) = rest.split_once('"')?;
            Some((cx.parse().ok()?, cy.parse().ok()?))
        })
        .collect();
    assert!(!centers.is_empty(), "coded markers draw circles");

    let min_x = centers.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let max_x = centers
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = centers.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let max_y = centers
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max);

    assert!(
        (0.5 * (min_x + max_x) - 0.5 * page_w).abs() < 1e-3,
        "content centered horizontally"
    );
    assert!(
        (0.5 * (min_y + max_y) - 0.5 * page_h).abs() < 1e-3,
        "content centered vertically"
    );
}

#[test]
fn a_target_too_large_for_the_page_is_rejected() {
    let err = TargetLayout::rect_24x24()
        .render_target_svg(&TargetRenderOptions::default().with_page(PageSpec::new(PageSize::A4)))
        .expect_err("a 343 mm board does not fit A4");
    let TargetGenerationError::ContentExceedsPage {
        content_mm,
        printable_mm,
    } = err
    else {
        panic!("expected ContentExceedsPage, got {err:?}");
    };
    assert!(content_mm[0] > printable_mm[0]);
}

#[test]
fn a_margin_that_consumes_the_page_is_rejected() {
    let err = fixture_target()
        .render_target_svg(
            &TargetRenderOptions::default()
                .with_page(PageSpec::new(PageSize::A4).with_margin_mm(150.0)),
        )
        .expect_err("300 mm of margin leaves nothing on A4");
    assert!(matches!(
        err,
        TargetGenerationError::EmptyPrintableArea { .. }
    ));
}

#[test]
fn a_non_finite_custom_page_is_rejected() {
    let err = fixture_target()
        .render_target_svg(&TargetRenderOptions::default().with_page(PageSpec::new(
            PageSize::Custom {
                width_mm: f64::NAN,
                height_mm: 100.0,
            },
        )))
        .expect_err("NaN page size");
    assert!(matches!(err, TargetGenerationError::InvalidPageSize { .. }));
}

// ── Artifact bundle ─────────────────────────────────────────────────────────

#[test]
fn bundle_agrees_with_the_individual_renderers() {
    let target = fixture_target();
    let options = TargetRenderOptions::default().with_png_dpi(96.0);
    let bundle = target
        .render_target_artifacts(&options)
        .expect("render bundle");

    assert_eq!(bundle.svg_text, target.render_target_svg(&options).unwrap());
    assert_eq!(bundle.dxf_text, target.render_target_dxf());
    assert_eq!(bundle.json_text, format!("{}\n", target.to_json_string()));

    // The PNG bytes must decode back to the same raster as render_target_png.
    let decoded = load_from_memory(&bundle.png_bytes)
        .expect("bundle png decodes")
        .into_luma8();
    let direct = target.render_target_png(&options).expect("direct png");
    assert_eq!(decoded.dimensions(), direct.dimensions());
    assert_eq!(decoded.as_raw(), direct.as_raw());
}

#[test]
fn bundle_png_carries_the_requested_dpi() {
    let bundle = fixture_target()
        .render_target_artifacts(&TargetRenderOptions::default().with_png_dpi(96.0))
        .expect("render bundle");
    let reader = PngDecoder::new(Cursor::new(&bundle.png_bytes))
        .read_info()
        .expect("png header");
    let dims = reader
        .info()
        .pixel_dims
        .expect("pHYs chunk carries the print scale");
    assert_eq!(dims.unit, Unit::Meter);
    // 96 dpi = 96 / 25.4 px per mm = 3779.5 px per meter.
    assert_eq!(dims.xppu, 3780);
    assert_eq!(dims.yppu, 3780);
}

#[test]
fn bundle_matches_the_committed_fixtures() {
    let bundle = fixture_target()
        .render_target_artifacts(&TargetRenderOptions::default().with_png_dpi(96.0))
        .expect("render bundle");
    assert_eq!(
        normalize_text_newlines(&bundle.json_text),
        normalize_text_newlines(EXPECTED_JSON)
    );
    assert_eq!(
        normalize_text_newlines(&bundle.svg_text),
        normalize_text_newlines(EXPECTED_SVG)
    );
}

// ── Codebook profile ────────────────────────────────────────────────────────

fn coded_hex_with_profile(
    rows: usize,
    long_row_cols: usize,
    codebook_profile: CodebookProfile,
) -> Result<TargetLayout, ringgrid::TargetValidationError> {
    TargetLayout::new(
        "profile_probe",
        LatticeGeometry::Hex(HexGeometry {
            rows,
            long_row_cols,
            pitch_mm: 8.0,
        }),
        RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 3.2,
        },
        MarkerCoding::Coded16(CodedRingSpec {
            ring_width_mm: 1.152,
            codebook_profile,
            id_assignment: None,
        }),
        None,
    )
}

#[test]
fn extended_profile_lifts_the_893_cell_cap() {
    // 40 x 40 hex ≈ 1560 cells: over the baseline table, inside the extended one.
    assert!(
        coded_hex_with_profile(40, 40, CodebookProfile::Base).is_err(),
        "baseline table holds 893 codewords"
    );
    let extended =
        coded_hex_with_profile(40, 40, CodebookProfile::Extended).expect("extended table fits");
    assert!(extended.n_cells() > 893);
}

#[test]
fn extended_profile_renders_codewords_the_baseline_table_does_not_have() {
    let extended = coded_hex_with_profile(40, 40, CodebookProfile::Extended).expect("valid");
    // Rendering a >893 target proves the renderer reads the target's profile:
    // the baseline table would be out of bounds for these IDs.
    let svg = extended
        .render_target_svg(&TargetRenderOptions::default())
        .expect("render extended");
    assert!(svg.contains("data-id=\"1000\""));
}

#[test]
fn base_profile_targets_keep_their_exact_json() {
    // `codebook_profile` is skipped when it is Base, so specs written before the
    // field existed round-trip byte for byte.
    let json = fixture_target().to_json_string();
    assert!(
        !json.contains("codebook_profile"),
        "baseline profile must not appear in the spec"
    );
    assert_eq!(
        normalize_text_newlines(&format!("{json}\n")),
        normalize_text_newlines(EXPECTED_JSON)
    );
}

#[test]
fn extended_profile_round_trips_through_the_target_spec() {
    let extended = coded_hex_with_profile(5, 5, CodebookProfile::Extended).expect("valid");
    let json = extended.to_json_string();
    assert!(
        json.contains("\"codebook_profile\": \"extended\""),
        "{json}"
    );

    let back = TargetLayout::from_json_str(&json).expect("round trip");
    let MarkerCoding::Coded16(spec) = back.coding() else {
        panic!("still coded");
    };
    assert_eq!(spec.codebook_profile, CodebookProfile::Extended);
}

#[test]
fn detection_config_follows_the_targets_codebook_profile() {
    let extended = coded_hex_with_profile(5, 5, CodebookProfile::Extended).expect("valid");
    let config = ringgrid::DetectConfig::default().with_target(extended);
    assert_eq!(
        config.advanced.decode.codebook_profile,
        CodebookProfile::Extended,
        "decoding must use the table the target was printed from"
    );

    let base = ringgrid::DetectConfig::default().with_target(fixture_target());
    assert_eq!(base.advanced.decode.codebook_profile, CodebookProfile::Base);
}

#[test]
fn dxf_is_page_independent() {
    // Documented contract: the DXF is board-frame millimeters for fabrication,
    // so page, margin and scale bar must not reach it. Guarding it here means
    // a future page-plumbing change cannot silently start shifting CAD output.
    let target = fixture_target();
    let baseline = target.render_target_dxf();

    for options in [
        TargetRenderOptions::default(),
        TargetRenderOptions::default()
            .with_page(PageSpec::new(PageSize::A4).with_margin_mm(10.0))
            .with_scale_bar(false),
        TargetRenderOptions::default()
            .with_page(PageSpec::new(PageSize::Letter).with_orientation(PageOrientation::Landscape))
            .with_png_dpi(600.0),
    ] {
        let bundle = target
            .render_target_artifacts(&options)
            .expect("render bundle");
        assert_eq!(bundle.dxf_text, baseline, "DXF changed with {options:?}");
    }
}
