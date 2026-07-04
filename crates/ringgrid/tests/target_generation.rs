use image::load_from_memory;
use png::{Decoder as PngDecoder, Unit};
use ringgrid::{
    CodedRingSpec, HexGeometry, LatticeGeometry, MarkerCoding, PngTargetOptions, RingGeometry,
    SvgTargetOptions, TargetLayout,
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
        .render_target_svg(&SvgTargetOptions::default())
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
        .render_target_png(&PngTargetOptions {
            dpi: 96.0,
            ..PngTargetOptions::default()
        })
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
        .write_target_svg(&svg_path, &SvgTargetOptions::default())
        .expect("write fixture svg");
    target
        .write_target_png(
            &png_path,
            &PngTargetOptions {
                dpi: 96.0,
                ..PngTargetOptions::default()
            },
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
