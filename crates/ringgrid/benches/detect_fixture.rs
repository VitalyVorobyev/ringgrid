use criterion::{criterion_group, criterion_main, Criterion};
use image::ImageReader;
use ringgrid::{BoardLayout, Detector};
use serde::Deserialize;
use std::hint::black_box;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoardLayoutSpecV3 {
    schema: String,
    name: String,
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    marker_outer_radius_mm: f32,
    marker_inner_radius_mm: f32,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_image_path() -> PathBuf {
    repo_root().join("testdata/target_3_split_00.png")
}

fn fixture_board_path() -> PathBuf {
    repo_root().join("testdata/board_ringgrid.json")
}

fn load_fixture_image() -> image::GrayImage {
    ImageReader::open(fixture_image_path())
        .expect("open fixture image")
        .decode()
        .expect("decode fixture image")
        .to_luma8()
}

fn load_fixture_board() -> BoardLayout {
    let board_path = fixture_board_path();
    let raw = std::fs::read_to_string(&board_path).expect("read fixture board json");

    if let Ok(board) = BoardLayout::from_json_str(&raw) {
        return board;
    }

    let legacy: BoardLayoutSpecV3 =
        serde_json::from_str(&raw).expect("fixture board must parse as legacy v3");
    assert_eq!(legacy.schema, "ringgrid.target.v3");

    let canonical = BoardLayout::default();
    assert!((legacy.pitch_mm - canonical.pitch_mm).abs() < 1e-6);
    assert_eq!(legacy.rows, canonical.rows);
    assert_eq!(legacy.long_row_cols, canonical.long_row_cols);
    assert!((legacy.marker_outer_radius_mm - canonical.marker_outer_radius_mm).abs() < 1e-6);
    assert!((legacy.marker_inner_radius_mm - canonical.marker_inner_radius_mm).abs() < 1e-6);

    BoardLayout::with_name(
        legacy.name,
        legacy.pitch_mm,
        legacy.rows,
        legacy.long_row_cols,
        legacy.marker_outer_radius_mm,
        legacy.marker_inner_radius_mm,
        canonical.marker_ring_width_mm,
    )
    .expect("legacy fixture board geometry must remain valid")
}

fn bench_detect_fixture(c: &mut Criterion) {
    let image = load_fixture_image();
    let detector = Detector::new(load_fixture_board());

    c.bench_function("propose_target_3_split_00", |b| {
        b.iter(|| {
            let proposals = detector.propose(black_box(&image));
            black_box(proposals.len())
        })
    });

    c.bench_function("detect_target_3_split_00", |b| {
        b.iter(|| {
            let result = detector.detect(black_box(&image));
            black_box(result.detected_markers.len())
        })
    });
}

criterion_group!(detect_fixture, bench_detect_fixture);
criterion_main!(detect_fixture);
