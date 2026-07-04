use criterion::{Criterion, criterion_group, criterion_main};
use image::ImageReader;
use ringgrid::{Detector, TargetLayout};
use std::hint::black_box;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_image_path() -> PathBuf {
    repo_root().join("testdata/target_3_split_00.png")
}

fn fixture_target_path() -> PathBuf {
    repo_root().join("testdata/board_ringgrid.json")
}

fn load_fixture_image() -> image::GrayImage {
    ImageReader::open(fixture_image_path())
        .expect("open fixture image")
        .decode()
        .expect("decode fixture image")
        .to_luma8()
}

fn load_fixture_target() -> TargetLayout {
    let path = fixture_target_path();
    let raw = std::fs::read_to_string(&path).expect("read fixture target json");
    // v4 board_spec.json files auto-migrate to the v5 model on load.
    TargetLayout::from_json_str(&raw).expect("fixture target json must parse (v4/v5)")
}

fn bench_detect_fixture(c: &mut Criterion) {
    let image = load_fixture_image();
    let detector = Detector::new(load_fixture_target());

    c.bench_function("propose_target_3_split_00", |b| {
        b.iter(|| {
            let proposals = detector.propose(black_box(&image));
            black_box(proposals.len())
        })
    });

    c.bench_function("detect_target_3_split_00", |b| {
        b.iter(|| {
            let result = detector
                .detect(black_box(&image))
                .expect("supported target");
            black_box(result.detected_markers.len())
        })
    });
}

criterion_group!(detect_fixture, bench_detect_fixture);
criterion_main!(detect_fixture);
