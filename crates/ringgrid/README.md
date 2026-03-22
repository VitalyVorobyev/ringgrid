# ringgrid

Pure-Rust detector for dense coded ring calibration targets on a hex lattice.
Detects markers with subpixel edge precision, decodes 16-sector binary IDs from
a shipped baseline 893-codeword profile (with an opt-in extended profile
available for larger ID spaces), fits ellipses via Fitzgibbon's direct method with
RANSAC, corrects projective center bias, and estimates a board-to-image
homography. No OpenCV dependency.

## Key Features

- **Subpixel edge detection** — gradient-based radial sampling produces edge points fed to a direct ellipse fit, yielding subpixel-accurate marker localization
- **Projective center correction** — recovers the true projected center from inner/outer conic pencil geometry, correcting the systematic bias of ellipse-fit centers
- **Consistency-first ID correction** — verifies decoded IDs against local hex-lattice structure, clears contradictory IDs, and recovers safe missing IDs before global filtering
- **Stable baseline IDs plus opt-in extension** — shipped `base` profile keeps 893 stable IDs at minimum cyclic Hamming distance 2; opt-in `extended` grows capacity to 2180 IDs with a weaker minimum distance of 1 without introducing new polarity ambiguity beyond the shipped baseline
- **Distortion-aware** — supports external camera models (Brown-Conrady) via the `PixelMapper` trait, or blind single-parameter self-undistort estimation
- **Pure Rust** — no C/C++ dependencies, no OpenCV bindings

## Pipeline Stages

Named stage order:
proposal -> local fit/decode -> dedup -> projective center -> `id_correction` -> optional global filter -> optional completion -> final homography refit.

## Installation

```toml
[dependencies]
ringgrid = "0.5"
```

## Rust Target Generation

The library can generate canonical target JSON plus printable SVG/PNG directly:

```rust,no_run
use ringgrid::{BoardLayout, PngTargetOptions, SvgTargetOptions};
use std::path::Path;

let board = BoardLayout::with_name("ringgrid_demo", 8.0, 15, 14, 4.8, 3.2, 1.152).unwrap();

board.write_json_file(Path::new("target.json")).unwrap();
board
    .write_target_svg(Path::new("target.svg"), &SvgTargetOptions::default())
    .unwrap();
board
    .write_target_png(
        Path::new("target.png"),
        &PngTargetOptions {
            dpi: 300.0,
            ..PngTargetOptions::default()
        },
    )
    .unwrap();
```

`render_target_svg` returns the SVG as a string, and `render_target_png` returns an in-memory grayscale `image::GrayImage` when you want to avoid file I/O. `write_target_png` embeds the requested DPI as PNG print metadata.

## Equivalent Command-Line Workflows

The Rust API above is equivalent to these command-line paths when you want the
same artifact set from the terminal instead of from application code.

Rust CLI:

```bash
cargo run -p ringgrid-cli -- gen-target \
  --out_dir tools/out/target_faststart \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

Python script from a repository checkout:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip maturin
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
./.venv/bin/python tools/gen_target.py \
  --out_dir tools/out/target_faststart \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

All three paths generate:

- `tools/out/target_faststart/board_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`

Use the generated JSON in detection:

```rust,no_run
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("tools/out/target_faststart/board_spec.json")).unwrap();
let detector = Detector::new(board);
```

Complete step-by-step target generation docs (Rust API, Rust CLI, Python script, and helper tools):
- https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html

## Simple Detection

```rust,no_run
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let detector = Detector::new(board);
let result = detector.detect(&image);

for marker in &result.detected_markers {
    if let Some(id) = marker.id {
        println!("Marker {id} at ({:.1}, {:.1})", marker.center[0], marker.center[1]);
    }
}
```

`result` is a `DetectionResult`. It contains the final marker list plus image
size, frame metadata, optional homography/RANSAC stats, and optional
`self_undistort` output. For the serialized JSON shape and field meanings, see:
- https://vitalyvorobyev.github.io/ringgrid/book/output-format.html

With a marker diameter hint for better scale tuning:

```rust,no_run
# use ringgrid::{BoardLayout, Detector};
# use std::path::Path;
# let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let detector = Detector::with_marker_diameter_hint(board, 32.0);
```

## Proposal-Only Diagnostics

When you want to inspect candidate centers before fit/decode, use the proposal
API directly:

```rust,no_run
use ringgrid::{BoardLayout, Detector, ProposalConfig};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let detector = Detector::with_marker_diameter_hint(board, 32.0);
let proposals = detector.propose(&image);
let diagnostics = detector.propose_with_heatmap(&image);

let result = ringgrid::proposal::find_ellipse_centers_with_heatmap(
    &image,
    &ProposalConfig {
        r_min: 4.0,
        r_max: 18.0,
        min_distance: 12.0,
        ..ProposalConfig::default()
    },
);

println!("{}", proposals.len());
println!("{:?}", diagnostics.image_size);
println!("{:?}", result.heatmap.len());
```

`ProposalResult.heatmap` is the post-Gaussian-smoothed vote accumulator
used for thresholding and NMS. Proposal tutorial and Python plotting workflow:
- https://vitalyvorobyev.github.io/ringgrid/book/detection-modes/proposal-diagnostics.html

## Adaptive Scale Detection

For scenes with large marker size variation, use adaptive multi-scale methods:

```rust,no_run
# use ringgrid::{BoardLayout, Detector, ScaleTiers};
# use std::path::Path;
# let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
# let detector = Detector::new(board);
# let image = image::open("photo.png").unwrap().to_luma8();
let result = detector.detect_adaptive(&image);
let result = detector.detect_adaptive_with_hint(&image, Some(32.0));
let result = detector.detect_multiscale(&image, &ScaleTiers::four_tier_wide());
```

Which method to choose:

| Situation | Recommended call | Why |
|---|---|---|
| Marker size unknown / mixed near-far scene | `detect_adaptive` | Probe + auto tier selection |
| Approximate diameter is known | `detect_adaptive_with_hint(..., Some(d))` | Skip probe and use focused two-tier bracket around `d` |
| Exact tier policy required (reproducible benchmarks) | `detect_multiscale(..., tiers)` | Full explicit control over tier set |
| Size range is tight and throughput matters | `detect` | Single-pass and fastest |

Inspect adaptive tiers before detecting:

```rust,no_run
# use ringgrid::{BoardLayout, Detector};
# use std::path::Path;
# let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
# let detector = Detector::new(board);
# let image = image::open("photo.png").unwrap().to_luma8();
let tiers = detector.adaptive_tiers(&image, Some(32.0));
let result = detector.detect_multiscale(&image, &tiers);
```

Adaptive scale guide:
- https://vitalyvorobyev.github.io/ringgrid/book/detection-modes/adaptive-scale.html

## Detection with Camera Model

When camera intrinsics and distortion coefficients are known, use `detect_with_mapper`
for distortion-aware detection via a two-pass pipeline:

```rust,no_run
use ringgrid::{
    BoardLayout, CameraIntrinsics, CameraModel, Detector, RadialTangentialDistortion,
};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();
let (w, h) = image.dimensions();

let camera = CameraModel {
    intrinsics: CameraIntrinsics {
        fx: 900.0, fy: 900.0,
        cx: w as f64 * 0.5, cy: h as f64 * 0.5,
    },
    distortion: RadialTangentialDistortion {
        k1: -0.15, k2: 0.05, p1: 0.001, p2: -0.001, k3: 0.0,
    },
};

let detector = Detector::new(board);
let result = detector.detect_with_mapper(&image, &camera);

for marker in &result.detected_markers {
    // center is always image-space
    println!("Image: ({:.1}, {:.1})", marker.center[0], marker.center[1]);
    // center_mapped is working-frame (undistorted)
    if let Some(mapped) = marker.center_mapped {
        println!("Working: ({:.1}, {:.1})", mapped[0], mapped[1]);
    }
}
```

## Self-Undistort (No Calibration Required)

When camera calibration is unavailable, ringgrid can estimate a single-parameter
division-model distortion correction from the detected markers:

```rust,no_run
use ringgrid::{BoardLayout, DetectConfig, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let mut cfg = DetectConfig::from_target(board);
cfg.self_undistort.enable = true;

let detector = Detector::with_config(cfg);
let result = detector.detect(&image);

if let Some(su) = &result.self_undistort {
    println!("Lambda: {:.3e}, applied: {}", su.model.lambda, su.applied);
}
```

## Custom PixelMapper

Implement the `PixelMapper` trait to plug in any distortion model:

```rust
use ringgrid::PixelMapper;

struct Identity;

impl PixelMapper for Identity {
    fn image_to_working_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        Some(p)
    }
    fn working_to_image_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        Some(p)
    }
}
```

Then use it with `detector.detect_with_mapper(&image, &mapper)`.

## Coordinate Frames

- `DetectedMarker.center` — always raw image pixel coordinates
- `DetectedMarker.center_mapped` — working-frame (undistorted) coordinates when a mapper is active
- `DetectedMarker.board_xy_mm` — board-space marker coordinates in millimeters for valid decoded IDs
- `DetectionResult.center_frame` / `homography_frame` — explicit frame metadata

## Documentation

- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) — comprehensive mdbook covering marker design, detection pipeline, mathematical foundations, and configuration
- [API Reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/) — rustdoc for all public types

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
