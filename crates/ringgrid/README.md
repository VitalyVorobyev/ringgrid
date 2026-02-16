# ringgrid

Pure-Rust detector for dense coded ring calibration targets on a hex lattice.
Detects markers with subpixel edge precision, decodes 16-sector binary IDs from
a 893-codeword codebook, fits ellipses via Fitzgibbon's direct method with
RANSAC, corrects projective center bias, and estimates a board-to-image
homography. No OpenCV dependency.

## Key Features

- **Subpixel edge detection** — gradient-based radial sampling produces edge points fed to a direct ellipse fit, yielding subpixel-accurate marker localization
- **Projective center correction** — recovers the true projected center from inner/outer conic pencil geometry, correcting the systematic bias of ellipse-fit centers
- **893 unique IDs** — 16-sector binary codebook with minimum cyclic Hamming distance of 5, enabling reliable identification under noise and partial occlusion
- **Distortion-aware** — supports external camera models (Brown-Conrady) via the `PixelMapper` trait, or blind single-parameter self-undistort estimation
- **Pure Rust** — no C/C++ dependencies, no OpenCV bindings

## Installation

```toml
[dependencies]
ringgrid = "0.1"
```

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

With a marker diameter hint for better scale tuning:

```rust,no_run
# use ringgrid::{BoardLayout, Detector};
# use std::path::Path;
# let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let detector = Detector::with_marker_diameter_hint(board, 32.0);
```

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
- `DetectionResult.center_frame` / `homography_frame` — explicit frame metadata

## Documentation

- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) — comprehensive mdbook covering marker design, detection pipeline, mathematical foundations, and configuration
- [API Reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/) — rustdoc for all public types

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
