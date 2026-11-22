# ringgrid

Pure-Rust detector for dense ring calibration targets on hex or rectangular
lattices. Detects markers with subpixel edge precision, decodes 16-sector binary
IDs from a shipped baseline 893-codeword profile (with an opt-in extended profile
available for larger ID spaces), fits ellipses via Fitzgibbon's direct method with
RANSAC, corrects projective center bias, and estimates a board-to-image
homography. No OpenCV dependency.

## Key Features

- **Subpixel edge detection** — gradient-based radial sampling produces edge points fed to a direct ellipse fit, yielding subpixel-accurate marker localization
- **Projective center correction** — recovers the true projected center from inner/outer conic pencil geometry, correcting the systematic bias of ellipse-fit centers
- **Photometric center refinement (opt-in)** — fits each marker's pixels with a model of the blurred printed pattern; on the synthetic blur benchmark the center RMS is 0.008–0.040 px against 0.449–0.736 px for the projective center (see [below](#photometric-center-refinement))
- **Consistency-first ID correction** — verifies decoded IDs against local hex-lattice structure, clears contradictory IDs, and recovers safe missing IDs before global filtering
- **Stable baseline IDs plus opt-in extension** — shipped `base` profile keeps 893 stable IDs at minimum cyclic Hamming distance 2; opt-in `extended` grows capacity to 2180 IDs with a weaker minimum distance of 1 without introducing new polarity ambiguity beyond the shipped baseline
- **Distortion-aware** — supports external camera models (Brown-Conrady) via the `PixelMapper` trait, or blind single-parameter self-undistort estimation
- **Compositional target model** — `TargetLayout` composes hex/rect lattices, coded (16-sector) or plain rings, and optional origin-dot fiducials; targets load and save as `ringgrid.target.v6` JSON (v4 and v5 files are no longer loaded; see the [migration notes](https://github.com/VitalyVorobyev/ringgrid/tree/main/docs/migrations/0.15-to-0.16.md))
- **Pure Rust** — no C/C++ dependencies, no OpenCV bindings

## Pipeline Stages

Named stage order:
proposal -> local fit/decode -> dedup -> projective center -> `id_correction` -> optional global filter -> optional completion -> final homography refit.

## Installation

```toml
[dependencies]
ringgrid = "0.18"
```

## Rust Target Generation

The library can generate canonical target JSON plus printable SVG/PNG directly:

```rust,no_run
use ringgrid::{TargetLayout, TargetRenderOptions};
use std::path::Path;

// One constructor per target-matrix row — `coded_hex`, `coded_rect`,
// `plain_hex`, `plain_rect` — each taking plain scalars and deriving a
// deterministic geometry-based name (`with_name` overrides it). Plain targets
// take an `OriginDots` argument: `Auto` places the origin-dot triad for you.
//
//   TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)
//
// `TargetLayout::new` gives full control over lattice, ring geometry, coding,
// and explicit origin fiducials.
let target = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).unwrap();

// One options value drives every format: page (size, orientation, margin),
// scale bar, and PNG resolution.
let options = TargetRenderOptions::default().with_png_dpi(300.0);

target.write_json_file(Path::new("target.json")).unwrap();
target.write_target_svg(Path::new("target.svg"), &options).unwrap();
target.write_target_png(Path::new("target.png"), &options).unwrap();
target.write_target_dxf(Path::new("target.dxf")).unwrap();
```

`render_target_svg` returns the SVG as a string, and `render_target_png` returns an in-memory grayscale `image::GrayImage` when you want to avoid file I/O. `write_target_png` embeds the requested DPI as PNG print metadata.

## Equivalent Command-Line Workflow

The Rust API above is equivalent to the published `ringgrid` CLI when you want
the same artifact set from the terminal instead of from application code:

```bash
cargo install ringgrid --features cli
ringgrid example --name hex_coded --out hex_coded.toml   # start from a recipe
ringgrid gen hex_coded.toml --out ./out/target           # SVG + PNG + DXF + JSON
```

Both paths write a v6 `target_spec.json`, the only target schema detection
loads. See
the [CLI Guide](https://vitalyvorobyev.github.io/ringgrid/book/cli-guide.html)
for the recipe format and every flag.

Use the generated JSON in detection:

```rust,no_run
use ringgrid::{Detector, TargetLayout};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
let detector = Detector::new(target);
```

Complete step-by-step target generation docs (Rust API, Rust CLI, Python script, and helper tools):
- https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html

## Screen targets

To show a target on a display (an e-reader, a monitor) instead of printing it,
fit it to the screen. `render_fitted_screen_target` scales a template target to
the requested drawn marker diameter, fits the grid with the most markers, and
renders a native-resolution 8-bit grayscale PNG with area-coverage edges and
unsnapped (fractional-pixel) marker centers:

```rust,no_run
use ringgrid::{ScreenTargetOptions, TargetLayout};

// width_px, height_px, pixel_pitch_mm (25.4 / ppi), marker_diameter_mm
let options = ScreenTargetOptions::new(825, 1200, 25.4 / 150.0, 10.752);
let screen = TargetLayout::default_hex().render_fitted_screen_target(&options)?;
std::fs::write("target_screen.png", &screen.png_bytes)?; // show 1:1
std::fs::write("target_spec.json", &screen.json_text)?; // detect against this
println!("{} markers", screen.metadata.marker_count);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The CLI equivalent is `ringgrid gen-screen`. In the project's simulation of a
166 dpi e-reader, a binary PNG at display resolution costs 5.0 µm of
marker-to-marker photometric center error, against 0.6 µm with 16 gray levels
and 0.5 µm on print. The guide covers why, how the fit works, and how to choose
the marker diameter:
- https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html#screen-targets

## Simple Detection

```rust,no_run
use ringgrid::{Detector, TargetLayout};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let detector = Detector::new(target);
let result = detector.detect(&image).unwrap();

for marker in &result.detected_markers {
    if let Some(id) = marker.id {
        println!("Marker {id} at ({:.1}, {:.1})", marker.center[0], marker.center[1]);
    }
}
```

`result` is a slim `DetectionResult`. It contains the final marker list plus
image size, frame metadata, optional homography, and optional `self_undistort`
output. Per-marker fit/decode metrics, edge points, and homography RANSAC stats
are an opt-in diagnostics channel — call `detector.detect_with_diagnostics(&image)`
to also get a `DetectionDiagnostics`. For the serialized JSON shape and field
meanings, see:
- https://vitalyvorobyev.github.io/ringgrid/book/output-format.html

With a marker diameter hint for better scale tuning:

```rust,no_run
# use ringgrid::{Detector, TargetLayout};
# use std::path::Path;
# let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
let detector = Detector::with_marker_diameter_hint(target, 32.0);
```

## Proposal-Only Diagnostics

When you want to inspect candidate centers before fit/decode, use the proposal
API directly:

```rust,no_run
use ringgrid::{Detector, ProposalConfig, TargetLayout};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let detector = Detector::with_marker_diameter_hint(target, 32.0);
let proposals = detector.propose(&image);
let diagnostics = detector.propose_with_heatmap(&image);

let result = ringgrid::find_ellipse_centers_with_heatmap(
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
# use ringgrid::{Detector, ScaleTiers, TargetLayout};
# use std::path::Path;
# let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
# let detector = Detector::new(target);
# let image = image::open("photo.png").unwrap().to_luma8();
let result = detector.detect_adaptive(&image).unwrap();
let result = detector.detect_adaptive_with_hint(&image, Some(32.0)).unwrap();
let result = detector.detect_multiscale(&image, &ScaleTiers::four_tier_wide()).unwrap();
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
# use ringgrid::{Detector, TargetLayout};
# use std::path::Path;
# let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
# let detector = Detector::new(target);
# let image = image::open("photo.png").unwrap().to_luma8();
let tiers = detector.adaptive_tiers(&image, Some(32.0));
let result = detector.detect_multiscale(&image, &tiers).unwrap();
```

Adaptive scale guide:
- https://vitalyvorobyev.github.io/ringgrid/book/detection-modes/adaptive-scale.html

## Photometric Center Refinement

`CircleRefinementMethod::Photometric` runs after projective-center correction
and fits each marker's pixels with a model of the blurred printed pattern
(Gaussian PSF, affine illumination, a local board-to-image map). It costs about
6–15 ms per marker fit in release, so it is opt-in; `ProjectiveCenter` stays
the default.

```rust,no_run
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};

let mut config = DetectConfig::from_target(TargetLayout::default_hex());
config.circle_refinement = CircleRefinementMethod::Photometric;
let detector = Detector::with_config(config);
```

Synthetic blur benchmark (720×540, default hex board with 0.84 mm strokes,
≈33 px markers, tilts 0/20/30°, an illumination gradient, a glare blob,
contrast 80 grey levels, σ = 2 noise). Center
RMS / p95 over decoded markers, px:

| blur σ | projective center | photometric |
|---:|---:|---:|
| 0.8 | 0.614 / 0.924 | 0.008 / 0.014 |
| 1.5 | 0.715 / 1.095 | 0.013 / 0.022 |
| 2.5 | 0.600 / 1.118 | 0.028 / 0.052 |
| 3.0 | 0.736 / 1.387 | 0.040 / 0.073 |

Use it when the centers feed a calibration, and whenever the optics blur the
markers. Full tables (including contrast 40 and real-image results):
- https://vitalyvorobyev.github.io/ringgrid/book/detection-pipeline/photometric-center.html

## Detection with Camera Model

When camera intrinsics and distortion coefficients are known, use `detect_with_mapper`
for distortion-aware detection via a two-pass pipeline:

```rust,no_run
use ringgrid::{
    CameraIntrinsics, CameraModel, Detector, RadialTangentialDistortion, TargetLayout,
};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
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

let detector = Detector::new(target);
let result = detector.detect_with_mapper(&image, &camera).unwrap();

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
use ringgrid::{DetectConfig, Detector, TargetLayout};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let mut cfg = DetectConfig::from_target(target);
cfg.self_undistort.enable = true;

let detector = Detector::with_config(cfg);
let result = detector.detect(&image).unwrap();

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
- `DetectedMarker.grid_coord` — lattice cell (`[q, r]` axial for hex, `[col, row]` for rect); both are centered on cell `(0, 0)`, and it is the only marker key on plain targets
- `DetectedMarker.board_xy_mm` — board-space marker coordinates in millimeters for valid decoded IDs
- `DetectionResult.center_frame` / `homography_frame` / `board_frame` — explicit frame metadata

## Documentation

- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) — comprehensive mdbook covering marker design, detection pipeline, mathematical foundations, and configuration
- [API Reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/) — rustdoc for all public types

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
