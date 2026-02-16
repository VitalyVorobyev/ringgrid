# ringgrid

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice.
It detects markers, decodes IDs, estimates homography, and exports structured JSON.

## Visual Overview

Target print example:

![Ringgrid target print](docs/assets/target_print.png)

Detection overlay example:

![Detection overlay example](docs/assets/det_overlay_0002.png)

## How It Works

ringgrid uses dual concentric ring markers with a 16-sector binary code band between the inner and outer rings. Each marker encodes a unique ID from a 893-codeword codebook with minimum cyclic Hamming distance of 5.

**Why rings?** Circles project to ellipses under perspective, and ellipse boundaries can be localized to subpixel precision via gradient-based edge sampling and direct algebraic fitting (Fitzgibbon's method). The dual-ring design enables projective center correction — recovering the true projected center from the inner/outer conic pencil, which corrects the systematic bias inherent in ellipse-fit centers. This yields significantly better accuracy than corner-based targets (checkerboards, ArUco) at oblique viewing angles.

**Detection pipeline** (13 stages): Scharr gradient voting proposes candidate centers → radial profile estimates outer radius → RANSAC ellipse fit → 16-sector code decode → inner ellipse fit → deduplication → projective center correction → RANSAC homography filter → H-guided refinement → completion at missing IDs → final homography refit.

## Documentation

- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) — comprehensive mdbook covering marker design, the 13-stage detection pipeline, mathematical foundations (Fitzgibbon fitting, DLT homography, RANSAC, projective center recovery, division distortion model), configuration, and usage
- [API Reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/) — rustdoc for all public types

## Quick Start

### 1. Build

```bash
cargo build --release
```

### 2. Install Python tooling deps (for synth/eval/viz)

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install numpy matplotlib
```

### 3. Generate one synthetic sample

```bash
./.venv/bin/python tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 1 --blur_px 1.0
```

### 4. Run detection

```bash
target/release/ringgrid detect \
  --image tools/out/synth_001/img_0000.png \
  --out tools/out/synth_001/det_0000.json

# Optional scale prior tuning:
#   --marker-diameter-min 18 --marker-diameter-max 48
# Legacy fixed-size mode:
#   --marker-diameter 32
```

### 5. Score against ground truth

```bash
./.venv/bin/python tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

### 6. Render detection overlay

```bash
tools/run_synth_viz.sh tools/out/synth_001 0
```

`tools/run_synth_viz.sh` auto-uses `.venv/bin/python` when present.

## Public API (v1)

All detection goes through `Detector` methods. No public free functions.

Stable surface (library users):
- `Detector` — entry point
- `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod` — configuration
- `DetectionResult`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, `RansacStats` — results
- `BoardLayout`, `BoardMarker`, `MarkerSpec` — geometry
- `CameraModel`, `CameraIntrinsics`, `PixelMapper` — camera/distortion
- `Ellipse` — conic geometry

Design constraints in v1:
- Target JSON is mandatory for high-level detector construction:
  `BoardLayout::from_json_file(...)` + `Detector::new(...)`.
- `Detector::detect(...)` is config-driven:
  `self_undistort.enable=false` runs single-pass, `true` runs self-undistort orchestration.
- `Detector::detect_with_mapper(...)` always uses the provided mapper and ignores
  `self_undistort` config.
- `DetectedMarker.center` is always image-space; mapper-frame center is optional in
  `DetectedMarker.center_mapped`.
- `DetectionResult` includes explicit `center_frame` / `homography_frame` metadata.
- Low-level math/pipeline modules are internal.
- Example target JSON: `crates/ringgrid/examples/target.json`.

Minimal usage:

```rust
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("crates/ringgrid/examples/target.json"))?;
let detector = Detector::new(board);
// let result = detector.detect(&gray_image);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Project Layout

```text
crates/
  ringgrid/
    src/
      lib.rs       # re-exports only (public API surface)
      api.rs       # Detector facade
      pipeline/    # stage orchestration: single_pass, multi_pass, run, fit_decode, finalize
      detector/    # per-marker primitives: proposal, fit, decode, dedup, filter, center correction, completion
      ring/        # ring-level sampling: edge, radius, projective center
      marker/      # codebook, decode, marker spec
      homography/  # DLT + RANSAC, refit utilities
      conic/       # ellipse types, fitting, RANSAC, eigenvalue solver
      pixelmap/    # camera models, PixelMapper trait, self-undistort
    examples/      # concise library usage examples
  ringgrid-cli/    # CLI binary: ringgrid
tools/
  gen_synth.py         # synthetic dataset generator
  run_synth_eval.py    # generate -> detect -> score
  score_detect.py      # scoring utility
  viz_detect.py        # DetectionResult overlay rendering
docs/
  assets/
  module_structure.md
  pipeline_analysis.md  # detailed pipeline architecture analysis
```

Module ownership and dependency direction are documented in `docs/module_structure.md`.
Detailed pipeline architecture is in `docs/pipeline_analysis.md`.

## Examples

Run crate examples from workspace root:

```bash
cargo run -p ringgrid --example basic_detect -- \
  crates/ringgrid/examples/target.json tools/out/synth_001/img_0000.png

cargo run -p ringgrid --example detect_with_camera -- \
  crates/ringgrid/examples/target.json tools/out/synth_001/img_0000.png

cargo run -p ringgrid --example detect_with_self_undistort -- \
  crates/ringgrid/examples/target.json tools/out/synth_001/img_0000.png
```

## Detection Modes

### Rust API

**Simple detection** (no distortion correction):

```rust,no_run
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let detector = Detector::new(board);
let result = detector.detect(&image::open("photo.png").unwrap().to_luma8());
```

**With camera model** (two-pass distortion-aware pipeline):

```rust,no_run
use ringgrid::{BoardLayout, CameraIntrinsics, CameraModel, Detector, RadialTangentialDistortion};
use std::path::Path;

let camera = CameraModel {
    intrinsics: CameraIntrinsics { fx: 900.0, fy: 900.0, cx: 640.0, cy: 480.0 },
    distortion: RadialTangentialDistortion { k1: -0.15, k2: 0.05, p1: 0.0, p2: 0.0, k3: 0.0 },
};
let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let detector = Detector::new(board);
let result = detector.detect_with_mapper(&image::open("photo.png").unwrap().to_luma8(), &camera);
```

**Self-undistort** (estimates distortion from markers, no calibration needed):

```rust,no_run
use ringgrid::{BoardLayout, DetectConfig, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
let mut cfg = DetectConfig::from_target(board);
cfg.self_undistort.enable = true;
let detector = Detector::with_config(cfg);
let result = detector.detect(&image::open("photo.png").unwrap().to_luma8());
```

### CLI Flags

- `--circle-refine-method none|projective-center` (default: projective-center)
- `--no-global-filter` — disable global homography filtering
- `--no-complete` — disable H-guided completion
- `--marker-diameter-min <px>` / `--marker-diameter-max <px>` — scale search range
- `--self-undistort` — enable self-undistort estimation (mutually exclusive with `--cam-*`)
- `--cam-fx/fy/cx/cy` + `--cam-k1/k2/k3/p1/p2` — external camera model

## Distortion Correction

ringgrid supports three distortion correction modes:

| Mode | Requires | Method |
|---|---|---|
| None | — | Single-pass in image coordinates |
| External camera | Calibrated intrinsics + distortion | Two-pass with `CameraModel` implementing `PixelMapper` |
| Self-undistort | Nothing | Estimates 1-parameter division model from detected markers |

Camera model and self-undistort are mutually exclusive. See the [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) for details.

## Metrics (Synthetic Scoring)

`tools/score_detect.py` reports several geometric metrics; the three key ones are:

- `center_error`: TP-only error between predicted `marker.center` and GT center in the selected frame (`--center-gt-key image|working|auto`).
- `homography_self_error`: homography self-consistency error (`project(H_est, board_xy_mm)` vs predicted marker center) in the selected evaluation frame.
- `homography_error_vs_gt`: absolute error between estimated `H` and GT projection (`project(H_est, board_xy_mm)` vs GT center in selected frame via `--homography-gt-key`).

Interpretation:

- Lower is better for all three.
- `homography_self_error` can be lower than `center_error`, because it measures consistency of `H` with detected centers, not absolute GT center error.
- For cross-run comparisons, evaluate all metrics in distorted image space.
- Use `--center-gt-key image --homography-gt-key image`.
- Prefer `--pred-center-frame auto --pred-homography-frame auto` so scorer uses detector-emitted `center_frame` / `homography_frame` metadata.

Distortion-aware eval example:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 3 \
  --blur_px 0.8 \
  --out_dir tools/out/r4_distortion_eval \
  --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --pass_camera_to_detector
```

Self-undistort eval example:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 3 \
  --blur_px 0.8 \
  --out_dir tools/out/r4_self_undistort_eval \
  --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --self_undistort
```

Self-undistort implementation notes:
- Two-pass flow: pass-1 detection, estimate division-model `lambda`, pass-2 with mapper if accepted.
- Primary objective is homography self-consistency in mapped working space (when enough decoded IDs exist).
- Fallback objective is robust conic-consistency (inner/outer Sampson residuals).
- Apply gates require meaningful improvement, non-trivial `|lambda|`, and reject boundary solutions.
- Default lambda search range is `[-8e-7, 8e-7]`.

## Performance Snapshots (Synthetic)

### Distortion Benchmark (Projective-Center, 3 Images)

Source:
- `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json`

Example distorted sample used in this benchmark:

![Distortion benchmark sample](docs/assets/distortion_benchmark_sample.png)

Run command:

```bash
./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0 --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --corrections none external self_undistort \
  --modes projective_center
```

Benchmark script defaults to `cargo run` (source-of-truth build). Use
`--use-prebuilt-binary` only when you intentionally want to benchmark an existing binary artifact.

Image-space metric snapshot:

| Correction | Precision | Recall | Center mean (px) | H self mean/p95 (px) | H vs GT mean/p95 (px) |
|---|---:|---:|---:|---:|---:|
| `none` | 1.000 | 0.974 | 0.232 | 1.030 / 2.961 | 1.345 / 3.611 |
| `external` | 1.000 | 1.000 | 0.078 | 0.075 / 0.146 | 0.020 / 0.029 |
| `self_undistort` | 1.000 | 1.000 | 0.078 | 0.210 / 0.426 | 0.193 / 0.408 |

Notes:
- Scripts now score in distorted image space for all three correction variants.
- For `self_undistort`, scoring frame is selected per-image from `self_undistort.applied` in detection JSON.
- On this synthetic distortion setup, self-undistort is much better than no correction, but still less accurate than external calibration parameters.

### Reference Benchmark (Clean, 3 Images)

Source: `tools/out/reference_benchmark_post_pipeline/summary.json`

Run command:

```bash
./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/reference_benchmark_post_pipeline \
  --n_images 3 \
  --blur_px 0.8 \
  --noise_sigma 0.0 \
  --marker_diameter 32.0 \
  --modes none projective_center
```

| Mode | Center mean (px) | H self mean/p95 (px) | H vs GT mean/p95 (px) |
|---|---:|---:|---:|
| `none` | 0.072 | 0.065 / 0.132 | 0.033 / 0.049 |
| `projective-center` | 0.054 | 0.051 / 0.098 | 0.019 / 0.030 |

### Regression Batch (10 images)

Source: `tools/out/regress_r2_batch/det/aggregate.json`

This set is intentionally harder (`blur_px=3.0`), and markers are visibly weak/blurred.

Example image from this stress set:

![Regression blur=3.0 sample](docs/assets/regression_blur3_sample.png)

Run command:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 10 \
  --blur_px 3.0 \
  --out_dir tools/out/regress_r2_batch \
  --marker_diameter 32.0
```

Snapshot:

| Metric | Value |
|---|---:|
| Images | 10 |
| Avg precision | 1.000 |
| Avg recall | 0.949 |
| Avg TP / image | 192.6 |
| Avg FP / image | 0.0 |
| Avg center error (px) | 0.278 |
| Avg H vs GT error (px) | 0.147 |
| Avg H self error (px) | 0.235 |

## Regenerate Embedded Assets

```bash
python3 tools/gen_codebook.py \
  --n 893 --seed 1 \
  --out_json tools/codebook.json \
  --out_rs crates/ringgrid/src/codebook.rs

# Board target is runtime JSON (`ringgrid.target.v3`), no generated Rust module.
python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 \
  --rows 15 --long_row_cols 14 \
  --board_mm 200.0 \
  --json_out tools/board/board_spec.json
```

Then rebuild:

```bash
cargo build --release
```

## Development Checks

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
