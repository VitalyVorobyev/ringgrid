# ringgrid

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice.
It detects markers, decodes IDs, estimates homography, and exports structured JSON.

## Visual Overview

Target print example:

![Ringgrid target print](docs/assets/target_print.png)

Detection overlay example (`tools/out/synth_002/img_0002.png`):

![Detection overlay example](docs/assets/det_overlay_0002.png)

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
  --out tools/out/synth_001/det_0000.json \
  --marker-diameter 32.0
```

### 5. Score against ground truth

```bash
./.venv/bin/python tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

### 6. Render detection debug overlay

```bash
tools/run_synth_viz.sh tools/out/synth_001 0
```

`tools/run_synth_viz.sh` auto-uses `.venv/bin/python` when present.

## Project Layout

```text
crates/
  ringgrid-core/   # detection algorithms and result structures
  ringgrid-cli/    # CLI binary: ringgrid
tools/
  gen_synth.py         # synthetic dataset generator
  run_synth_eval.py    # generate -> detect -> score
  score_detect.py      # scoring utility
  viz_detect_debug.py  # debug overlay rendering
docs/
  ARCHITECTURE.md
```

## Detection Modes

Core refinement selector:

- `--circle-refine-method none`
- `--circle-refine-method projective-center` (default)
- `--circle-refine-method nl-board`

When `nl-board` is used, select solver via:

- `--nl-solver lm`
- `--nl-solver irls`

Other commonly used toggles:

- `--no-global-filter`
- `--no-refine`
- `--no-complete`
- `--debug-json <path>`
- `--debug-store-points`

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
- Set predicted frame explicitly via `--pred-center-frame image|working` and `--pred-homography-frame image|working`.
- Benchmark scripts in this repository set these frame flags explicitly (no `auto`), so reported numbers are frame-consistent and reproducible.

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

## Performance Snapshots (Synthetic)

### Distortion Benchmark (Projective-Center, 3 Images)

Source:
- `tools/out/r4_benchmark_distorted_nomapper_pc/summary.json`
- `tools/out/r4_benchmark_distorted_mapper_pc/summary.json`

Example distorted sample used in this benchmark:

![Distortion benchmark sample](docs/assets/distortion_benchmark_sample.png)

Run commands:

```bash
./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/r4_benchmark_distorted_nomapper_pc \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0 --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --modes projective_center

./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/r4_benchmark_distorted_mapper_pc \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0 --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --pass-camera-to-detector \
  --modes projective_center
```

Image-space metric snapshot:

| Pipeline | Precision | Recall | Center mean (px) | H self mean/p95 (px) | H vs GT mean/p95 (px) |
|---|---:|---:|---:|---:|---:|
| No mapper | 1.000 | 0.975 | 0.237 | 1.053 / 2.903 | 1.359 / 3.341 |
| Mapper enabled | 1.000 | 1.000 | 0.079 | 0.077 / 0.155 | 0.022 / 0.031 |

### Reference Benchmark (Clean, 3 Images)

Source: `tools/out/reference_benchmark/summary.json`

Run command:

```bash
./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/reference_benchmark \
  --n_images 3 \
  --blur_px 0.8 \
  --noise_sigma 0.0 \
  --marker_diameter 32.0
```

| Mode | Center mean (px) | H self mean/p95 (px) | H vs GT mean/p95 (px) |
|---|---:|---:|---:|
| `none` | 0.072 | 0.065 / 0.132 | 0.032 / 0.046 |
| `projective-center` | 0.054 | 0.051 / 0.101 | 0.019 / 0.027 |
| `nl-board + lm` | 0.047 | 0.027 / 0.089 | 0.038 / 0.052 |
| `nl-board + irls` | 0.069 | 0.061 / 0.122 | 0.034 / 0.048 |

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
| Avg center error (px) | 0.280 |
| Avg H vs GT error (px) | 0.150 |
| Avg H self error (px) | 0.230 |

## CI Workflows

Draft CI is configured under `.github/workflows/`:

- `ci.yml`
  - rust formatting/lint/tests on Ubuntu
  - synthetic smoke eval (`tools/run_synth_eval.py --n 1`)
  - cross-platform build/test on macOS + Windows
- `audit.yml`
  - weekly `cargo audit`
- `publish-docs.yml`
  - publish Rustdoc to GitHub Pages

## Regenerate Embedded Assets

```bash
python3 tools/gen_codebook.py \
  --n 893 --seed 1 \
  --out_json tools/codebook.json \
  --out_rs crates/ringgrid-core/src/codebook.rs

python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 --board_mm 200.0 \
  --json_out tools/board/board_spec.json \
  --rust_out crates/ringgrid-core/src/board_spec.rs
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
