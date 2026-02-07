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
python3 -m pip install -U pip
python3 -m pip install numpy matplotlib
```

### 3. Generate one synthetic sample

```bash
python3 tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 1 --blur_px 1.0
```

### 4. Run detection

```bash
target/release/ringgrid detect \
  --image tools/out/synth_001/img_0000.png \
  --out tools/out/synth_001/det_0000.json \
  --debug-json tools/out/synth_001/debug_0000.json \
  --marker-diameter 32.0
```

### 5. Score against ground truth

```bash
python3 tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

### 6. Render detection debug overlay

```bash
tools/run_synth_viz.sh tools/out/synth_001 0
```

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

## Performance Snapshots (Synthetic)

### Regression Batch (10 images)

Source: `tools/out/regress_r2_batch/det/aggregate.json`

Run command:

```bash
python3 tools/run_synth_eval.py \
  --n 10 \
  --skip_gen \
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
| Avg center error (px) | 0.254 |
| Avg reprojection error (px) | 0.199 |

### Refinement Strategy Comparison (`synth_002`, image 0002)

Source: `tools/out/synth_002/_cmp_recheck/score_*.json`

All modes below reached 203 TP / 0 FP on this sample.

| Mode | Center mean (px) | Legacy center mean (px) | Homography mean / p95 (px) |
|---|---:|---:|---:|
| `none` | 0.102 | 0.102 | 0.076 / 0.157 |
| `projective-center` | 0.078 | 0.097 | 0.060 / 0.128 |
| `nl-board + lm` | 0.086 | 0.102 | 0.040 / 0.103 |
| `nl-board + irls` | 0.099 | 0.102 | 0.071 / 0.140 |

Notes:

- `projective-center` currently matches `tools/out/synth_002/det_0002.json` exactly.
- `none` and `projective-center` are not equivalent: all 203 centers moved on this sample.

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
