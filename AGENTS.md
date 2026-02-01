# ringgrid — Agent Notes

## Project overview

`ringgrid` is a Rust workspace for detecting dense, all-coded ring calibration markers arranged on a hex (triangular) lattice. The core detector is pure Rust (no OpenCV bindings) and is paired with small Python utilities for generating synthetic datasets, running evaluation loops, scoring, and visualization.

The Rust CLI (`ringgrid`) loads an image, runs the end-to-end ring detection pipeline (proposal → edge sampling → ellipse fit → decode → dedup → optional homography RANSAC filtering + refinement), and writes results as JSON (`ringgrid_core::DetectionResult`).

## Local setup

### Rust

- Rust toolchain (edition 2021) and `cargo`

Common commands:
```bash
cargo build
cargo test
```

### Python tooling

- Python **3.10+** (some scripts use `X | None` type syntax)
- Required packages:
  - `numpy` (synthetic generation)
  - `matplotlib` (visualization)

Example venv setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install numpy matplotlib
```

## Main dev loops

### 1) Generate embedded codebook + board spec

These generators produce:
- JSON references for tooling under `tools/`
- Embedded Rust constants under `crates/ringgrid-core/src/` (rebuild after regenerating)

```bash
# Codebook: tools/codebook.json + crates/ringgrid-core/src/codebook.rs
python3 tools/gen_codebook.py \
  --n 893 --seed 1 \
  --out_json tools/codebook.json \
  --out_rs crates/ringgrid-core/src/codebook.rs

# Board spec: tools/board/board_spec.json + crates/ringgrid-core/src/board_spec.rs
python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 --board_mm 200.0 \
  --json_out tools/board/board_spec.json \
  --rust_out crates/ringgrid-core/src/board_spec.rs

# Rebuild to pick up embedded constants
cargo build
```

Sanity checks:
```bash
cargo run -- codebook-info
cargo run -- board-info
```

### 2) Generate a synthetic dataset

```bash
python3 tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 3 --blur_px 1.0
```

Outputs in `tools/out/synth_001/`:
- `img_0000.png`, …
- `gt_0000.json`, … (ground truth)
- `board_spec.json` (board layout used for the run)

### 3) Run the detector

```bash
cargo run -- detect \
  --image tools/out/synth_001/img_0000.png \
  --out tools/out/synth_001/det_0000.json \
  --debug tools/out/synth_001/debug_0000.json \
  --marker-diameter 32.0
```

Notes:
- Logging goes to stderr via `tracing`; use `RUST_LOG=debug` (or `info`, `trace`, etc.).
- `--debug` currently writes the same JSON payload as `--out` (both are `DetectionResult`).

### 4) Score detections

```bash
python3 tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

### One-command end-to-end eval

This runs: generate → detect → score and writes an aggregate summary.
```bash
python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_run
```

## Debugging workflow

- Reproduce a failing sample:
  - Use `tools/run_synth_eval.py --skip_gen --out_dir <same_dir>` to re-run detection/scoring on an existing dataset.
  - Or run `ringgrid detect` directly on `img_XXXX.png` while tweaking `--marker-diameter`, `--ransac-thresh-px`, `--no-global-filter`, `--no-refine`.
- Inspect artifacts:
  - Ground truth: `tools/out/<run>/synth/gt_XXXX.json`
  - Predictions: `tools/out/<run>/det/det_XXXX.json` (and `debug_XXXX.json`)
  - Scores: `tools/out/<run>/det/score_XXXX.json` and `aggregate.json`
- Visual sanity check (GT overlay):
  ```bash
  python3 tools/viz_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --gt tools/out/synth_001/gt_0000.json \
    --out tools/out/synth_001/viz_0000.png
  ```

## Conventions for new milestones / prompts

- Keep algorithms in `crates/ringgrid-core/`; keep CLI and file I/O in `crates/ringgrid-cli/`.
- Prefer `serde` structs for external JSON outputs (see `ringgrid_core::DetectionResult`).
- Avoid introducing OpenCV bindings (Rust or Python).
- Add/extend unit tests for math-heavy primitives (`conic`, `homography`, `codec`, etc.).
- When tracking planned work in code, follow the existing `TODO Milestone N:` pattern.

## CI / checks

No CI configuration is checked into this repo. Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets
cargo test
```

## Do / Don’t

- Do regenerate embedded artifacts via `tools/gen_codebook.py` and `tools/gen_board_spec.py` (don’t hand-edit generated Rust).
- Do keep the generator(s) + embedded Rust + decoder/scorer consistent if formats change.
- Don’t change the codebook/board ID conventions without updating:
  - the generator scripts
  - the embedded Rust modules
  - decoding + global filter logic
  - tests and scoring scripts
