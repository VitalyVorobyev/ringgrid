# ringgrid — Agent Notes

## Project overview

`ringgrid` is a Rust workspace for detecting dense, all-coded ring calibration markers arranged on a hex (triangular) lattice. The core detector is pure Rust (no OpenCV bindings) and is paired with Python utilities for synthetic generation, evaluation, scoring, and visualization.

The Rust CLI (`ringgrid`) loads an image, runs the end-to-end ring detection pipeline (proposal -> local fit/decode -> dedup -> optional homography filtering/refinement/completion), and writes results as JSON (`ringgrid::DetectionResult`).

## Local setup

### Rust

- Rust toolchain (edition 2021) and `cargo`

Common commands:
```bash
cargo build
cargo test
```

### Python tooling

- Python **3.10+**
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

### 1) Generate codebook + board spec

These generators produce:
- JSON references for tooling under `tools/`
- Embedded Rust constants only for codebook under `crates/ringgrid/src/`

```bash
# Codebook: tools/codebook.json + crates/ringgrid/src/codebook.rs
python3 tools/gen_codebook.py \
  --n 893 --seed 1 \
  --out_json tools/codebook.json \
  --out_rs crates/ringgrid/src/codebook.rs

# Board spec (runtime JSON): tools/board/board_spec.json
python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 \
  --rows 15 --long_row_cols 14 \
  --board_mm 200.0 \
  --json_out tools/board/board_spec.json

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
  --marker-diameter 32.0
```

Notes:
- Logging goes to stderr via `tracing`; use `RUST_LOG=debug` (or `info`, `trace`, etc.).
- Refinement runs when a homography is available; disable with `--no-refine`.

### 4) Score detections

```bash
python3 tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

### One-command end-to-end eval

This runs: generate -> detect -> score and writes an aggregate summary.
```bash
python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_run
```

## Public API (implemented)

`ringgrid` exposes a `Detector` struct as the primary entry point:

```rust
use ringgrid::{BoardLayout, Detector};

let board = BoardLayout::default();
let detector = Detector::new(board);                // default board
let detector = Detector::with_config(config);       // full control

let result = detector.detect(&image);
let result = detector.detect_with_mapper(&image, &mapper);

detector.config_mut().completion.enable = false;   // post-construction tuning
detector.config_mut().self_undistort.enable = true; // makes detect() run self-undistort flow
```

Notes:
- `detect()` is config-driven: self-undistort runs only when `config.self_undistort.enable=true`.
- `detect_with_mapper()` ignores `config.self_undistort` and always uses the provided mapper.

## Debugging workflow

- Reproduce a failing sample:
  - Use `tools/run_synth_eval.py --skip_gen --out_dir <same_dir>` to re-run detection/scoring on an existing dataset.
  - Or run `ringgrid detect` directly on `img_XXXX.png` while tweaking `--marker-diameter`, `--ransac-thresh-px`, `--no-global-filter`, `--no-refine`.
- Inspect artifacts:
  - Ground truth: `tools/out/<run>/synth/gt_XXXX.json`
  - Predictions: `tools/out/<run>/det/det_XXXX.json`
  - Scores: `tools/out/<run>/det/score_XXXX.json` and `aggregate.json`
- Visual sanity check (GT overlay):
  ```bash
  python3 tools/viz_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --gt tools/out/synth_001/gt_0000.json \
    --out tools/out/synth_001/viz_0000.png
  ```

- Visual sanity check (detection overlay):
  ```bash
  python3 tools/viz_detect.py \
    --image tools/out/synth_001/img_0000.png \
    --det_json tools/out/synth_001/det_0000.json \
    --out tools/out/synth_001/det_overlay_0000.png
  ```

## Conventions for new milestones / prompts

- Keep algorithms in `crates/ringgrid/`; keep CLI and file I/O in `crates/ringgrid-cli/`.
- Prefer `serde` structs for external JSON outputs (see `ringgrid::DetectionResult`).
- Avoid introducing OpenCV bindings (Rust or Python).
- Add/extend unit tests for math-heavy primitives (`conic`, `homography`, `codec`, etc.).

## CI / checks

Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Do / Don’t

- Do regenerate generated assets via `tools/gen_codebook.py` and `tools/gen_board_spec.py` (do not hand-edit generated Rust codebook).
- Do keep generator(s) + runtime target JSON schema + decoder/scorer consistent if formats change.
- Do keep configuration structs and default values as a single source of truth. If multiple stages use the same knobs, use one shared type instead of `*Params` + `*Config` mirrors.
- Do remove old structs/conversions immediately when consolidating APIs, and update all call sites to the shared type.
- Don’t change the codebook/board ID conventions without updating:
  - generator scripts
  - embedded Rust modules
  - decoding + global filter logic
  - tests and scoring scripts
- Don’t introduce logic duplication (especially thresholds/gates/defaults) across modules without a documented necessity.
- Don’t add adapter/conversion layers between near-identical structs unless they represent different semantic domains.

## Open decisions for maintainers

- Which advanced parameters should remain user-exposed in v1 vs internal-only?
- Should `min_marker_separation_px` be a direct user knob or derived from target geometry + marker scale?
- Should camera calibration remain optional in v1 API, or required for precision-oriented profiles?
