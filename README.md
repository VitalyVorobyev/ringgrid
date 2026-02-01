# ringgrid

Robust ring-grid calibration target detector in **Rust** (no OpenCV).

`ringgrid` detects dense, all-coded, two-edge ring calibration markers arranged on a hex (triangular) lattice. The detector decodes marker IDs via an embedded 16-bit codebook and can apply a board-aware global filter using homography RANSAC (with optional refinement).

## Key features (current)

- Pure Rust detector (`image` / `imageproc` / `nalgebra`), no OpenCV bindings.
- Candidate proposals via gradient-voting radial symmetry.
- Radial edge sampling to recover inner/outer ring edges.
- Robust ellipse fitting (direct fit + optional RANSAC fallback).
- 16-sector decoding with cyclic rotation matching (tries all 16 rotations) and inverted-polarity fallback.
- Deduplication (by center proximity, then by best confidence per ID).
- Optional global homography RANSAC filter using the embedded board spec, plus a 1-iteration “refine-with-H” pass.
- Python tools for synthetic data generation, scoring, and visualization.

## Repository structure

```
crates/
  ringgrid-core/        # Core detection algorithms + JSON output structs
  ringgrid-cli/         # CLI binary: ringgrid
tools/
  gen_codebook.py       # Generate tools/codebook.json + embedded Rust codebook.rs
  gen_board_spec.py     # Generate tools/board/board_spec.json + embedded Rust board_spec.rs
  gen_synth.py          # Synthetic dataset generator (img_XXXX.png + gt_XXXX.json)
  run_synth_eval.py     # End-to-end: generate → detect → score
  score_detect.py       # Score ringgrid detect JSON vs synthetic GT
  viz_debug.py          # Visualize GT overlay for synthetic samples
docs/
  ARCHITECTURE.md       # Background + notes (includes planned milestones)
```

## Quickstart

### 1) Rust build

```bash
cargo build --release
```

Useful during development:
```bash
RUST_LOG=info cargo run -- --help
```

### 2) Python env (for tools)

Python **3.10+** and:
```bash
python3 -m pip install numpy matplotlib
```

### 3) Generate a tiny synthetic dataset

```bash
python3 tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 1 --blur_px 1.0
```

### 4) Run the detector on one image

```bash
target/release/ringgrid detect \
  --image tools/out/synth_001/img_0000.png \
  --out tools/out/synth_001/det_0000.json \
  --debug tools/out/synth_001/debug_0000.json \
  --marker-diameter 32.0
```

### 5) Score vs ground truth

```bash
python3 tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

### End-to-end (generate → detect → score)

```bash
python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_run
```

## CLI reference (current)

`ringgrid` subcommands:

- `ringgrid detect`
  - Required: `--image <path> --out <path>`
  - Optional: `--debug <path>`
  - Tuning: `--marker-diameter <px>`
  - Global filter: `--ransac-thresh-px <px>`, `--ransac-iters <n>`, `--no-global-filter`
  - Refinement: `--no-refine`
- `ringgrid codebook-info` — print embedded codebook stats.
- `ringgrid board-info` — print embedded board spec summary.
- `ringgrid decode-test --word 0xABCD` — decode a raw 16-bit word against the embedded codebook.

## Output artifacts

### Detection JSON (`ringgrid detect`)

`ringgrid detect` writes a `ringgrid_core::DetectionResult` JSON object:

- Top-level:
  - `detected_markers`: list of detections
  - `image_size`: `[width, height]`
  - `homography`: optional `[[f64;3];3]` (row-major), present when fitted
  - `ransac`: optional RANSAC stats, present when fitted
- Per marker (`DetectedMarker`):
  - `id`: optional (absent if decoding rejected)
  - `confidence`: `0..1`
  - `center`: `[x, y]` in pixels
  - `ellipse_outer`: optional `{ center_xy, semi_axes, angle }`
  - `ellipse_inner`: optional `{ center_xy, semi_axes, angle }`
  - `fit`: edge/fit quality metrics
  - `decode`: optional `{ observed_word, best_id, best_rotation, best_dist, margin, decode_confidence }`

Note: `--debug` currently writes the same JSON payload as `--out`.

### Synthetic ground truth (`tools/gen_synth.py`)

For each image `img_XXXX.png`, `gen_synth.py` writes `gt_XXXX.json` with (high level):

- Top-level: `image_file`, `image_size`, `seed`, `blur_px`, `homography`, `board_mm`, `pitch_mm`, `outer_radius_mm`, `inner_radius_mm`, `markers`, …
- Per marker: `id`, `q`, `r`, `board_xy_mm`, `true_image_center`, `outer_ellipse`, `inner_ellipse`, `visible`

### Scoring output (`tools/score_detect.py`)

`score_detect.py` writes a JSON report containing:

- Counts: `n_gt`, `n_pred`, `n_tp`, `n_fp`, `n_miss`, `n_pred_with_id`, `n_pred_no_id`
- Metrics: `precision`, `recall`
- Center error stats (TP only): `center_error.mean|median|p95|max` (when any TP exist)
- Decode histogram: `decode_dist_histogram`
- Diagnostics: `missed_ids_top20`, `false_positives_top20`
- Optional: `ransac_stats` (copied from the detection JSON when present)

### End-to-end eval outputs (`tools/run_synth_eval.py`)

Under `--out_dir tools/out/eval_run`:

- `tools/out/eval_run/synth/` — `img_XXXX.png`, `gt_XXXX.json`
- `tools/out/eval_run/det/` — `det_XXXX.json`, `debug_XXXX.json`, `score_XXXX.json`, `aggregate.json`

## Regenerating embedded assets

The detector uses an **embedded** codebook and board spec (Rust constants in `crates/ringgrid-core/src/`).

```bash
python3 tools/gen_codebook.py \
  --n 893 --seed 1 \
  --out_json tools/codebook.json \
  --out_rs crates/ringgrid-core/src/codebook.rs

python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 --board_mm 200.0 \
  --json_out tools/board/board_spec.json \
  --rust_out crates/ringgrid-core/src/board_spec.rs

cargo build --release
```

## Roadmap (from TODOs in repo)

- Milestone 2: implement and integrate `ringgrid_core::preprocess`, `ringgrid_core::edges`, and `ringgrid_core::lattice`.
- Milestone 3: implement `ringgrid_core::refine` (LM refinement) and add Criterion benchmarks.
- Cleanup: reconcile the legacy `ringgrid_core::codec::decode_marker_id` TODO stub with the active decoder in `ringgrid_core::ring::decode`.

## License

No `LICENSE` file is present in this repository. If you intend this to be open source, add an explicit license.

## Contributing

- Keep algorithmic work in `crates/ringgrid-core/` and CLI wiring in `crates/ringgrid-cli/`.
- Keep JSON output changes backwards-aware (update tooling in `tools/` as needed).
- Run:
  ```bash
  cargo fmt --all
  cargo clippy --all-targets
  cargo test
  ```

## TODO / needs clarification

- Physical/printable board assets (SVG/PDF) and exact marker drawing spec are not tracked in this repo (only codebook + center layout + synthetic renderer).
- `docs/ARCHITECTURE.md` includes planned milestone notes and may not match the current end-to-end pipeline 1:1.
