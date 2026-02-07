# ringgrid — Agent Notes

## Project overview

`ringgrid` is a Rust workspace for detecting dense, all-coded ring calibration markers arranged on a hex (triangular) lattice. The core detector is pure Rust (no OpenCV bindings) and is paired with Python utilities for synthetic generation, evaluation, scoring, and visualization.

The Rust CLI (`ringgrid`) loads an image, runs the end-to-end ring detection pipeline (proposal -> local fit/decode -> dedup -> optional homography filtering/refinement/completion), and writes results as JSON (`ringgrid_core::DetectionResult`).

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
  --debug-json tools/out/synth_001/debug_0000.json \
  --marker-diameter 32.0
```

Notes:
- Logging goes to stderr via `tracing`; use `RUST_LOG=debug` (or `info`, `trace`, etc.).
- `--debug-json` writes `ringgrid.debug.v1` (versioned debug dump).
- `--debug` is deprecated (alias for `--debug-json`).
- NL refinement (board-plane circle fit) runs when a homography is available; disable with `--no-nl-refine`.

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

## Architecture review snapshot (2026-02-07)

### 1) Mixed-responsibility hotspots

- `crates/ringgrid-core/src/ring/detect/debug_pipeline.rs` (~727 LOC): still mixes stage execution with debug-schema mapping and serialization shaping.
- `crates/ringgrid-core/src/ring/detect/completion.rs` (~634 LOC): completion logic + gates + debug mapping in one module.
- `crates/ringgrid-core/src/refine/pipeline.rs` (~524 LOC): per-marker refine flow still long, although now isolated from API/types.
- `crates/ringgrid-core/src/conic.rs` (~1180 LOC): core model types, conversion, direct fit, generalized eigen solver, cubic root solver, and RANSAC in one file.
- `crates/ringgrid-cli/src/main.rs` (~400 LOC): CLI parsing and parameter-scaling policy are tightly coupled (`run_detect` uses many arguments).

### 2) Redundant logic / data paths

- Core dedup/global-filter/refine-H duplication has been reduced by delegating to shared module paths.
- Marker assembly is still repeated across regular and debug flows (`non_debug/stage_fit_decode.rs`, `debug_pipeline.rs`, `completion.rs`, `refine_h.rs`).
- `inner_estimate.rs` and `outer_estimate.rs` duplicate radial aggregation/peak machinery (`aggregate`, `per_theta_peak_r`).
- Radial outer-edge probing still exists in both `ring/detect/*` and `refine/*` with slightly different gates.
- `ring/edge_sample.rs::sample_edges` is currently unused by production pipeline (used only by its unit tests).

### 3) Refactoring plan (ordered)

1. Completed: split `ring/detect.rs` into focused modules while keeping behavior identical (synthetic aggregate parity checks kept exact).
2. Completed: split `refine.rs` into focused modules (`refine/math.rs`, `refine/sampling.rs`, `refine/solver.rs`, `refine/pipeline.rs`).
3. Next: introduce shared builder helpers for marker construction (`FitMetrics`, `DecodeMetrics`, `EllipseParams`) and reuse across regular/debug flows.
4. Next: consolidate radial profile utilities into one reusable module (`ring/radial_profile.rs`) used by inner/outer/refine.
5. Next: reduce CLI argument plumbing by introducing a small config adapter:
   - `CliDetectArgs -> DetectPreset + DetectOverrides -> DetectConfig`.

## Missing feature plan: ellipse center correction via vanishing line pole

Current behavior uses outer ellipse center directly; this is projectively biased. Planned first implementation:

Policy decision (v1): run center correction for all accepted local fits whenever a valid homography is available (not only decoded/RANSAC-inlier subset).

1. Build conic matrix `C` from fitted ellipse coefficients (`ellipse_to_conic`).
2. Obtain board-plane vanishing line `l` from fitted homography `H`:
   - `l ~ H^{-T} * [0, 0, 1]^T`.
3. Compute center candidate as pole of `l` w.r.t. `C`:
   - `p ~ C^{-1} * l` (or `adj(C) * l` for robustness).
4. Dehomogenize `p` to image coordinates.
5. Compute this for outer and inner ellipse when both available; fuse with quality weighting and gating.
6. If no valid homography or unstable conic inversion, fall back to current center estimate.

Acceptance checks:
- New synthetic perspective stress test where corrected center error decreases vs baseline.
- Keep decode and RANSAC inlier counts non-regressing on existing eval runs.

## Camera calibration / distortion plan

Goal: allow undistortion of edge samples for higher precision.

1. Add `camera` module with explicit calibration structs for radial-tangential distortion.
2. Make camera parameters optional in `DetectConfig` and output metadata.
3. Add distortion-aware sampling helper used by radial sampling and refinement.
4. Start with point-wise undistortion/remap during sampling (avoid full-image remap blur for precision path).
5. Add synthetic-distortion eval mode in `tools/gen_synth.py` and score scripts.

Scope decision (v1): radial-tangential only.

## Public API target shape

Design target for `ringgrid-core`:

- Keep a simple entrypoint for common usage.
- Expose expert controls in nested structs; avoid flat parameter sprawl.

Proposed API direction:

- `Detector` object holding immutable target/codebook/camera context.
- `target` comes from runtime JSON and is mandatory in public API v1.
- `Detector::detect(&GrayImage) -> DetectionResult`
- `Detector::detect_with_debug(&GrayImage, DebugOptions) -> (DetectionResult, DebugDumpV1)`
- `DetectionOptions` with two tiers:
  - Stable user-facing fields (small set).
  - `advanced` optional sub-structs for proposal/edge/decode/refine internals.

Initial stable parameter surface (v1, provisional):

- `marker_diameter_px`
- `min_marker_separation_px` (maps to dedup/min-distance semantics)
- `enable_global_filter`
- `ransac_reproj_thresh_px`
- `enable_completion`
- `enable_nl_refine`
- `decode_min_confidence`
- `camera` (optional radial-tangential calibration)

Initial advanced surface (v1, provisional):

- Proposal internals (`r_min/r_max`, gradient threshold, NMS)
- Outer/inner estimator internals (search windows, polarity, theta/radial sample counts)
- Decode sampling internals (band ratio, samples per sector, radial rings)
- Completion gates (ROI radius, reproj gate, arc coverage, fit confidence, attempts)
- Homography internals (`max_iters`, seed, min inliers)
- NL refine internals (`max_iters`, huber delta, min_points, reject shift, H-refit loop)

## Target specification format (planned)

Move to a versioned runtime JSON schema (while keeping generated Rust embedding support):

- `schema`: version string (for example `ringgrid.target.v1`)
- `name`, `units`
- `board`: either explicit marker list (`id`, `xy_mm`, optional `q/r`) or parametric hex layout descriptor
- `marker_geometry`: outer/inner/code-band radii, sector count
- `coding`: codebook metadata + codewords (or external reference)
- optional: tolerances / expected visibility masks

Keep `tools/gen_board_spec.py` and embedded constants as one backend of this schema, not a separate format.
Public API decision (v1): runtime target JSON is mandatory (no embedded-board fallback at API boundary).

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

- Visual sanity check (detection debug overlay):
  ```bash
  python3 tools/viz_detect_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --debug_json tools/out/synth_001/debug_0000.json \
    --out tools/out/synth_001/det_overlay_0000.png
  ```

## Conventions for new milestones / prompts

- Keep algorithms in `crates/ringgrid-core/`; keep CLI and file I/O in `crates/ringgrid-cli/`.
- Prefer `serde` structs for external JSON outputs (see `ringgrid_core::DetectionResult`).
- Avoid introducing OpenCV bindings (Rust or Python).
- Add/extend unit tests for math-heavy primitives (`conic`, `homography`, `codec`, etc.).

## CI / checks

No CI configuration is checked into this repo. Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Do / Don’t

- Do regenerate embedded artifacts via `tools/gen_codebook.py` and `tools/gen_board_spec.py` (do not hand-edit generated Rust).
- Do keep generator(s) + embedded Rust + decoder/scorer consistent if formats change.
- Do keep debug schema evolution versioned (`ringgrid.debug.v1`, `v2`, ...).
- Don’t change the codebook/board ID conventions without updating:
  - generator scripts
  - embedded Rust modules
  - decoding + global filter logic
  - tests and scoring scripts

## Open decisions for maintainers

- Which advanced parameters should remain user-exposed in v1 vs internal-only?
- Should `min_marker_separation_px` be a direct user knob or derived from target geometry + marker scale?
- Should camera calibration remain optional in v1 API, or required for precision-oriented profiles?
