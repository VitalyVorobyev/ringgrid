# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice. It detects markers, decodes 16-sector IDs, estimates homography, and exports structured JSON. No OpenCV bindings — all image processing is in Rust. The goal is to provide a polished external Rust library API.

## Workspace Structure

Cargo workspace (edition 2024, MSRV 1.88) with two crate members plus excluded bindings:
- `crates/ringgrid/` — detection algorithms, math primitives, result types
- `crates/ringgrid-cli/` — CLI binary (`ringgrid`) with clap-based argument parsing
- `crates/ringgrid-py/` — PyO3 Python bindings (excluded from workspace, built via maturin)
- `crates/ringgrid-wasm/` — WebAssembly bindings (excluded from workspace, built via wasm-pack)
- `tools/` — Python utilities for synthetic data generation, evaluation, scoring, and visualization
- `py/` — Python scripts for rtv3d benchmarking and overlay visualization

Key external dependency: `radsym` crate provides the proposal-stage center detection (fused RSD with Scharr gradient). Python package management uses `uv` (virtualenv at `.venv`).

## Engineering principles & critical review

Hold every change to a high design bar, and actively keep this workspace
from drifting back into undisciplined, copy-paste "agentic slop." Apply
these principles by default — not only when explicitly asked:

- **SOLID** — one responsibility per type/module; extend behavior through
  the detector / refiner / orientation traits, not by editing parallel
  match arms; depend on trait abstractions (`DenseDetector`, the refiner
  traits), not concretions.
- **DRY / single source of truth** — one canonical definition per concept.
  Lower config to core params in exactly one place; never let a
  threshold's or parameter's meaning be duplicated or diverge across
  crates.
- **KISS & YAGNI** — choose the simplest design that meets the actual
  requirement; do not add config knobs, enum variants, or abstraction
  layers for hypothetical future needs.
- **Make illegal states unrepresentable** — push invariants into the type
  system (enum-with-payload over a discriminator + parallel fields;
  `Option` / newtypes over magic sentinels) so misuse fails to compile.
- **Least astonishment & orthogonality** — APIs do what their names say;
  keep independent concerns independent (orientation is a cross-cutting
  stage, not a sub-mode of one detector).
- **Minimal, honest public surface** — expose only what callers need; keep
  diagnostics and internals out of the stable API (see *Public surface
  hygiene*).

**Be critical of every proposal — including the user's.** Treat a request
as the start of a design discussion, not an instruction to implement
verbatim. Before coding, review it against the principles above and the
existing architecture; if it would introduce duplication, leak internals,
add a dominated alternative, widen the public surface needlessly, or
otherwise degrade the design, **say so and offer the cleaner alternative**
rather than building it as-stated. When the better design needs a breaking
change and the crate is pre-1.0, prefer the better design (see *Decisive
cleanup*). Every change should leave the workspace's design and style
better than it found them.

## Module Layout

```
crates/ringgrid/src/
├── lib.rs              # Re-exports only (public API surface)
├── api.rs              # Detector facade + propose_with_* free functions
├── board_layout.rs     # Board geometry (hex lattice layout, JSON loader)
├── target_generation.rs # SVG/PNG target rendering
├── proposal/           # Center proposal generation (delegates to radsym)
│   ├── mod.rs          # Adapter: radsym fused RSD → ringgrid Proposal types
│   ├── config.rs       # ProposalConfig (translated to radsym RsdConfig)
│   └── tests.rs        # Behavioral tests
├── detector/           # Per-marker detection primitives
│   ├── config.rs       # DetectConfig and all sub-configs
│   ├── outer_fit/      # RANSAC ellipse fitting (Fitzgibbon direct LS)
│   │   ├── mod.rs, sampling.rs, scoring.rs, solver.rs
│   ├── inner_fit.rs    # Inner ring ellipse fit
│   ├── inner_as_outer_recovery.rs # Recover inner ring from outer-fit failures
│   ├── marker_build.rs # DetectedMarker, FitMetrics structs + builder
│   ├── dedup.rs        # Spatial + ID-based deduplication
│   ├── global_filter.rs# RANSAC homography filter
│   ├── completion.rs   # Conservative fits at missing H-projected IDs
│   ├── center_correction.rs # Projective center application
│   └── id_correction/  # BFS hex-neighbor ID consensus
│       ├── mod.rs, engine.rs, workspace.rs, types.rs, index.rs
│       ├── local.rs, homography.rs, vote.rs, math.rs
│       ├── bootstrap.rs, consistency.rs, cleanup.rs, diagnostics.rs
├── ring/               # Ring-level sampling & estimation primitives
│   ├── outer_estimate.rs    # Radius hypotheses via radial profile peaks
│   ├── inner_estimate.rs    # Inner ring scale estimation
│   ├── radial_profile.rs    # Radial intensity profile sampling
│   ├── radial_estimator.rs  # Radial profile estimator
│   ├── edge_sample.rs       # Edge point sampling (distortion-aware)
│   └── projective_center.rs # Unbiased center from inner+outer conics
├── marker/             # Code decoding & marker specification
│   ├── decode.rs       # 16-sector code sampling → codebook match
│   ├── codec.rs        # Codebook matching logic
│   ├── codebook.rs     # Generated 893-codeword table (don't hand-edit)
│   └── marker_spec.rs  # MarkerSpec type
├── pipeline/           # Detection pipeline orchestration
│   ├── mod.rs          # DetectionResult struct, module glue, re-exports
│   ├── run.rs          # Top-level orchestrator + two-pass/self-undistort/multiscale
│   ├── fit_decode.rs   # Proposals → fit → decode → dedup
│   ├── finalize.rs     # Center correction → global filter → completion → final H
│   ├── scale_probe.rs  # Ring angular-variance sweep → dominant radius estimates
│   ├── prelude.rs      # Common imports for pipeline modules
│   └── result.rs       # DetectionResult type
├── homography/         # Homography estimation & utilities
│   ├── core.rs         # DLT + RANSAC, RansacStats
│   ├── utils.rs        # Refit, reprojection error, matrix conversion
│   └── correspondence.rs # Point correspondence matching
├── conic/              # Ellipse/conic math
│   ├── types.rs        # Ellipse type
│   ├── fit.rs          # Direct ellipse fitting (Fitzgibbon)
│   ├── ransac.rs       # RANSAC ellipse fitting
│   └── eigen.rs        # Generalized eigenvalue solver
├── pixelmap/           # Camera, distortion, pixel mapping
│   ├── cameramodel.rs  # CameraModel, CameraIntrinsics
│   ├── distortion.rs   # RadialTangentialDistortion, DivisionModel
│   └── self_undistort/ # Self-undistort estimation
│       ├── mod.rs, config.rs, result.rs
│       ├── estimator.rs, objective.rs, optimizer.rs, policy.rs
```

## Detection Pipeline Architecture

### Single-pass stages (in order)

1. **Proposal** (`proposal/mod.rs`) — radsym fused RSD (Scharr gradient + magnitude voting + NMS) → candidate centers
2. **Outer Estimate** (`ring/outer_estimate.rs`) — radius hypotheses via radial profile peaks
3. **Outer Fit** (`detector/outer_fit/`) — RANSAC ellipse fitting (Fitzgibbon direct LS)
4. **Decode** (`marker/decode.rs`) — 16-sector code sampling → codebook match (893 codewords)
5. **Inner Fit** (`detector/inner_fit.rs`) — inner ring ellipse fit
6. **Dedup** (`detector/dedup.rs`) — spatial + ID-based deduplication
7. **Projective Center** (`detector/center_correction.rs`) — correct fit-decode marker centers (once per marker)
8. **ID Correction** (`detector/id_correction/`) — BFS hex-neighbor consensus
9. **Global Filter** (`detector/global_filter.rs`) — RANSAC homography if ≥4 decoded markers (uses corrected centers)
10. **Completion** (`detector/completion.rs`) — conservative fits at missing H-projected IDs + projective center for new markers
11. **Final H Refit** — refit homography from all corrected centers

Pipeline orchestration: stages 1–6 in `pipeline/fit_decode.rs`, stages 7–11 in `pipeline/finalize.rs`, top-level sequencing in `pipeline/run.rs`.

### Multi-scale / adaptive pipeline

For multi-scale detection (`detect_multiscale`, `detect_adaptive`, `detect_adaptive_with_hint`), the pipeline is split at the finalize boundary:

- **`finalize_premerge`** — runs stages 7–8 (projective center + ID correction) only; returns `Vec<DetectedMarker>` without global filter, completion, or H refit. Called once per tier.
- **`merge_multiscale_markers`** (`detector/dedup.rs`) — size-consistency-aware NMS across all tier outputs. Prefers markers whose outer radius matches the neighborhood median (k=6 hex-lattice neighbors); confidence breaks ties.
- **`finalize_postmerge`** — runs stages 9–11 (global filter + completion + final H refit) exactly once on the merged pool.

Scale probe (`pipeline/scale_probe.rs`): ring angular-variance sweep at top-K gradient proposals over 20 geometric radius candidates (4–110 px). High variance at a radius indicates the code band (alternating bright/dark sectors). Code-band midpoint ≈ 0.8× outer ring radius; results feed `ScaleTiers::from_detected_radii`.

Default `MarkerScalePrior` is **[14, 66] px**.

## Public API

All detection goes through the `Detector` struct (`api.rs`). Standalone proposal functions are also available:
- `find_ellipse_centers`, `find_ellipse_centers_with_heatmap` — low-level proposal generation
- `propose_with_marker_scale`, `propose_with_marker_diameter` (+ heatmap variants) — board-aware proposal helpers

Key public types:
- `Detector` — entry point
- `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod` — configuration
- `ScaleTier`, `ScaleTiers` — multi-scale tier configuration (presets: `four_tier_wide`, `two_tier_standard`, `single`)
- `DetectionResult`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, `RansacStats` — results
- `BoardLayout`, `BoardMarker`, `MarkerSpec` — geometry
- `CameraModel`, `CameraIntrinsics`, `PixelMapper` — camera/distortion
- `Ellipse` — conic geometry
- `Proposal`, `ProposalConfig`, `ProposalResult` — proposal types

## Build & Development Commands

```bash
# Build
cargo build --release

# Tests
cargo test --workspace

# Lint & format
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Python bindings (uses uv for package management)
# VIRTUAL_ENV must point at .venv so maturin installs into it, not system Python.
cd crates/ringgrid-py && VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --release && cd ../..

# WASM bindings
wasm-pack build crates/ringgrid-wasm --target web --release

# Run detector
cargo run -- detect --image <path> --out <path> --marker-diameter 32.0

# End-to-end synthetic eval (generate → detect → score)
.venv/bin/python tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_run

# Score a single detection
.venv/bin/python tools/score_detect.py --gt <gt.json> --pred <det.json> --gate 8.0 --out <score.json>

# Regenerate embedded codebook/board constants (don't hand-edit these)
.venv/bin/python tools/gen_codebook.py --n 893 --seed 1 --out_json tools/codebook.json --out_rs crates/ringgrid/src/codebook.rs
.venv/bin/python tools/gen_board_spec.py --pitch_mm 8.0 --rows 15 --long_row_cols 14 --board_mm 200.0 --json_out tools/board/board_spec.json
cargo build  # rebuild after regenerating
```

## Key Math Modules

- `conic/` — ellipse/conic types (`types.rs`), direct fitting (`fit.rs`), generalized eigenvalue solver (`eigen.rs`), RANSAC (`ransac.rs`)
- `homography/` — DLT with Hartley normalization + RANSAC (`core.rs`), refit & reprojection utilities (`utils.rs`)
- `ring/projective_center.rs` — unbiased center recovery from inner+outer conics

## Center Correction Strategies

Single-choice selector: `none` | `projective_center`

CLI: `--circle-refine-method {none,projective-center}`

Center correction is applied once per marker: before global filter for fit-decode markers, and after completion for newly completed markers only. Logic is in `pipeline/finalize.rs`.

## Camera / Distortion Support

Optional radial-tangential distortion model (`pixelmap/`). When camera intrinsics are provided:
- Two-pass detection: pass-1 without mapper, pass-2 with mapper using pass-1 seeds
- Final centers mapped back to image space; ellipse/homography in working frame
- `PixelMapper` trait allows custom distortion adapters

Self-undistort mode estimates a division-model distortion correction from detected ellipse edge points, then re-runs detection with the estimated mapper.

## Versioning

The workspace version is defined once in `Cargo.toml` under `[workspace.package]`.
`ringgrid` and `ringgrid-cli` inherit it via `version.workspace = true`.

When bumping the version, update **four** locations:
1. `Cargo.toml` — `[workspace.package] version` (single source of truth for workspace crates)
2. `crates/ringgrid-py/Cargo.toml` — `version` field + `ringgrid` dependency version
3. `crates/ringgrid-py/pyproject.toml` — `project.version` field
4. `crates/ringgrid-wasm/Cargo.toml` — `version` field + `ringgrid` dependency version

CI workflows (`.github/workflows/publish-crates.yml`, `release-pypi.yml`) verify
version consistency between the git tag and these files using `tomllib`. The CI
scripts resolve `version.workspace = true` by falling back to the workspace root.

## Feature Flags (ringgrid crate)

- `std` (default) — enables file I/O (`from_json_file`, `write_json_file`, `write_target_svg`, `write_target_png`) and the `png` dependency. Disable for WASM targets.
- WASM crate uses `default-features = false` to exclude `std`.

## Decision-Making Rules

- **Never draw conclusions without reproducible evidence.** Every claim about correctness, performance, or behavior must be backed by a test, benchmark, or concrete reproducing example. "I believe this works" is not acceptable — run the code, show the output.
- **When in doubt, stop and ask.** If the task is ambiguous, the expected behavior is unclear, or there are multiple reasonable interpretations, do not guess. Ask the user for clarification before proceeding.
- **Do not speculate about root causes.** When debugging, reproduce the issue first, then form a hypothesis, then verify it with a test. Do not propose fixes based on reading code alone — run the failing case and confirm the diagnosis.

## Conventions

- Algorithms go in `ringgrid`; CLI/file I/O in `ringgrid-cli`
- External JSON uses `serde` structs (see `DetectionResult` in `pipeline/mod.rs`)
- Never introduce OpenCV bindings
- `codebook.rs` is generated; board target is runtime JSON (`tools/board/board_spec.json`) — regenerate via Python scripts, never hand-edit generated Rust
- Logging via `tracing` crate; control with `RUST_LOG=debug|info|trace`
- `lib.rs` is purely re-exports; type definitions live at their construction sites
- Keep one source of truth for shared configs/defaults: if a config is used by multiple stages, define it once and reuse it directly.
- Avoid mirrored structs (`*Params` vs `*Config`) that carry the same fields/defaults; consolidate into one type unless domains are genuinely different.
- Avoid duplicated thresholds/gating logic across modules; centralize constants and semantics.
- If a translation layer is unavoidable, document why it exists and ensure only one side owns defaults.
- Python package management uses `uv`. Virtual environment at `.venv`. Use `.venv/bin/python` and `.venv/bin/uv pip install` for all Python operations.

## CI / checks

GitHub Actions workflows in `.github/workflows/`:
- `ci.yml` — fmt, clippy, test, WASM build smoke test on push/PR
- `publish-crates.yml` — publish ringgrid + ringgrid-cli to crates.io on tag
- `release-pypi.yml` — build and publish Python wheels to PyPI on tag
- `release-npm.yml` — build and publish WASM npm package on tag
- `release.yml` — create GitHub release
- `publish-docs.yml` — build and deploy mdBook docs
- `audit.yml` — dependency audit

Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Regression Gate

The `/regression-gate` skill runs the full regression suite (invoke after significant changes):

1. Build release + Python bindings
2. Run 3 synthetic benchmarks + 1 real-world benchmark (rtv3d, local-only):
   - `bash tools/run_reference_benchmark.sh`
   - `bash tools/run_distortion_benchmark.sh`
   - `bash tools/run_blur3_benchmark.sh`
   - `.venv/bin/python tools/run_rtv3d_eval.py` (skipped if `data/rtv3d` absent)
3. Validate against `tools/ci/regression_baseline.json`
4. Run `tools/ci/maintainability_guardrails.py`

Baselines: `tools/ci/regression_baseline.json` (synthetic + rtv3d thresholds), `tools/ci/maintainability_baseline.json` (function size, dead code, doc coverage).
