# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice. It detects markers, decodes 16-sector IDs, estimates homography, and exports structured JSON. No OpenCV bindings — all image processing is in Rust. The goal is to provide a polished external Rust library API (charuco-style).

## Workspace Structure

Cargo workspace with two crates:
- `crates/ringgrid-core/` — detection algorithms, math primitives, result types
- `crates/ringgrid-cli/` — CLI binary (`ringgrid`) with clap-based argument parsing
- `tools/` — Python utilities for synthetic data generation, evaluation, scoring, and visualization

## Build & Development Commands

```bash
# Build
cargo build --release

# Tests
cargo test --workspace --all-features

# Lint & format
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Run detector
cargo run -- detect --image <path> --out <path> --marker-diameter 32.0

# End-to-end synthetic eval (generate → detect → score)
python3 tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_run

# Score a single detection
python3 tools/score_detect.py --gt <gt.json> --pred <det.json> --gate 8.0 --out <score.json>

# Regenerate embedded codebook/board constants (don't hand-edit these)
python3 tools/gen_codebook.py --n 893 --seed 1 --out_json tools/codebook.json --out_rs crates/ringgrid-core/src/codebook.rs
python3 tools/gen_board_spec.py --pitch_mm 8.0 --board_mm 200.0 --json_out tools/board/board_spec.json --rust_out crates/ringgrid-core/src/board_spec.rs
cargo build  # rebuild after regenerating
```

## Detection Pipeline Architecture

The core pipeline flows through these stages in order:

1. **Proposal** (`ring/proposal.rs`) — Scharr gradient voting + NMS → candidate centers
2. **Outer Estimate** (`ring/outer_estimate.rs`) — radius hypotheses via radial profile peaks
3. **Outer Fit** (`ring/detect/outer_fit.rs`) — RANSAC ellipse fitting (Fitzgibbon direct LS)
4. **Decode** (`ring/decode.rs`) — 16-sector code sampling → codebook match (893 codewords)
5. **Inner Estimate** (`ring/inner_estimate.rs`) — inner ring ellipse
6. **Dedup** (`ring/pipeline/dedup.rs`) — spatial + ID-based deduplication
7. **Global Filter** (`ring/pipeline/global_filter.rs`) — RANSAC homography if ≥4 decoded markers
8. **H-guided Refinement** (`ring/detect/refine_h.rs`) — local refit at H-projected priors
9. **Completion** (`ring/detect/completion.rs`) — conservative fits at missing H-projected IDs
10. **NL Board-plane Refine** (`refine/`) — per-marker nonlinear circle-center optimization

## Key Math Modules

- `conic/` — ellipse/conic types (`types.rs`), direct fitting (`fit.rs`), generalized eigenvalue solver (`eigen.rs`), RANSAC (`ransac.rs`)
- `homography.rs` — DLT with Hartley normalization + RANSAC
- `projective_center.rs` — unbiased center recovery from inner+outer conics
- `refine/` — LM (tiny-solver) and IRLS solvers for board-plane circle fitting

## Center Correction Strategies

Single-choice selector (no chaining): `none` | `projective_center` | `nl_board`

CLI: `--circle-refine-method {none,projective-center,nl-board}`
NL solver: `--nl-solver {lm,irls}`

## Camera / Distortion Support

Optional radial-tangential distortion model (`camera.rs`). When camera intrinsics are provided:
- Two-pass detection: pass-1 without mapper, pass-2 with mapper using pass-1 seeds
- Final centers mapped back to image space; ellipse/homography in working frame
- `PixelMapper` trait allows custom distortion adapters

## Conventions

- Algorithms go in `ringgrid-core`; CLI/file I/O in `ringgrid-cli`
- External JSON uses `serde` structs (see `DetectionResult` in `lib.rs`)
- Never introduce OpenCV bindings
- `codebook.rs` and `board_spec.rs` are generated — regenerate via Python scripts, never hand-edit
- Debug schema is versioned (`ringgrid.debug.v1`)
- Logging via `tracing` crate; control with `RUST_LOG=debug|info|trace`
- Debug collection is runtime-gated via `Option<&DebugCollectConfig>` — no compile-time feature flag needed

## Refactoring Plan (Internals-First → API)

### Phase 1: Internal Cleanup (completed)

**R5A — Split conic.rs (~1265 LOC → 4 modules):** completed
- `conic/types.rs`: ConicCoeffs, Ellipse, Conic2D, conversions, normalization
- `conic/fit.rs`: Direct ellipse fitting (Fitzgibbon), design matrix construction
- `conic/ransac.rs`: RANSAC wrapper, sampling, inlier counting
- `conic/eigen.rs`: Generalized eigenvalue solver (GEP), cubic root solver, null vector

**R5B — Unify fit-decode-build pipeline:** completed
- Extracted shared helpers (`fit_metrics_with_inner`, `inner_ellipse_params`) into `marker_build.rs`
- `stage_fit_decode`, `completion.rs`, and `refine_h.rs` now use shared helpers

**R5C — Extract radial profile helpers:** completed (R2)
- Shared `ring/radial_profile.rs` used by inner/outer estimators

**R5D — Unify gradient polarity enums:** completed (R2)
- Single `GradPolarity` enum in `marker_spec.rs`

**R5E — Move parameter scaling to core:** completed (R2)
- `DetectConfig::from_marker_diameter_px(f32)` in core

**R5F — Simplify refine/pipeline.rs:** deferred to Phase 4
**R5G — Unify ellipse representation:** deferred to Phase 4

### Phase 2: Remove debug-trace feature gate (completed)

**R6 — Runtime debug collection:**
- Removed all `#[cfg(feature = "debug-trace")]` annotations across 14 files
- Merged `run()` / `run_with_debug()` pairs into single functions with `Option<&DebugCollectConfig>`
- Removed `FitDecodeDebugOutput` — `FitDecodeCoreOutput` serves both roles
- Debug CLI flags (`--debug-json`, `--debug-store-points`) are always available
- Removed `debug-trace` feature from both Cargo.toml files
- 60 tests pass (2 previously feature-gated tests now always compiled)

### Phase 3: Runtime Target Specification (completed)

**R7A — `BoardLayout` struct:**
- Created `board_layout.rs` with `BoardLayout` and `BoardMarker` structs
- HashMap-based ID lookup, serde JSON support, `Default` from embedded `board_spec` constants
- Methods: `xy_mm(id)`, `n_markers()`, `marker_outer_radius_mm()`, `marker_ids()`, `from_json_file()`

**R7B — Config integration:**
- Added `board: BoardLayout` field to `DetectConfig`
- Initialized from `BoardLayout::default()` in all config constructors

**R7C — Pipeline threading:**
- Replaced all 8 direct `board_spec::` call sites across 6 pipeline files with `&BoardLayout` parameters
- Files updated: `global_filter.rs`, `homography_utils.rs`, `completion.rs`, `refine_h.rs`, `refine/pipeline.rs`, `stage_finalize.rs`
- `board_spec.rs` remains as data source for `BoardLayout::default()` only

**R7E — CLI target flag:**
- Added `--target <path.json>` to load board layout from JSON at runtime
- Without `--target`, embedded default is used (backward compatible)

### Phase 4: Completion Refactor + Public API (completed)

**R8A — Refactor `complete_with_h`:**
- Extracted 8+ helper functions from the 530-line monolithic function in `completion.rs`
- New helpers: `CandidateQuality`, `compute_candidate_quality`, `check_decode_gate`, `check_quality_gates`, `make_attempt_record`, `build_pre_gate_fit_debug`, `build_success_fit_debug`, `build_edges_debug`, `points_debug`
- Added shared `outer_estimation_debug` and `inner_estimation_debug` helpers to `debug_conv.rs`, replacing 5 duplicated inline constructions across `completion.rs` and `stage_fit_decode.rs`
- Main function reduced to orchestration; no behavior change (66 tests pass)

**R8B — `Detector` API:**
- Created `detector.rs` with `Detector` and `TargetSpec` structs
- `TargetSpec` wraps `BoardLayout` (extension point for codebook/geometry variants)
- `Detector` methods: `new(marker_diameter_px)`, `with_target(target, diameter)`, `with_config(config)`, `config()`, `config_mut()`
- Detection: `detect(&GrayImage)`, `detect_with_camera(&GrayImage, &CameraModel)`, `detect_with_mapper(&GrayImage, &dyn PixelMapper)`, `detect_with_debug(&GrayImage, &DebugCollectConfig)`
- All methods delegate to existing `ring::detect::*` free functions; no behavior change
- Re-exported as `ringgrid_core::Detector` and `ringgrid_core::TargetSpec`

## Correctness Notes (updated)

1. **Two-pass mapper frame consistency**: fixed. Two-pass outputs now remain in mapper working coordinates; retained pass-1 fallback markers are remapped to working frame and non-preservable geometry fields are dropped to avoid mixed-frame output.
2. **Completion gate ordering**: fixed. Completion now applies the standard quality gates first; decode mismatch is recorded as an acceptance note instead of using a separate stricter pre-gate.
3. **H refit best-so-far selection**: fixed. Final H refit now compares candidate vs current reprojection error and keeps the better transform instead of blindly replacing with the last refit.
