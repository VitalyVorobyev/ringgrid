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

# Build with debug dump support (enables --debug-json CLI flag)
cargo build --features debug-trace

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

- `conic.rs` (~1265 LOC) — ellipse/conic fitting, generalized eigenvalue solver, RANSAC
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
- `debug-trace` feature is compile-time gated (default off) — will be replaced by observer pattern

## Refactoring Plan (Internals-First → API)

### Phase 1: Internal Cleanup

**R5A — Split conic.rs (~1265 LOC → 3-4 modules):**
- `conic/types.rs`: ConicCoeffs, Ellipse, Conic2D, conversions, normalization
- `conic/fit.rs`: Direct ellipse fitting (Fitzgibbon), design matrix construction
- `conic/ransac.rs`: RANSAC wrapper, sampling, inlier counting
- `conic/eigen.rs`: Generalized eigenvalue solver (GEP), cubic root solver, null vector

**R5B — Unify fit-decode-build pipeline:**
- Extract shared "single marker fit" function from `stage_fit_decode`, `completion.rs`, and `refine_h.rs` (currently 3 copies of outer_fit → decode → inner_fit → marker_build)
- All three call sites become thin wrappers around the shared function with different priors/gates

**R5C — Extract radial profile helpers:**
- Move duplicated code from `outer_estimate.rs` and `inner_estimate.rs` into `radial_profile.rs`:
  - `compute_radial_derivative_curve()` — identical in both modules
  - `smooth_curve_3point()` — identical in both modules
  - `compute_theta_consistency()` — identical logic with same heuristic

**R5D — Unify gradient polarity enums:**
- Merge `InnerGradPolarity` and `OuterGradPolarity` into single `GradPolarity` enum (identical variants: DarkToLight, LightToDark, Auto)

**R5E — Move parameter scaling to core:**
- Create `DetectConfig::from_marker_diameter(f32)` in core (currently in CLI `build_detect_config()`)
- Name and document magic scaling constants (0.4, 1.7, 0.8, 2.0, 0.3, 2.5, etc.)

**R5F — Simplify refine/pipeline.rs:**
- Replace 11-exit-path + 15-param `make_record()` with `RefineError` enum
- Core function returns `Result<RefineOk, RefineError>`, record built from the result
- Separate validation, geometry, and record creation

**R5G — Unify ellipse representation:**
- Decide canonical internal type (likely keep `Ellipse` from conic.rs)
- `EllipseParams` becomes a serde wrapper with `From<Ellipse>` conversion only at API boundary
- Eliminate field-name confusion (`center_xy` vs `cx/cy`)

### Phase 2: Observer Pattern for Debug

**R6A — Design `DetectionObserver` trait:**
```rust
pub trait DetectionObserver {
    fn on_proposal(&mut self, proposals: &[Proposal]) {}
    fn on_marker_fit(&mut self, id: usize, result: &MarkerFitResult) {}
    fn on_global_filter(&mut self, h: &Homography, stats: &RansacStats) {}
    fn on_completion(&mut self, stats: &CompletionStats) {}
    fn on_refine(&mut self, record: &MarkerRefineRecord) {}
    // ... etc
}
```

**R6B — Eliminate debug code duplication:**
- Remove `run_with_debug()` variants from stage_fit_decode and stage_finalize
- Replace with single code path that calls observer methods at key points
- `NullObserver` (default) has zero-cost via monomorphization
- `DebugDumpObserver` replaces current debug-trace feature

**R6C — Remove `debug-trace` feature gate:**
- Observer pattern makes it unnecessary
- Debug output availability controlled by which observer is passed, not compile-time features

### Phase 3: Target Specification

**R7A — Predefined dictionary approach:**
- Single large dictionary (893+ codewords, current codebook)
- Board layout selects a subset of IDs from the dictionary
- Board layout provided as runtime JSON (marker positions, geometry)

**R7B — Design `TargetSpec` struct:**
```rust
pub struct TargetSpec {
    pub dictionary: &'static Dictionary,  // predefined, embedded
    pub board: BoardLayout,               // runtime JSON: marker positions + geometry
    pub marker_geometry: MarkerGeometry,  // radii, sector count
}
```

**R7C — Runtime board layout:**
- JSON schema for board layout (marker positions in mm, optional hex coordinates)
- Remove hard dependency on compile-time `board_spec::xy_mm()` in detection pipeline
- `board_spec.rs` becomes one provider of `BoardLayout`, not the only one

### Phase 4: Public API Design

**R8A — `Detector` object:**
```rust
pub struct Detector {
    target: TargetSpec,
    config: DetectConfig,
}
impl Detector {
    pub fn new(target: TargetSpec) -> Self;
    pub fn with_config(target: TargetSpec, config: DetectConfig) -> Self;
    pub fn detect(&self, image: &GrayImage) -> DetectionResult;
    pub fn detect_with_camera(&self, image: &GrayImage, camera: &CameraModel) -> DetectionResult;
    pub fn detect_with_observer(&self, image: &GrayImage, observer: &mut dyn DetectionObserver) -> DetectionResult;
}
```

**R8B — Clean output types:**
- `DetectedMarker`: id, center, confidence (always present); ellipses and metrics behind accessor methods
- `DetectionResult`: markers, homography (as `[[f64;3];3]`), image_size
- Feature-gated `nalgebra` re-exports for consumers who want Matrix3 interop

**R8C — Configuration tiers:**
- Simple: `DetectConfig::default()` + `marker_diameter_px` (covers 90% of use cases)
- Advanced: nested structs for proposal/edge/decode/refine internals
- `DetectConfig::from_marker_diameter(px)` handles all scaling

**R8D — Error handling:**
- Public functions return `Result<DetectionResult, DetectionError>`
- `DetectionError` enum with variants for invalid config, image too small, etc.
- Detection with zero markers is success (empty result), not error

## Known Correctness Issues (to fix during refactoring)

1. **Two-pass mapper ellipse inconsistency**: `map_centers_to_image_space()` maps marker centers but not ellipse parameters — output mixes coordinate frames
2. **Completion gate ordering**: Decode-mismatch path uses 0.35x reproj_gate before normal gates, creating inconsistent acceptance
3. **H refit convergence**: Loop exits with original H if first iteration degrades; should track best-so-far
