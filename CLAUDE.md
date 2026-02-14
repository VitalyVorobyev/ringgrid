# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice. It detects markers, decodes 16-sector IDs, estimates homography, and exports structured JSON. No OpenCV bindings — all image processing is in Rust. The goal is to provide a polished external Rust library API.

## Workspace Structure

Cargo workspace with two crates:
- `crates/ringgrid/` — detection algorithms, math primitives, result types
- `crates/ringgrid-cli/` — CLI binary (`ringgrid`) with clap-based argument parsing
- `tools/` — Python utilities for synthetic data generation, evaluation, scoring, and visualization

## Module Layout

```
crates/ringgrid/src/
├── lib.rs              # Re-exports only (public API surface)
├── api.rs              # Detector (primary entry point)
├── board_layout.rs     # Board geometry (hex lattice layout, JSON loader)
├── detector/           # Per-marker detection primitives
│   ├── config.rs       # DetectConfig and sub-configs
│   ├── proposal.rs     # Scharr gradient voting + NMS → candidate centers
│   ├── outer_fit.rs    # RANSAC ellipse fitting (Fitzgibbon direct LS)
│   ├── inner_fit.rs    # Inner ring ellipse fit
│   ├── marker_build.rs # DetectedMarker, FitMetrics structs + builder
│   ├── dedup.rs        # Spatial + ID-based deduplication
│   ├── global_filter.rs# RANSAC homography filter
│   ├── refine_h.rs     # H-guided local refit
│   ├── completion.rs   # Conservative fits at missing H-projected IDs
│   └── center_correction.rs # Projective center application
├── ring/               # Ring-level sampling & estimation primitives
│   ├── outer_estimate.rs    # Radius hypotheses via radial profile peaks
│   ├── inner_estimate.rs    # Inner ring scale estimation
│   ├── radial_profile.rs    # Radial intensity profile sampling
│   ├── edge_sample.rs       # Edge point sampling (distortion-aware)
│   └── projective_center.rs # Unbiased center from inner+outer conics
├── marker/             # Code decoding & marker specification
│   ├── decode.rs       # 16-sector code sampling → codebook match
│   ├── codec.rs        # Codebook matching logic
│   ├── codebook.rs     # Generated 893-codeword table (don't hand-edit)
│   └── marker_spec.rs  # MarkerSpec type
├── pipeline/           # Detection pipeline orchestration
│   ├── mod.rs          # DetectionResult struct, module glue, re-exports
│   ├── run.rs          # Top-level run() orchestrator (proposals → finalize)
│   ├── fit_decode.rs   # Proposals → fit → decode → dedup
│   ├── finalize.rs     # Global filter → refine → complete → assemble
│   └── two_pass.rs     # Two-pass + self-undistort orchestration
├── homography/         # Homography estimation & utilities
│   ├── core.rs         # DLT + RANSAC, RansacStats
│   └── utils.rs        # Refit, reprojection error, matrix conversion
├── conic/              # Ellipse/conic math
│   ├── types.rs        # Ellipse type
│   ├── fit.rs          # Direct ellipse fitting (Fitzgibbon)
│   ├── ransac.rs       # RANSAC ellipse fitting
│   └── eigen.rs        # Generalized eigenvalue solver
├── pixelmap/           # Camera, distortion, pixel mapping
│   ├── cameramodel.rs  # CameraModel, CameraIntrinsics
│   ├── distortion.rs   # RadialTangentialDistortion, DivisionModel
│   └── self_undistort.rs # Self-undistort estimation
└── debug_dump.rs       # Debug JSON schema (feature-gated)
```

## Detection Pipeline Architecture

The core pipeline flows through these stages in order:

1. **Proposal** (`detector/proposal.rs`) — Scharr gradient voting + NMS → candidate centers
2. **Outer Estimate** (`ring/outer_estimate.rs`) — radius hypotheses via radial profile peaks
3. **Outer Fit** (`detector/outer_fit.rs`) — RANSAC ellipse fitting (Fitzgibbon direct LS)
4. **Decode** (`marker/decode.rs`) — 16-sector code sampling → codebook match (893 codewords)
5. **Inner Estimate** (`ring/inner_estimate.rs`) — inner ring ellipse
6. **Dedup** (`detector/dedup.rs`) — spatial + ID-based deduplication
7. **Projective Center** (`detector/center_correction.rs`) — 1st pass: correct fit-decode marker centers
8. **Global Filter** (`detector/global_filter.rs`) — RANSAC homography if ≥4 decoded markers (uses corrected centers)
9. **H-guided Refinement** (`detector/refine_h.rs`) — local refit at H-projected priors
10. **Projective Center** — 2nd pass: reapply after refinement (new ellipses)
11. **Completion** (`detector/completion.rs`) — conservative fits at missing H-projected IDs
12. **Projective Center** — 3rd pass: correct completion-only markers
13. **Final H Refit** — refit homography from all corrected centers

Pipeline orchestration: stages 1–6 in `pipeline/fit_decode.rs`, stages 7–13 in `pipeline/finalize.rs`, top-level sequencing in `pipeline/run.rs`. Two-pass and self-undistort logic in `pipeline/two_pass.rs`.

## Public API

All detection goes through the `Detector` struct (`api.rs`). No public free functions.

Key public types:
- `Detector` — entry point
- `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod` — configuration
- `DetectionResult`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, `RansacStats` — results
- `BoardLayout`, `BoardMarker`, `MarkerSpec` — geometry
- `CameraModel`, `CameraIntrinsics`, `PixelMapper` — camera/distortion
- `Ellipse` — conic geometry

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
python3 tools/gen_codebook.py --n 893 --seed 1 --out_json tools/codebook.json --out_rs crates/ringgrid/src/codebook.rs
python3 tools/gen_board_spec.py --pitch_mm 8.0 --rows 15 --long_row_cols 14 --board_mm 200.0 --json_out tools/board/board_spec.json
cargo build  # rebuild after regenerating
```

## Key Math Modules

- `conic/` — ellipse/conic types (`types.rs`), direct fitting (`fit.rs`), generalized eigenvalue solver (`eigen.rs`), RANSAC (`ransac.rs`)
- `homography/` — DLT with Hartley normalization + RANSAC (`core.rs`), refit & reprojection utilities (`utils.rs`)
- `ring/projective_center.rs` — unbiased center recovery from inner+outer conics

## Center Correction Strategies

Single-choice selector: `none` | `projective_center`

CLI: `--circle-refine-method {none,projective-center}`

Center correction is applied in three passes: before global filter (fit-decode markers), after H-guided refinement (recompute with new ellipses), and after completion (new markers only). All passes are in `pipeline/finalize.rs`.

## Camera / Distortion Support

Optional radial-tangential distortion model (`pixelmap/`). When camera intrinsics are provided:
- Two-pass detection: pass-1 without mapper, pass-2 with mapper using pass-1 seeds
- Final centers mapped back to image space; ellipse/homography in working frame
- `PixelMapper` trait allows custom distortion adapters

Self-undistort mode estimates a division-model distortion correction from detected ellipse edge points, then re-runs detection with the estimated mapper.

## Conventions

- Algorithms go in `ringgrid`; CLI/file I/O in `ringgrid-cli`
- External JSON uses `serde` structs (see `DetectionResult` in `pipeline/mod.rs`)
- Never introduce OpenCV bindings
- `codebook.rs` is generated; board target is runtime JSON (`tools/board/board_spec.json`) — regenerate via Python scripts, never hand-edit generated Rust
- Debug schema is versioned (`ringgrid.debug.v7`)
- Logging via `tracing` crate; control with `RUST_LOG=debug|info|trace`
- Debug collection is feature-gated via `cli-internal` feature flag
- `lib.rs` is purely re-exports; type definitions live at their construction sites

## CI / checks

Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
