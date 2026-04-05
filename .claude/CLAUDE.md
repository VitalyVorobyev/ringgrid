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
├── proposal/           # Center proposal generation (delegates to radsym)
│   ├── mod.rs          # Adapter: radsym fused RSD → ringgrid Proposal types
│   ├── config.rs       # ProposalConfig (translated to radsym RsdConfig)
│   └── tests.rs        # Behavioral tests
├── detector/           # Per-marker detection primitives
│   ├── config.rs       # DetectConfig and sub-configs
│   ├── outer_fit.rs    # RANSAC ellipse fitting (Fitzgibbon direct LS)
│   ├── inner_fit.rs    # Inner ring ellipse fit
│   ├── marker_build.rs # DetectedMarker, FitMetrics structs + builder
│   ├── dedup.rs        # Spatial + ID-based deduplication
│   ├── global_filter.rs# RANSAC homography filter
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
│   ├── run.rs          # Top-level orchestrator + two-pass/self-undistort/multiscale flow
│   ├── fit_decode.rs   # Proposals → fit → decode → dedup
│   ├── finalize.rs     # Center correction → global filter → completion → final H
│   ├── scale_probe.rs  # Ring angular-variance sweep → dominant radius estimates
│   ├── prelude.rs      # Common imports for pipeline modules
│   └── result.rs       # DetectionResult type
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
```

## Detection Pipeline Architecture

### Single-pass stages (in order)

1. **Proposal** (`proposal/mod.rs`) — radsym fused RSD (Scharr gradient + magnitude voting + NMS) → candidate centers
2. **Outer Estimate** (`ring/outer_estimate.rs`) — radius hypotheses via radial profile peaks
3. **Outer Fit** (`detector/outer_fit.rs`) — RANSAC ellipse fitting (Fitzgibbon direct LS)
4. **Decode** (`marker/decode.rs`) — 16-sector code sampling → codebook match (893 codewords)
5. **Inner Fit** (`detector/inner_fit.rs`) — inner ring ellipse fit
6. **Dedup** (`detector/dedup.rs`) — spatial + ID-based deduplication
7. **Projective Center** (`detector/center_correction.rs`) — correct fit-decode marker centers (once per marker)
8. **Global Filter** (`detector/global_filter.rs`) — RANSAC homography if ≥4 decoded markers (uses corrected centers)
9. **Completion** (`detector/completion.rs`) — conservative fits at missing H-projected IDs + projective center for new markers
10. **Final H Refit** — refit homography from all corrected centers

Pipeline orchestration: stages 1–6 in `pipeline/fit_decode.rs`, stages 7–10 in `pipeline/finalize.rs`, top-level sequencing in `pipeline/run.rs`.

### Multi-scale / adaptive pipeline

For multi-scale detection (`detect_multiscale`, `detect_adaptive`, `detect_adaptive_with_hint`), the pipeline is split at the finalize boundary:

- **`finalize_premerge`** — runs stages 7 (projective center) only; returns `Vec<DetectedMarker>` without global filter, completion, or H refit. Called once per tier.
- **`merge_multiscale_markers`** (`detector/dedup.rs`) — size-consistency-aware NMS across all tier outputs. Prefers markers whose outer radius matches the neighborhood median (k=6 hex-lattice neighbors); confidence breaks ties.
- **`finalize_postmerge`** — runs stages 8–10 (global filter + completion + final H refit) exactly once on the merged pool.

Scale probe (`pipeline/scale_probe.rs`): ring angular-variance sweep at top-K gradient proposals over 20 geometric radius candidates (4–110 px). High variance at a radius indicates the code band (alternating bright/dark sectors). Code-band midpoint ≈ 0.8× outer ring radius; results feed `ScaleTiers::from_detected_radii`.

Default `MarkerScalePrior` is **[14, 66] px** (updated from [20, 56] in this release, +11.4% on the rtv3d dataset).

## Public API

All detection goes through the `Detector` struct (`api.rs`). No public free functions.

Key public types:
- `Detector` — entry point
- `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod` — configuration
- `ScaleTier`, `ScaleTiers` — multi-scale tier configuration (presets: `four_tier_wide`, `two_tier_standard`, `single`)
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
`ringgrid-py` is excluded from the workspace — its version must be updated manually
in **three** places when bumping:
- `crates/ringgrid-py/Cargo.toml` (`version` field + `ringgrid` dependency version)
- `crates/ringgrid-py/pyproject.toml` (`project.version` field)

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

## CI / checks

Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
