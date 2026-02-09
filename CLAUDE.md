# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice. It detects markers, decodes 16-sector IDs, estimates homography, and exports structured JSON. No OpenCV bindings — all image processing is in Rust. The goal is to provide a polished external Rust library API.

## Workspace Structure

Cargo workspace with two crates:
- `crates/ringgrid/` — detection algorithms, math primitives, result types
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
python3 tools/gen_codebook.py --n 893 --seed 1 --out_json tools/codebook.json --out_rs crates/ringgrid/src/codebook.rs
python3 tools/gen_board_spec.py --pitch_mm 8.0 --rows 15 --long_row_cols 14 --board_mm 200.0 --json_out tools/board/board_spec.json
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
10. **Projective Center** (`projective_center.rs`) — unbiased center recovery from inner+outer conics

## Key Math Modules

- `conic/` — ellipse/conic types (`types.rs`), direct fitting (`fit.rs`), generalized eigenvalue solver (`eigen.rs`), RANSAC (`ransac.rs`)
- `homography.rs` — DLT with Hartley normalization + RANSAC
- `projective_center.rs` — unbiased center recovery from inner+outer conics

## Center Correction Strategies

Single-choice selector: `none` | `projective_center`

CLI: `--circle-refine-method {none,projective-center}`

## Camera / Distortion Support

Optional radial-tangential distortion model (`camera.rs`). When camera intrinsics are provided:
- Two-pass detection: pass-1 without mapper, pass-2 with mapper using pass-1 seeds
- Final centers mapped back to image space; ellipse/homography in working frame
- `PixelMapper` trait allows custom distortion adapters

## Conventions

- Algorithms go in `ringgrid`; CLI/file I/O in `ringgrid-cli`
- External JSON uses `serde` structs (see `DetectionResult` in `lib.rs`)
- Never introduce OpenCV bindings
- `codebook.rs` is generated; board target is runtime JSON (`tools/board/board_spec.json`) — regenerate via Python scripts, never hand-edit generated Rust
- Debug schema is versioned (`ringgrid.debug.v3`)
- Logging via `tracing` crate; control with `RUST_LOG=debug|info|trace`
- Debug collection is runtime-gated via `Option<&DebugCollectConfig>` — no compile-time feature flag needed

## CI / checks

Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
