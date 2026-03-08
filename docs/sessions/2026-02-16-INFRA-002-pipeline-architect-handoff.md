# Handoff: Pipeline Architect -> Project Lead

- **Task:** INFRA-002: Decompose Self-Undistort into Focused Modules
- **Date:** 2026-02-16
- **Branch:** code_quality

## Work Completed

- Decomposed monolithic `pixelmap/self_undistort.rs` into focused modules under `crates/ringgrid/src/pixelmap/self_undistort/`:
  - `config.rs` (configuration contract)
  - `result.rs` (serialized result contract)
  - `objective.rs` (conic objective + homography objective helpers)
  - `optimizer.rs` (1D search)
  - `policy.rs` (apply/reject policy gates and validation)
  - `estimator.rs` (orchestration and strategy selection)
  - `tests.rs` (behavior-preserving synthetic/unit tests)
- Preserved top-level entrypoint behavior: `estimate_self_undistort(...) -> Option<SelfUndistortResult>`.
- Removed monolithic source file: `crates/ringgrid/src/pixelmap/self_undistort.rs`.

## API Surface Diff

### Public types / methods

- **Added:** none
- **Removed:** none
- **Changed signatures:** none

### Public contract notes

- `SelfUndistortConfig`, `SelfUndistortResult`, and `estimate_self_undistort` remain available through `crate::pixelmap` re-exports.
- `Detector` API and `DetectConfig` fields are unchanged.

## Backward Compatibility

- **Caller compatibility:** non-breaking for existing Rust API callers.
- **Serialization compatibility:** `SelfUndistortResult` field schema unchanged.
- **Migration section:** no migration required (no public API break introduced).

## Module Boundary Verification

- Algorithmic/self-undistort logic remains in `crates/ringgrid/`.
- No file I/O or CLI concerns were added to library modules.
- `lib.rs` remains pure re-exports with no new type definitions.

## Pipeline Stage Flow

- Stage ordering and detect orchestration are unchanged.
- Self-undistort still runs as: baseline pass -> estimate mapper -> optional seeded rerun.

## Validation Results

### Quality gates

- `cargo fmt --all` ✅
- `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `cargo test --workspace --all-features` ✅

### Required PERF/accuracy scripts

- `bash tools/run_blur3_benchmark.sh` ✅
  - `tools/out/eval_blur3_post_pipeline/det/aggregate.json`
  - avg center error: `0.3148150563710771 px`
  - avg recall: `0.9477832512315271`
  - avg precision: `1.0`
- `bash tools/run_reference_benchmark.sh` ✅
  - `tools/out/reference_benchmark_post_pipeline/summary.json`
  - `projective_center__none` avg center: `0.0528657736611285 px`
- `bash tools/run_distortion_benchmark.sh` ✅
  - `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json`
  - `projective_center__external` avg center: `0.0766663227286656 px`
  - `projective_center__self_undistort` avg center: `0.07783403833012702 px`

## Notes

- INFRA-002 focused on architecture and separation of concerns; no intentional behavior changes were introduced.
- Existing synthetic tests in self-undistort path were preserved and relocated.

## Recommended Handoff

- **To Validation Engineer:** independent replay/sign-off on validation suite artifacts.
- **To Performance Engineer:** optional profiling replay if additional guardrails are desired after modularization.
