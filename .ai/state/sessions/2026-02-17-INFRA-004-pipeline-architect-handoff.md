# Handoff: Pipeline Architect -> Project Lead

- **Task:** INFRA-004: Deduplicate Homography Correspondence/Stats Utilities
- **Date:** 2026-02-17
- **Branch:** code_quality

## Executive Summary

- Introduced a shared internal utility layer for homography correspondence assembly and reprojection-stat helpers in `crates/ringgrid/src/homography/correspondence.rs`.
- Rewired all three target call sites to the shared utilities:
  - `crates/ringgrid/src/detector/global_filter.rs`
  - `crates/ringgrid/src/homography/utils.rs`
  - `crates/ringgrid/src/pixelmap/self_undistort/objective.rs`
- Made destination-frame semantics explicit at utility boundaries (`image` vs `working`) and normalized duplicate-ID policy via explicit enum options.
- Added focused unit coverage for correspondence edge cases, including duplicate-ID confidence/tie handling.
- Validation gates and synth eval checks passed with zero aggregate metric deltas versus baseline.

## Scope Delivered

### Core refactor

- Added `homography::correspondence` internal module containing:
  - correspondence collection with explicit frame + duplicate policy
  - reprojection error vector helper
  - shared mean/P95 and masked-inlier mean helpers
- Re-exported shared helpers at `pub(crate)` scope through `crates/ringgrid/src/homography/mod.rs`.
- Removed materially duplicated loops/logic in global-filter, homography-utils, and self-undistort objective paths.

### Tests added

- New correspondence utility tests in `crates/ringgrid/src/homography/correspondence.rs`:
  - missing ID / out-of-board ID / invalid center filtering
  - duplicate-ID highest-confidence selection with deterministic tie behavior
  - mapper-style closure drop behavior
  - reprojection stat helper behavior (empty/non-empty and finite-inlier mean)

## API Surface Diff

### Public API

- **Added public types/methods:** none
- **Removed public types/methods:** none
- **Changed public signatures:** none

### Internal API

- Added new `pub(crate)` homography utility items only (no `pub` exports).
- `lib.rs` remains re-exports only; no new type definitions added there.

## Backward Compatibility

- **Rust caller compatibility:** non-breaking (no public surface change).
- **Serialization compatibility:** unchanged (`DetectionResult` schema unchanged).
- **Migration requirement:** none.

## Module Boundary Verification

- All new logic remains in `crates/ringgrid/` algorithm/internal utility modules.
- No CLI/file-I/O concerns introduced into library code.
- Pipeline stage ordering is unchanged; this is an internal deduplication/refactor.

## Pipeline Stage Flow

- Stage 8 (Global Filter) and Stage 10 (Final H Refit stats path) behavior is preserved.
- No sequencing changes in `pipeline` orchestration.
- Frame semantics are now explicit in correspondence collection calls.

## Validation Results

### Required quality gates

- `cargo fmt --all` ✅
- `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `cargo test --workspace --all-features` ✅
  - `ringgrid` tests: `100 passed`
  - `ringgrid-cli` tests: `4 passed`
  - doc tests: `5 passed`

### Synthetic eval gate (required for pipeline-impacting change)

Command:

```bash
.venv/bin/python3 tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_infra004_check
```

Comparison:
- Baseline: `tools/out/eval_check/det/aggregate.json`
- Refactor: `tools/out/eval_infra004_check/det/aggregate.json`

Metric deltas (refactor - baseline):
- `avg_precision`: `+0.000000000`
- `avg_recall`: `+0.000000000`
- `avg_center_error`: `+0.000000000 px`
- `avg_homography_error_vs_gt`: `+0.000000000 px`
- `avg_homography_self_error`: `+0.000000000 px`

Threshold result:
- center mean regression `<= +0.01 px`: ✅
- precision/recall non-decrease: ✅

## Changed Files

- `crates/ringgrid/src/homography/correspondence.rs` (new)
- `crates/ringgrid/src/homography/mod.rs`
- `crates/ringgrid/src/homography/utils.rs`
- `crates/ringgrid/src/detector/global_filter.rs`
- `crates/ringgrid/src/pixelmap/self_undistort/objective.rs`

## Recommendation

- **Recommendation to Project Lead:** accept `INFRA-004`.
- Acceptance criteria are met: shared utility introduced and adopted in all scoped modules, duplication materially removed, frame semantics explicit, tests added, and validation gates are green with no measured regression.
