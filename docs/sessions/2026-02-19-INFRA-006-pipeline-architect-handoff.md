# Handoff: Pipeline Architect -> Project Lead

- **Task:** INFRA-006: Split Outer-Fit Responsibilities and Remove Hardcoded Solver Knobs
- **Date:** 2026-02-19
- **Branch:** main

## Executive Summary

- Split monolithic `detector/outer_fit.rs` into focused internal modules: orchestration (`mod.rs`), sampling (`sampling.rs`), solver (`solver.rs`), and scoring (`scoring.rs`).
- Removed hardcoded outer-fit RANSAC literals from implementation; solver now reads all robust-fit knobs from shared detector config.
- Added `OuterFitConfig` to `DetectConfig` with behavior-preserving defaults and re-exported it in the public API surface.
- Kept completion-specific edge-policy behavior explicit while reusing the same shared outer-fit core.
- Preserved reject-reason semantics and behavior; validation/eval gates passed with zero metric deltas versus baseline.

## API Surface Diff

### Added public types

- `OuterFitConfig` (`crates/ringgrid/src/detector/config.rs`)

### Changed public fields/signatures

- `DetectConfig` now includes:
  - `outer_fit: OuterFitConfig`
- Re-exports updated:
  - `crates/ringgrid/src/detector/mod.rs` now re-exports `OuterFitConfig`
  - `crates/ringgrid/src/lib.rs` now re-exports `OuterFitConfig`

### Removed public items

- None.

## Backward Compatibility Assessment

- **Source compatibility:** soft-breaking for callers that construct `DetectConfig` via exhaustive struct literal without `..Default::default()`. Constructor-based and update-syntax callers remain compatible.
- **Behavior compatibility:** preserved by defaults (`min_direct_fit_points=6`, `min_ransac_points=8`, `ransac={max_iters:200,inlier_threshold:1.5,min_inliers:6,seed:42}`).
- **Serialization compatibility:** unchanged for `DetectionResult` and detection output schema.

## Module Boundary Verification

- All algorithm/config refactor work is contained in `crates/ringgrid/`.
- No CLI or file-I/O logic was added to the library.
- `lib.rs` remains re-exports-only.

## Pipeline Stage Notes

- Stage sequence is unchanged (10-stage flow preserved).
- Internal responsibility split for impacted stages:
  - Stage 3 (Outer Fit): now explicitly separated into sampling / solver / scoring / orchestration units.
  - Stage 9 (Completion): still uses a completion-specific edge policy, but shares the same fit/decode scoring core.

## Acceptance Criteria Mapping

- Outer-fit responsibilities split into focused units/modules: ✅
- Hardcoded outer-fit solver knobs removed and sourced from config/defaults: ✅
- Completion behavior remains explicit without duplicating baseline core logic: ✅
- Reject-reason semantics/diagnostics stable: ✅
- Unit tests for config sourcing and behavior parity added: ✅

## Validation Results

### Required quality gates

- `cargo fmt --all` ✅
- `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `cargo test --workspace --all-features` ✅
  - `ringgrid`: 105 passed
  - `ringgrid-cli`: 4 passed
  - doc tests: 5 passed

### Synthetic eval gate

Command:

```bash
.venv/bin/python3 tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_infra006_check
```

Comparison:
- Baseline: `tools/out/eval_check/det/aggregate.json`
- New: `tools/out/eval_infra006_check/det/aggregate.json`

Metric deltas (new - baseline):
- `avg_precision`: `+0.000000000`
- `avg_recall`: `+0.000000000`
- `avg_center_error`: `+0.000000000 px`
- `avg_homography_error_vs_gt`: `+0.000000000 px`
- `avg_homography_self_error`: `+0.000000000 px`

Threshold result:
- center mean regression `<= +0.01 px`: ✅
- precision/recall non-decrease (`>-0.005` gate): ✅

## Files Changed

- `crates/ringgrid/src/detector/config.rs`
- `crates/ringgrid/src/detector/mod.rs`
- `crates/ringgrid/src/detector/outer_fit.rs` (removed)
- `crates/ringgrid/src/detector/outer_fit/mod.rs` (added)
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` (added)
- `crates/ringgrid/src/detector/outer_fit/scoring.rs` (added)
- `crates/ringgrid/src/detector/outer_fit/solver.rs` (added)
- `crates/ringgrid/src/lib.rs`
- `.ai/state/sessions/2026-02-19-INFRA-006-pipeline-architect-handoff.md`

## Recommendation

- **Recommendation to Project Lead:** accept `INFRA-006`.
- Task goals are met with improved module boundaries, centralized outer-fit solver config, and no measured regression in quality/eval metrics.
