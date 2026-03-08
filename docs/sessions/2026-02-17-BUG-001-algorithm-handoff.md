# Handoff: Algorithm Engineer -> Project Lead

- **Task:** BUG-001: Fix Decode Config Drift and Expose Hidden Thresholds
- **Date:** 2026-02-17
- **Branch:** code_quality

## Executive Summary

- Resolved `DecodeConfig` doc/default drift in `crates/ringgrid/src/marker/decode.rs` by moving decode defaults to explicit constants and wiring docs + `Default` to the same source.
- Promoted hidden decode thresholding controls to explicit `DecodeConfig` fields:
  - `min_decode_contrast`
  - `threshold_max_iters`
  - `threshold_convergence_eps`
- Removed decode-time hidden constants for contrast gate and iterative thresholding stop criteria; behavior is now deterministic and configuration-backed.
- Added deterministic unit coverage for default config contract, serde backward compatibility, low-contrast rejection, and threshold-loop guard behavior.
- Required quality gates and synthetic eval gate passed; no precision/recall regression and center-error delta is within threshold.

## Scope Delivered

### Module updates

- `crates/ringgrid/src/marker/decode.rs`

### Implementation details

- `DecodeConfig` now exposes all decode thresholding controls that previously lived as hidden implementation constants.
- New decode config fields are serde-safe for older serialized configs via field-level defaults:
  - `#[serde(default = "DecodeConfig::default_min_decode_contrast")]`
  - `#[serde(default = "DecodeConfig::default_threshold_max_iters")]`
  - `#[serde(default = "DecodeConfig::default_threshold_convergence_eps")]`
- Added `compute_iterative_two_means_threshold(...)` helper for deterministic bounded Lloyd updates:
  - bounded by `threshold_max_iters`
  - converges by `threshold_convergence_eps`
- Added finite-value guards for decode thresholds:
  - non-finite `min_decode_contrast` falls back to default
  - non-finite epsilon falls back to default

## Mathematical Justification

- Thresholding remains a 1D two-cluster Lloyd iteration (iterative 2-means), which is unchanged algorithmically from prior behavior.
- Exposing iteration cap and convergence epsilon externalizes previously hidden termination controls without changing the estimator family.
- Contrast gate remains `max(sector_intensities) - min(sector_intensities)`; promoting its threshold to config removes ambiguity in decode acceptance behavior.

## Metrology / Frame Invariants

- No geometry-frame semantics changed:
  - decode still samples in image coordinates (or mapper-backed image sampling path),
  - no center coordinate conventions were modified.
- Pixel-center convention remains unchanged (`i as f32` sampling coordinates).

## Acceptance Criteria Assessment

| Acceptance Criterion | Status | Evidence |
|---|---|---|
| `DecodeConfig` rustdoc values exactly match runtime defaults | ✅ | Docs now point to `DecodeConfig::DEFAULT_*` constants used by `Default` |
| Hidden decode constants resolved consistently (explicit config or fixed invariant + docs/tests) | ✅ | Hidden constants promoted to explicit config fields with defaults and tests |
| Decode thresholding behavior deterministic and unit-tested | ✅ | Added threshold helper + tests for max-iter and epsilon guards |
| Regression tests for default path, low-contrast rejection, threshold-loop guards | ✅ | New tests in `marker/decode.rs` (`decode_config_*`, `decode_low_contrast_*`, `threshold_loop_*`) |
| `cargo test --workspace --all-features` passes | ✅ | Pass |
| `cargo clippy --all-targets --all-features -- -D warnings` clean | ✅ | Pass |
| Synthetic eval metrics within thresholds | ✅ | Precision/recall unchanged; center-mean delta `-0.000080256 px` vs baseline |

## Validation Results

### Quality gates

- `cargo fmt --all`: pass
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo test --workspace --all-features`: pass

### Synthetic eval gate

Command run:

```bash
.venv/bin/python3 tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_check
```

Aggregate (`tools/out/eval_check/det/aggregate.json`):

- `avg_precision`: `1.0`
- `avg_recall`: `1.0`
- `avg_center_error`: `0.067911275143036 px`
- `avg_homography_error_vs_gt`: `0.025470891648532997 px`
- `avg_homography_self_error`: `0.06349050970186912 px`

Baseline comparison (clean `HEAD` worktree, same command):

- precision delta: `+0.000000000`
- recall delta: `+0.000000000`
- center-mean delta: `-0.000080256 px`
- homography-vs-GT delta: `+0.000062706 px`
- homography-self delta: `-0.000088730 px`

Threshold check:

- Center-error regression <= `+0.01 px`: ✅
- Precision/recall regression <= `0.005` absolute: ✅

## Performance / Allocation Notes

- Decode hot loop did not introduce new per-sector/per-sample dynamic allocations.
- Threshold helper operates on fixed-size `[f32; 16]` inputs and scalar accumulators only.

## Public API Impact

- Public config surface changed: **yes** (`DecodeConfig` gained three public fields with defaults).
- No detector method signatures changed.
- No output schema (`DetectionResult`) changes.

## Files Changed

| File | Change |
|------|--------|
| `crates/ringgrid/src/marker/decode.rs` | Exposed decode threshold knobs, unified defaults/docs, replaced hidden constants, added deterministic threshold helper/tests |
| `.ai/state/sessions/2026-02-17-BUG-001-algorithm-handoff.md` | This handoff report |

## Recommendation

- Ready for Project Lead review.
- Conditional per spec: because public decode config knobs were added, request Pipeline Architect API check before final close if strict workflow enforcement is desired.
