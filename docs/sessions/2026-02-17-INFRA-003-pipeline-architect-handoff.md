# Handoff: Pipeline Architect -> Project Lead

- **Task:** INFRA-003: Replace Stringly Reject Reasons with Typed Enums
- **Date:** 2026-02-17
- **Branch:** code_quality

## Executive Summary

- Replaced stringly reject/failure channels in scoped INFRA-003 modules with typed enums plus structured diagnostic payloads.
- Preserved existing detector behavior and public API surface.
- Validation gates passed (`fmt`, `clippy -D warnings`, `test --workspace --all-features`).
- Required blur=3 synthetic eval passed with **zero metric delta** vs baseline aggregate.

## Scope Delivered

### Modules updated (per spec)

- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid/src/detector/inner_fit.rs`
- `crates/ringgrid/src/detector/outer_fit.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/pipeline/fit_decode.rs`

### Core implementation changes

- Added typed reject enums and stable code formatting (`reason.code()` / `Display`) for:
  - decode rejects
  - inner-fit rejects/failures
  - outer-fit rejects
  - completion gate rejects and decode notices
  - fit/decode pipeline candidate reject aggregation
- Replaced `Result<_, String>` and `Option<String>` reason channels in scoped modules with typed reason structs/enums.
- Preserved dynamic numeric context as structured payloads (e.g., observed vs threshold/limit fields), instead of embedding numbers into free-form strings.
- Updated fit/decode rejection aggregation to typed keys with deterministic ordering by `(count desc, code asc)`.
- Added targeted unit tests for representative reason mapping and enum serialization stability.

## Acceptance Criteria Assessment

| Acceptance Criterion | Status | Evidence |
|---|---|---|
| String-based reason channels replaced in scoped modules | ✅ | Scoped modules no longer use `Result<_, String>`/`Option<String>` for reject channels |
| Stable machine-readable reject codes | ✅ | Typed enums + `snake_case` serde + stable `code()`/`Display` mappings |
| `pipeline/fit_decode.rs` aggregation keyed by typed reasons + stable ordering | ✅ | `CandidateRejectReason` map and deterministic sort in `crates/ringgrid/src/pipeline/fit_decode.rs` |
| Numeric context preserved structurally | ✅ | Context enums added across decode/inner-fit/outer-fit/completion |
| Unit tests cover mapping + serialization formatting | ✅ | New tests in updated modules (decode, inner_fit, outer_fit, completion, fit_decode) |
| `cargo test --workspace --all-features` passes | ✅ | Pass |
| `cargo clippy --all-targets --all-features -- -D warnings` clean | ✅ | Pass |
| Synthetic eval thresholds satisfied | ✅ | Blur=3 run matches baseline exactly (see Metrics section) |

## API Surface Diff

### Public types / methods

- **Added:** none
- **Removed:** none
- **Changed public signatures:** none

### Public contract notes

- `Detector` facade unchanged.
- `lib.rs` remains re-exports only.
- No public JSON schema/version changes required for `DetectionResult`.

## Backward Compatibility

- **Caller compatibility:** non-breaking for existing Rust API users.
- **Serialization compatibility:** no externally documented result schema changes in public output types.
- **Migration:** none required.

## Module Boundary Verification

- All changes remain in `crates/ringgrid/` algorithm/pipeline modules.
- No CLI / clap / file-I/O logic introduced into library code.
- No public type definitions added to `lib.rs`.

## Validation Results

### Required quality gates

- `cargo fmt --all` ✅
- `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `cargo test --workspace --all-features` ✅

### Required synthetic eval gate (blur=3)

Run executed on 2026-02-17:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 10 --blur_px 3.0 --marker_diameter 32.0 \
  --out_dir tools/out/eval_infra003_blur3
```

Compared:
- Baseline: `tools/out/eval_blur3_post_pipeline/det/aggregate.json`
- New: `tools/out/eval_infra003_blur3/det/aggregate.json`

Metric deltas (new - baseline):
- `avg_precision`: `+0.000000000000`
- `avg_recall`: `+0.000000000000`
- `avg_center_error`: `+0.000000000000 px`
- `avg_homography_error_vs_gt`: `+0.000000000000 px`
- `avg_homography_self_error`: `+0.000000000000 px`

Threshold result:
- Center mean regression `<= +0.01 px`: ✅
- Precision/recall regression `<= 0.005` absolute: ✅

## Risks / Notes

- `ring::outer_estimate` and `ring::inner_estimate` still emit string reasons internally; INFRA-003 scope intentionally normalized typed reasons at the detector/decode/completion/pipeline boundaries listed in the task spec.
- If maintainers want full end-to-end typed reasons across all estimator layers, that should be tracked as follow-up infra work.

## Changed Files (Code)

- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid/src/detector/inner_fit.rs`
- `crates/ringgrid/src/detector/outer_fit.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/pipeline/fit_decode.rs`

## Recommendation

- **Recommendation to Project Lead:** **Accept INFRA-003**.
- Task goals are met, validation gates pass, and no public API break was introduced.
