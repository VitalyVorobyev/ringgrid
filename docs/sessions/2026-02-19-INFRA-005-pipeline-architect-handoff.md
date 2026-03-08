# Handoff: Pipeline Architect -> Project Lead

- **Task:** INFRA-005: Harden BoardLayout Invariants and Loading Errors
- **Date:** 2026-02-19
- **Branch:** code_quality

## Executive Summary

- Hardened `BoardLayout` invariants by making marker storage private and exposing read-only marker accessors.
- Removed public invariant-repair footgun (`build_index()`); index consistency is now maintained internally.
- Replaced weak string/boxed board-loading failures with typed, machine-matchable error enums.
- Propagated typed board-load errors through `Detector::from_target_json_file*`.
- Added focused test coverage for typed `io` / JSON-parse / validation error mapping and retained default-board behavior guarantees.

## API Surface Diff

### Added public types

- `BoardLayoutValidationError` (`crates/ringgrid/src/board_layout.rs`)
- `BoardLayoutLoadError` (`crates/ringgrid/src/board_layout.rs`)

### Changed public signatures

- `BoardLayout::from_json_file(path: &Path)`:
  - from `Result<Self, Box<dyn std::error::Error>>`
  - to `Result<Self, BoardLayoutLoadError>`
- `Detector::from_target_json_file(path: &Path)`:
  - from `Result<Self, Box<dyn std::error::Error>>`
  - to `Result<Self, BoardLayoutLoadError>`
- `Detector::from_target_json_file_with_scale(...)`:
  - from `Result<Self, Box<dyn std::error::Error>>`
  - to `Result<Self, BoardLayoutLoadError>`
- `Detector::from_target_json_file_with_marker_diameter(...)`:
  - from `Result<Self, Box<dyn std::error::Error>>`
  - to `Result<Self, BoardLayoutLoadError>`

### Invariant-safety surface changes

- `BoardLayout.markers` is now private.
- Added read-only accessors:
  - `BoardLayout::markers(&self) -> &[BoardMarker]`
  - `BoardLayout::marker(&self, id: usize) -> Option<&BoardMarker>`
  - `BoardLayout::marker_by_index(&self, index: usize) -> Option<&BoardMarker>`

### Re-exports

- `lib.rs` now re-exports:
  - `BoardLayoutLoadError`
  - `BoardLayoutValidationError`

## Backward Compatibility Assessment

- **Source compatibility:** API-breaking for callers that:
  - mutated `board.markers` directly;
  - called `build_index()` explicitly;
  - relied on boxed-error signatures for board/detector target-file constructors.
- **Behavior compatibility (valid inputs):** preserved.
  - Default board shape/count/lookup semantics unchanged.
  - `xy_mm`, marker geometry generation, and downstream pipeline behavior remain unchanged.
- **Serialization compatibility:** unchanged (`DetectionResult` and result schema unaffected).

## Module Boundary Verification

- Algorithmic and API-contract changes were implemented in `crates/ringgrid/`:
  - `crates/ringgrid/src/board_layout.rs`
  - `crates/ringgrid/src/api.rs`
  - `crates/ringgrid/src/lib.rs`
- CLI-only adaptation was limited to `crates/ringgrid-cli/src/main.rs` (board-info read path update), preserving crate separation.
- `lib.rs` remains re-exports-only.

## Acceptance Criteria Mapping

- `BoardLayout` invariant safety by API design: ✅
  - external mutable marker access removed.
- Manual `build_index()` footgun removed from public usage: ✅
  - no public repair API required.
- Typed load/validation errors with matchable variants: ✅
  - explicit `Io`, `JsonParse`, `Validation` + structured validation variants.
- Detector target-file constructors propagate typed board errors: ✅
  - signatures now return `BoardLayoutLoadError`.
- Valid-input behavior preserved: ✅
  - existing default-board geometry tests still pass.
- Unit tests cover typed invalid JSON/spec mapping: ✅
  - added `io`/parse/validation mapping tests.

## Validation Results

### Required quality gates

- `cargo fmt --all` ✅
- `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `cargo test --workspace --all-features` ✅
  - `ringgrid`: 103 passed
  - `ringgrid-cli`: 4 passed
  - doc tests: 5 passed

### Synthetic eval gate

Command:

```bash
.venv/bin/python3 tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_infra005_check
```

Comparison:
- Baseline: `tools/out/eval_check/det/aggregate.json`
- New: `tools/out/eval_infra005_check/det/aggregate.json`

Metric deltas (new - baseline):
- `avg_precision`: `+0.000000000`
- `avg_recall`: `+0.000000000`
- `avg_center_error`: `+0.000000000 px`
- `avg_homography_error_vs_gt`: `+0.000000000 px`
- `avg_homography_self_error`: `+0.000000000 px`

Threshold result:
- center mean regression `<= +0.01 px`: ✅
- precision/recall non-decrease: ✅

## Changed Files

- `crates/ringgrid/src/board_layout.rs`
- `crates/ringgrid/src/api.rs`
- `crates/ringgrid/src/lib.rs`
- `crates/ringgrid-cli/src/main.rs`
- `.ai/state/sessions/2026-02-19-INFRA-005-pipeline-architect-handoff.md`

## Recommendation

- **Recommendation to Project Lead:** accept `INFRA-005`.
- Task goals are met with explicit typed contracts, hardened board invariants, and no measured detection/regression impact.
