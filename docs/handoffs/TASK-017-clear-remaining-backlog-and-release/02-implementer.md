# Implementer Report - TASK-017-clear-remaining-backlog-and-release

- Task ID: `TASK-017-clear-remaining-backlog-and-release`
- Backlog ID: `INFRA-008, ALGO-009, ALGO-010, ALGO-011, INFRA-013`
- Role: `implementer`
- Date: `2026-03-11`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-017-clear-remaining-backlog-and-release/01-architect.md`
- `crates/ringgrid-cli/src/main.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/detector/id_correction/math.rs`
- `crates/ringgrid/src/detector/id_correction/mod.rs`
- `crates/ringgrid/src/detector/outer_fit/mod.rs`
- `crates/ringgrid/src/detector/outer_fit/sampling.rs`
- `crates/ringgrid/src/pipeline/finalize.rs`
- `book/src/cli-guide.md`
- `book/src/detection-modes/external-mapper.md`
- `docs/tuning-guide.md`
- `CHANGELOG.md`
- `docs/backlog.md`

## Summary
- Implemented the additive CLI calibration-file path from `INFRA-008` with conflict validation and JSON-shape compatibility for direct `CameraModel` payloads and top-level `{ "camera": ... }` wrappers.
- Landed the three remaining algorithm backlog items in the existing detector pipeline without adding new public config knobs: local-affine completion seeding, expected-radius outer-ray screening, and final axis-ratio outlier cleanup.
- Updated release-facing docs and bookkeeping for `0.5.0`, including changelog coverage, CLI/tuning guidance, and backlog closure.
- Ran the full local validation baseline plus focused regression tests for each changed behavior.

## Decisions Made
- Reused existing `id_correction` affine math for completion seeding instead of introducing a second affine solver.
- Implemented `--calibration` as an additive CLI input path and made it mutually exclusive with inline `--cam-*` flags and `--self-undistort`.
- Cleared only marker identity metadata for axis-ratio outliers rather than deleting markers outright so the late finalize pipeline stays localized and existing collection behavior is preserved.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-cli/src/main.rs` - added calibration JSON loading, input validation, and CLI tests.
- `crates/ringgrid/src/detector/completion.rs` - added local-affine completion seed selection with homography fallback tests.
- `crates/ringgrid/src/detector/id_correction/math.rs` - exposed reusable affine helpers for completion.
- `crates/ringgrid/src/detector/id_correction/mod.rs` - re-exported affine helpers for detector-internal reuse.
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` - added expected-radius outer-ray pre-screen logic and regression coverage.
- `crates/ringgrid/src/detector/outer_fit/mod.rs` - threaded expected radius into the updated sampling path.
- `crates/ringgrid/src/pipeline/finalize.rs` - added post-collection axis-ratio outlier cleanup and tests.
- `book/src/cli-guide.md` - documented `--calibration`, precedence rules, and CLI examples.
- `book/src/detection-modes/external-mapper.md` - documented CLI compatibility with `CameraModel` JSON files.
- `docs/tuning-guide.md` - replaced stale `--camera-model` wording with the shipped CLI surface.
- `CHANGELOG.md` - documented all user-visible changes included in `0.5.0`.
- `docs/backlog.md` - closed the remaining backlog items for the release slice.

## Validation / Tests
- Commands run:
  - `cargo test -p ringgrid local_affine_completion_seed -- --nocapture`
  - `cargo test -p ringgrid projected_completion_seed_falls_back_to_h -- --nocapture`
  - `cargo test -p ringgrid collect_outer_edge_points_rejects_rays_far_from_expected_radius -- --nocapture`
  - `cargo test -p ringgrid axis_ratio_filter_clears_strong_outliers -- --nocapture`
  - `cargo test -p ringgrid axis_ratio_filter_keeps_in_family_markers -- --nocapture`
  - `cargo test -p ringgrid-cli calibration_file_loads_direct_camera_model_shape -- --nocapture`
  - `cargo test -p ringgrid-cli calibration_file_loads_detector_output_wrapper_shape -- --nocapture`
  - `cargo test -p ringgrid-cli calibration_file_rejects_mixed_inline_camera_flags -- --nocapture`
  - `cargo fmt --all`
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- Results:
  - Focused regression tests passed for all four backlog behaviors.
  - `cargo fmt --all --check` passed after formatting the modified Rust sources.
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
  - `cargo test --workspace --all-features` passed with the workspace green, including `158` `ringgrid` unit tests, `4` `target_generation` integration tests, `9` CLI tests, and `5` doctests.
  - `cargo doc --workspace --all-features --no-deps` and `cargo test --doc --workspace` passed.
  - `mdbook build book` passed.
  - Python surface validation passed: typing artifacts check succeeded, `maturin develop` succeeded, and `pytest crates/ringgrid-py/tests -q` passed with `31 passed`.

## Risks / Open Questions
- `docs/backlog.md` now describes the release as complete only if the final local `v0.5.0` tag is actually cut on this tree.
- The new axis-ratio cleanup intentionally clears IDs instead of removing markers outright.
  - Impact: this is the smallest safe change for the current finalize ordering, but a future pipeline cleanup could choose full marker removal if downstream consumers require it.

## Next Handoff
- To: `Reviewer`
- Requested action: verify the changed detector paths against the architect acceptance criteria, confirm the release/docs bookkeeping is truthful, and reproduce at least the highest-risk validation on the final tree before approval.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1: completed CLI calibration loading and tests in `crates/ringgrid-cli/src/main.rs`.
- Architect step 2: completed all three algorithmic robustness changes and their targeted tests in `completion.rs`, `outer_fit/sampling.rs`, and `finalize.rs`, with affine reuse in `id_correction`.
- Architect step 3: updated release-facing docs/bookkeeping and ran the full local validation baseline.

### Changes Made
- Added a `--calibration <file.json>` CLI flag that deserializes either a direct `CameraModel` JSON payload or a detector-output wrapper containing a top-level `camera` field.
- Added explicit camera-input conflict checks so `--calibration` cannot be mixed with inline `--cam-*` flags and no external camera path can be mixed with `--self-undistort`.
- Reused the structural-ID local affine fit to seed completion from the 3-4 nearest decoded neighbors in board coordinates before falling back to global homography projection.
- Added an outer-ray expected-radius gate that rejects contaminated rays before ellipse fitting when they drift more than 40% from the expected outer radius.
- Added a final axis-ratio consistency pass that clears decoded identity for strong inner/outer ratio outliers relative to the global non-completion median.
- Updated the mdBook, tuning guide, changelog, and backlog to document the shipped behavior and release contents.

### Files Changed
- `CHANGELOG.md` - promoted the upcoming release notes into the `0.5.0` entry and documented the completed backlog work.
- `book/src/cli-guide.md` - documented the new calibration-file path and conflict rules.
- `book/src/detection-modes/external-mapper.md` - documented JSON examples and wrapper compatibility.
- `crates/ringgrid-cli/src/main.rs` - added calibration parsing, validation, and CLI unit tests.
- `crates/ringgrid-py/pyproject.toml` - aligned Python package metadata to `0.5.0`.
- `crates/ringgrid/src/detector/completion.rs` - added local-affine seeding helpers and tests.
- `crates/ringgrid/src/detector/id_correction/math.rs` - exposed affine-fit helpers for detector reuse.
- `crates/ringgrid/src/detector/id_correction/mod.rs` - re-exported the detector-internal helpers.
- `crates/ringgrid/src/detector/outer_fit/mod.rs` - threaded expected outer radius into the sampling API.
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` - added outer-ray contamination rejection and tests.
- `crates/ringgrid/src/pipeline/finalize.rs` - added axis-ratio cleanup and regression tests.
- `docs/backlog.md` - closed the remaining backlog items for the release slice.
- `docs/tuning-guide.md` - replaced stale CLI wording.

### Deviations From Plan
- Implemented the `ALGO-011` reject path by clearing marker identity metadata rather than removing entire marker records.
  - Reason: this keeps the change localized inside late finalization and avoids wider reorder/refactor risk while still preventing bad IDs from shipping.
  - Impact: downstream consumers still see the fitted marker, but it is no longer treated as a decoded board correspondence.

### Tests Added/Updated
- `crates/ringgrid-cli/src/main.rs` - added calibration JSON parsing and conflict-handling unit tests.
- `crates/ringgrid/src/detector/completion.rs` - added local-affine seed selection and fallback tests.
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` - added expected-radius contamination rejection coverage.
- `crates/ringgrid/src/pipeline/finalize.rs` - added axis-ratio outlier cleanup tests.

### Commands Run
- `cargo test -p ringgrid local_affine_completion_seed -- --nocapture`
- `cargo test -p ringgrid projected_completion_seed_falls_back_to_h -- --nocapture`
- `cargo test -p ringgrid collect_outer_edge_points_rejects_rays_far_from_expected_radius -- --nocapture`
- `cargo test -p ringgrid axis_ratio_filter_clears_strong_outliers -- --nocapture`
- `cargo test -p ringgrid axis_ratio_filter_keeps_in_family_markers -- --nocapture`
- `cargo test -p ringgrid-cli calibration_file_loads_direct_camera_model_shape -- --nocapture`
- `cargo test -p ringgrid-cli calibration_file_loads_detector_output_wrapper_shape -- --nocapture`
- `cargo test -p ringgrid-cli calibration_file_rejects_mixed_inline_camera_flags -- --nocapture`
- `cargo fmt --all`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo doc --workspace --all-features --no-deps`
- `cargo test --doc --workspace`
- `mdbook build book`
- `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
- `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
- `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q`

### Results
- All targeted regression tests passed.
- All required Rust formatting, lint, test, rustdoc, and doctest checks passed.
- `mdbook build book` passed after the CLI/docs updates.
- Python binding/docs validation passed, including successful editable install and `31` passing Python tests.

### Remaining Concerns
- The release bookkeeping in `docs/backlog.md` must stay aligned with the actual git tag/commit state when the release is cut.

### Handoff To Reviewer
- Confirm the additive CLI JSON loading is unambiguous and does not weaken existing camera-mode validation.
- Confirm the completion, outer-fit, and finalize changes match the architect acceptance criteria without introducing broader pipeline regressions.
- Confirm the release docs reflect the shipped tree truthfully before approval.
