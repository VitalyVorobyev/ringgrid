# Implementer Report - TASK-018-rust-cli-target-generation-equivalence

- Task ID: `TASK-018-rust-cli-target-generation-equivalence`
- Backlog ID: `n/a`
- Role: `implementer`
- Date: `2026-03-12`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-018-rust-cli-target-generation-equivalence/01-architect.md`
- `crates/ringgrid-cli/src/main.rs`
- `crates/ringgrid/src/board_layout.rs`
- `crates/ringgrid/src/target_generation.rs`
- `crates/ringgrid/tests/target_generation.rs`
- `tools/gen_target.py`
- `tools/tests/test_gen_target.py`
- `README.md`
- `book/src/cli-guide.md`
- `book/src/fast-start.md`
- `book/src/target-generation.md`
- `crates/ringgrid/README.md`
- `crates/ringgrid-py/README.md`
- `CHANGELOG.md`

## Summary
- Added a new Rust CLI subcommand, `ringgrid gen-target`, that generates canonical `board_spec.json` plus printable SVG/PNG from direct geometry arguments.
- Matched the dedicated Python tool contract on geometry/output options and artifact naming, including the same default output directory and basename.
- Added fixture-based CLI regression tests that prove Rust CLI target generation is equivalent to the committed canonical target-generation artifacts.
- Updated the root README, mdBook, Rust crate README, Python package README, and changelog so Rust API, Rust CLI, and Python script generation are documented as equivalent paths.

## Decisions Made
- Kept the CLI implementation thin by delegating all rendering and file-writing work to the existing `ringgrid` library methods.
- Preserved Python-script-compatible underscore flag names on the Rust CLI and added hyphenated aliases for users who expect the rest of the CLI's style.
- Documented this work under `CHANGELOG.md` as `[Unreleased]` because the feature is implemented but not yet released/tagged.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-cli/src/main.rs` - added `gen-target`, command handler, and parity-focused CLI tests.
- `README.md` - updated repo quick-start guidance to show Rust CLI, Python script, and Rust API target generation.
- `book/src/cli-guide.md` - documented `ringgrid gen-target`.
- `book/src/fast-start.md` - updated fast-start to present the three equivalent generation paths.
- `book/src/target-generation.md` - rewrote the chapter around Rust CLI, Python script, and Rust API equivalence.
- `crates/ringgrid/README.md` - aligned Rust crate target-generation docs with the new CLI and current Python script path.
- `crates/ringgrid-py/README.md` - linked the Python target-generation section to the equivalent Rust CLI and Rust API surfaces.
- `CHANGELOG.md` - added an `[Unreleased]` section for the new CLI target-generation command and doc alignment.

## Validation / Tests
- Commands run:
  - `cargo test -p ringgrid-cli gen_target -- --nocapture`
  - `cargo test -p ringgrid-cli calibration_file_loads_direct_camera_model_shape -- --nocapture`
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
  - `./.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
  - `git diff --check`
- Results:
  - Focused Rust CLI target-generation tests passed, including fixture parity, generated-name behavior, invalid geometry/option rejection, and Python-style flag parsing.
  - `cargo fmt --all --check` passed after formatting the CLI changes.
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
  - `cargo test --workspace --all-features` passed with the workspace green, including `158` `ringgrid` unit tests, `4` Rust target-generation integration tests, `13` CLI tests, and `5` doctests.
  - `cargo doc --workspace --all-features --no-deps` and `cargo test --doc --workspace` passed.
  - `mdbook build book` passed after the docs rewrite.
  - Python checks passed: typing artifacts are current, `maturin develop` succeeded, `pytest crates/ringgrid-py/tests -q` passed with `31 passed`, and `pytest tools/tests/test_gen_target.py -q` passed with `3 passed`.
  - `git diff --check` passed.

## Risks / Open Questions
- The new feature is documented in `[Unreleased]` only.
  - Impact: if maintainers want this included in the next tagged release, release notes and tag timing should be handled in a separate release task.

## Next Handoff
- To: `Reviewer`
- Requested action: verify that `ringgrid gen-target` is truly a thin CLI over the Rust generation engine, confirm the fixture parity coverage is adequate for the equivalence claim, and review the doc updates for consistency across Rust API, Rust CLI, and Python script guidance.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1: added `ringgrid gen-target` with Python-tool-compatible flags and implemented fixture-based CLI tests in `crates/ringgrid-cli/src/main.rs`.
- Architect step 2: updated the README, mdBook, Rust crate README, Python package README, and changelog so the three target-generation paths use the same geometry/output story.
- Architect step 3: ran the full validation baseline plus the dedicated Python `tools/gen_target.py` test suite.

### Changes Made
- Added a new `GenTarget` CLI subcommand and `CliGenTargetArgs` for direct board geometry and print options.
- Implemented `run_gen_target(...)`, which constructs `BoardLayout` from geometry and writes `board_spec.json`, SVG, and PNG using the existing Rust writer methods.
- Kept underscore long flags (`--pitch_mm`, `--long_row_cols`, `--margin_mm`, etc.) for parity with `tools/gen_target.py`, while also accepting hyphenated aliases.
- Added CLI regression tests for:
  - Python-style flag parsing,
  - fixture parity for JSON/SVG/PNG output,
  - deterministic generated-name behavior,
  - invalid geometry, margin, and DPI rejection.
- Updated the docs to make Rust API, Rust CLI, and Python script target generation explicit peer workflows.

### Files Changed
- `crates/ringgrid-cli/src/main.rs` - added `gen-target`, its handler, and new tests.
- `README.md` - repo quick-start now presents Rust CLI, Python script, and Rust API generation side by side.
- `book/src/cli-guide.md` - added `ringgrid gen-target` command docs.
- `book/src/fast-start.md` - added the three equivalent generation paths to the first-run path.
- `book/src/target-generation.md` - restructured the chapter around equivalent canonical generation paths and their shared options.
- `crates/ringgrid/README.md` - updated command-line examples and target-generation guidance.
- `crates/ringgrid-py/README.md` - tied the Python target-generation section to the equivalent Rust CLI and Rust API paths.
- `CHANGELOG.md` - added `[Unreleased]` documentation for the new CLI command and docs alignment.

### Deviations From Plan
- None.

### Tests Added/Updated
- `crates/ringgrid-cli/src/main.rs`
  - `gen_target_subcommand_parses_python_style_flags`
  - `gen_target_writes_committed_fixture_outputs`
  - `gen_target_uses_generated_name_when_name_is_omitted`
  - `gen_target_rejects_invalid_geometry_and_options`

### Commands Run
- `cargo test -p ringgrid-cli gen_target -- --nocapture`
- `cargo test -p ringgrid-cli calibration_file_loads_direct_camera_model_shape -- --nocapture`
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
- `./.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
- `git diff --check`

### Results
- All required Rust, docs, mdBook, and Python validation commands passed.
- The new Rust CLI target-generation tests passed and lock the equivalence claim against the committed canonical fixture set.

### Remaining Concerns
- None beyond the normal release/bookkeeping decision for when to ship the feature.

### Handoff To Reviewer
- Focus on the new `gen-target` CLI contract, especially the parity between its flags/outputs and `tools/gen_target.py`.
- Verify that the documentation now gives users a clear, non-conflicting story for Rust API, Rust CLI, and Python script target generation.
