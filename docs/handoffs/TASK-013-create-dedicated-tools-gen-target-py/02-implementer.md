# Implementer Report - TASK-013-create-dedicated-tools-gen-target-py

- Task ID: `TASK-013-create-dedicated-tools-gen-target-py`
- Backlog ID: `INFRA-012`
- Role: `implementer`
- Date: `2026-03-10`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/01-architect.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/implementer/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/tests-synthetic-fixtures/SKILL.md`
- `tools/gen_synth.py`
- `tools/gen_board_spec.py`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid/tests/target_generation.rs`
- `README.md`
- `book/src/fast-start.md`
- `book/src/target-generation.md`

## Summary
- Added `tools/gen_target.py` as a thin repo-root CLI that constructs a board with `ringgrid.BoardLayout.from_geometry(...)` and emits `board_spec.json`, SVG, and PNG through the existing installed-package generation methods.
- Added deterministic subprocess coverage in `tools/tests/test_gen_target.py` for fixture parity, generated-name behavior, parent-directory creation, and invalid geometry/option failures.
- Updated the root README and mdbook target-generation pages so `gen_target.py` is the dedicated JSON/SVG/PNG path, while `gen_synth.py` remains the synth/ground-truth workflow.
- Re-ran the full required validation baseline plus the architect-specific tool checks and a manual smoke run; all commands passed.

## Decisions Made
- Kept `tools/gen_target.py` as a pure Python wrapper over the shipped `ringgrid` package surface.
  - Reason: the architect plan explicitly scoped this task as a repo-level UX layer, not a new Rust or PyO3 feature.
- Required the full direct-geometry argument set (`pitch_mm`, `rows`, `long_row_cols`, outer radius, inner radius`) and did not reintroduce `board_mm` or derived radius defaults.
  - Reason: that is the locked backlog contract for `INFRA-012`, and it avoids reviving the older synth-oriented geometry semantics in the dedicated tool.
- Fixed the JSON output name to `board_spec.json` and used `--basename` only for SVG/PNG.
  - Reason: it preserves the current docs/artifact naming convention while keeping the CLI small.
- Added an explicit import error path for missing `ringgrid` installs.
  - Reason: the new script depends on the local Python binding being installed into the active environment, and a raw `ModuleNotFoundError` would be too opaque for the intended repo workflow.

## Files/Modules Affected (Or Expected)
- `tools/gen_target.py` - new direct-geometry CLI that emits canonical JSON plus printable SVG/PNG through the existing Python target-generation API.
- `tools/tests/test_gen_target.py` - deterministic subprocess-level tests against the committed compact target-generation fixtures and error cases.
- `README.md` - fast-start target-generation section now points print-only users to `gen_target.py` and keeps `gen_synth.py` for synth workflows.
- `book/src/fast-start.md` - fast-start flow now uses `gen_target.py` and explains the local binding prerequisite.
- `book/src/target-generation.md` - generation-path overview, dedicated-tool examples, and flag reference now cover `gen_target.py` alongside `gen_synth.py` and `gen_board_spec.py`.
- `crates/ringgrid-py/python/ringgrid/_api.py` - unchanged dependency surface consumed by the new tool.
- `crates/ringgrid/tests/fixtures/target_generation/*` - reused unchanged as the deterministic parity baseline.

## Validation / Tests
- Commands run:
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k target_generation`
  - `cargo fmt --all --check`
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `mdbook build book`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
  - `cargo test -p ringgrid --test target_generation`
  - `.venv/bin/python tools/gen_target.py --pitch_mm 8 --rows 3 --long_row_cols 4 --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 --name fixture_compact_hex --dpi 96 --out_dir /tmp/ringgrid_target_smoke`
- Results:
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release` passed and reinstalled the local editable `ringgrid` package for CPython 3.14.
  - `.venv/bin/python -m pytest tools/tests/test_gen_target.py -q` passed (`3 passed`).
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k target_generation` passed (`3 passed`).
  - `cargo fmt --all --check` passed.
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check` passed (`typing artifacts are up to date`).
  - `mdbook build book` passed.
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
  - `cargo test --workspace --all-features` passed (`143` Rust unit tests, `4` target-generation integration tests, `6` CLI tests, and workspace doctests all green).
  - `cargo doc --workspace --all-features --no-deps` passed.
  - `cargo test --doc --workspace` passed (`5` doctests).
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q` passed (`28 passed`).
  - `cargo test -p ringgrid --test target_generation` passed (`4 passed`).
  - Manual smoke run of `tools/gen_target.py` passed and wrote the three expected artifacts under `/tmp/ringgrid_target_smoke`.

## Risks / Open Questions
- `tools/gen_target.py` depends on the active Python environment already having `ringgrid` installed. The script now fails with an explicit setup message, but this remains a runtime dependency users must satisfy.
- The fast-start docs for repo checkouts now assume a local `maturin develop` step. That is intentional for the dedicated script, but users who only want synth datasets can still use the lighter `gen_synth.py` path with NumPy.
- The new tool is intentionally opinionated: it always writes JSON, SVG, and PNG together. If users later need partial-output modes, that should be handled as a separate follow-up rather than expanding this initial script ad hoc.

## Next Handoff
- To: `Reviewer`
- Requested action: verify that `tools/gen_target.py` stays thin over the existing Python target-generation API, confirm the subprocess tests lock parity against the committed fixtures without duplicating rendering logic, and check that the README/book updates accurately distinguish `gen_target.py`, `gen_synth.py`, and `gen_board_spec.py`.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1: implemented `tools/gen_target.py` as a narrow `argparse` wrapper over `ringgrid.BoardLayout.from_geometry(...)`, `to_spec_json(...)`, `write_svg(...)`, and `write_png(...)`.
- Architect step 2: added deterministic black-box tool tests under `tools/tests/test_gen_target.py` using the committed compact target-generation fixture geometry and explicit print options.
- Architect step 3: updated `README.md`, `book/src/fast-start.md`, and `book/src/target-generation.md` to introduce the dedicated tool while preserving `gen_synth.py` as the synth-specific path.
- Added one extra low-risk check inside step 2 for omitted-name behavior because the architect assumptions explicitly called out deterministic default naming.

### Changes Made
- Added `tools/gen_target.py`.
  - Requires direct board geometry args plus optional `--name`, `--out_dir`, `--basename`, `--dpi`, `--margin_mm`, and `--no-scale-bar`.
  - Creates the output directory if needed.
  - Writes `board_spec.json`, `<basename>.svg`, and `<basename>.png`.
  - Prints the emitted artifact paths and board summary on success.
  - Reports a clear setup message when `ringgrid` is not installed in the active environment.
- Added `tools/tests/test_gen_target.py`.
  - Runs the new script as a subprocess from the repo root.
  - Verifies JSON/SVG text parity against `fixture_compact_hex`.
  - Verifies PNG `pHYs` metadata and decoded grayscale pixels against the committed PNG fixture.
  - Verifies omitted-name generation uses the deterministic geometry-derived name.
  - Verifies invalid geometry and invalid margin fail with nonzero exit status.
- Updated `README.md`.
  - Swapped the print-only fast-start from `gen_synth.py` to `gen_target.py`.
  - Documented the local binding install step and the dedicated-script flag set.
  - Kept an explicit note pointing synth/ground-truth users back to `gen_synth.py`.
- Updated `book/src/fast-start.md` and `book/src/target-generation.md`.
  - Added the local binding prerequisite for the dedicated script.
  - Reframed the overview to three generation paths: `gen_target.py`, `gen_synth.py`, `gen_board_spec.py`.
  - Added the dedicated script’s flag reference and examples.

### Files Changed
- `tools/gen_target.py` - new direct-geometry target-generation CLI.
- `tools/tests/test_gen_target.py` - new deterministic subprocess test suite for the CLI.
- `README.md` - fast-start now uses the dedicated tool for JSON/SVG/PNG generation.
- `book/src/fast-start.md` - mdbook fast-start updated to the new dedicated path.
- `book/src/target-generation.md` - mdbook target-generation reference updated for the dedicated tool and tool-selection guidance.
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/02-implementer.md` - implementation handoff report for review.

### Deviations From Plan
- None.
- The implementation stayed within the architected scope and did not require any Rust or PyO3 changes.

### Tests Added/Updated
- `tools/tests/test_gen_target.py`
  - `test_gen_target_matches_committed_fixture_outputs` locks JSON/SVG/PNG parity for the compact fixture geometry (`8.0`, `3`, `4`, `4.8`, `3.2`, `name=fixture_compact_hex`, `dpi=96`, `margin=0`).
  - `test_gen_target_uses_generated_name_when_name_is_omitted` locks the deterministic default-name contract.
  - `test_gen_target_rejects_invalid_geometry_and_options` locks explicit CLI failure for invalid rows and negative margin.

### Commands Run
- `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
- `.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
- `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k target_generation`
- `cargo fmt --all --check`
- `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
- `mdbook build book`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo doc --workspace --all-features --no-deps`
- `cargo test --doc --workspace`
- `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- `cargo test -p ringgrid --test target_generation`
- `.venv/bin/python tools/gen_target.py --pitch_mm 8 --rows 3 --long_row_cols 4 --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 --name fixture_compact_hex --dpi 96 --out_dir /tmp/ringgrid_target_smoke`

### Results
- The dedicated script emits the same compact fixture JSON/SVG/PNG artifacts as the shared target-generation engine for identical inputs.
- The CLI creates missing output directories and surfaces invalid inputs as nonzero errors.
- The docs now expose `gen_target.py` as the dedicated JSON/SVG/PNG path without collapsing `gen_synth.py` and `gen_board_spec.py` into one story.
- The full required validation baseline passed after the changes.

### Remaining Concerns
- The dedicated script intentionally assumes the local Python binding is installed into the active environment; the improved import error message and updated docs reduce that friction but do not remove the dependency.
- Because this task stayed thin, the script does not offer partial-output modes or DXF; any expansion there should be a separate backlog item.

### Handoff To Reviewer
- Focus on `tools/gen_target.py` to confirm it is only CLI glue and does not duplicate geometry/rendering semantics from `gen_synth.py`.
- Focus on `tools/tests/test_gen_target.py` to confirm the fixture-based subprocess coverage is deterministic and sufficiently strict on JSON/SVG/PNG parity.
- Focus on `README.md`, `book/src/fast-start.md`, and `book/src/target-generation.md` to confirm the docs now clearly route users to the right generation tool for their workflow.
