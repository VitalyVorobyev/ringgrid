# Architect Report - TASK-018-rust-cli-target-generation-equivalence

- Task ID: `TASK-018-rust-cli-target-generation-equivalence`
- Backlog ID: `n/a`
- Role: `architect`
- Date: `2026-03-12`
- Status: `ready_for_implementer`

## Inputs Consulted
- User request in this thread
- `crates/ringgrid-cli/src/main.rs`
- `crates/ringgrid/src/board_layout.rs`
- `crates/ringgrid/src/target_generation.rs`
- `crates/ringgrid/tests/target_generation.rs`
- `tools/gen_target.py`
- `tools/tests/test_gen_target.py`
- `README.md`
- `book/src/cli-guide.md`
- `book/src/target-generation.md`
- `book/src/fast-start.md`
- `crates/ringgrid/README.md`
- `crates/ringgrid-py/README.md`

## Summary
- The Rust library already supports canonical target JSON plus printable SVG/PNG, but the shipped Rust CLI does not expose that surface.
- The Python script `tools/gen_target.py` is the current dedicated command-line path and defines the practical target-generation contract that users see today.
- The requested change should add an additive Rust CLI subcommand that mirrors the Python tool's geometry/options and writes the same artifact set.
- Documentation must present Rust API, Rust CLI, and Python script as equivalent target-generation paths with the same geometry inputs and output expectations.

## Decisions Made
- Add a new Rust CLI subcommand rather than changing existing commands.
  - Reason: this is additive and keeps current detection workflows stable.
- Match the `tools/gen_target.py` contract as closely as possible in flag names, defaults, and generated files.
  - Reason: users asked for equivalence, and the Python tool is already the shipped command-line reference path.
- Keep the Rust library API unchanged except for examples/docs.
  - Reason: the core file-oriented generation API already exists and does not need expansion to satisfy the request.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-cli/src/main.rs` - add `gen-target` subcommand, command handler, and CLI tests.
- `crates/ringgrid/tests/fixtures/target_generation/*` - reuse the committed fixture set as the shared parity oracle.
- `README.md` - advertise the three equivalent target-generation paths.
- `crates/ringgrid/README.md` - align Rust API examples and generated artifact naming with the shared target-generation story.
- `crates/ringgrid-py/README.md` - keep Python installed-package path aligned with the same geometry/output contract.
- `book/src/cli-guide.md` - document `ringgrid gen-target`.
- `book/src/target-generation.md` - present Rust CLI, Rust API, and Python script as equivalent options.
- `book/src/fast-start.md` - add or route users to the new Rust CLI quick path.

## Validation / Tests
- Commands run:
  - Not run in architect stage.
- Results:
  - Not run in architect stage.

## Risks / Open Questions
- Exact text output from the Rust CLI does not need to be byte-for-byte identical to `gen_target.py`, but the generated artifacts do.
  - Impact: tests should lock JSON/SVG/PNG parity, while docs can describe human-readable console output more loosely.
- Rust CLI `PathBuf` parsing and Rust API constructor validation may emit slightly different error strings than the Python wrapper.
  - Impact: keep semantics aligned and test the important failure classes rather than overfitting exact wording across languages.

## Next Handoff
- To: `Implementer`
- Requested action: add the Rust CLI target-generation command with parity-focused tests, then update the docs so all three generation paths are clearly equivalent and easy to follow.

---

## Architect Required Sections

### Problem Statement
- Users can already generate canonical target files from the Rust library and via Python tooling, but the Rust CLI lacks a dedicated target-generation command.
- Documentation currently routes users primarily through Python tooling, which makes the Rust CLI look incomplete even though the underlying Rust engine already supports the same artifact generation.
- The project needs one coherent target-generation story where Rust API, Rust CLI, and Python script all describe the same board geometry inputs and the same output artifact set.

### Scope
- In scope:
  - Add `ringgrid gen-target` to the Rust CLI.
  - Mirror the dedicated Python tool contract for geometry, output directory, base filename, DPI, margin, and scale-bar toggle.
  - Write `board_spec.json`, `<basename>.svg`, and `<basename>.png` from the Rust CLI in one run.
  - Add parity-focused tests that compare Rust CLI outputs to the committed target-generation fixtures.
  - Update user-facing docs so Rust API, Rust CLI, and Python script generation paths are clearly documented as equivalent.
- Out of scope:
  - Replacing or removing `tools/gen_target.py`.
  - Changing the canonical target JSON schema or SVG/PNG rendering logic.
  - Adding new public target-generation APIs to the Rust crate unless a small helper is strictly needed for CLI reuse.

### Constraints
- Keep crate boundaries intact: file generation logic stays in `ringgrid`, CLI argument parsing and command orchestration stay in `ringgrid-cli`.
- Preserve current target-generation artifact semantics and defaults.
- Prefer matching `tools/gen_target.py` flag names and defaults to minimize mental context switching for users.
- Reuse the committed target-generation fixtures as the shared oracle for artifact parity.

### Assumptions
- CLI subcommand naming should be `gen-target` to match the existing Python script naming.
- Artifact equivalence means identical JSON/SVG text and identical PNG pixel content plus DPI metadata for the same geometry/options.
- The docs can present the three paths side-by-side even if the exact code snippets differ by language.

### Affected Areas
- `crates/ringgrid-cli/src/main.rs` - new CLI surface, file-writing flow, and tests.
- `crates/ringgrid/tests/fixtures/target_generation/*` - shared output oracle for parity tests.
- `README.md` - top-level user routing and quick-start examples.
- `crates/ringgrid/README.md` - Rust API generation examples and terminology.
- `crates/ringgrid-py/README.md` - Python target-generation section alignment.
- `book/src/cli-guide.md` - new CLI command reference.
- `book/src/target-generation.md` - three-path equivalence guidance.
- `book/src/fast-start.md` - quick-start path for Rust CLI and/or explicit comparison.

### Plan
1. Add `ringgrid gen-target` with Python-tool-compatible flags and write-path behavior, then cover it with fixture-based CLI tests.
2. Update README and mdBook pages so Rust API, Rust CLI, and Python script target-generation paths use the same geometry, defaults, artifact names, and output descriptions.
3. Run the full validation baseline and produce reviewer-ready handoff reports.

### Acceptance Criteria
- `ringgrid gen-target` exists and accepts the same core geometry/output flags as `tools/gen_target.py`:
  - `--pitch_mm`
  - `--rows`
  - `--long_row_cols`
  - `--marker_outer_radius_mm`
  - `--marker_inner_radius_mm`
  - `--name`
  - `--out_dir`
  - `--basename`
  - `--dpi`
  - `--margin_mm`
  - `--no-scale-bar`
- For the committed compact fixture geometry, Rust CLI generation produces canonical `board_spec.json`, SVG, and PNG outputs equivalent to the committed fixture set.
- Rust CLI generation creates parent directories and validates invalid margin/DPI/geometry inputs coherently.
- Docs clearly explain all three supported target-generation paths:
  - Rust API
  - Rust CLI app
  - Python script / Python package path
- Examples across the docs use aligned geometry and artifact naming so users can move between the three paths without translation work.

### Test Plan
- Add Rust CLI unit/integration-style tests in `crates/ringgrid-cli/src/main.rs` for:
  - committed fixture parity,
  - deterministic generated-name behavior,
  - invalid geometry/option rejection.
- Reuse the existing Rust target-generation fixture set under `crates/ringgrid/tests/fixtures/target_generation/`.
- Re-run existing Python tool tests to ensure the documented equivalence story remains true:
  - `./.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
- Full local baseline:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q`

### Out Of Scope
- Deprecating Python-based generation.
- Adding a separate Rust CLI command for JSON-only board generation.
- Broader CLI redesign beyond the new target-generation subcommand and related docs.

### Handoff To Implementer
1. Add a `gen-target` subcommand to `ringgrid-cli` that constructs `BoardLayout` from direct geometry and writes JSON/SVG/PNG using the existing Rust target-generation API.
2. Keep its flags and defaults aligned with `tools/gen_target.py` wherever feasible.
3. Add deterministic CLI tests using the committed compact target-generation fixtures and negative cases for invalid geometry/options.
4. Update the root README, crate READMEs, and mdBook pages so Rust API, Rust CLI, and Python script generation are presented as equivalent workflows with aligned examples.
5. Run the full validation baseline and record the results for review.
