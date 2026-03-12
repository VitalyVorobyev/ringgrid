# Reviewer Report - TASK-018-rust-cli-target-generation-equivalence

- Task ID: `TASK-018-rust-cli-target-generation-equivalence`
- Backlog ID: `n/a`
- Role: `reviewer`
- Date: `2026-03-12`
- Status: `complete`

## Inputs Consulted
- `docs/handoffs/TASK-018-rust-cli-target-generation-equivalence/01-architect.md`
- `docs/handoffs/TASK-018-rust-cli-target-generation-equivalence/02-implementer.md`
- `crates/ringgrid-cli/src/main.rs`
- `README.md`
- `book/src/cli-guide.md`
- `book/src/fast-start.md`
- `book/src/target-generation.md`
- `crates/ringgrid/README.md`
- `crates/ringgrid-py/README.md`
- `CHANGELOG.md`
- `git diff --check`
- validation command output from the current tree

## Summary
- The new Rust CLI target-generation command is additive, thin, and correctly delegates to the existing Rust target-generation engine.
- The new CLI tests are appropriately tied to the committed canonical target-generation fixture set, which makes the equivalence claim concrete instead of descriptive only.
- The docs now present Rust API, Rust CLI, and Python script generation as parallel surfaces over the same geometry/output contract.
- No blocking issues were found.

## Decisions Made
- Accept the use of Python-script-compatible underscore flags on the Rust CLI, with hyphenated aliases retained for CLI ergonomics.
  - Reason: this satisfies the user's equivalence requirement without weakening the overall CLI usability.
- Accept `[Unreleased]` changelog documentation without forcing a release/tag in this task.
  - Reason: the task request was feature implementation and documentation, not release execution.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-cli/src/main.rs` - reviewed subcommand shape, handler implementation, and CLI tests.
- `README.md` - reviewed top-level user routing across the three target-generation paths.
- `book/src/cli-guide.md` - reviewed `gen-target` command documentation.
- `book/src/fast-start.md` - reviewed first-run guidance for the three paths.
- `book/src/target-generation.md` - reviewed the detailed equivalence story and shared option reference.
- `crates/ringgrid/README.md` - reviewed Rust API and command-line guidance alignment.
- `crates/ringgrid-py/README.md` - reviewed Python-side equivalence notes.
- `CHANGELOG.md` - reviewed unreleased feature documentation.

## Validation / Tests
- Commands run:
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
  - All required Rust, docs, mdBook, and Python validation commands passed on the current tree.
  - `cargo test --workspace --all-features` passed, including the new `ringgrid-cli` target-generation tests and the pre-existing Rust target-generation integration tests.
  - `pytest tools/tests/test_gen_target.py -q` still passes, so the Python tool contract remained stable while the Rust CLI was added.
  - `git diff --check` passed.

## Risks / Open Questions
- The CLI and docs now make a strong equivalence claim across three surfaces.
  - Impact: future target-generation changes should update the Rust CLI tests and the Python tool tests together to avoid drift.

## Next Handoff
- To: `Human`
- Requested action: merge or commit the approved feature when ready; release separately if you want it included in the next version.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the addition of `ringgrid gen-target` and the related documentation changes that align Rust API, Rust CLI, and Python script target generation.

### Inputs Reviewed
- `docs/handoffs/TASK-018-rust-cli-target-generation-equivalence/01-architect.md`
- `docs/handoffs/TASK-018-rust-cli-target-generation-equivalence/02-implementer.md`
- The code and docs listed above.
- Reproduced the full local validation baseline plus the dedicated Python `tools/gen_target.py` tests.

### What Was Checked
- CLI correctness for geometry parsing, output-path construction, and delegation to the Rust writer methods.
- Coverage quality for the new fixture-based CLI tests.
- Documentation consistency across the three target-generation surfaces.
- Coherence of the validation evidence on the final tree.

### Findings
- No blocking findings.

### Test Assessment
- Adequate. The Rust CLI tests lock parser behavior, deterministic name generation, invalid-option rejection, and committed fixture parity. Existing Rust target-generation integration tests and Python tool tests remain green.

### Risks
- Future target-generation feature work could drift between Rust CLI docs, Python script docs, and fixture tests if not kept together.
  - Mitigation: continue treating the committed target-generation fixture set and `tools/tests/test_gen_target.py` as the shared contract oracle.

### Required Changes Or Approval Notes
- Approval note: no additional implementation work is required for this task.

### Final Verdict
- `approved`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: commit or merge the approved feature, and release it later if desired.
