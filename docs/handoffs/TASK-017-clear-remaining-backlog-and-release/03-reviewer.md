# Reviewer Report - TASK-017-clear-remaining-backlog-and-release

- Task ID: `TASK-017-clear-remaining-backlog-and-release`
- Backlog ID: `INFRA-008, ALGO-009, ALGO-010, ALGO-011, INFRA-013`
- Role: `reviewer`
- Date: `2026-03-11`
- Status: `complete`

## Inputs Consulted
- `docs/handoffs/TASK-017-clear-remaining-backlog-and-release/01-architect.md`
- `docs/handoffs/TASK-017-clear-remaining-backlog-and-release/02-implementer.md`
- `git diff --stat`
- `git diff --check`
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
- The implementation matches the architect scope: additive CLI calibration loading, three bounded detector robustness changes, and release/doc bookkeeping.
- The changed detector paths stayed localized and reused existing math/contracts instead of introducing duplicate solvers or new user knobs.
- The full release validation baseline was reproduced on the current tree and passed cleanly.
- No blocking correctness or completeness issues were found in review.

## Decisions Made
- Accept the `ALGO-011` implementation that clears decoded identity instead of deleting markers.
  - Reason: it satisfies the architect intent of preventing bad correspondences from surviving while keeping the finalize change localized and low-risk.
- Treat the remaining release step as bookkeeping, not a code-review blocker.
  - Reason: the code/docs/test tree is ready; the only remaining action is to commit and tag the approved state.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-cli/src/main.rs` - reviewed additive CLI parsing path, shape compatibility, and conflict checks.
- `crates/ringgrid/src/detector/completion.rs` - reviewed local-affine seed path and fallback behavior.
- `crates/ringgrid/src/detector/id_correction/math.rs` - reviewed affine helper reuse boundary.
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` - reviewed expected-radius pre-screen behavior.
- `crates/ringgrid/src/pipeline/finalize.rs` - reviewed axis-ratio cleanup behavior and downstream metadata clearing.
- `CHANGELOG.md` / `docs/backlog.md` / `book` / `docs/tuning-guide.md` - reviewed release/documentation accuracy for the shipped feature set.

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
  - `git diff --check`
- Results:
  - `cargo fmt --all --check` passed.
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
  - `cargo test --workspace --all-features` passed with `158` library unit tests, `4` target-generation integration tests, `9` CLI unit tests, and `5` doctests green.
  - `cargo doc --workspace --all-features --no-deps` and `cargo test --doc --workspace` passed.
  - `mdbook build book` passed.
  - Python validation passed: typing artifacts up to date, `maturin develop` succeeded, and `pytest` passed with `31 passed in 2.78s`.
  - `git diff --check` passed with no whitespace or conflict-marker issues.

## Risks / Open Questions
- `docs/backlog.md` states that the local `v0.5.0` tag is part of closure.
  - Impact: the approved tree should be committed and tagged immediately so that release bookkeeping remains truthful.

## Next Handoff
- To: `Human`
- Requested action: cut the release commit and local annotated `v0.5.0` tag from this approved tree, then publish externally if desired.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the combined closure of `INFRA-008`, `ALGO-009`, `ALGO-010`, `ALGO-011`, and `INFRA-013` against the architect plan and the actual code/docs diff.

### Inputs Reviewed
- `docs/handoffs/TASK-017-clear-remaining-backlog-and-release/01-architect.md`
- `docs/handoffs/TASK-017-clear-remaining-backlog-and-release/02-implementer.md`
- The modified Rust, Python metadata, mdBook, changelog, and backlog files listed above.
- Reproduced release-baseline command output on the current tree.

### What Was Checked
- CLI correctness for calibration JSON loading, precedence, and mutual exclusion rules.
- Completion behavior for local-affine seeding and explicit homography fallback.
- Outer-fit contamination screening tied to expected radius.
- Finalize behavior for ratio-based decoded-marker cleanup and metadata consistency after clearing IDs.
- Documentation alignment with the shipped CLI and release scope.
- Validation completeness for the required Rust, mdBook, and Python release baseline.

### Findings
- No blocking findings.

### Test Assessment
- Adequate. The implementation includes targeted regression tests for each new behavior, and the full project release baseline was reproduced successfully.

### Risks
- Release closure still depends on committing and tagging this exact approved tree.
  - Mitigation: perform the release commit and create the annotated `v0.5.0` tag immediately after approval.

### Required Changes Or Approval Notes
- Approval note: no code or documentation rework is required before release.
- Minor follow-up: commit and tag the approved tree so the `INFRA-013` closure note remains accurate.

### Final Verdict
- `approved_with_minor_followups`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: create the release commit and annotated `v0.5.0` tag on the approved tree.
