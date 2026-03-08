# Reviewer Report - TASK-010-rust-target-generation-api

- Task ID: `TASK-010-rust-target-generation-api`
- Role: `reviewer`
- Date: `2026-03-08`
- Status: `complete`

## Inputs Consulted
- previous `docs/handoffs/TASK-010-rust-target-generation-api/03-reviewer.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/01-architect.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/02-implementer.md`
- `.agents/skills/reviewer/SKILL.md`
- `docs/templates/task-handoff-report.md`
- `Cargo.toml`
- `crates/ringgrid/Cargo.toml`
- `crates/ringgrid/src/target_generation.rs`
- `crates/ringgrid/tests/target_generation.rs`
- `crates/ringgrid/README.md`

## Summary
- Re-reviewed the `TASK-010` implementation after the prior `changes_requested` verdict on the PNG writer path.
- Verified that `write_target_png` now uses an explicit PNG encoder and writes PNG bytes independently of the filename suffix.
- Verified that the written PNG now carries DPI-derived `pHYs` metadata and that the integration test asserts that contract directly.
- Reproduced the target-generation integration suite and the full `ringgrid` crate test suite on the revised tree.
- No blocking correctness, contract, or validation gaps remain within the architected scope.

## Decisions Made
- Accepted the explicit `png` crate encoder in the library implementation.
  - Reason: the prior `image.save(...)` path could not preserve the print-oriented PNG metadata contract required by the task.
- Accepted the “always write PNG bytes regardless of suffix” API contract.
  - Reason: it removes extension-driven ambiguity and is now explicitly enforced by the file-writer regression test.

## Files/Modules Reviewed
- `crates/ringgrid/src/target_generation.rs` - explicit PNG encoding path, error mapping, and writer contract.
- `crates/ringgrid/tests/target_generation.rs` - regression coverage for PNG signature, `pHYs` metadata, non-`.png` suffix behavior, and decoded pixel parity.
- `Cargo.toml` - workspace-level `png` dependency addition.
- `crates/ringgrid/Cargo.toml` - crate-level adoption of the new encoder dependency.
- `crates/ringgrid/README.md` - public note for PNG print metadata behavior.

## Validation / Tests
- Commands run:
  - `cargo test -p ringgrid --test target_generation`
  - `cargo test -p ringgrid`
- Results:
  - `cargo test -p ringgrid --test target_generation` passed (`4 passed`).
  - `cargo test -p ringgrid` passed (`143` unit tests, `4` integration tests, `5` doc tests).

## Risks / Open Questions
- No blocking risks found within the task scope.
- Residual note: PNG byte-for-byte output remains intentionally non-contractual, but the architect-required contracts are now covered at the format, metadata, dimension, and decoded-pixel levels.

## Next Handoff
- To: `Human`
- Requested action: mark `TASK-010-rust-target-generation-api` complete and continue backlog execution.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the `TASK-010` rework against the architect acceptance criteria and the prior reviewer finding on PNG file output semantics.
- Focused on the changed writer path, the added dependency, the strengthened integration test, and reproduced validation.

### Inputs Reviewed
- `docs/handoffs/TASK-010-rust-target-generation-api/01-architect.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/02-implementer.md`
- previous `docs/handoffs/TASK-010-rust-target-generation-api/03-reviewer.md`
- changed code in `crates/ringgrid/src/target_generation.rs`
- changed tests in `crates/ringgrid/tests/target_generation.rs`
- dependency updates in `Cargo.toml` and `crates/ringgrid/Cargo.toml`

### What Was Checked
- `write_target_png` now encodes through an explicit PNG writer instead of extension-driven `image.save(...)`.
- The writer sets PNG physical pixel dimensions from `PngTargetOptions::dpi`.
- The public contract choice for non-`.png` paths is explicit and covered by test.
- The file-writer regression test checks file-level PNG behavior rather than only decoded raster parity.
- The revised implementation still satisfies the broader task validation requirements via the full crate test suite.

### Findings
- No findings.

### Test Assessment
- Adequate.
- The prior gap is closed because the integration test now verifies:
  - PNG signature,
  - `pHYs`/pixel-dimension metadata,
  - PNG output from a non-`.png` suffix path,
  - decoded pixel parity against the committed fixture.
- Reproduced `cargo test -p ringgrid` also confirms the writer change did not regress the rest of the crate.

### Risks
- No blocking risks found.

### Required Changes Or Approval Notes
- Approved as implemented.

### Final Verdict
- `approved`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: close `TASK-010-rust-target-generation-api` and proceed to the next backlog item.
