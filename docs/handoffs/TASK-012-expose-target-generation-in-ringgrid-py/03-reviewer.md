# Reviewer Report - TASK-012-expose-target-generation-in-ringgrid-py

- Task ID: `TASK-012-expose-target-generation-in-ringgrid-py`
- Backlog ID: `INFRA-011`
- Role: `reviewer`
- Date: `2026-03-09`
- Status: `complete`

## Inputs Consulted
- previous `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/03-reviewer.md`
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/01-architect.md`
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/02-implementer.md`
- `.agents/skills/reviewer/SKILL.md`
- `docs/templates/task-handoff-report.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid-py/python/ringgrid/__init__.py`
- `crates/ringgrid-py/python/ringgrid/__init__.pyi`
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template`
- `crates/ringgrid-py/README.md`
- `README.md`

## Summary
- Re-reviewed `INFRA-011` after the prior `changes_requested` verdict on stale `_spec_json` use in the new Python target-generation methods.
- Verified that `BoardLayout.to_spec_json(...)`, `write_svg(...)`, and `write_png(...)` now normalize from current public spec fields and refresh the in-memory snapshot before emitting outputs.
- Verified the new mutation regression test closes the previously reproduced inconsistency between `to_spec_dict()` and emitted target-generation outputs.
- Reviewed the implementer’s full local CI baseline and reproduced the highest-risk Python checks during review.
- No blocking findings remain within the architected scope.

## Decisions Made
- Accepted the implementer’s chosen mutable-board contract for the new target-generation API.
  - Reason: it resolves the reviewer finding without broadening scope into a larger Python API redesign.
- Accepted the normalization approach in `BoardLayout._refresh_from_current_spec_fields()`.
  - Reason: it keeps the Rust engine authoritative, updates derived markers together with canonical spec JSON, and applies uniformly across `to_spec_json(...)`, `write_svg(...)`, and `write_png(...)`.

## Files/Modules Reviewed
- `crates/ringgrid-py/python/ringgrid/_api.py` - mutable-board normalization helper plus the new spec/SVG/PNG emission paths.
- `crates/ringgrid-py/tests/test_api.py` - fixture parity, error-contract coverage, and the new mutation regression.
- `crates/ringgrid-py/src/lib.rs` - unchanged Rust-backed bridge for canonical spec JSON and SVG/PNG generation.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - unchanged typing surface for the new public API.
- `crates/ringgrid-py/README.md` and `README.md` - installed-package target-generation docs updated in the original implementation.

## Validation / Tests
- Commands reviewed from implementer evidence:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- Commands reproduced during review:
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k 'target_generation or mutated_spec_fields'`
  - `.venv/bin/python - <<'PY' ... PY` manual mutation flow for `BoardLayout.write_svg(...)`, `BoardLayout.write_png(...)`, and `BoardLayout.to_spec_json(...)`
- Results:
  - Reviewed evidence shows the implementer completed the full required local CI baseline successfully.
  - `generate_typing_artifacts.py --check` passed (`typing artifacts are up to date`).
  - The focused Python regression slice passed (`3 passed`).
  - Manual mutation reproduction now behaves correctly:
    - mutated `name` and `pitch_mm` flow through emitted canonical spec JSON,
    - `write_svg(...)` refreshes marker positions,
    - `write_png(...)` refreshes marker positions and still writes PNG bytes.

## Risks / Open Questions
- `BoardLayout` remains a mutable public dataclass, and the broader detection/config flow still snapshots `board._spec_json` at core construction time. That is a pre-existing Python-surface sharp edge outside this task’s bounded target-generation fix.
- The book target-generation chapter still emphasizes the repo-tool workflow; this documentation drift remains outside the scoped deliverable for `INFRA-011`.

## Next Handoff
- To: `Human`
- Requested action: mark `TASK-012-expose-target-generation-in-ringgrid-py` complete and continue the backlog.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the rework against the architect acceptance criteria and the prior reviewer finding on stale `_spec_json` use in the new Python target-generation methods.
- Focused on correctness of the mutable-board emission contract, regression coverage, and coherence of the recorded local CI baseline.

### Inputs Reviewed
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/01-architect.md`
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/02-implementer.md`
- previous `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/03-reviewer.md`
- changed code in `crates/ringgrid-py/python/ringgrid/_api.py`
- changed tests in `crates/ringgrid-py/tests/test_api.py`

### What Was Checked
- The new target-generation API remains additive, `BoardLayout`-centered, and thin over the Rust engine.
- `BoardLayout._refresh_from_current_spec_fields()` rebuilds canonical spec JSON from current public spec fields and refreshes derived marker state.
- `to_spec_json(...)`, `write_svg(...)`, and `write_png(...)` all consume that refreshed canonical spec instead of stale cached `_spec_json`.
- The new mutation regression test exercises post-construction mutation and verifies emitted outputs follow the mutated board.
- The original fixture-parity and error-contract coverage still remains intact.
- The implementer’s recorded local CI baseline is coherent, and the highest-risk Python checks are reproducible.

### Findings
- No findings.

### Test Assessment
- Adequate.
- The prior gap is closed because the updated tests now cover:
  - canonical JSON parity on the nominal fixture path,
  - SVG/PNG parity on the nominal fixture path,
  - invalid option and write-error handling,
  - post-construction mutation of public board spec fields and its effect on emitted outputs.
- Review-time reproduction confirmed the shared normalization helper is exercised by the writer methods as well as `to_spec_json(...)`.

### Risks
- Residual risk: mutable `BoardLayout` semantics outside the target-generation path are still broader than this task addressed, especially around detector/config core construction from cached spec JSON.
- Residual risk: book docs still lag the installed-package target-generation story.

### Required Changes Or Approval Notes
- Approved as implemented.
- Non-blocking follow-up note: if Python `BoardLayout` mutability becomes a recurring source of confusion, a future task should either document or harden the broader mutation semantics across `DetectConfig`/`Detector`, not just target generation.

### Final Verdict
- `approved`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: close `TASK-012-expose-target-generation-in-ringgrid-py` / `INFRA-011` and proceed to the next backlog item.
