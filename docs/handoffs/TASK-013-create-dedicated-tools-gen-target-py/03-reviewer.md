# Reviewer Report - TASK-013-create-dedicated-tools-gen-target-py

- Task ID: `TASK-013-create-dedicated-tools-gen-target-py`
- Backlog ID: `INFRA-012`
- Role: `reviewer`
- Date: `2026-03-10`
- Status: `complete`

## Inputs Consulted
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/01-architect.md`
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/02-implementer.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/reviewer/SKILL.md`
- `tools/gen_target.py`
- `tools/tests/test_gen_target.py`
- `README.md`
- `book/src/fast-start.md`
- `book/src/target-generation.md`
- `crates/ringgrid/tests/target_generation.rs`
- implementer-recorded validation evidence in `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/02-implementer.md`

## Summary
- Reviewed the `INFRA-012` implementation against the architected scope of a thin repo-root CLI wrapper, deterministic subprocess parity tests, and docs updates for the dedicated target-generation path.
- Confirmed the new `tools/gen_target.py` delegates all target generation to the existing Python `BoardLayout` surface and does not duplicate geometry/rendering logic from `gen_synth.py`.
- Confirmed the dedicated tool tests lock JSON/SVG/PNG parity against the committed compact fixture baseline and cover generated-name plus explicit CLI-failure behavior.
- Reproduced the highest-risk checks (`tools/tests/test_gen_target.py`, `cargo test -p ringgrid --test target_generation`, `mdbook build book`) and spot-checked direct invalid-DPI CLI failure behavior; no blocking issues found.

## Decisions Made
- Accepted the implementer’s choice to keep `tools/gen_target.py` as CLI glue only.
  - Reason: it matches the architect’s “thin wrapper” constraint and preserves `INFRA-010`/`INFRA-011` as the single source of target-generation semantics.
- Accepted the doc updates that move print-only fast-start guidance from `gen_synth.py` to `gen_target.py`.
  - Reason: they close the exact discoverability gap this task was created to solve while still keeping `gen_synth.py` documented for synth workflows.
- Treated the full CI baseline as sufficiently evidenced by the implementer report and reproduced the high-risk target-generation/doc checks rather than rerunning every baseline command.
  - Reason: the changed surface is tightly scoped to one Python tool, its tests, and user docs; the reproduced checks cover the riskier behavioral paths directly.

## Files/Modules Reviewed
- `tools/gen_target.py` - CLI contract, error handling, and delegation to the installed-package `ringgrid` API.
- `tools/tests/test_gen_target.py` - deterministic subprocess parity coverage and negative cases.
- `README.md` - repo-root fast-start workflow and tool-selection messaging.
- `book/src/fast-start.md` - mdbook fast-start flow for dedicated target generation.
- `book/src/target-generation.md` - overall tool-selection/reference chapter for JSON/SVG/PNG generation.
- `crates/ringgrid/tests/target_generation.rs` - shared Rust fixture contract that the dedicated tool must remain aligned with.

## Validation / Tests
- Commands reviewed from implementer evidence:
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
- Commands reproduced during review:
  - `.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
  - `cargo test -p ringgrid --test target_generation`
  - `mdbook build book`
  - `.venv/bin/python tools/gen_target.py --pitch_mm 8 --rows 3 --long_row_cols 4 --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 --dpi 0 --out_dir /tmp/ringgrid_target_bad_dpi`
- Results:
  - The implementer’s recorded baseline is coherent and includes every required command from the reviewer workflow.
  - Reproduced tool tests passed (`3 passed`).
  - Reproduced Rust target-generation integration tests passed (`4 passed`).
  - Reproduced `mdbook build book` passed.
  - Direct invalid-DPI CLI reproduction failed cleanly with exit code `1` and an explicit validation message (`dpi must be finite and > 0`), matching the architected error-contract expectations.

## Risks / Open Questions
- `tools/gen_target.py` still depends on the active environment already having the local `ringgrid` binding installed. That is documented and the script now fails clearly, but it remains an operational prerequisite rather than a self-contained tool.
- The new tool intentionally writes JSON, SVG, and PNG together. That is consistent with the architected scope, but future requests for partial-output modes should be handled as a separate task to keep this CLI stable.

## Next Handoff
- To: `Human`
- Requested action: close `TASK-013-create-dedicated-tools-gen-target-py` / `INFRA-012` and proceed with the next backlog item.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the implementation against the architect plan’s three deliverables:
  - thin repo-root `tools/gen_target.py` wrapper over the existing Python target-generation API,
  - deterministic subprocess parity/error tests,
  - README/mdbook updates that make the dedicated tool discoverable without replacing synth workflows.
- Focused on correctness of CLI behavior, parity against the committed fixture contract, clarity of failure modes, and documentation consistency.

### Inputs Reviewed
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/01-architect.md`
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/02-implementer.md`
- changed files in `tools/`, `tools/tests/`, `README.md`, and `book/src/`
- existing shared target-generation fixture contract in `crates/ringgrid/tests/target_generation.rs`

### What Was Checked
- `tools/gen_target.py` uses `ringgrid.BoardLayout.from_geometry(...)`, `to_spec_json(...)`, `write_svg(...)`, and `write_png(...)` rather than reimplementing generation logic.
- The CLI surface matches the architected direct-geometry contract and output naming policy.
- Tool tests exercise fixture parity, deterministic default naming, and explicit invalid-input failure.
- Docs now distinguish:
  - `gen_target.py` for dedicated JSON/SVG/PNG generation,
  - `gen_synth.py` for synth datasets and optional print output,
  - `gen_board_spec.py` for JSON-only output.
- The implementer’s validation record covers the full required baseline, and the highest-risk checks are reproducible.

### Findings
- No findings.

### Test Assessment
- Adequate.
- The new subprocess tests are appropriately black-box for this task and are strict on the architect’s core acceptance surface:
  - canonical JSON parity,
  - SVG text parity,
  - PNG metadata/pixel parity,
  - generated-name behavior,
  - explicit invalid CLI failure.
- Review-time reproduction confirmed the dedicated tool remains aligned with the shared Rust fixture contract and that the docs still build after the workflow changes.

### Risks
- Residual risk: the tool’s dependency on an installed local binding may still trip up contributors who skip the setup step, though the docs and runtime error are now explicit.
- Residual risk: the dedicated CLI is intentionally narrow and does not expose partial-output modes; that is a product-scope constraint, not an implementation defect.

### Required Changes Or Approval Notes
- Approved as implemented.
- Non-blocking follow-up note: if maintainers later want `tools/gen_target.py` to work without a preinstalled local binding, that should be handled as a separate packaging/bootstrap task rather than expanding this thin wrapper.

### Final Verdict
- `approved`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: mark `INFRA-012` complete in the backlog and continue to the next queued item.
