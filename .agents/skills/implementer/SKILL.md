---
name: implementer
description: Execute an approved architecture handoff with minimal, production-ready code and tests, then prepare a reviewer-ready implementation report. Use when coding is required from a task plan. Require a mandatory task-id and write only docs/handoffs/<task-id>/02-implementer.md.
---

# Implementer

## Purpose
Implement the Architect plan with focused code changes, explicit validation, and a reviewer-ready handoff.

## Required Inputs
- `task-id` (mandatory). Format: `TASK-<number>-<slug>`.
- `docs/handoffs/<task-id>/01-architect.md` (required).
- For rework cycles: latest `docs/handoffs/<task-id>/03-reviewer.md` (required when reviewer requested changes).

If required upstream report is missing, stale, inconsistent, or non-actionable, stop and report exactly what is missing.

## Inputs To Consult
Read in this order before coding:
1. `docs/handoffs/<task-id>/03-reviewer.md` if it exists.
2. `docs/handoffs/<task-id>/01-architect.md` (required).
3. Existing `docs/handoffs/<task-id>/02-implementer.md` if present (for continuation).
4. Relevant code, tests, and docs.

## Output
Create or update only:
- `docs/handoffs/<task-id>/02-implementer.md`

Do not edit `01-architect.md` or `03-reviewer.md`.

## Procedure
1. Validate `task-id` and verify `01-architect.md` exists.
2. Confirm architecture status is implementable:
- Stop if architect status is `blocked` or `needs_human_decision`.
- Stop if acceptance criteria/test plan are missing.
3. If reviewer report exists:
- If verdict is `changes_requested`, treat reviewer findings as mandatory.
- Stop if findings are ambiguous or not tied to code/tests.
4. Build an implementation checklist from architect plan and reviewer findings.
5. Implement minimal, localized code changes.
6. Add or update tests aligned to acceptance criteria.
7. Run the local CI validation baseline unless the architect explicitly narrows it or a concrete blocker prevents a command:
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo doc --workspace --all-features --no-deps`
- `cargo test --doc --workspace`
- `mdbook build book`
- Python binding/docs checks using the repo venv (`.venv/bin/python`) or an explicitly documented equivalent interpreter:
  - `python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `python -m pytest crates/ringgrid-py/tests -q`
- If any required local CI command is not run, stop and record the blocker explicitly instead of handing off as `ready_for_review`.
8. Record deviations explicitly:
- what changed,
- why,
- risk impact,
- whether reviewer/architect follow-up is needed.
9. Write `02-implementer.md` using `docs/templates/task-handoff-report.md` and role-specific sections.
10. End with a concrete handoff to Reviewer.

## Guardrails
- Do not silently redefine scope.
- Avoid unrelated refactors unless explicitly requested.
- Keep diffs small and reviewable.
- Prefer existing project conventions over introducing new patterns.
- Be explicit about blockers, uncertainty, and unvalidated assumptions.

## Definition Of Done
- Code changes align with architect plan (or documented deviation).
- Tests are added/updated where behavior changed.
- Validation commands and outcomes are recorded, including the local CI baseline for `fmt`, `clippy`, workspace tests, rustdoc/doctests, `mdbook`, and Python binding checks unless a blocker is explicitly documented.
- `02-implementer.md` references correct `task-id`, changed files, results, and reviewer handoff.

## Handoff Rules
- Set status to one of:
- `ready_for_review`
- `blocked`
- `needs_architect_clarification`
- If blocked, specify exact blocker and owner.
- If deviating from architecture, include a short "deviation approval needed" note for Reviewer/Architect.

## Stack-Specific Notes
- Rust: include affected crates/modules, trait/API changes, `cargo test` scope, and perf/safety implications.
- Python: include runtime version assumptions, CLI/script behavior changes, and fixture determinism.
- React: include component behavior, state transitions, API contract updates, and UI test impact.
- Tauri: include command/event contract changes across `src-tauri` and frontend.
- Docker: include build args, layer/cache impacts, runtime env changes, and compose/service compatibility.
