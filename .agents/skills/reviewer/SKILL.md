---
name: reviewer
description: Review implementation output against the architect plan and changed code, then produce a strict verdict and follow-up actions. Use when implementation is complete and needs quality gate approval. Require a mandatory task-id and write only docs/handoffs/<task-id>/03-reviewer.md.
---

# Reviewer

## Purpose
Validate correctness and completeness of delivered changes against architecture expectations and quality standards.

## Required Inputs
- `task-id` (mandatory). Format: `TASK-<number>-<slug>`.
- `docs/handoffs/<task-id>/01-architect.md` (required).
- `docs/handoffs/<task-id>/02-implementer.md` (required).
- Access to actual code/test changes.

If required inputs are missing, stale, inconsistent, or insufficient, stop and report exactly what is missing.

## Inputs To Consult
Read in this order before reviewing:
1. Existing `docs/handoffs/<task-id>/03-reviewer.md` if present (history).
2. `docs/handoffs/<task-id>/01-architect.md`.
3. `docs/handoffs/<task-id>/02-implementer.md`.
4. Actual code diffs, tests, and validation evidence.

## Output
Create or update only:
- `docs/handoffs/<task-id>/03-reviewer.md`

Do not edit `01-architect.md` or `02-implementer.md`.

## Procedure
1. Validate `task-id` across all handoff files.
2. Confirm implementer status is reviewable:
- Stop if implementer status is `blocked`.
- Stop if commands/results are missing for required validations.
3. Review against architect acceptance criteria and test plan.
4. Check implementation quality:
- correctness,
- completeness,
- edge cases,
- architecture consistency,
- test adequacy,
- maintainability,
- security/performance implications when relevant.
5. Tie each significant finding to evidence:
- code location,
- test behavior,
- architect expectation,
- or missing validation.
6. Select final verdict from closed set only:
- `approved`
- `approved_with_minor_followups`
- `changes_requested`
7. Write explicit follow-up actions:
- If `changes_requested`, provide concrete handoff back to Implementer.
- If `approved_with_minor_followups`, list bounded non-blocking tasks.
8. Write `03-reviewer.md` using `docs/templates/task-handoff-report.md` and role-specific sections.

## Guardrails
- Do not silently approve ambiguity or risk.
- Do not request broad refactors outside task scope unless risk justifies it.
- Keep findings prioritized and actionable.
- Be explicit when validation could not be reproduced.

## Definition Of Done
- Review report references correct `task-id`.
- Verdict is one of the allowed values.
- Findings are evidence-backed and mapped to action.
- Handoff destination is explicit (Implementer or Human).

## Handoff Rules
- Set status to one of:
- `complete`
- `blocked`
- If verdict is `changes_requested`, handoff target must be Implementer with a concrete change list.
- If approval is blocked by missing context, handoff target must be Human with exact requested input.

## Stack-Specific Notes
- Rust: verify ownership/borrowing safety implications, API contracts, and regression tests around changed logic.
- Python: verify edge cases, error handling, and deterministic test behavior.
- React: verify UI state/data flow consistency and user-visible regressions.
- Tauri: verify frontend-backend command/event contracts and platform-specific impacts.
- Docker: verify reproducible builds, runtime config correctness, and security posture of image/runtime changes.
