# Task Handoffs Workflow

This directory stores role handoffs for a strict `Architect -> Implementer -> Reviewer` workflow.

## Naming Rules
- Task id is mandatory for every invocation.
- Task id format: `TASK-<number>-<slug>`.
- One task directory per task id:
  - `docs/handoffs/<task-id>/`
- Fixed report files per role:
  - `docs/handoffs/<task-id>/01-architect.md`
  - `docs/handoffs/<task-id>/02-implementer.md`
  - `docs/handoffs/<task-id>/03-reviewer.md`

## File Ownership
- Architect reads available upstream context and writes only `01-architect.md`.
- Implementer reads architect/reviewer context and writes only `02-implementer.md`.
- Reviewer reads architect + implementer outputs and writes only `03-reviewer.md`.

No role edits another role's report file.

## Required Metadata In Every Report
- Title
- Task ID
- Role
- Date
- Status
- Inputs consulted
- Summary
- Decisions made
- Files/modules affected (or expected)
- Validation/tests
- Risks/open questions
- Next handoff

Use `docs/templates/task-handoff-report.md`.

## Task Lifecycle
1. Architect creates `01-architect.md` with scope, plan, acceptance criteria, and test strategy.
2. Implementer executes plan, changes code/tests, and updates `02-implementer.md`.
3. Reviewer checks code/tests vs plan and publishes verdict in `03-reviewer.md`.
4. If verdict is `changes_requested`, loop back to Implementer.
5. If major scope ambiguity appears during rework, Implementer can hand off back to Architect.
6. If critical product decisions are missing, hand off to Human.

## Who Reads Which File
- Architect reads all existing files in task folder (especially prior `03-reviewer.md` for replans).
- Implementer must read `01-architect.md` and latest `03-reviewer.md` if present.
- Reviewer must read `01-architect.md` and `02-implementer.md`.

## Allowed Reviewer Verdicts (Closed Set)
- `approved`
- `approved_with_minor_followups`
- `changes_requested`

No other verdict strings are allowed.

## Rework Loop Rules
- `changes_requested` must include concrete, implementable actions.
- Implementer updates only `02-implementer.md` and addresses each requested change explicitly.
- Reviewer re-reviews and updates only `03-reviewer.md` with a new verdict.
- If review findings are ambiguous or not actionable, the Reviewer report is insufficient and must be fixed before implementation resumes.

## Missing/Stale/Inconsistent Inputs
A role must stop (not guess) when any of the following is true:
- Required upstream file is missing.
- Task ids differ across reports.
- Required sections are absent in upstream report.
- Upstream status indicates blocked or pending human decision.
- Report content is clearly inconsistent with current task scope.

When stopping, state exactly what is missing and who must provide it.
