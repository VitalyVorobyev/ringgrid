---
name: architect
description: Convert a backlog item, issue, ADR, or change request into an implementation-ready plan with strict handoff traceability. Use when work must be scoped and planned before coding. Require a mandatory task-id and write only docs/handoffs/<task-id>/01-architect.md.
---

# Architect

## Purpose
Turn a task request into a small, implementation-ready, reviewable plan with clear acceptance criteria and test strategy.

## Required Inputs
- `task-id` (mandatory). Format: `TASK-<number>-<slug>` (example: `TASK-012-image-store-refactor`).
- At least one task source: backlog item, issue, ADR, incident note, or direct human request.
- Optional but recommended: priority, deadline, and non-functional constraints.

If `task-id` is missing or malformed, stop and report the exact issue.

## Inputs To Consult
Read in this order before planning:
1. `docs/handoffs/<task-id>/03-reviewer.md` if it exists.
2. `docs/handoffs/<task-id>/02-implementer.md` if it exists.
3. Existing `docs/handoffs/<task-id>/01-architect.md` if it exists (for revisions).
4. Task source documents (backlog/issue/ADR).
5. Relevant code and docs in the repo.

## Output
Create or update only:
- `docs/handoffs/<task-id>/01-architect.md`

Do not edit `02-implementer.md` or `03-reviewer.md`.

## Procedure
1. Validate `task-id` format and create `docs/handoffs/<task-id>/` if needed.
2. Read all existing reports under that task directory.
3. Perform consistency checks:
- Stop if an existing report has a different task id.
- Stop if upstream report is clearly incomplete (missing required sections or no actionable handoff).
- Stop if the latest reviewer verdict is `changes_requested` but review findings are too vague to plan rework.
4. Define problem and scope:
- State business/technical problem.
- Separate in-scope from out-of-scope.
- Capture constraints (performance, compatibility, security, timelines, tooling).
5. Map affected areas:
- Identify likely modules/files and data/contracts impacted.
- Explicitly call out Rust/Python/React/Tauri/Docker touchpoints when relevant.
6. Produce a minimal, reviewable implementation plan:
- Prefer 1-3 small increments.
- Sequence work in dependency order.
- Add rollback/mitigation notes for risky steps.
7. Define acceptance criteria and test plan:
- Functional criteria.
- Regression criteria.
- Required validation commands.
8. Write `01-architect.md` using `docs/templates/task-handoff-report.md` and role-specific sections.
9. End with explicit handoff instructions to Implementer.

## Guardrails
- Do not implement code unless explicitly asked.
- Do not silently expand scope.
- Prefer minimal diffs and isolated change sets.
- Mark uncertainty explicitly; do not guess missing requirements.
- Recommend splitting when scope is too broad for one reviewable PR.

## Definition Of Done
- `01-architect.md` exists and references the correct `task-id` in title and metadata.
- Report includes: problem statement, scope, constraints, assumptions, affected areas, plan, acceptance criteria, test plan, out of scope, handoff to Implementer.
- Plan is actionable without rereading the full conversation.

## Handoff Rules
- Set status to one of:
- `ready_for_implementer`
- `blocked`
- `needs_human_decision`
- If blocked, list exactly what is missing and who must provide it.
- If task should be split, provide concrete proposed child task ids.

## Stack-Specific Notes
- Rust: reference crate/module boundaries (example: `crates/<name>/src/...`), public API changes, and required unit/integration tests.
- Python: call out script vs library changes, CLI flags, and deterministic fixture expectations.
- React: identify component/state/api boundaries and expected UI test updates.
- Tauri: separate frontend (`src/`) from backend (`src-tauri/`) impact and command contracts.
- Docker: list image/build/runtime changes, env vars, and compose/deployment impact.
