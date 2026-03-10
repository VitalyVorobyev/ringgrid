# Task Handoffs Workflow

This directory stores role handoffs for a strict `Architect -> Implementer -> Reviewer` workflow.

## Naming Rules
- Backlog ids and workflow task ids are different on purpose:
  - backlog ids live in `docs/backlog.md` (`INFRA-011`, `DOCS-003`, ...),
  - workflow task ids live under `docs/handoffs/` (`TASK-012-expose-target-generation-in-ringgrid-py`, ...).
- Task id format: `TASK-<number>-<slug>`.
- One task directory per workflow task id:
  - `docs/handoffs/<task-id>/`
- Fixed report files per role:
  - `docs/handoffs/<task-id>/01-architect.md`
  - `docs/handoffs/<task-id>/02-implementer.md`
  - `docs/handoffs/<task-id>/03-reviewer.md`

## Backlog Mapping
- A backlog item may map to one workflow task directory at a time.
- When starting from a backlog item and no handoff exists yet:
  - mint the next unused `TASK-<number>-<slug>` id,
  - derive the slug from the backlog title,
  - record the source backlog id in report metadata as `Backlog ID: <ID>`.
- When a backlog item already has exactly one matching handoff directory, reuse that `task-id`.
- If multiple handoff directories appear to map to the same backlog item, stop and ask the human which one should continue.

## File Ownership
- Architect reads available upstream context and writes only `01-architect.md`.
- Implementer reads architect/reviewer context and writes only `02-implementer.md`.
- Reviewer reads architect + implementer outputs and writes only `03-reviewer.md`.

No role edits another role's report file.

## Required Metadata In Every Report
- Title
- Task ID
- Backlog ID (required when sourced from `docs/backlog.md`; otherwise `n/a` is acceptable)
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

## Orchestrator
`tools/orchestrate_handoffs.py` is the control-plane runner for this workflow. It never edits role reports directly; it only:
- initializes per-task control metadata,
- records human architect approval,
- reads report state,
- chooses the next legal role transition,
- optionally runs one Codex role turn at a time.

### Control File
For automated or semi-automated task routing, keep a per-task control file at:
- `docs/handoffs/<task-id>/orchestrator.json`

This file is owned by the orchestrator and human approvals, not by Architect, Implementer, or Reviewer.

Current fields:
- `task_id`
- `brief`
- `input_refs`
- `architect_approval_required`
- `approved_architect_report_mtime`
- `max_changes_requested`
- `changes_requested_count`
- `last_seen_reviewer_report_mtime`
- `history`

### Safe Usage Pattern
Initialize a task:
```bash
python3 tools/orchestrate_handoffs.py init TASK-012-image-store-refactor \
  --brief "Refactor image storage boundaries and remove duplicate write paths." \
  --input-ref "docs/backlog.md" \
  --input-ref "Issue #42"
```

If the human starts from a backlog id like `INFRA-011`, resolve or mint the `TASK-*` id first, then use that id consistently for the handoff directory and orchestrator control file.

Inspect all task states:
```bash
python3 tools/orchestrate_handoffs.py status
```

Approve an architect report after human review:
```bash
python3 tools/orchestrate_handoffs.py approve TASK-012-image-store-refactor architect
```

Preview the next machine-actionable role:
```bash
python3 tools/orchestrate_handoffs.py run --task-id TASK-012-image-store-refactor
```

Execute one bounded role turn:
```bash
python3 tools/orchestrate_handoffs.py run --execute --max-steps 1
```

### Automation Guidance
- Prefer recurring one-step runs over a single long-lived endless loop.
- Keep `--max-steps 1` for scheduled automation unless you have a strong reason otherwise.
- Leave architect approval enabled by default; disable it only for tasks where fully autonomous planning is acceptable.
- Keep `max_changes_requested` low so repeated review loops force a human check instead of running indefinitely.

### State Machine
- No `01-architect.md` plus valid `orchestrator.json` seed: run Architect.
- `01-architect.md` ready but not human-approved: stop for Human.
- Architect approved and `02-implementer.md` missing or stale: run Implementer.
- Implementer ready and `03-reviewer.md` missing or stale: run Reviewer.
- Reviewer verdict `changes_requested`: run Implementer again until loop limit is reached.
- Reviewer verdict `approved` or `approved_with_minor_followups`: stop as complete.
- Any `blocked`, `needs_human_decision`, task-id mismatch, or malformed report: stop for Human.
