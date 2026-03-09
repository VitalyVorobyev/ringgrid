# <Role> Report - <TASK-ID>

- Task ID: `<TASK-ID>`
- Backlog ID: `<INFRA-011|DOCS-003|n/a>`
- Role: `<architect|implementer|reviewer>`
- Date: `<YYYY-MM-DD>`
- Status: `<role-specific status>`

## Inputs Consulted
- `<file path, issue link, ADR, commit, command output>`

## Summary
- `<2-5 bullets of current state and result>`

## Decisions Made
- `<decision>` - `<reason>`

## Files/Modules Affected (Or Expected)
- `<path>` - `<why it matters>`

## Validation / Tests
- Commands run:
  - `<command>`
- Results:
  - `<pass/fail/not run + short evidence>`

## Risks / Open Questions
- `<risk or unknown>` - `<impact + owner>`

## Next Handoff
- To: `<Implementer|Reviewer|Architect|Human>`
- Requested action: `<specific next step>`

---

## Architect Required Sections

### Problem Statement
- `<what problem this task solves and why now>`

### Scope
- In scope:
  - `<item>`
- Out of scope:
  - `<item>`

### Constraints
- `<technical/business constraints>`

### Assumptions
- `<assumption>`

### Affected Areas
- `<module/path>` - `<expected change type>`

### Plan
1. `<small reviewable step>`
2. `<small reviewable step>`

### Acceptance Criteria
- `<observable criterion>`

### Test Plan
- `<unit/integration/e2e/manual checks>`

### Out Of Scope
- `<explicit non-goal>`

### Handoff To Implementer
- `<ordered implementation instructions>`

---

## Implementer Required Sections

### Plan Followed
- `<reference architect steps and mapping>`

### Changes Made
- `<what was implemented>`

### Files Changed
- `<path>` - `<change summary>`

### Deviations From Plan
- `<none>`
- or `<deviation>` - `<reason + impact>`

### Tests Added/Updated
- `<test file>` - `<coverage intent>`

### Commands Run
- `<build/test/lint command>`

### Results
- `<what passed/failed and evidence>`

### Remaining Concerns
- `<known limitation or follow-up>`

### Handoff To Reviewer
- `<what to focus on during review>`

---

## Reviewer Required Sections

### Review Scope
- `<what changes and claims were reviewed>`

### Inputs Reviewed
- `<architect report>`
- `<implementer report>`
- `<code/tests/diffs>`

### What Was Checked
- `<correctness/completeness/edge cases/...>`

### Findings
- `<severity>` `<finding>` - `<evidence and expected fix>`

### Test Assessment
- `<adequate/inadequate + why>`

### Risks
- `<risk>` - `<impact + mitigation>`

### Required Changes Or Approval Notes
- `<if changes requested: concrete checklist>`
- `<if approved: any minor follow-ups>`

### Final Verdict
- Allowed values only:
  - `approved`
  - `approved_with_minor_followups`
  - `changes_requested`

### Handoff To Implementer Or Human
- To: `<Implementer|Human>`
- Requested action: `<specific required follow-up>`
