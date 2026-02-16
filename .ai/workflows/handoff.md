# Handoff Protocol

Rules for transferring work between agent roles.

## When to Hand Off

Hand off when your phase in the workflow is complete. Do not accumulate multiple phases — complete one, hand off, then the next role picks up.

## Handoff Steps

1. **Fill the handoff template.** Copy `templates/handoff-note.md` and fill in all sections.

2. **Save to sessions.** Name: `state/sessions/YYYY-MM-DD-TASKID-from-to.md`
   - Example: `state/sessions/2026-02-15-FEAT-003-algorithm-validation.md`

3. **Update the backlog.** Set the task status in `state/backlog.md` to reflect current state and note the active role.

4. **Include test results.** Every handoff must include:
   - `cargo test` pass/fail
   - `cargo clippy` clean/warnings
   - Synthetic eval summary (if run)
   - Benchmark numbers (if run)

## Handoff Note Requirements

### Minimum Fields (always required)
- Task ID and title
- Work completed (bullet list)
- Files changed (paths with one-line descriptions)
- Test results (cargo test, clippy)
- Recommended next steps for the receiving role

### Accuracy Fields (required if algorithm or pipeline changes)
- Center error: mean, p50, p95
- Decode success rate
- Homography reprojection error

### Performance Fields (required if hot path changes)
- Benchmark name and before/after numbers
- Allocation count change
- PERF validation gate artifacts/deltas:
  - blur=3 synth eval (`n=10`) via `run_blur3_benchmark.sh`
  - `run_reference_benchmark.sh`
  - `run_distortion_benchmark.sh`
- Link to filled `.ai/templates/accuracy-report.md`

## Handoff Flow by Workflow

All workflows start with **Project Lead** dispatching and end with **Project Lead** closing.

### Planning
```
Project Lead ↔ Human (discuss, specify, dispatch, track)
```

### Feature Development
```
Project Lead → Pipeline Architect → Algorithm Engineer → Pipeline Architect → Validation Engineer → (Performance Engineer) → Pipeline Architect → Project Lead
```

### Bug Fix
```
Project Lead → Validation Engineer → Algorithm Engineer → Validation Engineer → (Performance Engineer) → Pipeline Architect → Project Lead
```

### Performance Optimization
```
Project Lead → Performance Engineer → (Algorithm Engineer) → Validation Engineer → Performance Engineer → Project Lead
```

### Algorithm Improvement
```
Project Lead → Algorithm Engineer → Pipeline Architect → Algorithm Engineer → Validation Engineer → Performance Engineer → Pipeline Architect → Project Lead
```

Roles in parentheses are conditional — only involved if their expertise is needed.

## Reading a Handoff

When you receive a handoff:
1. Read the handoff note completely
2. Start from "Recommended Next Steps"
3. If anything is unclear, check the task spec in `state/sessions/` or the original backlog entry
4. If still unclear, write your questions as a new handoff note back to the sender before proceeding
