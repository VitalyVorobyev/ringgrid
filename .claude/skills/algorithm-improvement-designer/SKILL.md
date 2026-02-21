---
name: algorithm-improvement-designer
description: Document an algorithm limitation, propose a mathematically justified improvement, and produce an ADR + handoff package for integration.
---

## When to use
Replacing or enhancing a mathematical primitive (ellipse fitting, decode strategy, center correction, RANSAC variant, etc.) where the limitation is known and a better approach exists.

## Prerequisites
- Limitation is clearly identified (accuracy ceiling, robustness issue, theoretical gap).
- Task exists in `state/backlog.md`.

## Inputs to consult
- Current implementation location(s) and call sites
- Existing synthetic eval baselines (or create minimal baseline if missing)
- `templates/adr.md`

## Outputs (deliverables)
- **Algorithm design document** (problem → proposal → math spec → risks)
- **ADR** in the repo format (from `templates/adr.md`)
- **Handoff note → Pipeline Architect** (integration constraints request)

## What to include in the design doc
- Current algorithm summary:
  - What it does, where it lives, known failure modes
  - Baseline accuracy on synthetic eval (mean/p50/p95 if available)
- Relevant literature / known improvements (citations + 1–2 sentence takeaway each)
- Mathematical specification of the new method:
  - Inputs/outputs, objective function (if any), invariants
  - Failure behavior and numerical stability notes
- Affected modules + downstream pipeline stages
- Test plan outline (accuracy + robustness + edge cases)

## ADR requirements
ADR must capture:
- Options considered (≥2) and tradeoffs
- Decision drivers (accuracy/robustness/perf/complexity)
- Compatibility plan (coexist behind toggle during validation, if applicable)
- Evidence plan (what data will decide adopt vs reject)

## Handoff note template (to Pipeline Architect)
Include:
- Proposed config shape (field/enum/new struct) **recommendation**
- Whether old/new should coexist behind a toggle during validation
- Migration intention (toggle → validate → remove old or keep fallback)
- Any public API/type changes expected (ideally none)

## Stop conditions
If you cannot write a clear math spec or cannot define measurable acceptance metrics, do not proceed to implementation—revise the design until it’s testable.
