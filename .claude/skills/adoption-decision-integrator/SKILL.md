---
name: adoption-decision-integrator
description: Decide adopt vs option vs reject based on evidence (accuracy/robustness/perf), integrate accordingly, and update docs + ADR with outcome.
---

## When to use
After you have:
- Accuracy/robustness evidence (A/B synthetic)
- Performance report (benchmarks)

## Prerequisites
- ADR exists and is linked to the task
- Benchmark report is complete
- Tests are passing

## Outputs (deliverables)
- Decision recorded (Adopt / Adopt as option / Reject)
- Integration changes:
  - Adopt: remove old path, new becomes default
  - Adopt as option: keep both with clear config semantics
  - Reject: remove new code
- Updated ADR with outcome + evidence summary
- Updated pipeline docs (e.g. `CLAUDE.md`) if applicable
- Handoff note → Project Lead with decision + evidence summary

## Evidence review checklist
- Accuracy: improved / same / regressed
- Robustness: better / same / worse
- Performance: faster / same / slower
- Operational complexity: config, maintenance, failure modes

## Decision rules of thumb
- **Adopt** if accuracy/robustness improve and perf is acceptable (or can be recovered).
- **Adopt as option** only if both variants have clear, durable use cases.
- **Reject** if benefits are marginal or costs/risks dominate; document why.

## ADR update requirements
Add:
- Final decision
- Evidence summary (key metrics + links to reports)
- Any follow-up tasks (cleanup, docs, optimization, removal timeline)

## Handoff note template (to Project Lead)
- Decision + rationale
- Evidence highlights (accuracy/robustness/perf deltas)
- What changed in defaults / config behavior
- Any migration notes / risks
- Verification commands to re-run
