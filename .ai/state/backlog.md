# Backlog

## Status Values
- `todo` — not started
- `in-progress` — actively being worked
- `blocked` — waiting on something
- `done` — completed

## Priority Values
- `P0` — blocking release or correctness
- `P1` — next up
- `P2` — planned
- `P3` — someday

---

## Active Sprint

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| PERF-002 | in-progress | P1 | perf | Optimize proposal hotspot (`detector::proposal::find_proposals`) | Performance Engineer | Primary optimization task from PERF-001 baseline (`~61.11%` hotspot share); target >=10% reduction with validation gates |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| PERF-004 | todo | P2 | perf | Optimize outer-fit/outer-estimate hotspot group | Performance Engineer + Algorithm Engineer | Ordered candidate #2; focus on radial/profile temp allocations and per-candidate sorting overhead; same validation gates as PERF-002 |
| PERF-005 | todo | P2 | perf | Optimize inner-fit hotspot group | Performance Engineer | Ordered candidate #3; focus on per-theta profile aggregation overhead; same validation gates as PERF-002 |
| PERF-003 | todo | P2 | perf | Standardize perf validation suite (blur=3 batch + reference/distortion scripts) | Validation Engineer | Ensure every PERF optimization handoff includes blur=3 eval, reference benchmark, and distortion benchmark outputs |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| — | — | — | — | — | — | — |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| PERF-001 | 2026-02-16 | perf | Establish comprehensive performance tracing baseline and benchmark harness | Completed baseline report + hotspot ranking + follow-up plan. Allocation counts documented via proxy (RSS) due `xctrace`/SIP tooling limits |
| FEAT-001 | 2026-02-16 | feature | Normalize marker center API and simplify finalize flow | Closed by human decision: marginal regression accepted for now; follow-up perf/accuracy work tracked in PERF tasks |
| INFRA-001 | 2026-02-15 | infra | Set up `.ai/` agent workflow | Bootstrap ADR-001 |
