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
| PERF-001 | in-progress | P1 | perf | Establish comprehensive performance tracing baseline and benchmark harness | Performance Engineer | Next deliverable: baseline report with Criterion numbers, flamegraph top-3 hotspots, allocation profile, and optimization recommendations |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| PERF-002 | todo | P1 | perf | Optimize top hotspot identified by PERF-001 baseline | Performance Engineer | Start only after baseline evidence is published; target >=10% improvement on selected hotspot |
| PERF-003 | todo | P2 | perf | Standardize perf validation suite (blur=3 batch + reference/distortion scripts) | Validation Engineer | Make post-change validation consistently include `run_reference_benchmark.sh` and `run_distortion_benchmark.sh` |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| — | — | — | — | — | — | — |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| FEAT-001 | 2026-02-16 | feature | Normalize marker center API and simplify finalize flow | Closed by human decision: marginal regression accepted for now; follow-up perf/accuracy work tracked in PERF tasks |
| INFRA-001 | 2026-02-15 | infra | Set up `.ai/` agent workflow | Bootstrap ADR-001 |
