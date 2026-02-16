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
| FEAT-001 | in-progress | P0 | feature | Normalize marker center API and simplify finalize flow | Pipeline Architect | Breaking API change: `center` always image-space, add `center_mapped`, remove `center_projective*`, remove `refine_with_h` |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| PERF-001 | todo | P2 | perf | Add Criterion benchmarks for core hot paths (proposal, ellipse fit, radial profile) | Performance Engineer | Foundation for future optimization work |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| — | — | — | — | — | — | — |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| INFRA-001 | 2026-02-15 | infra | Set up `.ai/` agent workflow | Bootstrap ADR-001 |
