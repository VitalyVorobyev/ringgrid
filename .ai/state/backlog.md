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
| PERF-003 | in-progress | P1 | perf | Standardize perf validation suite (blur=3 batch + reference/distortion scripts) | Validation Engineer | Convert recurring perf-validation commands into standardized, repeatable validation workflow/reporting artifacts |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| — | — | — | — | — | — | — |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| — | — | — | — | — | — | — |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| PERF-005 | 2026-02-16 | perf | Optimize inner-fit hotspot group | Completed: `inner_fit_64r_96t_nomapper` `68.051 us -> 59.358 us` (`-12.77%`), `inner_fit_64r_96t_mapper` `93.909 us -> 82.433 us` (`-12.22%`); validation gates passed |
| PERF-004 | 2026-02-16 | perf | Optimize outer-fit/outer-estimate hotspot group | Completed: `outer_estimate_64r_48t_nomapper` `34.365 us -> 16.996 us` (`-50.54%`), `outer_estimate_64r_48t_mapper` `40.082 us -> 23.310 us` (`-41.85%`); validation gates passed |
| PERF-002 | 2026-02-16 | perf | Optimize proposal hotspot (`detector::proposal::find_proposals`) | Accepted: `proposal_1280x1024` `42.404 ms → 33.612 ms` (`-20.73%`), `proposal_1920x1080` `60.614 ms → 48.357 ms` (`-20.22%`); required validation gates passed |
| PERF-001 | 2026-02-16 | perf | Establish comprehensive performance tracing baseline and benchmark harness | Completed baseline report + hotspot ranking + follow-up plan. Allocation counts documented via proxy (RSS) due `xctrace`/SIP tooling limits |
| FEAT-001 | 2026-02-16 | feature | Normalize marker center API and simplify finalize flow | Closed by human decision: marginal regression accepted for now; follow-up perf/accuracy work tracked in PERF tasks |
| INFRA-001 | 2026-02-15 | infra | Set up `.ai/` agent workflow | Bootstrap ADR-001 |
