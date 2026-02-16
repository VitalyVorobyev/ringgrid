# Handoff: Project Lead → Performance Engineer

- **Task:** PERF-002: Optimize Proposal Hotspot (`detector::proposal::find_proposals`)
- **Date:** 2026-02-16
- **Branch:** performance

## Work Completed

- Assessed PERF-001 baseline outputs and accepted task closeout.
- Promoted PERF-002 to Active Sprint as in-progress.
- Created task spec: `.ai/state/sessions/2026-02-16-PERF-002-spec.md`.

## Key Findings

- PERF-001 baseline identifies proposal stage as top hotspot:
  - `detector::proposal::find_proposals` ~`61.11%` wall-time share.
- Baseline benchmark anchors for proposal:
  - `proposal_1280x1024`: `42.404 ms`
  - `proposal_1920x1080`: `60.614 ms`
- Validation gates for all optimization work are already defined and mandatory:
  - blur=3.0 synth eval
  - `run_reference_benchmark.sh`
  - `run_distortion_benchmark.sh`

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved PERF-001 to Done and promoted PERF-002 to Active Sprint |
| `.ai/state/sessions/2026-02-16-PERF-002-spec.md` | Added full PERF-002 task specification |
| `.ai/state/sessions/2026-02-16-PERF-002-lead-performance-engineer.md` | Added dispatch handoff for PERF-002 |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| PERF baseline center context | see `.ai/state/sessions/2026-02-16-PERF-001-baseline-report.md` |
| Validation gate policy | mean center error delta must remain <= `+0.01 px` |

## Performance State

| Benchmark | Result |
|-----------|--------|
| `proposal_1280x1024` baseline | `42.404 ms` |
| `proposal_1920x1080` baseline | `60.614 ms` |
| Proposal hotspot share baseline | `61.11%` |

## Open Questions

- Which proposal-loop changes deliver best speedup-risk tradeoff first: pixel access/layout, accumulator updates, or NMS pass structure?
- Can we reduce proposal share materially without harming recall on blur=3 and distortion benchmarks?

## Recommended Next Steps

1. Optimize `detector::proposal::find_proposals` hot loops with allocation-safe patterns.
2. Measure before/after deltas on proposal benches and report against PERF-002 thresholds.
3. Re-capture representative flamegraph and compare proposal wall-time share to PERF-001 baseline.
4. Hand off to Validation Engineer with benchmark data and required validation commands.

## Blocking Issues

None.
