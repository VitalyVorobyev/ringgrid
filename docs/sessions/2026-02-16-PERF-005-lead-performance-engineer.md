# Handoff: Project Lead → Performance Engineer

- **Task:** PERF-005: Optimize Inner-Fit Hotspot Group
- **Date:** 2026-02-16
- **Branch:** performance

## Work Completed

- Assessed PERF-004 outputs and accepted closeout.
- Promoted PERF-005 to Active Sprint as in-progress.
- Created task spec: `.ai/state/sessions/2026-02-16-PERF-005-spec.md`.

## Key Findings

- PERF-004 delivered large wins in outer-estimate/outer-fit paths, so optimization focus shifts to inner-fit path costs.
- PERF-001 and later profiling indicate inner-fit remains a meaningful contributor, especially in mapper-enabled runs.
- Validation gates are unchanged and mandatory for PERF-005:
  - blur=3.0 synth eval (`n=10`)
  - `run_reference_benchmark.sh`
  - `run_distortion_benchmark.sh`

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved PERF-005 to Active Sprint and kept downstream PERF tasks queued |
| `.ai/state/sessions/2026-02-16-PERF-005-spec.md` | Added full PERF-005 task specification |
| `.ai/state/sessions/2026-02-16-PERF-005-lead-performance-engineer.md` | Added dispatch handoff for PERF-005 |
| `.ai/state/sessions/2026-02-16-PERF-004-lead-human.md` | Added PERF-004 closure note for human |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| PERF-004 blur=3 center mean | `0.31674058408388994 px` |
| PERF-004 blur=3 precision/recall | `1.000 / 0.9458128078817735` |
| Accuracy policy | investigate if mean center delta > `+0.01 px` |

## Performance State

| Benchmark | Result |
|-----------|--------|
| PERF-004 outer-estimate result | `outer_estimate_64r_48t_nomapper`: `-50.54%`, `outer_estimate_64r_48t_mapper`: `-41.85%` |
| Current focus | inner-fit hotspot group |

## Open Questions

- Which deterministic microbench design best represents inner-fit cost under mapper and non-mapper conditions?
- How much of remaining mapper overhead is inside inner-fit path vs surrounding stage overhead?

## Recommended Next Steps

1. Add deterministic inner-fit/inner-estimate benchmarks to `hotpaths.rs`.
2. Capture pre-change baseline benchmarks and flamegraphs (mapper + non-mapper).
3. Apply implementation-level optimizations first (scratch reuse, loop tightening, allocation elimination).
4. Report before/after benchmark and flamegraph deltas against PERF-005 criteria.
5. Hand off to Validation Engineer with required validation commands.
6. If semantics-changing adjustments are needed, hand off to Algorithm Engineer with evidence.

## Blocking Issues

None.

---

## Execution Update (Performance Engineer, 2026-02-16)

Completed:

1. Added deterministic PERF-005 benchmarks in `crates/ringgrid/benches/hotpaths.rs`:
   - `inner_estimate_64r_96t_nomapper`
   - `inner_estimate_64r_96t_mapper`
   - `inner_fit_64r_96t_nomapper`
   - `inner_fit_64r_96t_mapper`
2. Optimized hot loops in:
   - `crates/ringgrid/src/ring/inner_estimate.rs`
   - `crates/ringgrid/src/detector/inner_fit.rs`
3. Re-benchmarked with fixed Criterion command:
   - `inner_fit_64r_96t_nomapper`: `68.051 us` -> `59.358 us` (`-12.77%`)
   - `inner_fit_64r_96t_mapper`: `93.909 us` -> `82.433 us` (`-12.22%`)
4. Captured PERF-005 flamegraphs:
   - `.ai/state/sessions/2026-02-16-PERF-005-detect-flamegraph.svg`
   - `.ai/state/sessions/2026-02-16-PERF-005-detect-flamegraph-mapper.svg`
5. Completed required validation gates:
   - `./.venv/bin/python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf005_blur3`
   - `bash tools/run_reference_benchmark.sh`
   - `bash tools/run_distortion_benchmark.sh`
6. Re-ran quality gates:
   - `cargo fmt --all`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`

Detailed report:
- `.ai/state/sessions/2026-02-16-PERF-005-performance-handoff.md`
