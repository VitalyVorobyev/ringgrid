# Handoff: Project Lead → Performance Engineer

- **Task:** PERF-004: Optimize Outer-Fit / Outer-Estimate Hotspot Group
- **Date:** 2026-02-16
- **Branch:** performance

## Work Completed

- Assessed PERF-002 results and accepted closeout.
- Promoted PERF-004 to Active Sprint as in-progress.
- Created task spec: `.ai/state/sessions/2026-02-16-PERF-004-spec.md`.

## Key Findings

- PERF-002 successfully reduced proposal hotspot cost and throughput now shifts focus to the next dominant stage cluster.
- PERF-001 baseline identified outer-fit as a top contributor (`~15.28%` in representative non-mapper profile).
- Mapper-enabled mode remains materially slower; outer-estimate/outer-fit path is a likely contributor and should be profiled explicitly in PERF-004.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved PERF-002 to Done and promoted PERF-004 to Active Sprint |
| `.ai/state/sessions/2026-02-16-PERF-004-spec.md` | Added full PERF-004 task specification |
| `.ai/state/sessions/2026-02-16-PERF-004-lead-performance-engineer.md` | Added dispatch handoff for PERF-004 |
| `.ai/state/sessions/2026-02-16-PERF-002-lead-human.md` | Added PERF-002 closure note for human |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| PERF-002 blur=3 center mean | `0.31549159136017835 px` |
| PERF-002 blur=3 precision/recall | `1.000 / 0.9467980295566502` |
| Accuracy policy | investigate if mean center delta > `+0.01 px` |

## Performance State

| Benchmark | Result |
|-----------|--------|
| PERF-002 proposal result | `proposal_1280x1024`: `-20.73%`, `proposal_1920x1080`: `-20.22%` vs PERF-001 baseline |
| Current focus | outer-estimate/outer-fit hotspot group (candidate #2) |

## Open Questions

- Which concrete microbenchmarks best represent outer-estimate/outer-fit compute cost on this code path?
- How much of mapper overhead is attributable to edge/radial sampling and can it be reduced without algorithmic drift?

## Recommended Next Steps

1. Capture fresh pre-change benchmarks/flamegraphs for outer-estimate/outer-fit-related paths (mapper and non-mapper).
2. Apply implementation-level optimizations first (buffer reuse, loop structure, branch reduction, temporary allocation elimination).
3. Report before/after benchmark deltas and stage-share changes.
4. Hand off to Validation Engineer with required validation commands:
   - blur=3 eval (`n=10`)
   - `run_reference_benchmark.sh`
   - `run_distortion_benchmark.sh`
5. If improvements require model/sampling semantics changes, create evidence-backed handoff to Algorithm Engineer.

## Blocking Issues

None.

---

## Execution Update (Performance Engineer, 2026-02-16)

Completed:

1. Added deterministic PERF-004 benchmarks in `crates/ringgrid/benches/hotpaths.rs`:
   - `outer_estimate_64r_48t_nomapper`
   - `outer_estimate_64r_48t_mapper`
2. Optimized hot loops in:
   - `crates/ringgrid/src/ring/outer_estimate.rs`
   - `crates/ringgrid/src/ring/radial_profile.rs`
   - `crates/ringgrid/src/ring/edge_sample.rs`
3. Re-benchmarked with the fixed Criterion command:
   - `outer_estimate_64r_48t_nomapper`: `34.365 us` -> `16.996 us` (`-50.54%`)
   - `outer_estimate_64r_48t_mapper`: `40.082 us` -> `23.310 us` (`-41.85%`)
4. Captured PERF-004 flamegraphs:
   - `.ai/state/sessions/2026-02-16-PERF-004-detect-flamegraph.svg`
   - `.ai/state/sessions/2026-02-16-PERF-004-detect-flamegraph-mapper.svg`
5. Completed required validation gates:
   - `./.venv/bin/python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf004_blur3`
   - `bash tools/run_reference_benchmark.sh`
   - `bash tools/run_distortion_benchmark.sh`
6. Re-ran quality gates:
   - `cargo fmt --all`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`

Detailed report and acceptance evidence:
- `.ai/state/sessions/2026-02-16-PERF-004-performance-handoff.md`
