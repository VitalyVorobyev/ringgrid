# Handoff: Performance Engineer → Validation Engineer

- **Task:** PERF-001: Comprehensive Performance Tracing and Data-Driven Hot-Path Optimization
- **Date:** 2026-02-16
- **Branch:** release

## Work Completed

- Added deterministic Criterion harness and required named benchmarks in `crates/ringgrid/benches/hotpaths.rs`.
- Added bench wiring (`criterion` dev-dependency + `[[bench]]`) in `crates/ringgrid/Cargo.toml`.
- Captured a representative `detect()` flamegraph artifact:
  - `.ai/state/sessions/2026-02-16-PERF-001-detect-flamegraph.svg`
- Optimized `conic::fit` hot path by replacing dynamic `DMatrix` construction + `D^T D` multiply with direct fixed-size `SMatrix<6,6>` scatter accumulation.
- Re-ran benchmarks and full quality gates (`fmt`, `clippy -D warnings`, `test --workspace --all-features`).
- Ran deterministic synthetic eval batch (`n=5`, `blur_px=3.0`) at:
  - `tools/out/eval_perf_2026-02-16/det/aggregate.json`

## Key Findings

- Flamegraph top wall-time hotspots (`detect()` run):
  - `ringgrid::detector::proposal::find_proposals` ~61.27%
  - `ringgrid::detector::outer_fit::fit_outer_candidate_from_prior_with_edge_cfg` ~16.20%
  - `ringgrid::detector::inner_fit::fit_inner_ellipse_from_outer_hint` ~13.38%
- `ellipse_fit_50pts` improved from `515.37 ns` to `263.85 ns` (`-48.80%`).
- I attempted allocation tracing with `cargo instruments -t alloc`, but `xctrace` failed in this environment with:
  - `Cannot write to specified standard output file: /dev/??`

## Files Changed

| File | Change |
|------|--------|
| `crates/ringgrid/Cargo.toml` | Added Criterion dev dependency and `hotpaths` bench target |
| `crates/ringgrid/benches/hotpaths.rs` | Added deterministic benchmarks: proposal/radial-profile/ellipse-fit |
| `crates/ringgrid/src/conic/fit.rs` | Removed dynamic `DMatrix` hot-loop allocation; direct fixed-size scatter accumulation |
| `.ai/state/sessions/2026-02-16-PERF-001-detect-flamegraph.svg` | Added flamegraph artifact |
| `.ai/state/sessions/2026-02-16-PERF-001-performance-validation-handoff.md` | This handoff/report |

## Test Results

- **cargo fmt --all:** clean
- **cargo clippy --all-targets --all-features -- -D warnings:** clean
- **cargo test --workspace --all-features:** pass
  - ringgrid unit tests: 80 passed
  - ringgrid-cli unit tests: 4 passed
  - doc tests: 5 passed
- **Synthetic eval:** pass (execution successful, metrics captured)
  - command: `./.venv/bin/python3 tools/run_synth_eval.py --n 5 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf_2026-02-16`

## Accuracy State

| Metric | Value |
|--------|-------|
| Center error (mean) | `0.3204 px` (`tools/out/eval_perf_2026-02-16/det/aggregate.json`) |
| Center error (p50) | `0.2972 px` (per-image medians; aggregate file provides per-image stats) |
| Center error (p95) | `0.7446 px` (worst per-image p95 on this 5-image run) |
| Decode success rate | precision `1.000`, recall `0.9399` |
| Homography reproj error | self-error mean `0.2797 px`, vs-GT mean `0.1455 px` |

Notes:
- This optimization is algebraically equivalent (scatter accumulation order changed, model unchanged).
- No direct evidence of a fit-specific accuracy regression was observed in tests/eval, but validation should still compare against the exact baseline batch used for FEAT/PERF tracking.

## Performance State

Benchmark command used for baseline and after:
- `cargo bench -p ringgrid --bench hotpaths -- --warm-up-time 1 --measurement-time 3 --sample-size 20`

| Benchmark | Baseline | After | Change |
|-----------|----------|-------|--------|
| `proposal_1280x1024` | `42.370 ms` | `42.689 ms` | `+0.75%` |
| `proposal_1920x1080` | `58.441 ms` | `60.027 ms` | `+2.71%` |
| `radial_profile_32r_180a` | `12.637 us` | `13.365 us` | `+5.76%` |
| `ellipse_fit_50pts` | `515.37 ns` | `263.85 ns` | `-48.80%` |

Notes:
- The targeted optimization was only in `conic::fit`; unaffected benches showed run-to-run drift on this host.
- A longer follow-up run (`warm-up=2s`, `measurement=5s`, `sample-size=30`) showed unchanged behavior for non-target benches and confirmed `ellipse_fit_50pts` remained around `~268 ns`.

## Allocation Profile

- Function-level allocation change (from code-path analysis):
  - Before: `fit_conic_direct` materialized dynamic `DMatrix (n x 6)` and dynamic `S = D^T D`.
  - After: uses stack-allocated `SMatrix<6,6>` and direct accumulation; no dynamic design-matrix allocation in this loop.
- For `n=50`, removed dynamic matrix payload is approximately:
  - `50 * 6 * 8 = 2400` bytes (`D`) plus `6 * 6 * 8 = 288` bytes (`S`) per fit, excluding allocator metadata and temporary overhead.
- Full per-`detect()` allocation counts remain **not measured** due the `xctrace` failure noted above.

## Open Questions

- Do we want a dedicated in-process allocation counter (custom global allocator in a profiling binary) to satisfy strict per-`detect()` allocation accounting in CI?
- Should the next optimization target be proposal voting (`find_proposals`), given flamegraph dominance (~61%)?

## Recommended Next Steps

1. Validation Engineer: run the standard deterministic comparison batch used in FEAT/PERF tracking and confirm center-error delta <= `+0.01 px` mean.
2. If validation passes, continue PERF-001 with proposal hot-path optimization (gradient + voting + NMS), then re-benchmark.
3. Add robust allocation profiling path (or environment fix for `cargo instruments`) to close the per-`detect()` allocation deliverable.

## Blocking Issues

None.
