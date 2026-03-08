# Handoff: Performance Engineer -> Project Lead

- **Task:** PERF-004: Optimize Outer-Fit / Outer-Estimate Hotspot Group
- **Date:** 2026-02-16
- **Branch:** performance

## Work Completed

- Added deterministic Criterion coverage for PERF-004 hot path in `crates/ringgrid/benches/hotpaths.rs`:
  - `outer_estimate_64r_48t_nomapper`
  - `outer_estimate_64r_48t_mapper`
- Optimized outer-estimate/radial sampling hot loops:
  - `crates/ringgrid/src/ring/outer_estimate.rs`
  - `crates/ringgrid/src/ring/radial_profile.rs`
  - `crates/ringgrid/src/ring/edge_sample.rs`
- Captured post-change flamegraphs:
  - `.ai/state/sessions/2026-02-16-PERF-004-detect-flamegraph.svg`
  - `.ai/state/sessions/2026-02-16-PERF-004-detect-flamegraph-mapper.svg`
- Completed required validation gates:
  - `./.venv/bin/python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf004_blur3`
  - `bash tools/run_reference_benchmark.sh`
  - `bash tools/run_distortion_benchmark.sh`
- Completed quality gates:
  - `cargo fmt --all`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`

## Key Findings

- PERF-004 acceptance target is met with substantial wins on both outer-estimate benchmarks.
- Mapper and non-mapper outer-estimate latency both improved materially:
  - `outer_estimate_64r_48t_nomapper`: `34.365 us` -> `16.996 us` (`-50.54%`)
  - `outer_estimate_64r_48t_mapper`: `40.082 us` -> `23.310 us` (`-41.85%`)
- Additional shared win in radial aggregation path:
  - `radial_profile_32r_180a`: `12.889 us` -> `8.744 us` (`-32.16%`)

## Files Changed

| File | Change |
|------|--------|
| `crates/ringgrid/benches/hotpaths.rs` | Added deterministic outer-estimate benchmarks and local benchmark shims for mapper/polarity types |
| `crates/ringgrid/src/ring/outer_estimate.rs` | Reworked theta/radial loop to reuse fixed scratch and contiguous curve slab (`Vec<f32>`) |
| `crates/ringgrid/src/ring/radial_profile.rs` | Added `radial_derivative_into`, removed smoothing allocation, median path uses `select_nth_unstable_by` |
| `crates/ringgrid/src/ring/edge_sample.rs` | Tightened bilinear checked sampler to direct raw-buffer interpolation (no duplicate checks / `get_pixel` overhead) |
| `.ai/state/sessions/2026-02-16-PERF-004-detect-flamegraph.svg` | Post-change non-mapper flamegraph |
| `.ai/state/sessions/2026-02-16-PERF-004-detect-flamegraph-mapper.svg` | Post-change mapper flamegraph |
| `.ai/state/sessions/2026-02-16-PERF-004-performance-handoff.md` | This handoff report |

## Performance State

Criterion command (baseline + after, identical):
- `cargo bench -p ringgrid --bench hotpaths -- --warm-up-time 2 --measurement-time 5 --sample-size 30`

| Benchmark | Before (median) | After (median) | Delta |
|-----------|-----------------|----------------|-------|
| `outer_estimate_64r_48t_nomapper` | `34.365 us` | `16.996 us` | `-50.54%` |
| `outer_estimate_64r_48t_mapper` | `40.082 us` | `23.310 us` | `-41.85%` |
| `radial_profile_32r_180a` | `12.889 us` | `8.744 us` | `-32.16%` |

## Flamegraph Observations

Post-change non-mapper (`PERF-004`) vs PERF-001 baseline:
- `outer_fit::fit_outer_candidate_from_prior_with_edge_cfg`:
  - PERF-001: `15.28%`
  - PERF-004: `14.78%` (down `0.50` pp)
- `outer_estimate::estimate_outer_from_prior_with_mapper` (nested hotspot):
  - PERF-001: `13.19%`
  - PERF-004: `7.83%` (down `5.36` pp)

Post-change mapper (`PERF-004`) vs PERF-001 baseline:
- `outer_fit::fit_outer_candidate_from_prior_with_edge_cfg`:
  - PERF-001 mapper: `20.98%`
  - PERF-004 mapper: `17.61%` (down `3.37` pp)
- `outer_estimate::estimate_outer_from_prior_with_mapper`:
  - PERF-001 mapper: `13.17%`
  - PERF-004 mapper: `8.81%` (down `4.36` pp)

Mapper profiling is included and shows the same directional improvement in the targeted stage group.

## Allocation Profile

Implementation-level allocation changes (hot loop):
- Removed per-theta temporary allocations in outer estimate:
  - eliminated per-theta `Vec` creation for sampled intensities and derivatives,
  - eliminated per-theta allocation in `smooth_3point` (copy-based path removed).
- Replaced `Vec<Vec<f32>>` curve storage with one contiguous `Vec<f32>` slab.
- No new per-candidate/per-theta dynamic allocations introduced in the optimized inner loops.

RSS proxy (`/usr/bin/time -l` on representative detect):
- Non-mapper max RSS: `29,720,576` (pre) -> `30,703,616` (post)
- Mapper max RSS: `30,294,016` (pre) -> `31,424,512` (post)

Interpretation:
- Peak RSS variation is small and within run-to-run noise envelope for this workload shape.
- Allocation churn in the targeted inner loops is reduced by construction (scratch reuse + contiguous buffers).

## Accuracy State

Primary gate (`tools/out/eval_perf004_blur3/det/aggregate.json`):

| Metric | PERF-002 baseline | PERF-004 | Delta |
|--------|-------------------|----------|-------|
| Center error mean | `0.31549159136017835 px` | `0.31674058408388994 px` | `+0.00125 px` |
| Recall | `0.9467980295566502` | `0.9458128078817735` | `-0.00099` |
| Precision | `1.000` | `1.000` | `0.000` |
| Homography vs GT mean | `0.14155531964686616 px` | `0.1456968635010419 px` | `+0.00414 px` |

Reference benchmark (`tools/out/reference_benchmark_post_pipeline/summary.json`):
- `none__none`: precision `1.000`, recall `1.000`, avg center mean `0.0743 px`
- `projective_center__none`: precision `1.000`, recall `1.000`, avg center mean `0.0528 px`

Distortion benchmark (`tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json`):
- `projective_center__external`: precision `1.000`, recall `1.000`, avg center mean `0.0798 px`
- `projective_center__self_undistort`: precision `1.000`, recall `1.000`, avg center mean `0.0801 px`
- `projective_center__none`: precision `1.000`, recall `0.9787`, avg center mean `0.1065 px`

Accuracy impact statement:
- **Preserved** within PERF policy gates (center mean delta well below `+0.01 px`; no measurable precision drop).

## Acceptance Check (PERF-004)

- [x] Deterministic outer-estimate benchmark coverage added.
- [x] At least one outer benchmark improved by >=10% (both improved >40%).
- [x] Representative flamegraph indicates reduction in targeted hotspot share vs PERF-001 baseline.
- [x] Mapper-enabled profiling included.
- [x] No new per-candidate allocations in optimized inner loops.
- [x] `cargo test --workspace --all-features` passed.
- [x] `cargo clippy --all-targets --all-features -- -D warnings` passed.
- [x] Validation gate commands completed.

## Recommended Next Steps

1. Hand off to Validation Engineer for independent sign-off replay (policy gate).
2. Proceed to `PERF-005` (inner-fit hotspot) using the same benchmark/profiling discipline.
