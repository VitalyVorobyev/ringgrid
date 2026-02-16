# Handoff: Performance Engineer -> Project Lead

- **Task:** PERF-005: Optimize Inner-Fit Hotspot Group
- **Date:** 2026-02-16
- **Branch:** performance

## Work Completed

- Added deterministic inner-path benchmark coverage in `crates/ringgrid/benches/hotpaths.rs`:
  - `inner_estimate_64r_96t_nomapper`
  - `inner_estimate_64r_96t_mapper`
  - `inner_fit_64r_96t_nomapper`
  - `inner_fit_64r_96t_mapper`
- Optimized inner hot loops in:
  - `crates/ringgrid/src/ring/inner_estimate.rs`
  - `crates/ringgrid/src/detector/inner_fit.rs`
  - `crates/ringgrid/src/ring/radial_profile.rs` (bench/clippy compatibility annotations only)
- Captured PERF-005 flamegraphs:
  - `.ai/state/sessions/2026-02-16-PERF-005-detect-flamegraph.svg`
  - `.ai/state/sessions/2026-02-16-PERF-005-detect-flamegraph-mapper.svg`
- Completed quality gates:
  - `cargo fmt --all`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
- Completed required validation gates:
  - `./.venv/bin/python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf005_blur3`
  - `bash tools/run_reference_benchmark.sh`
  - `bash tools/run_distortion_benchmark.sh`

## Key Findings

- PERF-005 acceptance threshold is met on inner-fit benchmarks (>=10% improvement vs PERF-005 baseline).
- Main hot-loop changes:
  - removed per-theta temporary `Vec` churn in inner estimate sampling path,
  - reused fixed profile/derivative scratch buffers in inner fit,
  - reduced trig overhead with iterative theta rotation in inner loops,
  - kept algorithm semantics unchanged (same estimator/fit flow).

## Files Changed

| File | Change |
|------|--------|
| `crates/ringgrid/benches/hotpaths.rs` | Added deterministic inner-estimate/inner-fit benchmarks and local bench shims |
| `crates/ringgrid/src/ring/inner_estimate.rs` | Inner sampling/aggregation loop tightening (scratch reuse, contiguous storage, reduced per-theta overhead) |
| `crates/ringgrid/src/detector/inner_fit.rs` | Inner point sampling loop tightening (buffer reuse, reduced per-theta work) |
| `crates/ringgrid/src/ring/radial_profile.rs` | `#[allow(dead_code)]` for shared helper fns used by benches/tests variants |
| `.ai/state/sessions/2026-02-16-PERF-005-detect-flamegraph.svg` | Post-change non-mapper flamegraph |
| `.ai/state/sessions/2026-02-16-PERF-005-detect-flamegraph-mapper.svg` | Post-change mapper flamegraph |
| `.ai/state/sessions/2026-02-16-PERF-005-performance-handoff.md` | This handoff report |

## Performance State

Criterion command (baseline + after, identical):
- `cargo bench -p ringgrid --bench hotpaths -- --warm-up-time 2 --measurement-time 5 --sample-size 30`

PERF-005 baseline (first run after adding inner benches):
- `inner_estimate_64r_96t_nomapper`: `40.726 us`
- `inner_estimate_64r_96t_mapper`: `54.155 us`
- `inner_fit_64r_96t_nomapper`: `68.051 us`
- `inner_fit_64r_96t_mapper`: `93.909 us`

Final after (latest run on final code):
- `inner_estimate_64r_96t_nomapper`: `37.951 us` (`-6.81%`)
- `inner_estimate_64r_96t_mapper`: `51.225 us` (`-5.41%`)
- `inner_fit_64r_96t_nomapper`: `59.358 us` (`-12.77%`)
- `inner_fit_64r_96t_mapper`: `82.433 us` (`-12.22%`)

Acceptance target check:
- At least one inner benchmark improved by >=10%: **met** (`inner_fit_*` both >10%).

## Flamegraph Observations

Reference baseline (PERF-001):
- non-mapper `inner_fit::fit_inner_ellipse_from_outer_hint`: `13.89%`
- mapper `inner_fit::fit_inner_ellipse_from_outer_hint`: `22.93%`

PERF-005 post-change:
- non-mapper `inner_fit::fit_inner_ellipse_from_outer_hint`: `11.50%` (down `2.39` pp)
- mapper `inner_fit::fit_inner_ellipse_from_outer_hint`: `19.38%` (down `3.55` pp)

Additional inner-stage signal:
- non-mapper `inner_estimate::estimate_inner_scale_from_outer_with_mapper`: `7.96%`
- mapper `inner_estimate::estimate_inner_scale_from_outer_with_mapper`: `11.25%`

Mapper-enabled profiling is included and shows reduced inner-fit hotspot share versus PERF-001 reference.

## Allocation Profile

Hot-loop allocation behavior:
- eliminated per-theta `Vec` allocation in inner-fit profile collection,
- eliminated per-theta derivative vector allocation in inner-fit,
- reused fixed scratch buffers for profile/derivative paths,
- no new per-candidate/per-theta dynamic allocations introduced in optimized inner loops.

RSS proxy (`/usr/bin/time -l`, representative detect image):
- PERF-004 context (pre):
  - non-mapper max RSS: `30,703,616`
  - mapper max RSS: `31,424,512`
- PERF-005 post:
  - non-mapper max RSS: `30,638,080`
  - mapper max RSS: `31,195,136`

Interpretation:
- memory footprint shape is stable/slightly lower; optimization is primarily CPU-loop efficiency.

## Accuracy State

Primary gate (`tools/out/eval_perf005_blur3/det/aggregate.json`) vs PERF-004 context:

| Metric | PERF-004 | PERF-005 | Delta |
|--------|----------|----------|-------|
| Center error mean | `0.31674058408388994 px` | `0.3148150563710771 px` | `-0.00193 px` |
| Recall | `0.9458128078817735` | `0.9477832512315271` | `+0.00197` |
| Precision | `1.000` | `1.000` | `0.000` |
| Homography vs GT mean | `0.1456968635010419 px` | `0.1443926835841835 px` | `-0.00130 px` |

Reference benchmark (`tools/out/reference_benchmark_post_pipeline/summary.json`):
- `none__none`: precision `1.000`, recall `1.000`, avg center mean `0.0743 px`
- `projective_center__none`: precision `1.000`, recall `1.000`, avg center mean `0.0529 px`

Distortion benchmark (`tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json`):
- `projective_center__external`: precision `1.000`, recall `1.000`, avg center mean `0.0767 px`
- `projective_center__self_undistort`: precision `1.000`, recall `1.000`, avg center mean `0.0778 px`
- `projective_center__none`: precision `1.000`, recall `0.9770`, avg center mean `0.0948 px`

Accuracy impact statement:
- **Preserved** (center-mean delta is within gate and no measurable precision degradation).

## Acceptance Check (PERF-005)

- [x] Deterministic inner-fit/inner-estimate benchmark coverage added.
- [x] At least one inner benchmark improved by >=10% vs PERF-005 baseline.
- [x] Non-mapper and mapper flamegraphs show reduced inner-fit share vs PERF-001 reference.
- [x] Mapper-enabled profiling included.
- [x] No new per-candidate/per-theta allocations introduced in optimized inner loops.
- [x] `cargo test --workspace --all-features` passed.
- [x] `cargo clippy --all-targets --all-features -- -D warnings` passed.
- [x] Required validation commands completed.

## Recommended Next Steps

1. Hand off to Validation Engineer for independent replay/sign-off.
2. Move to PERF-003 (validation suite standardization) or next hotspot as prioritized.
