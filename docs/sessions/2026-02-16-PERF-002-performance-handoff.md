# Handoff: Performance Engineer → Project Lead

- **Task:** PERF-002: Optimize Proposal Hotspot (`detector::proposal::find_proposals`)
- **Date:** 2026-02-16
- **Branch:** performance

## Work Completed

- Optimized `find_proposals` hot loops in `crates/ringgrid/src/detector/proposal.rs`:
  - switched gradient reads from `get_pixel()` to contiguous raw-buffer access (`as_raw()`),
  - avoided unnecessary sqrt work for below-threshold gradients (`threshold_sq` gate),
  - tightened vote deposition path with in-bounds bilinear helper,
  - precomputed NMS neighbor offsets to remove repeated radius checks in the NMS inner loop.
- Re-ran deterministic Criterion benchmarks before/after with fixed command:
  - `cargo bench -p ringgrid --bench hotpaths -- --warm-up-time 2 --measurement-time 5 --sample-size 30`
- Captured post-change flamegraphs:
  - `.ai/state/sessions/2026-02-16-PERF-002-detect-flamegraph.svg`
  - `.ai/state/sessions/2026-02-16-PERF-002-detect-flamegraph-mapper.svg`
- Completed mandatory validation gates:
  - `./.venv/bin/python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf002_blur3`
  - `bash tools/run_reference_benchmark.sh`
  - `bash tools/run_distortion_benchmark.sh`
- Re-ran quality gates:
  - `cargo fmt --all`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`

## Key Findings

- Proposal benchmarks improved well beyond PERF-002 targets:
  - `proposal_1280x1024`: `42.404 ms` baseline → `33.612 ms` (`-20.73%`)
  - `proposal_1920x1080`: `60.614 ms` baseline → `48.357 ms` (`-20.22%`)
- Proposal hotspot share in representative non-mapper flamegraph decreased:
  - PERF-001 baseline: `61.11%`
  - PERF-002: `57.89%` (down `3.22` pp)
- Mapper-enabled share also decreased:
  - prior mapper profile: `44.88%`
  - PERF-002 mapper profile: `39.38%` (down `5.50` pp)
- Mapper mode remains materially slower in wall-clock terms, but absolute runtime improved:
  - non-mapper mean: `0.13875 s` → `0.12500 s` (`-9.9%`)
  - mapper mean: `0.19500 s` → `0.18750 s` (`-3.8%`)

## Files Changed

| File | Change |
|------|--------|
| `crates/ringgrid/src/detector/proposal.rs` | Proposal hot-loop optimization (raw access + tighter vote/NMS loops) |
| `.ai/state/sessions/2026-02-16-PERF-002-detect-flamegraph.svg` | Post-change non-mapper flamegraph |
| `.ai/state/sessions/2026-02-16-PERF-002-detect-flamegraph-mapper.svg` | Post-change mapper flamegraph |
| `.ai/state/sessions/2026-02-16-PERF-002-performance-handoff.md` | This handoff report |

## Test Results

- **cargo fmt:** clean
- **cargo clippy:** clean (`-D warnings`)
- **cargo test:** pass
  - ringgrid unit tests: 80 passed
  - ringgrid-cli unit tests: 4 passed
  - doc tests: 5 passed

## Accuracy State

Primary gate (`blur=3.0`, `n=10`) from `tools/out/eval_perf002_blur3/det/aggregate.json`:

| Metric | Value |
|--------|-------|
| Center error (mean) | `0.31549159136017835 px` |
| Decode success rate | precision `1.000`, recall `0.9467980295566502` |
| Homography self-error (mean) | `0.2787741806511102 px` |
| Homography vs GT (mean) | `0.14155531964686616 px` |

Reference benchmark summary (`tools/out/reference_benchmark_post_pipeline/summary.json`):
- `none__none`: precision `1.000`, recall `1.000`, avg center mean `0.0743 px`
- `projective_center__none`: precision `1.000`, recall `1.000`, avg center mean `0.0528 px`

Distortion benchmark summary (`tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json`):
- `projective_center__external`: precision `1.000`, recall `1.000`, avg center mean `0.0797 px`
- `projective_center__self_undistort`: precision `1.000`, recall `1.000`, avg center mean `0.0790 px`
- `projective_center__none`: precision `1.000`, recall `0.9787`, avg center mean `0.1062 px`

Accuracy impact statement:
- **Preserved** on required validation gates (no material regression observed).

## Performance State

Criterion (median, same command before/after in this turn):

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| `proposal_1280x1024` | `40.614 ms` | `33.612 ms` | `-17.24%` |
| `proposal_1920x1080` | `57.134 ms` | `48.357 ms` | `-15.36%` |

Acceptance check versus PERF-001 baseline:

| Benchmark | PERF-001 baseline | PERF-002 after | Delta |
|-----------|-------------------|----------------|-------|
| `proposal_1280x1024` | `42.404 ms` | `33.612 ms` | `-20.73%` |
| `proposal_1920x1080` | `60.614 ms` | `48.357 ms` | `-20.22%` |

## Allocation Profile

- No per-pixel, per-row, or per-candidate allocations were introduced.
- Added small per-call helper buffers in proposal stage:
  - `radii` (`Vec<f32>`, length roughly `r_max - r_min + 1`)
  - `nms_offsets` (`Vec<isize>`, bounded by NMS disk area)
- These allocations occur once per `find_proposals` call and are outside hot inner loops.

## Recommended Next Steps

1. Move to PERF-004 (outer-fit / outer-estimate hotspot group) with the same benchmark discipline.
2. If mapper throughput is now priority, target pixel-mapper-heavy segments in outer/inner fit (`edge_sample` + radial aggregation under mapper mode).

## Blocking Issues

None.
