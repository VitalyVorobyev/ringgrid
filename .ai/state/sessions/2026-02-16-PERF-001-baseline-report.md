# Benchmark Report: PERF-001

- **Date:** 2026-02-16
- **Baseline commit:** `7598db4`
- **Platform:** macOS 26.3 (25D125), Apple M4 Pro, Darwin 25.3.0
- **Toolchain:** `rustc 1.93.0`, `cargo 1.93.0`
- **Build:** `cargo build --release`

## Summary

Deterministic Criterion baseline and fresh flamegraphs were captured for both plain `detect()` and mapper-enabled `detect_with_mapper`; proposal stage remains dominant while mapper mode adds substantial total latency.

## Criterion Baseline

Command:
- `cargo bench -p ringgrid --bench hotpaths -- --warm-up-time 2 --measurement-time 5 --sample-size 30`

| Benchmark | Baseline (median) |
|-----------|-------------------|
| `proposal_1280x1024` | `42.404 ms` |
| `proposal_1920x1080` | `60.614 ms` |
| `radial_profile_32r_180a` | `12.602 us` |
| `ellipse_fit_50pts` | `267.55 ns` |

## Flamegraph Analysis

Artifact:
- `.ai/state/sessions/2026-02-16-PERF-001-detect-flamegraph.svg`

Command:
- `cargo flamegraph --release -p ringgrid-cli --bin ringgrid -o .ai/state/sessions/2026-02-16-PERF-001-detect-flamegraph.svg -- detect --image tools/out/synth_002/img_0000.png --out /tmp/ringgrid_det_perf001.json --marker-diameter 32.0`

Top-3 wall-time hotspots (single representative run):
1. `ringgrid::detector::proposal::find_proposals` — `61.11%`
2. `ringgrid::detector::outer_fit::fit_outer_candidate_from_prior_with_edge_cfg` — `15.28%`
3. `ringgrid::detector::inner_fit::fit_inner_ellipse_from_outer_hint` — `13.89%`

## Mapper-Enabled Profile

Artifact:
- `.ai/state/sessions/2026-02-16-PERF-001-detect-flamegraph-mapper.svg`

Command:
- `cargo flamegraph --release -p ringgrid-cli --bin ringgrid -o .ai/state/sessions/2026-02-16-PERF-001-detect-flamegraph-mapper.svg -- detect --image tools/out/synth_002/img_0000.png --out /tmp/ringgrid_det_mapper.json --marker-diameter 32.0 --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 --cam-k1=-0.15 --cam-k2=0.05 --cam-p1=0.001 --cam-p2=-0.001 --cam-k3=0.0`

Top-3 wall-time hotspots (single representative mapper run):
1. `ringgrid::detector::proposal::find_proposals` — `44.88%`
2. `ringgrid::detector::inner_fit::fit_inner_ellipse_from_outer_hint` — `22.93%`
3. `ringgrid::detector::outer_fit::fit_outer_candidate_from_prior_with_edge_cfg` — `20.98%`

Mapper-specific signal in flamegraph:
- `ringgrid::pipeline::run::detect_with_mapper` frame dominates total run (`95.61%` inclusive).
- `ringgrid::pixelmap::cameramodel::CameraModel::distort_pixel` appears explicitly (`3.41%`), while additional mapper cost is folded into outer/inner estimation stacks.

Wall-clock comparison on same image (`tools/out/synth_002/img_0000.png`, 8 runs each, `/usr/bin/time -p`):

| Mode | Mean real time |
|------|----------------|
| no mapper | `0.13875 s` |
| mapper enabled | `0.19500 s` |

Relative increase with mapper: `+40.54%`.

## Allocation Profile

### Tooling Attempt (preferred)

Tried to capture allocation counts with Xcode Instruments Allocations (`xctrace`/`cargo instruments`), but this environment does not produce reliable CLI traces:
- short-lived process attach failure:
  - `Failed to attach to target process`
- wrapper-launch fallback blocked by SIP restrictions:
  - `Target process is marked restricted and cannot be traced while System Integrity Protection is enabled`

### Fallback Evidence (captured)

Measured memory footprint proxy via `/usr/bin/time -l`:

| Workload | Command | Max RSS |
|----------|---------|---------|
| Representative detect | `target/release/ringgrid detect --image tools/out/synth_002/img_0000.png --out /tmp/perf_alloc_std_det.json --marker-diameter 32.0` | `30,605,312 bytes` |
| Stress-size detect | `target/release/ringgrid detect --image tools/out/perf_alloc_large/img_0000.png --out /tmp/perf_alloc_large_det.json --marker-diameter 32.0` | `368,918,528 bytes` |

Notes:
- This satisfies baseline memory-shape documentation, but not exact allocation-count attribution.
- Follow-up recommendation: add an in-process allocation-counter harness for CLI detect path if strict allocation counts are required in CI.

## Ordered Optimization Candidates (for PERF-002+)

1. `detector::proposal::find_proposals` (owner: Performance Engineer)
   - Expected impact: high (`~61%` hotspot share)
   - Risk: medium (touches core candidate-recall stage)
   - Initial tactics: remove repeated pixel accessor overhead, hoist invariants, reduce branch pressure in voting/NMS loops, evaluate gradient buffer reuse.
2. `detector::outer_fit` + `ring::outer_estimate` (owner: Performance Engineer, Algorithm Engineer if math tradeoffs needed)
   - Expected impact: medium-high (`~15%` + shared radial/profile costs)
   - Risk: medium-high (edge selection quality coupling)
   - Initial tactics: reduce per-candidate sort/allocation work, cache reusable radial buffers.
3. `detector::inner_fit` (owner: Performance Engineer)
   - Expected impact: medium (`~14%`)
   - Risk: medium (inner edge robustness)
   - Initial tactics: avoid repeated temporary vectors in per-theta profile/aggregation path.

## Validation Requirements For PERF Follow-Ups

Every optimization handoff must include:
1. `./.venv/bin/python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir <...>`
2. `bash tools/run_reference_benchmark.sh`
3. `bash tools/run_distortion_benchmark.sh`
4. Mean center-error delta gate: escalate if `> +0.01 px` vs baseline.

## Verification

- `cargo fmt --all` (already clean in current workspace)
- `cargo clippy --all-targets --all-features -- -D warnings` (pass)
- `cargo test --workspace --all-features` (pass)
