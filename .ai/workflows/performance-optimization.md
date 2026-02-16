# Performance Optimization Workflow

Use this workflow for: latency reduction, throughput improvement, allocation elimination, or cache-locality optimization.

## Prerequisites
- Performance concern identified (profiling data, user report, or benchmark regression)
- Task added to `state/backlog.md`

## Phases

### 1. Baseline Measurement (Performance Engineer)

**Goal:** Establish quantified baseline and identify optimization targets.

**Steps:**
1. Establish Criterion benchmark baseline for affected functions:
   - Representative image sizes: 1280x1024, 1920x1080
   - Deterministic seeded inputs
2. Profile with flamegraph:
   ```bash
   cargo flamegraph --release -- detect --image <path> --out /dev/null --marker-diameter 32.0
   ```
3. Identify top-3 hotspots by wall-clock contribution
4. Document allocation profile (total allocations per `detect()` call)
5. Write handoff note → Algorithm Engineer (if algorithmic change needed) or proceed to implementation

**Deliverables:** Baseline benchmarks, profiling report, hotspot identification, handoff note

### 2. Implementation (Performance Engineer + Algorithm Engineer)

**Goal:** Optimize the identified hotspots while maintaining correctness.

**Roles:**
- **Performance Engineer:** Buffer reuse, branch elimination, memory layout, loop restructuring
- **Algorithm Engineer:** If algorithmic change needed (e.g., early termination in RANSAC, coarser-to-finer search, approximate computation)

**Steps:**
1. Implement optimization in the target module
2. Follow `/hotpath-rust` patterns:
   - Reuse scratch buffers (Vec fields on struct, clear + resize)
   - No per-row/per-col allocations in inner loops
   - Minimal unsafe with documented invariants
   - Keep safe reference path for testing
3. Add/update Criterion benchmarks for each changed function
4. Run benchmarks and record before/after:
   ```
   [bench_name]: baseline_ns → optimized_ns (±X%)
   ```
5. Write handoff note → Validation Engineer with benchmark results

**Deliverables:** Optimized code, Criterion benchmarks, before/after numbers, handoff note

### 3. Accuracy Verification (Validation Engineer)

**Goal:** Confirm optimization preserves detection accuracy.

**Steps:**
1. Run baseline synthetic eval:
   ```bash
   python3 tools/run_synth_eval.py --n 5 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf
   ```
2. Run challenging synthetic eval:
   ```bash
   python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_perf_blur3
   ```
3. Run reference and distortion benchmark scripts:
   ```bash
   bash tools/run_reference_benchmark.sh
   bash tools/run_distortion_benchmark.sh
   ```
4. Compare center error statistics against baseline:
   - Mean, p50, p95, max center error
   - Decode success rate
   - Homography reprojection error
5. **Flag any accuracy delta > 0.01 px mean center error** — requires Algorithm Engineer review
6. Fill in accuracy report from `templates/accuracy-report.md`, including blur=3.0 and reference/distortion script outputs
7. Write handoff note → Performance Engineer

**Deliverables:** Accuracy report, scoring comparison, handoff note

### 4. Finalize (Performance Engineer)

**Goal:** Document results and close.

**Steps:**
1. Fill in benchmark report from `templates/benchmark-report.md`
2. Document optimization in commit message:
   ```
   perf(proposal): reduce gradient voting allocation by reusing scratch buffer

   proposal_1280x1024: 1.2ms → 0.9ms (-25%)
   Accuracy: unchanged (center error mean 0.054 px)
   ```
3. Update `state/backlog.md` — mark task done
4. Human reviews and merges
