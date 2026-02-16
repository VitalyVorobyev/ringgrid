# Performance Optimization Workflow

Use this workflow for: latency reduction, throughput improvement, allocation elimination, or cache-locality optimization.

## Prerequisites
- Performance concern identified (profiling data, user report, or benchmark regression)
- Task added to `state/backlog.md`
- PERF validation uses the standardized suite in:
  - `.ai/workflows/perf-validation-suite-runbook.md`
  - `.ai/templates/accuracy-report.md`

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
1. Run the standardized blur=3.0 gate (`n=10`) script and snapshot baseline/after:
   ```bash
   bash tools/run_blur3_benchmark.sh
   rm -rf tools/out/eval_<label>_blur3
   cp -R tools/out/eval_blur3_post_pipeline tools/out/eval_<label>_blur3
   ```
2. Run reference benchmark script and preserve summary for each label:
   ```bash
   bash tools/run_reference_benchmark.sh
   cp tools/out/reference_benchmark_post_pipeline/summary.json tools/out/reference_benchmark_post_pipeline_<label>.summary.json
   ```
3. Run distortion benchmark script and preserve summary for each label:
   ```bash
   bash tools/run_distortion_benchmark.sh
   cp tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline_<label>.summary.json
   ```
4. Compare baseline vs after for all three gates:
   - Mean, p50, p95, max center error
   - Precision/recall deltas
   - Homography self and vs-GT deltas
   - Frame consistency (expected `image` frame in scorer output)
5. **Flag any blur=3 mean center delta > +0.01 px** — requires Algorithm Engineer review.
6. Flag any blur=3 homography mean delta (`self` or `vs-GT`) > `+0.02 px` for investigation/escalation.
7. Fill `.ai/templates/accuracy-report.md` with the three gate tables and artifact paths.
8. Write handoff note using `.ai/templates/handoff-note.md` including PERF gate artifact paths and deltas.

**Deliverables:** Accuracy report, gate artifact bundle, standardized handoff note

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
