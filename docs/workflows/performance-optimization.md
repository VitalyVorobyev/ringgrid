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
5. Run validation gates (see role spec) to verify accuracy is preserved
6. Write handoff note → Pipeline Architect with benchmark results and validation gate results

**Deliverables:** Optimized code, Criterion benchmarks, before/after numbers, validation gate results, handoff note

### 3. Finalize (Performance Engineer)

**Goal:** Document results and hand off.

**Steps:**
1. Fill in benchmark report from `templates/benchmark-report.md`
2. Document optimization in commit message:
   ```
   perf(proposal): reduce gradient voting allocation by reusing scratch buffer

   proposal_1280x1024: 1.2ms → 0.9ms (-25%)
   Accuracy: unchanged (center error mean 0.054 px)
   ```
3. Write handoff note → Project Lead with benchmark report and validation results

**Deliverables:** Benchmark report, handoff note

### 4. Close-Out (Project Lead)

**Goal:** Verify acceptance criteria, update tracking, present to human.

**Steps:**
1. Review all handoff notes for the task
2. Verify acceptance criteria from task spec are met
3. Verify validation gates passed (check reported results)
4. Update `state/backlog.md` — mark task done
5. Write session summary
6. Present to human for review and merge

**Deliverables:** Updated backlog, session summary
