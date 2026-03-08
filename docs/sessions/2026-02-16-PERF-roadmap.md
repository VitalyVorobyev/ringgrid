# PERF Roadmap (Project Lead)

- **Date:** 2026-02-16
- **Goal:** Establish rigorous profiling, then optimize by measured impact.

## Sequence

1. **PERF-001 (in-progress, Performance Engineer)**
   - Build deterministic Criterion harness and detect-level profiling baseline.
   - Produce hotspot ranking (top-3) + allocation profile.
   - Deliver optimization recommendation list with expected impact.
2. **PERF-002 (up next, Performance Engineer)**
   - Optimize highest-impact hotspot from PERF-001 data.
   - Record before/after benchmark deltas and implementation notes.
   - Hand off to Validation Engineer for accuracy/perf verification.
3. **PERF-003 (planned, Validation Engineer)**
   - Standardize performance validation suite for all perf-sensitive changes:
     - challenging blur=3.0 synth eval batch
     - `tools/run_reference_benchmark.sh`
     - `tools/run_distortion_benchmark.sh`
   - Ensure reports include comparable baseline vs after-change deltas.

## Handoff Guidance

- Start with **Performance Engineer** now (PERF-001 baseline phase).
- Hand off to **Algorithm Engineer** only if top bottleneck requires mathematical/algorithmic change.
- Keep **Pipeline Architect** involvement only if API/config/module-boundary changes become necessary.
