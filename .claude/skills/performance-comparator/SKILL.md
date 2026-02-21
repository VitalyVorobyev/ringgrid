---
name: performance-comparator
description: Benchmark old vs new algorithm, quantify latency impact, and write a benchmark report with an accuracy-per-microsecond tradeoff summary.
---

## When to use
After A/B implementation exists and correctness evidence is ready; now quantify runtime impact.

## Prerequisites
- A/B toggle exists (old/new selectable)
- Synthetic eval exists (accuracy + robustness) to pair with performance
- `templates/benchmark-report.md` is available

## Outputs (deliverables)
- Filled `templates/benchmark-report.md` (or a report file derived from it)
- Handoff note → Pipeline Architect + Algorithm Engineer with findings

## Benchmark requirements
1. Benchmark **old vs new** on representative inputs:
   - per-function latency
   - per-detect-call latency (end-to-end where relevant)
2. Report:
   - median and tail (p95) if feasible
   - variance notes (warmup, caching, CPU scaling)
3. If slower:
   - quantify **accuracy-per-microsecond** tradeoff
   - identify hotspots and whether optimization is plausible

## Report checklist
- Environment (CPU, build profile, feature flags)
- Inputs / dataset description
- Old vs new latency table
- Interpretation:
  - faster / same / slower
  - whether regression is acceptable given accuracy gains
- Reproduction instructions

## Handoff note template (to Pipeline Architect + Algorithm Engineer)
- Summary: adopt / caution / reject recommendation (performance-only)
- Key numbers (median, p95 deltas)
- Where time went (top functions, algorithmic reason)
- Next optimization ideas (only if needed and realistic)
