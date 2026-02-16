---
name: criterion-bench
description: Apply this skill when adding or updating Criterion benchmarks for ringgrid. It standardizes deterministic benchmark inputs, naming conventions, benchmark structure, and before/after reporting for fair performance comparisons.
metadata:
  short-description: Criterion benchmark setup and reporting
---

# Criterion Bench

Use this skill for benchmark harness creation, benchmark updates, and performance evidence in handoffs.

## Benchmark Scope Rules

1. Benchmark operations, not whole refactors.
- One benchmark file per hotspot family when possible.
2. Use deterministic inputs.
- Fixed seeds, fixed image sizes, fixed candidate counts.
3. Keep names comparable across revisions.
- Format: `operation_inputshape` (example: `proposal_1280x1024`).
4. Minimize benchmark noise.
- Avoid I/O and one-time setup inside measured loops.

## Setup Checklist

1. Add Criterion in crate dev-dependencies where benches live.
2. Add benchmark target entries if required by Cargo layout.
3. Place bench sources under `crates/ringgrid/benches/`.
4. Construct deterministic fixtures in benchmark setup code.
5. Benchmark both baseline and changed path when practical.

## Measurement Discipline

- Warmup and sample settings should stay stable between comparisons.
- Compare like-for-like inputs only.
- Report median or mean consistently within the same task.

## Required Reporting

For each benchmark include:
- Benchmark name.
- Before and after values.
- Percent change.
- Notes on input shape and fixture seed.

## Anti-Patterns

- Benchmarks whose input randomness changes per run.
- Mixing expensive setup into timed sections.
- Renaming benchmarks between commits and losing comparability.
