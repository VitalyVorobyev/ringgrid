---
name: tests-synthetic-fixtures
description: Use this skill when creating or updating deterministic synthetic tests and regression fixtures for ringgrid. It standardizes seeded data generation, measurable pass criteria, and reproducible Rust/Python validation flows.
metadata:
  short-description: Deterministic synthetic fixtures and regression tests
---

# Tests Synthetic Fixtures

Use this skill to turn bugs, regressions, and algorithm assumptions into reproducible test cases.

## Fixture Strategy

1. Start from the smallest failing case.
- Prefer compact numeric fixtures for math primitives.
- Use full synthetic images only when pipeline behavior is required.
2. Make determinism explicit.
- Fix RNG seeds and record generation parameters.
3. Separate "debug run artifacts" from committed fixtures.
- Iterate under `tools/out/...`.
- Commit only minimal assets needed for stable regression coverage.

## Rust-Focused Pattern

1. Add/extend unit tests near the touched module (`#[cfg(test)]`).
2. Use `approx` assertions and document numeric tolerances.
3. Cover one nominal case and at least one edge case (degenerate geometry, border proximity, sparse inliers, or extreme scale).

## Python/E2E Pattern

1. Generate deterministic synthetic input with fixed parameters.
2. Run detector with explicit config knobs used in the report.
3. Score with a fixed gate and record precision/recall + center/homography stats.
4. If this is a regression bug, encode the failing mode as a repeatable evaluation scenario.

## Required Reporting

For each fixture or scenario, record:
- Seed and generation parameters.
- Expected metrics and thresholds.
- Pass/fail delta versus baseline (if baseline exists).

## Common Mistakes

- Non-seeded generation that causes flaky tests.
- Large binary fixture commits when a generated fixture would suffice.
- Assertions on exact floats instead of tolerance-based checks.
