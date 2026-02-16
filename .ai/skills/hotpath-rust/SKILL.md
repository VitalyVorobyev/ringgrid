---
name: hotpath-rust
description: Use this skill for Rust performance work in ringgrid hot loops. It provides guardrails for allocation control, cache-friendly iteration, branch reduction, safe-fast patterns, and accuracy-preserving optimization.
metadata:
  short-description: Rust hot-loop optimization patterns
---

# Hotpath Rust

Use this skill when modifying proposal, sampling, fitting, RANSAC, or any per-pixel/per-candidate loop.

## Optimization Priorities

1. Remove repeated allocations.
- Reuse owned scratch buffers; `clear`/`resize` instead of recreate.
2. Improve memory access locality.
- Prefer contiguous traversal and predictable access patterns.
3. Reduce branch cost in inner loops.
- Hoist invariant checks and constants out of loop bodies.
4. Keep parallelism deliberate.
- Rayon only when task size amortizes scheduling overhead.
5. Preserve numerical behavior.
- Any measurable metric shift must be reported for validation.

## Safe-Fast Workflow

1. Establish the hotspot and baseline metric first.
2. Apply one optimization class at a time.
3. Re-run benchmarks and compare before/after.
4. Confirm no material accuracy regression.

## Unsafe Policy

- Prefer safe Rust.
- If `unsafe` is required, place the safety invariant directly above the block.
- Keep or add a safe reference path for correctness comparison.

## Reporting Format

Include:
- Hot path touched.
- Before/after timing and percent delta.
- Allocation behavior change.
- Accuracy impact (`preserved`, `changed`, or `needs validation`).

## Anti-Patterns

- Micro-optimizing without a measured baseline.
- Adding convenience allocations in inner loops.
- Broad refactors that mix architecture and optimization in one step.
