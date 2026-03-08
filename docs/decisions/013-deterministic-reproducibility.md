# DEC-013: Deterministic Reproducibility

**Status:** Active
**Date:** 2025

## Decision

All stochastic algorithms use seeded RNGs for deterministic reproducibility.

### RANSAC seeds

- `RansacConfig.seed` field controls the PRNG seed (default: 42).
- Outer ellipse RANSAC uses seed 42.
- Inner ellipse RANSAC uses seed 43 (different to avoid correlated failures).
- Homography RANSAC uses its own configurable seed.

### Test fixtures

- All test randomness uses `StdRng::seed_from_u64(N)` with documented seeds.
- Benchmark fixtures use seeded RNGs for stable performance baselines.
- Synthetic image generation (`tools/gen_synth.py`) accepts `--seed` for
  reproducible test images.

### Invariant

Given the same input image and `DetectConfig`, `Detector::detect` must
produce identical output across runs. No thread-local RNG, no
`rand::thread_rng()`, no time-based seeds in library code.
