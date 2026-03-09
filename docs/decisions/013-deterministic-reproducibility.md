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

### Generated artifacts

- Codebook generation is reproducible via `tools/gen_codebook.py`.
- The committed shipped codebook artifacts are generated with `--n 893 --seed 1`
  and written to `tools/codebook.json` and
  `crates/ringgrid/src/marker/codebook.rs`.
- The achieved minimum cyclic Hamming distance recorded in those generated
  outputs is part of the reproducible artifact contract.

### Invariant

Given the same input image and `DetectConfig`, `Detector::detect` must
produce identical output across runs. No thread-local RNG, no
`rand::thread_rng()`, no time-based seeds in library code.
