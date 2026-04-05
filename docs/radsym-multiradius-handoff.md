# Handoff: Multi-Radius Fused Voting for radsym (FRST + RSD)

## Problem

ringgrid replaced its internal proposal stage with radsym's RSD. The integration
is functionally correct (all 4 regression benchmarks pass), but RSD is **~6-10x
slower** than the old internal code on proposal generation due to per-radius
Gaussian blur overhead.

| Benchmark (1280x1024)  | Old (internal) | radsym RSD | Slowdown |
|-------------------------|----------------|------------|----------|
| proposal_1280x1024      | 24.8 ms        | 142 ms     | 5.7x     |
| proposal_1920x1080      | 33.8 ms        | 194 ms     | 5.7x     |
| propose_fixture (720x540) | 15.9 ms      | 34 ms      | 2.1x    |

The same bottleneck applies to FRST, which is even slower (~13x) due to
additional O_n normalization per radius.

## Root Cause

Both FRST and RSD process each radius independently:

```
for each radius n in config.radii:
    1. Vote: scan all strong-gradient pixels → per-radius accumulator
    2. Blur: Gaussian blur with sigma = smoothing_factor * n
    3. Accumulate: sum += per_radius_result
```

This means N full-image Gaussian blurs for N radii. The old ringgrid code did it
in **one** pass:

```
1. Vote: for each strong-gradient pixel, for each r in [r_min..r_max]:
     shared_accumulator[pixel ± gradient_dir * r] += 1
2. Blur: ONE Gaussian blur on the shared accumulator
3. NMS: extract peaks
```

The old code accumulates all radii into one buffer in one nested loop, then
blurs once. The per-radius Gaussian blur is the dominant cost — not the voting.

## Proposed Solution: Fused Multi-Radius Mode

Add `rsd_response_fused` and `frst_response_fused` functions that accumulate all
radii into a single shared buffer and apply one global blur, matching the old
algorithm's efficiency.

### API

```rust
/// RSD with all radii fused into a single voting + blur pass.
pub fn rsd_response_fused(
    gradient: &GradientField,
    config: &RsdConfig,
) -> Result<ResponseMap>

/// FRST with all radii fused into a single voting + blur pass.
/// Drops per-radius O_n normalization in favor of direct magnitude voting.
pub fn frst_response_fused(
    gradient: &GradientField,
    config: &FrstConfig,
) -> Result<ResponseMap>
```

Both return the same `ResponseMap` type — downstream NMS is unchanged.

### Algorithm (shared by both)

```
1. Allocate ONE shared accumulator (w × h), zero-initialized
2. Compute gradient threshold (absolute, from config)
3. For each pixel (x, y) with |gradient| >= threshold:
     mag = gradient magnitude
     (dx, dy) = normalized gradient direction
     For each radius n in config.radii:
       // Vote along gradient direction at distance n
       p_pos = (x + round(dx * n), y + round(dy * n))
       p_neg = (x - round(dx * n), y - round(dy * n))
       if polarity allows positive: accumulator[p_pos] += mag
       if polarity allows negative: accumulator[p_neg] += mag
4. ONE Gaussian blur with sigma = smoothing_factor * median(radii)
5. Wrap in ResponseMap
```

The inner loop over radii is **per pixel**, not per radius over all pixels.
This means one pass over the image instead of N passes.

### Key Design Decisions

1. **Single shared accumulator**: All radii vote into one buffer. This loses
   per-radius isolation but matches the behavior of ringgrid's old code, which
   never had per-radius separation and achieved excellent accuracy.

2. **Single global blur**: Instead of `sigma = kn * n` per radius, apply one
   blur with `sigma = smoothing_factor * median(radii)`. Validated by ringgrid's
   regression benchmarks.

3. **Magnitude-weighted voting for both**: Both fused variants use magnitude
   weighting (`+= mag`). For `frst_response_fused`, this means dropping the
   `|O_n|^alpha` orientation consistency check — the cost of per-radius O_n
   tracking negates the performance benefit. For ring/circle detection (where
   gradients converge symmetrically), the orientation check has negligible effect.

4. **Keep existing per-radius functions**: `rsd_response` and `frst_response`
   remain unchanged for use cases that need per-radius analysis (e.g., scale
   estimation, diagnostic heatmaps per radius).

### Expected Performance

The fused version does:
- 1 pass over pixels (inner loop: N radius offsets per pixel — fast, cache-friendly)
- 1 Gaussian blur (the previous dominant cost, now O(w×h) instead of O(N×w×h))
- 1 NMS pass

For the ringgrid benchmark configs (15-53 radii on 1280x1024):
- Expected: **~20-30ms** — matching or beating the old internal code (24.8ms)
- Current: 142-245ms (5.7-10x slower)

### Implementation Notes

- Both functions should live alongside their existing counterparts:
  - `rsd_response_fused` in `src/propose/rsd.rs`
  - `frst_response_fused` in `src/propose/frst.rs`
- Share the inner voting loop via a private helper to avoid duplication
- The `rayon` feature can parallelize the pixel loop over row chunks
- Re-export both at crate root: `pub use crate::propose::rsd::rsd_response_fused`

### Validation

ringgrid's regression benchmarks provide end-to-end validation:
```bash
# From the ringgrid repo:
bash tools/run_reference_benchmark.sh    # 2 modes, synthetic
bash tools/run_distortion_benchmark.sh   # 3 modes, synthetic + distortion
bash tools/run_blur3_benchmark.sh        # heavy blur stress test
.venv/bin/python tools/run_rtv3d_eval.py # 120 real-world tiles (local data)
# Check against tools/ci/regression_baseline.json
```

radsym's own `ringgrid_proposal_bench` benchmark can track raw RSD/FRST timing.

### Integration in ringgrid

Once available, the ringgrid adapter (`crates/ringgrid/src/proposal/mod.rs`)
switches one call:

```rust
// Before:
let response = radsym::rsd_response(&gradient, &rsd_config)?;

// After:
let response = radsym::rsd_response_fused(&gradient, &rsd_config)?;
```

No other changes needed. The `RsdConfig` type is shared.
