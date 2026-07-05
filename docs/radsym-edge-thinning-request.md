# Handoff: Edge-Thinning (Gradient-Direction NMS) for radsym RSD

## Problem

The proposal stage (radsym RSD voting) is the dominant cost in ringgrid
detection — **~50–54 % of end-to-end time** (per-stage `StageTimings`, Apple
M4 Pro, release build):

| image | proposal | total | proposal share |
|---|---|---|---|
| real 720×540, 78 markers | 15.2 ms | 28.5 ms | **53 %** |
| synthetic 1280×960, 203 markers | 32 ms | 65.8 ms | **49 %** |

Every above-threshold gradient pixel votes. Real edges are multi-pixel-wide
bands (3–5 px after blur), so the same boundary votes several times. A
Canny-style non-maximum-suppression pass along the gradient direction thins
those bands to single-pixel ridges **before** voting, which historically cut the
strong-edge count by **60–80 %** and reduced the voting workload
proportionally. ringgrid's pre-radsym internal proposal stage did exactly this.

ringgrid added an opt-in `ProposalConfig::radius_step` knob that subsamples the
radius axis, but it is **off by default**: at `radius_step = 2` it cuts proposal
time ~29 % while lowering real-world recall (rtv3d −2.9 %), so it does not pass
the accuracy gate as a default. The larger, orthogonal, *accuracy-preserving*
win — thinning the *gradient* axis (fewer voting pixels, same radii) — is only
reachable inside radsym.

## Why ringgrid cannot do this today

Edge-thinning must run on the gradient field that RSD votes from. As of the
current dependency (**radsym 0.4**, verified against
`crates/radsym/src/propose/rsd.rs` and `crates/radsym/src/core/gradient.rs`)
there is still no way to inject a thinned gradient:

- `RsdConfig` (`radii`, `gradient_threshold`, `polarity`, `smoothing_factor`)
  has **no edge-thinning / NMS knob**.
- `rsd_response_fused(gradient: &GradientField, config: &RsdConfig)` accepts only
  a `&GradientField`.
- `GradientField`'s `gx` / `gy` are private; the only accessors (`gx()`, `gy()`)
  return **read-only** views, and there is **no public constructor** (only
  `sobel_gradient` / `scharr_gradient`, which build from an image). So a caller
  cannot build a `GradientField` from externally-thinned components and pass it
  back in.

The net effect: the documented 60–80 % voting reduction cannot be applied from
ringgrid without a new radsym API (target: **radsym 0.5**).

## Proposed solution (any one of these unblocks ringgrid)

In rough order of preference:

1. **A thinning knob on `RsdConfig`.** Add an opt-in field, e.g.
   `pub edge_thinning: bool` (or an enum for the quantization scheme), so
   `rsd_response_fused` / `rsd_response` run gradient-direction NMS before
   voting. This keeps the policy inside radsym where the voting loop lives and is
   the smallest change for callers.

   ```rust
   pub struct RsdConfig {
       pub radii: Vec<u32>,
       pub gradient_threshold: Scalar,
       pub polarity: Polarity,
       pub smoothing_factor: Scalar,
       pub edge_thinning: bool, // new: NMS along gradient direction pre-vote
   }
   ```

2. **A standalone thinning transform.** Expose
   `pub fn thin_gradient(field: &GradientField) -> GradientField` (4-direction
   quantization, integer `mag²` comparisons, no `atan2`) so callers can thin and
   re-vote:

   ```rust
   let grad = radsym::scharr_gradient(&view)?;
   let thin = radsym::thin_gradient(&grad);
   let resp = radsym::rsd_response_fused(&thin, &cfg)?;
   ```

3. **A public `GradientField` constructor.** e.g.
   `GradientField::from_components(gx: OwnedImage<Scalar>, gy: OwnedImage<Scalar>)`
   (or `gx_mut()` / `gy_mut()`), letting callers implement thinning themselves
   and feed the result back. Most flexible, but pushes the algorithm to callers.

## Reference implementation (4-direction NMS, no atan2)

For each pixel with non-zero gradient:

1. Quantize the gradient direction to one of {0°, 45°, 90°, 135°} via integer
   ratio tests on `(gx, gy)`.
2. Compare the pixel's `gx² + gy²` against its two neighbors along that
   quantized direction.
3. Zero the pixel's gradient if it is not a local maximum.

All comparisons are integer `mag²`, so no floating-point or `atan2` overhead.

## Expected benefit

- Proposal stage could drop from ~50 % toward ~20 % of detection time (the
  60–80 % strong-edge reduction is the historical figure from ringgrid's own
  pre-radsym implementation).
- Composes with `radius_step` subsampling (already in ringgrid) — the two cut
  orthogonal axes (edge count × radius count).

## Acceptance

- Thinning is **opt-in** (default off) so existing results are unchanged unless
  requested.
- ringgrid will gate adoption on its full regression suite
  (`tools/ci/regression_baseline.json`): synthetic reference / distortion / blur
  benchmarks + the rtv3d real-world benchmark must hold their certified
  precision/recall before the thinned path becomes the default.

## Context

- Perf finding: `docs/reviews/2026-06-performance-profiling.md`
- Prior radsym handoff (fused multi-radius voting): `docs/radsym-multiradius-handoff.md`
- ringgrid adapter: `crates/ringgrid/src/proposal/mod.rs` (`compute_via_radsym`)
