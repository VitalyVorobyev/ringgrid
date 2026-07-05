# Handoff: Edge-Thinning (Gradient-Direction NMS) for radsym RSD

> **Status: delivered, not adopted.** radsym **0.4.1** shipped this request as
> option 2 below — the standalone
> `thin_gradient(&GradientField) -> Result<GradientField>` transform. ringgrid
> evaluated it and measured a **net ~14 % proposal-stage regression with no
> accuracy gain**, so it is deliberately **not** enabled (see *Outcome* below).
> This was always separate from the Gaussian-blur cost, which was the dominant
> proposal bottleneck and *was* resolved in 0.4.1 (vectorized separable blur +
> opt-in `unsafe-opt`).

## Problem

The proposal stage (radsym RSD voting) is the largest single cost in ringgrid
detection — **~44–52 % of end-to-end time** on the coded reference scenes
(per-stage `StageTimings`, Apple M4 Pro, radsym 0.4.1, median of 15):

| image | proposal | total | proposal share |
|---|---|---|---|
| real 720×540, 80 markers | 14.7 ms | 28.4 ms | **52 %** |
| synthetic 1280×960, 203 markers | 28.3 ms | 64.0 ms | **44 %** |

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

## Outcome (2026-07-05, radsym 0.4.1)

radsym 0.4.1 delivered **option 2** below: `thin_gradient`, a standalone
gradient-direction NMS transform (4-direction quantization, integer `mag²`
comparisons). ringgrid wired it into `proposal/mod.rs`
(`scharr_gradient → thin_gradient → rsd_response_fused`) and benchmarked it.

The premise did **not** hold on the current fused-RSD path. The historical
60–80 % voting reduction was measured against ringgrid's *pre-radsym*
per-radius design, which re-blurred once per radius and was pixel-count
dominated. radsym's fused RSD already does **one** shared blur, so the voting
loop is no longer the bottleneck thinning targets — and `thin_gradient` adds a
full-image pass whose cost exceeds the voting pixels it removes.

Controlled criterion A/B (same machine + thermal state, 3 s warm-up, 100
samples, `p < 0.05`):

| benchmark | thinning OFF | thinning ON | Δ |
|---|---:|---:|---:|
| `proposal_1280x1024` | 24.1 ms | 28.0 ms | **+14 % (slower)** |
| `proposal_1920x1080` | 36.9 ms | 42.9 ms | **+14 % (slower)** |

Accuracy is unchanged either way (synthetic reference recall/precision `1.000`,
mean centre error ≈ 0.081/0.084 px). Net: a ~14 % proposal-stage cost for no
accuracy benefit, so ringgrid **does not** call `thin_gradient`. `proposal/mod.rs`
carries a comment recording this. Should a future radsym make thinning cheaper
than the voting it saves (e.g. fused into the gradient or vote pass), re-run the
A/B and revisit.

## Historical: proposed solutions

The request offered three shapes; radsym shipped **#2** (`thin_gradient`) in
0.4.1. Kept here as the original handoff record.

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
