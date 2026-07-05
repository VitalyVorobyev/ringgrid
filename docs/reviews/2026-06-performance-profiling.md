# ringgrid performance profiling — 2026-06-25

> **Superseded (2026-07-05, radsym 0.4.1).** radsym 0.4.1 vectorized the
> separable Gaussian blur (bit-identical output) and added the opt-in
> `unsafe-opt` bounds-check elision — now enabled — cutting proposal-stage cost
> **~23 %** with unchanged detection accuracy. The re-measured stage split
> (Apple M4 Pro, rustc 1.95, median of 15) is:
>
> | image | proposal | fit + decode | finalize | total |
> |---|---|---|---|---|
> | real 720×540, 80 markers | 14.7 ms (**52 %**) | 9.1 ms (32 %) | 4.6 ms (16 %) | 28.4 ms |
> | synthetic 1280×960, 203 markers | 28.3 ms (**44 %**) | 26.2 ms (41 %) | 9.4 ms (15 %) | 64.0 ms |
>
> Updated Criterion primitives: `proposal_1280x1024` / `proposal_1920x1080`
> **22.9 / 35.0 ms**, `propose_target_3_split_00` **14.8 ms**,
> `detect_target_3_split_00` **28.9 ms**. The proposal stage is still the
> largest single stage but no longer clearly dominant on dense boards. The
> June-25 numbers below are kept as the historical record; the
> [live dashboard](https://vitalyvorobyev.github.io/ringgrid/performance/) and
> `docs/proposal-performance-analysis.md` are the current reference.

Method: Criterion micro-benchmarks (`crates/ringgrid/benches/`), the per-stage
`StageTimings` now exposed on the diagnostics channel, and the `ringgrid-dev
bench` subcommand (median of N repeats). Machine: Apple M4 Pro, release build.

> Headline: **the proposal stage (radsym RSD voting) is ~50–54 % of detection
> time** and is the only real bottleneck; per-marker primitives are already
> cheap (µs). The biggest available win is in the proposal stage, but it is
> accuracy-sensitive and is therefore scoped as a **benchmark-gated follow-up**,
> not bundled into this PR. Two behavior-preserving micro-optimizations that
> remove redundant work are included.

---

## Where the time goes

End-to-end stage split (`ringgrid bench`, median):

| image | proposal | fit + decode | finalize | total |
|---|---|---|---|---|
| real 720×540, 78 markers | 15.2 ms (**53 %**) | 8.8 ms (31 %) | 4.6 ms (16 %) | 28.5 ms |
| synthetic 1280×960, 203 markers | 32 ms (**49 %**) | 24.5 ms (37 %) | 8.9 ms (14 %) | 65.8 ms |

Criterion primitives (per call):

| bench | time |
|---|---|
| `propose_target_3_split_00` (whole proposal stage) | **15.4 ms** |
| `detect_target_3_split_00` (whole detect) | 28.5 ms |
| `proposal_1280x1024` / `proposal_1920x1080` | 32 ms / 48 ms (scales with area) |
| `radial_profile_32r_180a` | 8.5 µs |
| `outer_estimate` / `inner_estimate` | 18.7 µs / 37.5 µs |
| `inner_fit_64r_96t` | 59.8 µs |
| `ellipse_fit_50pts` | 299 ns |

Reading: the proposal stage scales with **image area** (per-pixel gradient +
RSD voting over a set of radii); `fit + decode` scales with **marker count**
(~95 µs/candidate across outer estimate → outer-fit RANSAC → inner fit →
decode); `finalize` is board-level reasoning (RANSAC homography, ID consensus,
completion). The individual fit primitives are already in the tens of µs.

## Proposal-stage finding: dead perf config

`ProposalConfig` exposes `edge_thinning` (documented: "cuts the strong-edge
count by 60–80 % and proportionally reduces the voting workload") and
`accum_sigma`, but **neither is read** by the radsym-0.2 adapter
(`proposal/mod.rs::compute_via_radsym`) — they are only ever *set* in struct
literals (`scale_probe.rs`, `proposal/tests.rs`). radsym 0.2's `RsdConfig` has
no edge-thinning knob, so the promised 60–80 % voting reduction is **not applied
today**. This is the single largest latent speedup: every above-threshold
gradient pixel votes (multi-pixel-wide edge bands), not thinned ridges.

Action: either wire edge-thinning into the proposal path (apply gradient-
direction NMS before voting, or adopt a radsym facility if one is added) or
deprecate the two vestigial fields. Wiring it is accuracy-sensitive (thinning
changes which pixels vote) → benchmark-gated follow-up.

## Optimization opportunities (ranked, with risk)

1. **Wire edge-thinning** (or thin gradients before voting) — potentially the
   advertised 60–80 % voting reduction → proposal could drop from ~50 % toward
   ~20 % of total. **Risk: detection-quality change; must pass the full gate.**
2. **Subsample voting radii** (`proposal/mod.rs::build_radii` uses *every*
   integer radius). Halving radii ≈ halves proposal cost (~25 % faster
   end-to-end) but reduces accumulator sensitivity. **Risk: accuracy; gate it.**
3. **Tune `proposal_downscale`** activation for mid-size images (only large
   images downscale today). **Risk: low-moderate.**

These three are deliberately **not** applied here: each changes detection
results and belongs in its own PR validated against the now-certified
`tools/ci/regression_baseline.json`, separate from this PR's dependency
migration and docs.

## Applied here (behavior-preserving — proposals/markers byte-identical)

1. **Skip the vote heatmap on the detection seed path**
   (`pipeline/run.rs::find_proposals_with_downscale`). The pipeline discards the
   heatmap, yet the old code computed it and, under downscaling, ran a
   full-image Triangle resize of it. Now the seed path uses
   `find_ellipse_centers` (no heatmap); a shared `downscale_setup` helper keeps
   the two paths DRY. Verified identical via
   `detector_proposal_apis_honor_proposal_downscale` (asserts seed-path
   proposals equal heatmap-path proposals). Benefits the large-image/downscale
   path; negligible at factor 1.
2. **Hoist the conic conversion out of RANSAC scoring**
   (`conic/types.rs`, `conic/ransac.rs`, `conic/fit.rs`).
   `Ellipse::sampson_distance` recomputed `to_conic()` (trig) for *every* point;
   the candidate is constant within a RANSAC iteration. A new
   `ConicCoeffs::sampson_distance` lets each scoring loop convert once and reuse.
   Measured **−3.7 %** on the `inner_fit_64r_96t` micro-benchmark; end-to-end is
   within the noise floor (the proposal stage dominates, and back-to-back
   benchmark + rtv3d runs introduced ~2 % thermal drift on this machine).

Both are correct, reviewer-flagged reductions of redundant work; 183 lib tests
pass, clippy `-D warnings` clean, and the fixture detects an identical 78/93.

## Verdict

Detection is already well-optimized at the primitive level; the lever that
matters is the proposal stage, where a real ~2× opportunity exists via
edge-thinning — but it trades against detection quality and must be landed as a
separately benchmark-gated change. This pass removes the clearly-wasted work
(heatmap on the seed path, per-point conic reconversion) without touching
detection results, and documents the gated path to the larger win.
