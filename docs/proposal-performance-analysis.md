# Proposal Performance And Alternatives

This note captures the current performance shape of the proposal stage in
`crates/ringgrid/src/proposal/` on the radsym 0.4 backend, the reasons it
costs what it costs, the status of the three performance levers under
consideration (direct `u8` gradients, per-proposal scale narrowing, and a
coarse-to-fine pyramid), and the most credible remaining alternatives.

The proposal stage no longer contains its own image-processing code. It is a
thin adapter (`proposal/mod.rs::compute_via_radsym`) over the external
`radsym` crate: `scharr_gradient` → `rsd_response_fused` (fused multi-radius
voting) → `extract_proposals` (NMS) → `suppress_proposals_by_distance`. Two
prior handoffs document how this backend evolved and should be read alongside
this note rather than duplicated here:

- `docs/radsym-multiradius-handoff.md` — why radsym gained a *fused*
  multi-radius mode (the original per-radius `rsd_response` re-blurred the
  accumulator once per radius and was 5.7–10x slower than ringgrid's old
  internal implementation; fusing radii into one accumulator + one blur
  closed that gap).
- `docs/radsym-edge-thinning-request.md` — the still-open ask (now tracked
  upstream as **radsym issue #16**, filed 2026-07-03) to thin gradient bands
  to single-pixel ridges before voting, which historically cut strong-edge
  count 60–80 % in ringgrid's pre-radsym implementation. This capability was
  lost in the migration to radsym and has not been restored.

## Current Snapshot

Benchmark command:

```bash
cargo bench -p ringgrid --bench hotpaths proposal_ -- --warm-up-time 2 --measurement-time 5 --sample-size 30
```

A second, real-image proposal benchmark lives in a separate bench binary:

```bash
cargo bench -p ringgrid --bench detect_fixture propose_ -- --warm-up-time 2 --measurement-time 5 --sample-size 30
```

Fresh numbers on `feat/algo-soundness-perf` (2026-07-04, Criterion median
estimate `[low, median, high]` of the confidence interval):

| Benchmark | Image | Median | 95% CI |
|---|---|---:|---:|
| `proposal_1280x1024` | synthetic, 1280×1024 | `29.909 ms` | `[29.829, 29.997] ms` |
| `proposal_1920x1080` | synthetic, 1920×1080 | `45.040 ms` | `[44.967, 45.122] ms` |
| `propose_target_3_split_00` | real fixture, 720×540, 78 markers | `14.877 ms` | `[14.850, 14.901] ms` |

The two synthetic fixtures use a fixed `ProposalConfig` (`r_min=4, r_max=18,
radius_step=1` → 15 voting radii) over a 14×10 ring grid regardless of image
size, so the 1920×1080 fixture has both more pixels *and* larger rings (more
edge pixels) than the 1280×1024 one; area grows ~1.58x (1.31 MP → 2.07 MP)
and measured time grows ~1.51x (29.9 ms → 45.0 ms), consistent with per-pixel
gradient and voting cost dominating.

These numbers are **not directly comparable** to the `37.199 ms` /
`53.121 ms` figures previously recorded in this document — those were
measured against the pre-radsym internal implementation (with edge-thinning
enabled) before the migration described in
`docs/radsym-multiradius-handoff.md`. For an apples-to-apples before/after
comparison of that migration, see the tables in that handoff doc and in
`docs/reviews/2026-06-performance-profiling.md` (which recorded
`32 ms` / `48 ms` for the same two synthetic benchmarks on 2026-06-25 —
in the same range as the numbers above, modulo normal run-to-run variance).

## Current Architecture (radsym 0.4, fused RSD)

The stage does four things, none of them optional:

1. **Gradient** — `radsym::scharr_gradient` computes `gx`/`gy` in one pass
   directly from the `&[u8]` backing the input `GrayImage`
   (`ImageView::from_slice(gray.as_raw(), w, h)`). No f32 image conversion or
   copy happens on either side of the boundary — see [Lever (a)](#a-direct-u8-sourcepixel--already-the-case) below.
2. **Threshold calibration** — one full-image scan for `max_magnitude()` to
   turn `ProposalConfig::grad_threshold` (a *relative* fraction) into an
   absolute gradient-magnitude threshold, and later one full-image scan of
   the accumulator for its max, to turn `min_vote_frac` into an absolute NMS
   threshold.
3. **Fused voting** (`rsd_response_fused` → `fused_voting_pass`) — a single
   pass over all pixels. Pixels below the gradient threshold are skipped via
   a squared-magnitude gate (no `sqrt` for rejected pixels). Each surviving
   pixel casts a vote at every configured radius, in both gradient directions
   (`Polarity::Both`), into one shared accumulator. This is followed by
   **one** Gaussian blur of the accumulator with
   `sigma = smoothing_factor * median(radii)`.
4. **Extraction** — `extract_proposals` runs local-peak NMS (radius capped at
   `min(min_distance, 10)` for efficiency) over the smoothed accumulator, then
   `suppress_proposals_by_distance` greedily thins the NMS output to the full
   `min_distance` and the `max_candidates` budget.

The dominant term is the voting loop in step 3:

```text
O(strong_gradient_pixels * radii_count)
```

with two accumulator writes per surviving pixel per radius (positive and
negative gradient-direction votes). `radii_count = floor(r_max - r_min) /
radius_step + 1` (via `build_radii`, `proposal/mod.rs`). The fused voting
pass is **single-threaded**: unlike the non-fused `rsd_response` (which
parallelizes per-radius accumulation with `rayon` when the `rayon` feature is
enabled), `fused_voting_pass` has no `rayon`-gated code path at all, and
ringgrid's `Cargo.toml` does not enable radsym's `rayon` feature regardless.

Steps 1, 2, and the blur in step 3 are each a single `O(w*h)` pass; step 4 is
a dense neighborhood scan over the accumulator followed by an `O(kept^2)`
(bounded by the NMS budget) greedy distance filter. None of these come close
to the voting loop's cost at realistic strong-edge counts.

## Track B: Status Of The Three Levers

### (a) Direct `u8` `SourcePixel` — already the case

`radsym::scharr_gradient<P: SourcePixel>` is generic over the source pixel
type, with `impl SourcePixel for u8` casting inline (`self as Scalar`) inside
the stencil loop. `find_ellipse_centers` builds `ImageView` directly from
`gray.as_raw()` — the `GrayImage`'s owned `u8` buffer — with no intermediate
`f32` image. There is nothing left to do here; this is not a future
optimization opportunity, it is the current, already-shipped behavior, and
this document should stop listing it as an open item.

### (b) Per-proposal scale narrowing — blocked upstream

`extract_proposals` (`radsym::propose::extract`) unconditionally sets
`scale_hint: None` on every `Proposal` it produces from the fused response
map:

```rust
peaks.into_iter().map(|peak| Proposal {
    seed: SeedPoint { position: peak.position, score: peak.score },
    scale_hint: None,
    polarity,
    source: response.source(),
})
```

This is structural, not an oversight: fusing all radii into one shared
accumulator (the whole point of `rsd_response_fused`, see the multi-radius
handoff) means there is no per-pixel record of *which* radius contributed the
winning vote at a given accumulator cell. Nothing downstream of the fused
pass can recover a per-proposal scale estimate.

ringgrid already has a scale-narrowing mechanism, but it operates at a
coarser granularity than "per proposal": `pipeline/scale_probe.rs` sweeps
ring angular-variance over the top-K gradient proposals across ~20 geometric
radius candidates (4–110 px) to find dominant board-level radii, and
`ScaleTiers::from_detected_radii` turns that into a small set of scale tiers
for `detect_adaptive`. This is a *board-level, post-hoc* estimate consumed to
pick which tier(s) to run — not a way to narrow the radii voted on *within* a
single `rsd_response_fused` call.

For radsym to provide real per-proposal scale hints on the fused path, the
fused voting pass would need to track, per accumulator cell, not just the
summed vote magnitude but also the radius that contributed the running
maximum (an argmax-style parallel buffer). Concretely this means:

- an additional `w*h` buffer to hold the winning radius index per cell —
  roughly doubling the accumulator's memory footprint;
- a compare-and-conditionally-write on *both* buffers at every vote
  destination instead of a single `+=`, roughly doubling the write bandwidth
  of the hot loop;
- a blur problem: the current single global Gaussian blur is only valid for
  the (continuous) magnitude accumulator. A per-cell radius index is not a
  quantity you can blur — the argmax buffer would either have to be read
  un-blurred (noisier peak-radius estimates) or the implementation would need
  a per-radius response array blurred independently, which reverts toward
  the `O(radii_count)` blur cost that fusion was specifically built to avoid.

That combination of memory, bandwidth, and blur-semantics cost is why this is
tracked as a future upstream ask rather than something ringgrid works around
locally — there's no equivalent transformation available on the ringgrid
side of the boundary.

### (c) Coarse-to-fine pyramid — not justified by current numbers

ringgrid already ships a one-level coarse/fine split, in two complementary
forms, both of which predate and remain independent of this measurement:

- **`ProposalDownscale`** (`detector/config/scale.rs`) downscales the image
  by an integer factor (1–4, `Auto` resolves from
  `marker_scale.diameter_min_px`) *before* proposal generation; every
  downstream stage (outer fit, decode, inner fit) runs at full resolution,
  with proposal coordinates scaled back up. This is a genuine two-level
  coarse(proposal)/fine(everything else) split — it is just not a
  multi-level image pyramid.
- **`ProposalConfig::radius_step`** subsamples the *radius* axis of the
  voting loop instead of the spatial axis (stride > 1, `r_max` always kept).
  As documented on the field itself: at `radius_step = 2`, proposal time
  drops ~29 % but real-world recall drops (rtv3d −2.9 %), so it fails the
  accuracy gate as a default and stays opt-in.

radsym itself has only a single-level analog of this pattern: a
`remap_proposal_to_image` helper (`propose/remap.rs`) that maps proposals
found on one `2^level` box-pyramid level back to base-image coordinates.
There is no cascading multi-level proposal search built into radsym either —
a caller (ringgrid) would have to run proposal generation at each pyramid
level and merge/dedupe results itself, which is architecturally similar to
what `ScaleTiers` already does by running full proposal + fit + decode passes
per scale band and merging with size-consistency-aware NMS
(`detector/dedup.rs::merge_multiscale_markers`).

A true multi-level pyramid would mainly pay off when a single
`rsd_response_fused` call needs a *wide* `r_min..r_max` span (large
`radii_count`), by spreading that span's cost across levels instead of
voting all radii in one full- or reduced-resolution pass. ringgrid already
avoids issuing such wide-span single calls in the wide-scale-range case: it
decomposes them into `ScaleTiers` (several narrow-range full-resolution
proposal passes, each with a small `radii_count`) or a single
`ProposalDownscale` factor (one reduced-resolution pass). Given the measured
numbers above — a synthetic 1280×1024 pass with a 15-radius span costs
~30 ms end-to-end, and `radius_step` already offers an accuracy-for-speed
knob on that same axis — a full pyramid's incremental win over the existing
`ScaleTiers` + `ProposalDownscale` combination is not demonstrated. It stays
a documented non-goal unless a concrete scenario surfaces where those two
existing mechanisms measurably underperform it.

## Critical Assessment Of The Current Approach

### Strengths

- no template bank or trained model required
- polarity-robust because votes go in both gradient directions
- naturally board-agnostic and usable before any decode or homography context
- straightforward to expose as a standalone diagnostic surface
  (`find_ellipse_centers_with_heatmap`)
- deterministic under fixed input/config
- fused voting (radsym 0.4) closed the earlier per-radius-blur regression —
  current cost is voting-bound, not blur-bound

### Weaknesses

- expensive for wide scale ranges (`radii_count` scales the dominant term
  close to linearly)
- still dense-image in nature even when only a modest number of markers exist
- writes into a large accumulator with limited locality
- depends on global max-gradient thresholding, which can be brittle when a
  few very strong edges dominate the image
- raw proposal score is useful for ranking but not strongly calibrated
- fused-path proposals carry no scale hint (`scale_hint: None` always) — see
  Lever (b) above
- the one accuracy-preserving lever with a large documented upside
  (edge-thinning, ~60–80 % strong-edge reduction) is blocked on radsym#16

Overall assessment: for `ringgrid` today, this remains a good default
proposal engine. It is simple, stable, and already integrated with the rest
of the pipeline. Its main open problem is the lost edge-thinning capability,
not a conceptual mismatch with the marker family.

## Alternatives

Trimmed to the entries still worth tracking; the rest (contour-first ellipse
hypotheses, blob/Hessian/LoG/DoG detectors, circle Hough, learned proposal
networks) were surveyed previously and remain not recommended for the same
reasons as before (weaker fit for ringgrid's robustness goals, scale
explosion, or a poor match to the pure-Rust deterministic design) — see prior
revisions of this document in git history for the full writeups if needed.

### Edge-thinning (radsym#16) — the main pending upstream lever

Canny-style gradient-direction NMS, thinning multi-pixel edge bands to
single-pixel ridges before voting. This is not a new idea to evaluate — it
is a capability ringgrid's pre-radsym implementation already had and lost in
the migration. Per the pre-migration measurements
(`docs/radsym-edge-thinning-request.md`), it cut strong-edge count 60–80 %
and reduced voting cost proportionally, with no accuracy trade-off (unlike
`radius_step`). It requires a radsym-side change (`RsdConfig` knob, a
standalone `thin_gradient` transform, or a public `GradientField`
constructor — see that doc for the three concrete proposals) because
`GradientField`'s `gx`/`gy` are `pub(crate)` with no public constructor, so
ringgrid cannot inject a thinned gradient today without forking radsym.

### FRST / Normalized Radial Symmetry

radsym already ships `frst_response` / `frst_response_fused`
(`radsym::propose::frst`) as a sibling to the RSD path ringgrid currently
uses — this is no longer a hypothetical alternative to prototype from
scratch, just an unused option in the same dependency. It keeps the same
"radial evidence accumulates at centers" intuition with an orientation
(`O_n`) consistency check RSD dropped for speed.

Pros: can normalize vote contributions better than raw-magnitude RSD;
often produces cleaner center peaks under clutter.

Cons: FRST's fused variant is documented as ~13x slower than the old
internal implementation before fusing (vs. RSD's 5.7–10x), due to
per-radius `O_n` normalization cost that fusing alone doesn't eliminate;
not automatically cheaper unless paired with a coarser scale strategy.

Assessment: worth a controlled recall/precision comparison against RSD on
the regression suite if RSD's accuracy (not its speed) becomes the limiting
factor; not a performance lever.

## Recommendation

Keep the current proposal algorithm (radsym 0.4 fused RSD) as the shipped
default.

Status of prior mitigations:

1. **Fused multi-radius voting** (radsym 0.4, `docs/radsym-multiradius-handoff.md`):
   shipped. Closed the per-radius-blur regression from the initial radsym
   migration.
2. **Edge thinning** (ALGO-016, historically 60–80 % strong-edge reduction):
   lost in the radsym migration, not yet restored. Tracked upstream as
   **radsym issue #16** (filed 2026-07-03). This is the highest-value pending
   lever — accuracy-preserving, unlike `radius_step`.
3. **Optional downscaling** (PERF-006): `ProposalDownscale` on
   `DetectConfig` remains available and unchanged; still the right tool for
   large images with a known coarse marker-scale floor.
4. **`radius_step` subsampling**: available, opt-in, off by default (costs
   real-world recall).

Near-term actions:

1. Track and, once available, adopt radsym#16 (edge-thinning) behind the
   existing regression gate (`tools/ci/regression_baseline.json`) before
   flipping any default.
2. Treat per-proposal scale hints (Lever b) as blocked on the same upstream
   surface — no local workaround exists once fusion is in place.
3. Do not build a ringgrid-side coarse-to-fine pyramid (Lever c) without a
   concrete case where `ScaleTiers` + `ProposalDownscale` demonstrably
   underperform it.

Not recommended as primary next steps:

- circle Hough replacement
- learned models
- contour-first rewrite without strong evidence from failure analysis
- a ringgrid-side pyramid duplicate of `ScaleTiers` / `ProposalDownscale`
