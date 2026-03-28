# Proposal Performance And Alternatives

This note captures the current performance shape of the proposal stage in
`crates/ringgrid/src/proposal/` (formerly `detector/proposal.rs`), the main
reasons it costs what it costs, and the most credible alternative approaches.

## Current Snapshot

Benchmark command:

```bash
cargo bench -p ringgrid --bench hotpaths proposal_ -- --warm-up-time 2 --measurement-time 5 --sample-size 30
```

Current benchmark snapshot on this tree:

| Benchmark | Median-ish result |
|---|---:|
| `proposal_1280x1024` | `37.199 ms` |
| `proposal_1920x1080` | `53.121 ms` |

Relevant historical profiling context from the existing PERF reports:

- proposal generation has been the dominant single-pass hotspot
- the proposal stage previously accounted for roughly `45%` to `61%` of total
  `detect()` wall time depending on mode and profile capture
- mapper-enabled runs remain slower overall, but proposal generation still
  consumes a large fraction of end-to-end latency because pass-1 still runs in
  image space before mapper-aware refinement

## Why The Current Algorithm Costs What It Costs

The stage does five things (the first is optional):

0. (optional) Canny-style edge thinning — gradient-direction NMS to thin
   multi-pixel edge bands to single-pixel ridges, reducing strong-edge count
   by 60–80% (enabled by default via `edge_thinning: true`)
1. compute full-image Scharr gradients
2. scan the gradient field once to find the maximum magnitude
3. for each strong-gradient pixel, cast votes at every integer radius from
   `r_min` to `r_max` in both gradient directions
4. blur the full accumulator and run two-step NMS (local peak + greedy distance suppression)

The dominant term is the voting loop:

```text
O(strong_gradient_pixels * radius_steps)
```

Where:

- `strong_gradient_pixels` grows with image area, edge density, blur, and scene clutter
- `radius_steps ~= floor(r_max - r_min) + 1`

This is why proposal cost rises sharply when the diameter prior is widened. A
wider prior both increases runtime and broadens the set of plausible peaks that
later stages must sort out.

## Cost Breakdown By Subsystem

At a high level the time is split across:

- Scharr gradients: full-image convolution twice
- vote accumulation: the main hot loop, branchy and memory-write-heavy
- Gaussian blur: full-image post-processing over the accumulator
- NMS: dense neighborhood comparisons over the smoothed accumulator

The implementation is already reasonably disciplined for this design:

- raw gradient buffer access instead of per-pixel accessor overhead
- squared-threshold gate before `sqrt`
- bilinear vote deposition in a tight helper
- precomputed circular NMS offsets

That means the remaining cost is mostly structural rather than "obvious missed
micro-optimizations".

## Failure Modes Of Wide Radius Ranges

Wide `r_min` / `r_max` ranges hurt in two ways.

### 1. Runtime inflation

Every additional integer radius adds two vote destinations per strong-gradient
pixel. The cost increase is close to linear in the radius span.

### 2. Proposal competition

Wide radius spans admit more partially consistent structures:

- ring edges at the wrong scale
- textured clutter with radial-looking local gradients
- multiple nearby peaks for the same true marker

That increases candidate crowding before fit/decode and can suppress true
markers indirectly when later caps such as `max_candidates` are active.

## Critical Assessment Of The Current Approach

### Strengths

- no template bank or trained model required
- polarity-robust because votes go in both gradient directions
- naturally board-agnostic and usable before any decode or homography context
- straightforward to expose as a standalone diagnostic surface
- deterministic under fixed input/config

### Weaknesses

- expensive for wide scale ranges
- still dense-image in nature even when only a modest number of markers exist
- writes into a large accumulator with limited locality
- depends on global max-gradient thresholding, which can be brittle when a few
  very strong edges dominate the image
- raw proposal score is useful for ranking but not strongly calibrated

Overall assessment:

- for `ringgrid` today, this remains a good default proposal engine
- it is simple, stable, and already integrated with the rest of the pipeline
- its main problem is cost under wide or heterogeneous scale priors, not basic
  conceptual mismatch with the marker family

## Alternatives

## FRST / Normalized Radial Symmetry

Fast Radial Symmetry Transform style methods are the closest conceptual
relative to the current approach.

Pros:

- same "radial evidence accumulates at centers" intuition
- can normalize vote contributions better than the current raw-magnitude sum
- often produces cleaner center peaks under clutter

Cons:

- still accumulator-based and still scale-sensitive
- normalization and orientation projection details materially affect recall
- not automatically cheaper unless paired with a coarser scale strategy

Assessment:

- strong candidate for future algorithm work
- especially attractive if combined with coarse-to-fine scale handling

## Coarse-To-Fine / Pyramid Proposal Search

Run a cheaper wide search at low resolution, then refine locally near peaks at
higher resolution.

Pros:

- attacks the biggest current problem directly: wide radius spans
- can reduce both vote count and NMS search area at full resolution
- preserves the current radial-symmetry family of ideas

Cons:

- more moving parts and more threshold interactions
- refinement logic must avoid losing small/weak markers
- harder to reason about exact recall guarantees

Assessment:

- the most credible performance-oriented follow-up
- likely the best next step if proposal latency becomes a priority again

## Contour-First Ellipse Hypotheses

Use connected edges / contours first, then fit ellipse-like candidates and score
them as potential rings.

Pros:

- potentially much less dense than full-image voting
- can reject obviously non-elliptic clutter earlier

Cons:

- contour extraction is brittle under blur, partial occlusion, and weak inner edges
- contour fragmentation becomes its own failure mode
- more coupling to later fit heuristics

Assessment:

- plausible for certain clean scenes
- weaker fit for `ringgrid`'s robustness goals than radial voting

## Blob / Hessian / LoG / DoG Center Detectors

Treat rings as scale-space blobs and recover centers from extremal responses.

Pros:

- mature literature and efficient implementations exist
- naturally multi-scale

Cons:

- ring markers are not generic isotropic blobs once perspective and contrast
  polarity are considered
- center localization quality under elliptical projection is less direct
- still needs substantial downstream verification

Assessment:

- useful as a sidecar baseline, not an obvious replacement

## Circle Hough

Classical circle Hough transform over edge maps.

Pros:

- conceptually simple
- well-known and easy to compare against

Cons:

- scale explosion is even more severe for the required radius range
- perspective-distorted rings are not circles in the image
- usually a poor match once markers become clearly elliptical

Assessment:

- not recommended for this project

## Learned Proposal Networks

Train a detector to output center heatmaps or ring keypoints directly.

Pros:

- can be very fast at inference
- can learn scene-specific clutter rejection

Cons:

- requires labeled data and retraining discipline
- weak match to the current pure-Rust, deterministic, low-dependency design
- introduces deployment and reproducibility complexity

Assessment:

- only worth considering if product constraints change substantially

## Recommendation

Keep the current proposal algorithm as the shipped default.

Implemented mitigations (2026-03):

1. **Edge thinning** (ALGO-016): Canny-style gradient-direction NMS reduces
   strong-edge count by 60–80%, proportionally reducing voting cost.
2. **Optional downscaling** (PERF-006): `ProposalDownscale` enum on `DetectConfig`
   allows auto or explicit downscaling before proposals. At factor=2, ~4x fewer
   pixels enter the voting loop.
3. **Standalone module** (ALGO-015): Proposal API extracted to `proposal/` with
   clean `find_ellipse_centers()` entry points and diagnostic heatmap output.

Near-term actions:

1. use the diagnostics tooling to understand recall/peak behavior on real
   failure cases
2. treat FRST-style normalization and coarse-to-fine search as the most
   credible future improvements

Not recommended as primary next steps:

- circle Hough replacement
- learned models
- contour-first rewrite without strong evidence from failure analysis
