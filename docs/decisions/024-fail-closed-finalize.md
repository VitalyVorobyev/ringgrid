# ADR-024: Fail-Closed Finalize

- **Status:** accepted
- **Date:** 2026-11-15
- **Author role:** algorithm-engineer
- **Amends:** ADR-023 (anchor acceptance, contradicted decodes, fallback
  ladder floor)

## Context

A wrong reported detection is worse than a missed one: a wrong ID, board
position or centre corrupts a camera calibration. ADR-023 was adopted on recall
evidence; its precision under hard conditions was not measured.

`tests/precision_audit.rs` is a ground-truth harness: 228 catalogue scenes
(default thin-band hex, coded rect, 1239-cell Extended hex, plain hex/rect with
origin dots; blur × diameter, tilt to 65°, radial distortion without a camera
model, partial views, an occluder, 1–6 markers, low contrast, a second board in
view, and a few fail-closed scenes outside the operating range; since added:
60 dark, low-dynamic-range scenes, group `dark`) plus a 450-scene
operating-range grid (blur σ ∈ {0.8, 1.5, 2.0} px × diameter {20, 24, 32, 44,
56} px × tilt {0, 30, 50}° × 2 poses × 5 boards). The scorer counts wrong IDs,
wrong absolute positions, relative label errors, wrong homographies (median
reprojection of the true cells off by more than half a lattice spacing) and
centres beyond 1 px / `max(2 px, 0.1 · diameter)`. Scenes export with ground
truth, so 0.15.0's CLI ran the same images.

Measured, 0.15.0 vs grid-first (ADR-023), operating-range grid: 226 vs 226
wrong IDs; catalogue: 0.15.0 and grid-first both report hundreds of wrong IDs
outside the operating range, grid-first adds wrong IDs on a second board in
view (hex/rect/Extended foreign scenes: 2→7, 1→4, 2→12) and on a strongly
distorted Extended view (0→1). Raw decodes cannot be trusted alone: the base
codebook's minimum cyclic distance is 2 (Extended: 1), so one or two bit
errors yield another valid word; of the 257 IDs both releases reported without
a homography on the harness, 197 (77 %) were wrong.

## Mechanisms and rules

| Mechanism (evidence) | Rule |
|---|---|
| Fallback without a homography passed raw decodes through (197 of 257 wrong; e.g. rect d20 b2: 39 wrong) | No homography → markers without IDs, board positions or homography |
| Fallback homography fitted to IDs that ID correction recovered around misread seeds (wrong ones had 1–4 own-decode inliers of 6–48; e.g. 7/15 at 0.09 px → 110 wrong IDs) | Accept only with `min_inliers` inliers carrying their own ring's decode |
| Fallback completion swept the board along a homography extrapolated from a near-side patch (rect 65°: 2 wrong + wrong H) | Fallback completion grows from the patch only |
| Majority anchor rejected decisive minority anchors under misreads (8 of 40 votes, runner-up 2) and sent them to the fallback | Winner must beat chance: `R · C(k, v) · n^(1−v) ≤ 1e-3` (R orientation-preserving symmetries, n cells, k decodes, v votes) |
| A labeling with a skipped row split correct decodes 7 / 6 between two placements; chance alone would accept the 7 (78 wrong IDs) | Runner-up must be at chance level (same expression ≥ 0.1) |
| Rings labeled or adopted at a cell their own on-board decode contradicts: a lattice slip under distortion, rings of a second board | Such a ring never takes the cell's ID (it returns to the pool); off-board decodes are misreads and are still overridden |
| Join homography bent by 2 misread candidates far from a 15-ring anchored patch → completion along it (365 wrong IDs) | Join needs `min_inliers` joined candidates and most identified rings as inliers |
| Foreign rings within 2–8 px of a hidden cell's projection passed the 10 px global floor | Geometric-verify global floor 2 px (MAD term still loosens it under distortion) |

Tried and rejected (operating-range recall cost, no precision gain left to
buy): contradicted-decode rejection at completion refits (Extended blur recall
0.93 → 0.57–0.66: Extended misreads land on board IDs); label gate back to 2 px
(fixes the distorted Extended scene, which the contradiction rule fixes too,
but plain rect distortion recall 0.92 → 0.54); dropping the whole-board sweep
in the grid path (−2 to −4 points, no wrong output removed); dropping ID
correction from the fallback (its recovered IDs then cannot reach a homography
anyway; without it, RANSAC over raw Extended decodes found chance 6-inlier
homographies: 958 wrong IDs on one scene).

## Owner requirements

**Perspective-corrected centres only.** With a centre correction configured,
a marker still carrying its outer-ellipse centre is not reported (it still
serves as lattice evidence: labeling, anchor votes, homography support).
Enforced once, on the finished result after the photometric stage, which can
still correct it. Operating range: outer-only markers were 18 % of reported
markers, median error 0.25–1.0 px vs 0.12–0.21 px projective; they held 51 % of
the centres beyond 1 px and 54 % of those beyond the loose gate (tilt and
distortion: 39 % of them beyond 1 px).

**Ellipse-shape consistency.** Geometric verify compares each outer ellipse's
axis ratio with the ratio the final homography's local Jacobian predicts:
`max(0.12, 6 σ)` around the view median. On the remaining errors: catches 38
of 46 wrong centres at 0.5 % false rejection (0 % under distortion). Other
forms measured: absolute residual 0.12 (40/46, 1.4 % false), size vs H
(1/46), orientation (≤ 7/46), neighbour-median axis ratio (≤ 6/46), the literal
monotone-along-a-line test (2–3/46 at 1–30 % false). It measures a different
quantity than the inner/outer axis-ratio filter, which is kept.

## Consequences

Operating range, after: 0 wrong IDs / positions / homographies on all 450 grid
scenes and every catalogue scene except three (a second board of the same or
another layout: 1 + 1 wrong ID, ≤ 0.15.0's 1 + 3; six isolated plain rings
labeled as neighbours in the relative frame: 5 label errors, as 0.15.0).
Recall (grid, default refinement): hex 0.831 → 0.694, Extended 0.858 → 0.708,
rect 0.808 → 0.697, plain hex 0.434 → 0.410, plain rect 0.609 → 0.498 — almost
all of it the corrected-centre contract; before it: 0.797 / 0.866 / 0.847 /
0.433 / 0.608. With `--refine photometric` the catalogue recall returns to the
pre-contract level (hex blur 0.878, rect blur 0.865, Extended blur 0.941) at 0
wrong outputs. 1–6 isolated markers are reported without IDs.

**Scope: one board per image.** A second board in view is not a supported
scenario (owner decision). The `foreign` scenes stay in the harness, scored
and printed but not gating; known limits at the time of writing:
`plainrect-foreign-same` 1 wrong ID (on a second-board ring).

### Per-marker confidence and `min_confidence` (accepted)

Owner request: a quality measure to trade recall for centre accuracy. The
owner's review of real images (11 wrong centres out of ~470, all two-conic)
found them low in `confidence`, near the image border and in the blurriest
fifth. `precision_quality_dump` (in-range and dark single-board scenes,
130 k reported markers, 1.6 % beyond 1 px) measured candidate predictors —
ROC AUC for a centre error beyond 1 px (beyond 0.5 px):

| predictor | AUC > 1 px | AUC > 0.5 px | bad removed at 10 % removed |
|---|---|---|---|
| reprojection residual vs final H | 0.950 | 0.913 | 0.96 |
| outer max angular gap | 0.926 | 0.781 | 0.75 |
| outer arc coverage | 0.902 | 0.763 | 0.70 |
| previous `confidence` | 0.893 | 0.771 | 0.71 |
| inner-edge theta consistency | 0.761 | 0.691 | 0.30 |
| border distance (radii) | 0.733 | 0.644 | 0.38 |
| shape residual vs H | 0.723 | 0.680 | 0.39 |
| inner/outer ratio vs view median | 0.705 | 0.637 | 0.32 |
| two-conic shift / radius | 0.666 | 0.617 | 0.28 |
| outer edge width (px) | 0.646 | 0.632 | 0.25 |
| contrast-to-noise, step, fit RMS, inliers, decode distance | 0.51–0.66 | — | ≤ 0.23 |

The reprojection residual is near-oracle on synthetic views (the homography
is exact there) and weaker on real ones (the owner's 11: 6 in the worst
10 %); it is not available when the marker is built, and was left out.
Logistic combinations (scene-split cross-validation): coverage + edge width
0.936; + border 0.941; + inner agreement 0.954 — each factor adds. The
closed form adopted (`marker_build::compute_marker_confidence`):
evidence × outer inliers × `1/(1 + rms/2)` × `exp(−gap/(π/4))` × inner
agreement × `1/(1 + (w/3 px)²)` × `min(1, d/2)`. AUC 0.950 (0.835 for
0.5 px), bad centres removed at 5 / 10 / 20 % of markers 0.64 / 0.87 / 0.95
(previous: 0.47 / 0.71 / 0.84); dark scenes 0.928 vs 0.823; tilt 0.930 vs
0.920. Redefining `confidence` also re-ranks dedup and ID votes: grid recall
moved −0.9 to +0.9 points, no single-board wrong output.

`DetectConfig::min_confidence` drops markers below it on the finished result
(with the corrected-centre rule; homography refit). Default 0.03 from the
curve (`precision_quality_dump`, recall / > 1 px / > 0.5 px / median):

| min_confidence | hex | Extended | rect | plain hex | plain rect |
|---|---|---|---|---|---|
| 0 | 0.679 / 2.5% / 15.5% / 0.20 | 0.688 / 1.6% / 11.2% / 0.18 | 0.688 / 1.6% / 11.2% / 0.18 | 0.410 / 2.0% / 11.0% / 0.16 | 0.478 / 0.9% / 6.6% / 0.15 |
| 0.02 | 0.672 / 2.0% / 14.7% / 0.20 | 0.683 / 1.3% / 10.6% / 0.18 | 0.681 / 1.0% / 10.3% / 0.18 | 0.407 / 1.4% / 10.1% / 0.16 | 0.474 / 0.5% / 5.7% / 0.15 |
| **0.03** | 0.661 / 1.5% / 13.7% / 0.20 | 0.675 / 1.0% / 10.0% / 0.18 | 0.672 / 0.7% / 9.5% / 0.18 | 0.402 / 1.0% / 9.3% / 0.16 | 0.469 / 0.4% / 5.2% / 0.15 |
| 0.05 | 0.642 / 0.8% / 11.9% / 0.19 | 0.655 / 0.6% / 8.7% / 0.17 | 0.651 / 0.4% / 8.0% / 0.17 | 0.395 / 0.6% / 7.9% / 0.15 | 0.461 / 0.2% / 4.3% / 0.14 |
| 0.08 | 0.593 / 0.3% / 9.0% / 0.17 | 0.606 / 0.2% / 6.3% / 0.16 | 0.601 / 0.1% / 5.7% / 0.16 | 0.378 / 0.1% / 5.9% / 0.14 | 0.434 / 0.1% / 3.0% / 0.14 |

Dark scenes, same columns:

| min_confidence | hex | Extended | rect | plain hex | plain rect |
|---|---|---|---|---|---|
| 0 | 0.564 / 1.5% / 16.2% / 0.27 | 0.533 / 1.7% / 14.3% / 0.26 | 0.539 / 1.7% / 12.1% / 0.25 | 0.214 / 0.9% / 8.3% / 0.21 | 0.173 / 0.7% / 6.5% / 0.21 |
| **0.03** | 0.528 / 0.6% / 13.9% / 0.26 | 0.498 / 0.6% / 11.7% / 0.25 | 0.495 / 0.6% / 9.1% / 0.23 | 0.189 / 0.0% / 5.6% / 0.21 | 0.172 / 0.2% / 4.5% / 0.20 |
| 0.05 | 0.446 / 0.3% / 11.4% / 0.24 | 0.440 / 0.3% / 9.3% / 0.24 | 0.437 / 0.3% / 7.6% / 0.22 | 0.172 / 0.0% / 5.5% / 0.21 | 0.171 / 0.0% / 4.0% / 0.20 |

Full sweep at the default: centres beyond 1 px on the operating-range grid
hex 314 → 182, Extended 435 → 272, rect 299 → 133, plain hex 231 → 118,
plain rect 215 → 81; grid recall −0.7 to −2.6 points, contrast groups −3 to
−9; recall floors re-measured.

### Outer-only centres from the homography (measured, rejected)

Of the visible cells missing while most lattice neighbours are reported
(hex ≥ 4 of 6, rect ≥ 6 of 8; `precision_surrounded_missing`, in-range
catalogue + grid: 7791 cells), 7358 (94 %) are reported once the
corrected-centre contract is lifted: fitted, identified and verified, dropped
for their outer-ellipse centre alone — 5645 without an inner ellipse, 1713
with one in that refinement-free run (no projective-centre gate rejects
anything in these scenes; the default run loses these rings upstream, cause
not yet identified). The rest: 88 geometric verify (74 shape, 14 position),
96 uncorrected + verify, 130 no outer fit, 36 never proposed, 65 a decode
contradicting the cell, 18 fitted but not labeled.

Tried: give such a labeled marker the pole of the homography's vanishing line
`H⁻ᵀ(0,0,1)ᵀ` with respect to its outer conic (exact under a homography; unit
tested), before the final refit and verify. Recall rose (grid: hex 0.694 →
0.789, Extended 0.708 → 0.860, rect 0.697 → 0.839, plain hex 0.410 → 0.430,
plain rect 0.498 → 0.604), but the correction moves the centre by a median
0.03–0.11 px (p95 ≤ 0.17 px): perspective is not what is wrong with these
centres — the outer edge fit is. Grid share beyond 1 px, new vs two-conic:
hex 10.8 % vs 2.7 %, Extended 6.5 % vs 1.6 %, rect 7.6 % vs 1.6 %, plain rect
3.3 % vs 0.9 % (median 0.25–0.37 vs 0.15–0.21 px); tilt groups 38–41 % vs
1.6–2.5 %; distortion median 0.76–1.08 vs 0.12 px. It also reported 3 new
wrong IDs on second-board scenes (hex/rect/Extended `foreign-same`): the
contract had been hiding rings of a second board labeled into this one. A
vanishing line from a DLT over each marker's two-conic neighbours (≥ 8 within
2 spacings) corrected no better (it only covered fewer, better-surrounded
markers; 2 of the 3 wrong IDs remained). Not shipped (the experiment is
commit `f91c764f`). The recall lever is the centre measurement itself
(inner-ellipse coverage, or the photometric fit), not the homography.

### Neighbour-supported inner retry (measured, rejected)

Every surrounded missing cell without an inner ellipse (`precision_inner_coverage`)
failed the inner-edge *estimate* (`estimate_not_ok`), 89 % with theta
consistency ≤ 0.2 (gate 0.25): the per-ray inner-edge peaks of a thin,
blurred inner band do not agree. 79 % are rings under 26 px.

Tried: for a labeled marker without an inner ellipse, search the inner edge
again with the window centred on the median inner/outer ratio of its
two-conic-centred lattice neighbours (≥ 3 within 2 spacings), each ray
required to show an interior radial extremum of the inner polarity, then the
ordinary fit gates and two-conic correction, before the final refit and
verify. No new wrong output; recall +0.3 to +1.0 point (grid: hex 0.694 →
0.701, Extended 0.708 → 0.717, rect 0.697 → 0.707, plain rect 0.498 →
0.505). The recovered centres are much worse than two-conic centres on the
same scenes — grid share beyond 1 px: hex 16.2 % vs 2.7 % (111 cells),
Extended 10.9 % vs 1.6 % (350), rect 12.7 % vs 1.7 % (260), plain rect
6.4 % vs 0.6 % (267); blur 4–6 % vs 0.6–2.3 %; tilt 18–25 % vs 1.6–2.5 %.
The estimate's consistency gate was rejecting inner edges that are not
measurable well enough for a calibration centre. Not shipped (commit
`a4f318a6`).

### Dark, low-contrast frames (measured, rejected)

Reproduced with a `dark` scene group (paper 55–75 grey levels, contrast
20–45, noise 1.5, rings 32–72 px): at contrast 20 every proposal is rejected,
mostly `no_valid_hypothesis`. The cause is the outer-edge ring-depth gate,
`edge_sample.min_ring_depth` = 0.08 of full scale — an absolute 20-grey-level
step across ±2 px; at 0.02 the same scenes fit 80–200 markers each. The
real-frame symptom (`outer_estimate_no_polarity_candidates`, the theta
consistency of the radius estimate) did not reproduce synthetically.

Tried, contrast-relative forms of the gate (rays whose step is ≥ f × the
candidate's median step; or `min(0.08, f × median)`, which only relaxes it
for low-contrast candidates). Dark recall rose by ~20 points (hex 0.564 →
0.77–0.78, Extended 0.535 → 0.73, rect 0.538 → 0.73–0.75) and contrast
groups by 8–19 points, but every form reported new wrong IDs on second-board views
(rect `foreign-same`/`foreign-other`: wrong IDs on primary rings; Extended
`foreign-same`/`-other`), `min(0.08, 0.5 × median)` a wrong homography (rect
grid d56 t50), and dark centres beyond 1 px rose (hex 13 → 59–115). Not
shipped (commit `5efb5466`); a second-board shape/size test is the
prerequisite.

The decode-anchored fallback's fixed `min_inliers` own-decode inliers: the
anchor's chance formula used verbatim (one free vote) passes the harness
(hex grid +0.6 point, no new wrong output) but is not valid for a
homography, which any four points fit — four votes are free. With four free
votes (`C(n, v) · n_cells^(4 − v) ≤ 1e-3`) 5 of 6 agreeing decodes are never
enough (hex: 6 · 203⁻¹ = 0.03), and larger decode sets need 7–8; the harness
is unchanged. Not shipped.

Follow-ups: projective-centre coverage at small, tilted rings (20 px at 55°:
143 of 390 identified rings have a corrected centre); a shape/size test for
rings of a second board of the same layout; a spacing-vs-ring-size check for
plain relative labelings.
