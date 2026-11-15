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
view, and a few fail-closed scenes outside the operating range) plus a 450-scene
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

### Outer-only centres from the homography (measured, rejected)

Of the visible cells missing while most lattice neighbours are reported
(hex ≥ 4 of 6, rect ≥ 6 of 8; `precision_surrounded_missing`, in-range
catalogue + grid: 7791 cells), 7358 (94 %) are reported once the
corrected-centre contract is lifted: fitted, identified and verified, dropped
for their outer-ellipse centre alone — 5645 without an inner ellipse, 1713
with one whose two-conic correction was rejected. The rest: 88 geometric verify (74 shape, 14 position), 96
uncorrected + verify, 130 no outer fit, 36 never proposed, 65 a decode
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
commit `f91c764f`). The recall lever is the
centre measurement itself — inner-ellipse coverage first (77 % of these
cells), then the two-conic gates (23 %), or the photometric fit — not the
homography.

Follow-ups: projective-centre coverage at small, tilted rings (20 px at 55°:
143 of 390 identified rings have a corrected centre); a shape/size test for
rings of a second board of the same layout; a spacing-vs-ring-size check for
plain relative labelings.
