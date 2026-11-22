# ADR-026: Recall of Large Plain Rings on Oblique Views

- **Status:** accepted
- **Amended by:** ADR-028 (remaining losses), ADR-029 (decision 3: no sharpness factor in a photometric centre's confidence)
- **Date:** 2026-11-18
- **Author role:** algorithm-engineer
- **Amends:** ADR-023 (grid labeling), ADR-024 (confidence, output contract)

## Context

A precision plate — 58 × 58 plain rings at 2 mm pitch, outer / inner radius
0.8 / 0.4 mm, an origin-dot triad — imaged by two 2048 × 1536 cameras from
frontal to strongly oblique (56 views, rings 105–123 px across the major
axis, axis ratio 0.57–0.81, a focus gradient of σ ≈ 0.8–6 px). Through
`detect_adaptive` with photometric centres (the calibration wrapper's
configuration) ringgrid 0.16 reported 6,324 of the 10,162 rings whose whole
outer ellipse lies in the image (recall 0.62). Four oblique views reported
0–2 rings with a board position, though each shows ~240 sharp rings.

Measured with `tools/score_coverage.py` and `tools/run_ipgs_eval.py`: the
expected set comes from a reference fit independent of the detector under
test (a relaxed single-pass run, its lattice basis checked against the ring
shapes, its origin found from the triad's dark dots in the image), and a
per-stage probe followed every expected ring through the pipeline.

Root causes, by rings lost:

1. **Confidence of blurred large rings.** The sharpness factor
   `1/(1 + (w / 3 px)²)` is absolute in the edge width `w`. Large rings
   blurred to `w ≈ 8 px` scored ~0.11 and fell below `min_confidence` (the
   smallest factor for ~95 % of the rings the contract dropped), although
   their photometric centres sat within 0.05 px (median) of the plate's
   homography. Confidence is never re-evaluated after the photometric fit.
2. **Outer fit on sharp, eccentric rings.** The fit searches each ray within
   ±1 px of the hypothesis's second-harmonic radius model, which is off by up
   to 3.3 px at an axis ratio of 0.56 (1.3 px at 0.70). A sharp edge was lost
   on most rays; a blurred one, wider than the error, survived. On oblique
   views the in-focus band was the band that never fitted (54–83 rings per
   view), and the hole split the labeled lattice into two components (72 more
   rings unlabeled on Cam1 161201).
3. **Sheared grid labels.** `projective_grid` labels centres from positions
   alone; on a strongly foreshortened rect view the image diagonal is as
   short as the foreshortened axis and it walked a valid but sheared basis
   of the point lattice. A homography fits those labels as tightly (RMS
   0.01 px), the triad can never match them, and three views stayed in the
   relative frame with 19 wrong-label suspects.
4. **Adaptive candidate scoring.** `mapped` counted markers with a decoded
   ID — always 0 on a plain target — and the score was taken before the
   output contract, so an unanchored candidate, or one whose labeled rings
   all fell below `min_confidence`, could win.
5. **Code-band scale probe on plain rings.** The probe reads the code band's
   angular variance; plain rings have none, its tiers (68–371 px for 110 px
   rings) found nothing, at the cost of a full multi-scale pass.

## Decision

1. **Lattice-basis check** (`pipeline/lattice_basis.rs`, called by
   `assign_plain_grid`, every target). Each outer ellipse is the image of a
   circle and measures the local affine map independently of the labels. The
   unit-determinant shapes the frame homography predicts are compared with
   the fitted ones (median Frobenius distance). Labels that predict them
   stand; otherwise the orientation-preserving unimodular relabeling (entries
   ≤ 2) that does, at least 2× better, re-indexes them; if none does, the
   labeling is refused (fail closed). Correct labelings measure a median of
   0.03 (p99 0.18, max 0.56) over the precision sweep and up to 0.26 on the
   plate; sheared ones 1.2–1.7. Gate 0.75.
2. **Ellipse-exact outer fit** (`detector/outer_fit`). The seed pass widens
   the refine window to the model's own error bound `A²/c0` (capped at half
   the dark band along the minor axis; a circle keeps the configured window)
   and, when the aggregated peak and the model's mean radius differ by more
   than the window, samples both centrings and keeps the one finding more
   edge points. Then every ray is re-sampled within the configured window
   around its intersection with the fitted ellipse and refit, up to two
   rounds while that finds more points. The accepted fit keeps every gate
   (`min_rays_with_ring`, angular gap, per-ray expected-radius gate).
3. **Ring-relative sharpness for photometric centres** (`marker_build.rs`,
   applied by the photometric stage on acceptance): the half-sharpness width
   becomes `max(3 px, 0.1875 r)`, `r` the outer mean semi-axis — identical
   for rings up to 32 px across. Edge-fitted centres keep the absolute
   factor.
4. **Adaptive scoring** counts markers with a board position and scores each
   candidate after the output contract.
5. **No code-band probe for plain targets** without a diameter hint.
6. **Completion refits anchored rings without a corrected centre** (labeled,
   projective correction failed): the cell is offered to completion's refit
   under the same gates and the ring is restored when the refit does not take
   it. Unresolved plain labelings are left alone.
7. **Coded completion refits need local support** (`detector/completion.rs`):
   a refit takes the ID of the cell its seed predicts, so on coded targets it
   is attempted only where at least 3 labeled rings that are not themselves
   refits lie within 3 lattice spacings. Far from any fitted ring the seed
   extrapolates — from the homography or a chain of refits — and uncorrected
   lens distortion shifts it by a whole cell at the periphery: with the
   outer-fit change on the ADR-027 decoder, refits on coded hex views at
   division-model distortion 0.20 / 0.25 sat exactly on the neighbouring ring
   with this cell's ID. Plain targets are exempt (their labels come from this
   growth; the gate costs a third of their distorted-view recall).

### Alternatives considered

- **Ring-relative sharpness for every centre.** Recovers the same plate
  rings, but over the precision harness (edge-fitted centres) the ROC AUC of
  confidence for centre errors beyond 1 px falls from 0.90 to 0.84 (0.76
  without the 3 px floor) and 29 more such centres are reported. Blur moves
  an edge-fitted centre in absolute pixels; it is a fitted parameter of the
  photometric model.
- **Only the ellipse-exact resampling, or only the wider seed window.**
  Resampling alone recovers 40 of ~80 lost rings per view (the seed fit
  often has too few points); the seed window alone 67, both together 69–73.
  The window alone leaves the final points searched in ±4 px; the resampling
  puts them back in ±1 px of an exact model.
- **Dropping the per-ray expected-radius gate in the resampling pass.**
  Recovered no plate ring and let 8 bogus double-size fits through on small
  rings at 60° (centre errors 7–10 px).
- **Feeding `projective_grid` oriented evidence (ellipse-rectified
  neighbour directions).** Would prevent the shear at the source, but needs
  per-ring neighbour estimation before labeling; the post-hoc check is exact
  (a relabeling changes no point) and fails closed.
- **Completion bounds gate on the predicted ellipse** (instead of centre ≥
  ROI + 10 px): measured first — over all 56 views and every adaptive
  candidate only 36 rejected seeds had a predicted ellipse inside the image.
  Not changed.
- **Changing the target model for the plate's triad offset.** The plate's
  triad sits 28 / 30 columns from the edges on one axis; ringgrid's 58 × 58
  auto layout centres it (29 / 29), so absolute labels differ from the
  plate's reference file by one column and one plate column lies off the
  layout. No view reaches the plate edge; not changed.

## Consequences

- Plain oblique views anchor and keep their sharp rings; a labeling no
  lattice basis reconciles with the ring shapes is refused rather than
  reported (coded targets then take the decode-anchored fallback).
- More markers on every target kind (precision sweep, on the ADR-027
  decoder: correct 118,039 → 132,352, centres off by more than 1 px
  955 → 335, gross centre errors 7 → 5), with wrong IDs, positions and labels
  unchanged on every in-scope scene; outer-ellipse centres move (refit
  edges), photometric centres by ≤ 0.007 px mean.
- Out of scope (ADR-024: one board per image), `rect-foreign-same` — a second
  copy of the same coded rect board in view — now reports the other copy
  (80 markers, consistent among themselves, counted as wrong IDs) where main
  reported the primary: the outer fit now fits enough of the tilted copy for
  it to win the labeling. Main's choice was no less arbitrary; a
  duplicate-board guard would be its own change.
- Confidence of photometric centres is ring-relative in its sharpness term;
  `min_confidence` keeps its meaning for edge-fitted centres.

### Remaining losses

Of the plate's 10,162 expected rings, 600 are still not reported (all 56
views anchored, no wrong label). By stage (per-stage probe, every view):

| stage | rings (border / interior) |
|---|---:|
| photometric centre accepted, confidence still < 0.03 through the edge-fit factors (inner-edge agreement 172+39, rim coverage 29+26) | 266 (65 / 201) |
| reported by another adaptive candidate, not the selected one | 184 (23 / 161) |
| removed by geometric verification (global residual test on the pre-photometric, blur-biased centre) | 126 (66 / 60) |
| no outer fit | 21 (14 / 7) |
| fitted, not grid-labeled | 3 |

Each needs its own change: a photometric-fit quality term in the confidence
(the edge-fit factors describe the edge ellipse, not the photometric
centre), verifying photometric configurations after the photometric stage,
and merging adaptive candidates instead of selecting one.

## Evidence

- IPGS plate, 56 views, `detect_adaptive` + photometric (sequential runs):
  detected 6,324 → 9,562 of 10,162 expected rings (recall 0.622 → 0.941),
  views anchored 53 → 56, wrong-label suspects 19 → 0, per-view homography
  RMS 0.034–0.071 px; mean runtime 12.2 / 10.6 s → 11.3 / 10.4 s (Cam1 /
  Cam2). Per fix, on 0.16: lattice basis 6,324 → 6,351 (19 → 0 wrong
  labels, 3 views anchored); outer fit → 7,981; photometric sharpness →
  9,557; adaptive scoring → 9,564; no plain probe → same rings, 1.1–1.9 s
  less per view; completion of uncorrected rings → 9,562 (neutral; +3 % on
  the precision sweep).
- rtv3d (coded, real): total decoded A / B / D 3,509 / 2,954 / 2,163 →
  3,976 / 3,710 / 2,625; of the rings both report, the IDs agree except one
  view where main reports seven IDs at one point (a degenerate homography)
  and this branch seven distinct rings.
- Synthetic benchmarks: 71 / 71 regression checks pass; blur2 recall
  0.921 → 0.976, centre errors unchanged or lower.
- `tests/plain_large_ring_recall.rs`: on main one view does not anchor
  (8 markers) and the other reports 31 of 79 rings; now 108 of 110 and 75 of
  79, every position true.
- Reproduce with `tools/run_ipgs_eval.py` (private data) and
  `tools/score_coverage.py`.
