# ADR-028: Verification on Refined Centres, Photometric Confidence, Joined Adaptive Candidates

- **Status:** accepted
- **Date:** 2026-11-18
- **Author role:** algorithm-engineer
- **Amends:** ADR-024 (geometric verification, confidence, output contract),
  ADR-026 (remaining losses)

## Context

After ADR-026 the IPGS precision plate (56 views, `detect_adaptive` with
photometric centres) reported 9,562 of its 10,162 fully visible rings, all
views anchored, no wrong label. ADR-026's per-stage probe attributed the 600
misses to:

| stage | rings |
|---|---:|
| photometric centre accepted, confidence < 0.03 through the edge-fit factors (inner-edge agreement, rim coverage) | 266 |
| reported by another adaptive candidate, not the selected one | 184 |
| removed by geometric verification on the pre-photometric centre | 126 |
| no outer fit / not grid-labeled | 24 |

Two further defects of the verification gate surfaced while fixing these:

- **The local test was affine.** It compared each centre with the midpoint of
  its opposite lattice neighbours, which is the image of the cell between
  them only under an affine map. Under perspective it misses by a term that
  grows with the square of the pitch in pixels: on
  `plainhex-grid-d56-b1.5-t30-s1` (precision harness) the five correct
  markers it removed sat 2.48–2.96 px from their midpoints, and the final
  homography's own second difference at those cells was 2.47–2.98 px. Only
  the scatter of blur-biased centres had kept the threshold above it; once
  photometric centres tightened it, correct markers fell.
- **Markers no test could reach.** With confidence no longer suppressing
  them, two rings at a corner of `hexext-distortion-l0.25` (uncorrected
  division-model distortion 0.25, no camera model) were reported with the
  neighbouring cell's ID: their centres were exact (0.16 / 0.18 px from a
  real ring), the labels a cell off. Neither had a labelled neighbour, so the
  local test could not see them, and the distortion had loosened the global
  test to 40.7 px (median residual 11 px) — above their 26 / 37 px.

## Decision

1. **Local test on the deviation field, after the global test**
   (`pipeline/geometric_verify.rs`). The second difference is taken of each
   marker's deviation from its cell's projection through the final
   homography (its centre when there is none): the homography's curvature
   cancels, leaving lens distortion's, centre noise and the ~1-pitch jump of
   a mis-celled marker. It runs on the markers the global residual and shape
   tests keep, so a gross blunder does not drag its correct neighbours over
   the tighter threshold. A marker with no opposite neighbour pair is tested
   one-sided — `2 f(c+d) − f(c+2d)` over the directions with both cells
   present, halved, which matches the midpoint residual in what it measures
   and in its noise — against the same threshold.
2. **Unverifiable markers are removed.** A labelled marker neither local test
   reaches is vouched for by the global test alone; where that threshold has
   grown to half a local pitch (the homography's image of one lattice
   spacing at the cell) it cannot tell the cell from the next, and the marker
   is removed (fail closed).
3. **Photometric configurations verify on refined centres**
   (`pipeline/finalize/`). Finalize only annotates the reprojection
   diagnostic; the output contract runs the same gate on the finished result,
   against the homography that best fits the photometric centres
   (finalize's final-H step over them alone). A homography fitted to every
   centre leans toward an edge-fitted centre a pixel or two off: on
   `hexext-partial-window-20` (27 markers) a projective centre 2.4 px off, its
   photometric fit not converged, sat 1.9 px from it and passed the 2 px
   floor. Adaptive candidates scored before the photometric stage are
   verified on their unrefined centres, as finalize would have.
4. **Photometric confidence** (`detector/marker_build.rs`). A marker keeps
   the factors its fits measured (`ConfidenceFactors`: evidence × outer fit,
   rim coverage, inner-edge agreement, border); `marker_confidence` is the one
   definition (edge-fitted confidences are bit-identical to before). For an
   accepted photometric centre rim coverage × inner-edge agreement is
   replaced by the fit's own precision
   `1/(1 + (σ / 0.1 px)²)`, `σ = √(tr Σ / 2)` from its centre covariance (which
   carries its residual level and correlation; convergence and residual are
   its acceptance gates), with ADR-026's ring-relative sharpness. Edge-fitted
   centres (outer ellipse, projective) are unchanged.
5. **Adaptive candidates are joined** (`pipeline/finalize/join.rs`). The
   best-scoring candidate takes the markers other candidates labelled at
   cells it lacks, from every candidate in the same board placement: an
   absolute board frame in the image frame, its labelled centres within 2 px
   (median) of the best candidate's homography — a placement a cell off or
   mirrored is a whole pitch away. The union's homography is refit and
   verified with finalize's own steps; photometric refinement still runs once,
   on the joined result. No candidate runs twice; the join costs one refit.

### Alternatives considered

- **Re-verify pre- and post-photometric, keep a marker passing either.**
  Two verdicts per marker and a second state on the record; verifying once
  on the best centres available is simpler and stricter.
- **Keep inner-edge agreement in the photometric confidence.** It is what
  hid the two corner mislabels above, by accident — it measures the edge
  sampler, not the label. Over the photometric precision catalogue the
  centre covariance ranks centre errors beyond 1 px at ROC AUC 0.999 against
  0.90 for the edge factors, with no accepted photometric centre off by more
  than 1 px. Labels are verification's job (decision 2).
- **Pool every candidate's tiers into one multi-scale merge.** Lets the
  wide single priors' double-size and inner-as-outer fits into one labelling;
  joining finished, anchored results keeps each candidate's own gates.
- **A completion bounds gate on the imaged ring instead of the ROI disc**
  (for the border misses). Measured on the `sharp_oblique` scene: the
  missed border rings are then attempted but the outer fit still fails (see
  *Remaining losses*). Not changed.

## Consequences

- Projective / none configurations: the verification changes (1, 2) only.
  Precision sweep: correct 132,352 → 132,496, wrong IDs / positions / labels
  unchanged (81 / 0 / 5, all out of scope or known), centres off > 1 px
  335 → 336.
- Photometric configurations report every centre the photometric fit vouches
  for. Their `confidence` changes meaning: a `min_confidence` tuned on
  photometric centres under 0.17-dev admits more markers.
- `detect_adaptive` can report markers from several candidates; the
  returned homography is refit over the union.

### Remaining losses

98 of the plate's 10,161 expected rings (0 wrong labels). Rerun at
`min_confidence = 0`: 60 are then reported (11 within two outer radii of the
image border, 49 interior; 30 of the 98 misses are on one view,
Cam1 161226), 38 are not detected at all (19 border, 19 interior). The border
fit failures trace to the outer estimate: its radial scan drops every ray
whose search window leaves the image (theta coverage 0.85 for a fully
visible ring 40 px from the bottom edge), the second-harmonic model is then
extrapolated across the hole, the edge points there are missed and the fit's
angular-gap gate (π/2) rejects it (reproduced on `plain_large_ring_recall`'s
`sharp_oblique`, cells [0, 8] and [-2, 9]). Truncating radial scans at the
image boundary is its own change.

## Evidence

- IPGS plate, `detect_adaptive` + photometric, per decision (cumulative):
  9,562 → 9,584 (1) → 9,634 (3) → 10,021 (4) → 10,063 (5) of 10,161–10,162
  (recall 0.941 → 0.990), 0 wrong labels throughout, per-view homography RMS
  0.04–0.07 px.
- Precision sweep, photometric refinement (harness knob, not committed):
  correct 143,659 → 146,367; centres off > 1 px 10 → 7; gross centre errors
  5 → 4; wrong IDs 82 → 81 (all out of scope: a second board in view). The
  one in-scope wrong ID before — a corner mislabel under distortion 0.25 —
  is removed by decision 2; decision 4 without decision 2 reported a second
  one (83).
- rtv3d (coded, real): total decoded A / B / D 3,976 / 3,710 / 2,625 →
  3,987 / 3,745 / 2,651. Synthetic benchmarks: 71 / 71 regression checks
  pass.
- `tests/plain_large_ring_recall.rs`: `sharp_oblique` 108 of 110,
  `focus_gradient` 79 of 79 (was 75).
