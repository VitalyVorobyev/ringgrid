# ADR-025: Plain Origin Anchoring by Anchor Cell

- **Status:** accepted
- **Date:** 2026-11-17
- **Author role:** algorithm-engineer
- **Amends:** the original origin enumeration and its cap

## Context

A plain board is anchored to its absolute frame by the origin-dot triad
around cell `(0, 0)` (ADR-021). The original resolver enumerated every placement
of the labeled patch on the board — each lattice symmetry × each translation
that keeps the patch on the board — fitted a homography per placement and
scored the projected dots, giving up beyond `MAX_CANDIDATES = 512`.

The number of placements grows with the board, not with the view: a
14 × 11-ring window of a 58 × 58 board admits ~17,000. Such a view — a
precision plate under a camera that sees a small part of it — stayed in the
relative frame with the triad plainly visible. Reproduced on main
(`tests/plain_partial_views.rs`, 14 × 11 window, 142 labeled rings): "too many
placement candidates". The precision harness's plain rect `partial` group had
recall 0.000 for the same reason.

## Decision

1. **Enumerate placements by anchor cell.** The triad surrounds board cell
   `(0, 0)`, so a placement is a symmetry `g` and the relative-frame cell `a`
   that carries the triad (`t = −g·a`). `a` runs over the labeled patch's
   coordinate bounds expanded by one step (`ANCHOR_SEARCH_MARGIN`); placements
   must keep the patch on the board (bounding box first, the full cell test
   only for placements that can still decide). The count follows the patch.
2. **One homography for scoring.** A DLT from relative-frame lattice mm
   (`LatticeGeometry::basis_mm · coord`) to pixels, fitted once. A placement
   re-expresses board mm by a lattice isometry, under which the normalized
   DLT is equivariant, so the old per-placement fit is needed only for the
   winner. A placement is orientation-preserving iff `det g` has the sign of
   that homography's Jacobian (this still admits the reflected placements
   when canonicalization left the relative frame mirrored).
3. **Acceptance unchanged:** weakest dot contrast ≥ 0.10 and a 0.05 margin
   over every other valid placement.
4. **Re-anchor after relative completion (plain only).** The labeler returns
   one connected component, which can stop short of the dots. An unresolved
   plain view is anchored once more after completion has grown the patch.

### Alternatives considered

- **Raise or drop the cap.** Correct, but the cost grows with the board (a
  DLT per placement) for a view whose evidence sits in one place.
- **Prune translations by projecting the triad into the image.** Still
  enumerates O(board) translations to find the few in view; the anchor-cell
  parameterisation reaches exactly those directly.
- **Detect dark blobs first, then match the triad pattern.** A second
  detector with its own thresholds; on the real plate there are many dust
  specks in gaps, and blur makes blob segmentation fragile. Verifying the
  predicted gaps measures the same darkness with the lattice as the prior.
- **Search the anchor far beyond the patch (extrapolate).** Under lens
  distortion an extrapolated homography can be off by a gap; the search would
  then pick the placement shifted to match — a wrong anchor by one cell. The
  margin keeps every dot within two steps of labeled rings; reach comes from
  the re-anchor on the completed patch instead.

## Consequences

- No cap; a small window of any board anchors like a small board. On small
  boards the true placement and its competitors are the same as before.
- The margin test now considers placements at and next to the patch only;
  dots extrapolated far from labeled rings no longer win or block.
- Views without the triad, with one or two dots in view, or with a speck
  that forms a second equally dark triad stay relative (unit and integration
  tests).

## Evidence

- `tests/plain_partial_views.rs`: 58 × 58 plain rect (2 mm pitch) through
  14 × 11, 20 × 16 and 30 × 30 windows at several offsets, rolls (incl. 90°,
  180°, 270°), tilt to 30°, Gaussian blur and a focus gradient; a 1,640-cell
  plain hex board: every board position equals the ground truth. Negatives:
  triad out of view, cut by the image border to two or one dot, specks in
  far gaps (correct anchor), a speck closing the triad's square (unresolved).
  On main every positive case stays relative.
- Precision harness (`precision_full_sweep`): no new wrong output; coded
  groups identical; plain rect recall grid 0.489 → 0.550, blur 0.459 → 0.535,
  dark 0.172 → 0.384, tilt 0.265 → 0.325, partial 0 → 0.152; plain hex grid
  0.402 → 0.415, dark 0.189 → 0.211, partial 0.614 → 0.777.
- Real images: 56 views of a 58 × 58 plain plate (2 mm pitch, ~137 px pitch,
  ~14 × 11 rings in view, two cameras, a focus gradient). Main anchors 0 / 56;
  this change 55 / 56 with projective and with photometric centres (the 56th
  has 2 detected rings). Labeled rings per anchored view: min 5, median 147.
  In every anchored view, under the reported labels the three darkest lattice
  gaps of the whole view are the triad (weakest triad contrast exceeds the
  darkest other gap by ≥ 0.55), so no view carries a label offset and all
  share one board frame. Board → image homography residual RMS median 0.31 px
  (max 0.84) projective, 0.25 px (0.58) photometric, no camera model.
- Timing (release, median of 5, `anchor_ms`): full 58 × 58 view (3,364
  rings) 1.60 → 0.54 ms; 14 × 11 synthetic window 0.64 → 0.20 ms; real view
  0.82 → 0.27 ms. Total detection time grows on views that now anchor
  (window 34.5 → 45.8 ms, real 351 → 366 ms): anchored completion visits the
  board's cells instead of growing the relative patch; full view unchanged
  (742 ms).
