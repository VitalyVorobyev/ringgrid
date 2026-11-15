# ADR-023: Grid-First Finalize for Every Target

- **Status:** accepted; anchor acceptance, contradicted decodes and the
  fallback floor amended by ADR-024
- **Date:** 2026-11-15
- **Author role:** algorithm-engineer
- **Supersedes:** the coded back half of ADR-017 (plain targets already ran
  this path); keeps ID correction only as a fallback

## Context

The finalize stage had two designs. Plain targets labeled every fitted ring
with lattice coordinates (`projective_grid::detect_grid`) and anchored the
labeling with origin dots. Coded targets ran ID correction (a hex-neighbor
voting engine, `detector/id_correction/`, 3,843 lines, hex only) and then a
RANSAC homography over **decoded** markers only, discarding every well-fitted
ring whose own code band did not decode; completion then refitted those cells
from scratch. Rect coded targets got no ID consistency check at all.

Measured on the old path: only 73% (rect coded) to 80% (Extended hex) of final
markers came from their own fit; raw accepted decodes are wrong far more often
than the outputs suggested (contradicted decodes among grid-labeled rings:
blur3 49.6%, Extended hex 26.2%, rtv3d 27–30%, mostly IDs not on the board,
nearly all at distance 0 with margin 1–2) — ID correction hid this.

## Decision

One back half for every target (`pipeline/finalize/grid.rs`):

1. **Label** every fitted ring, decoded or not (`pipeline/assign.rs`), in a
   canonical, orientation-preserving relative frame. The label residual gate
   is ringgrid's frame-RANSAC inlier threshold, not projective-grid's 2 px
   default (see the upstream defects below).
2. **Anchor** the relative frame to the board (`pipeline/anchor.rs`,
   `OriginEvidence`): coded targets by a vote of decoded IDs — each labeled,
   decoded ring votes, per lattice symmetry, for the translation mapping its
   cell onto its decoded cell; the winner needs ≥ 2 votes, a strict majority of
   the decoded labeled rings, a strict lead over the runner-up and an
   orientation-preserving homography (only such placements compete). Plain
   targets keep origin-dot verification.
3. **Identify**: every labeled ring takes its cell's ID; a contradicting
   decode is overridden and counted (the ring is an observation of that cell;
   a ring at a wrong place fails geometric verify).
4. **Join and complete**: dropped rings with a decode onto a free cell join
   through a RANSAC homography over identified + decoded rings (the global
   filter's terms); completion grows from the identified patch one lattice ring
   per round, adopting an already-fitted dropped ring within
   max(`reproj_gate_px`, half its outer radius) of the cell seed before
   refitting, then sweeps the whole board once.
5. Marker fixup (inner-as-outer recovery, axis-ratio filter), final H refit,
   geometric verify — shared with the fallback.

**Fallback ladder** (`finalize/fallback.rs`), when no grid is labeled (coded:
or none anchored): hex ID correction → global RANSAC filter over decodes →
completion → final H → verify. With fewer than four decoded correspondences
(or fewer than `ransac_homography.min_inliers` = 6 consistent ones) the
markers are returned without a homography; a single decoded marker keeps its
ID. Plain views without a grid return unlabeled markers.

## Evidence

Old path vs final (synthetic benchmarks; precision 1.0, ID accuracy 1.0 and 0
wrong IDs in every mode on both):

| Benchmark | Recall | Centre err (px) | H vs GT (px) | From own fit (`decoded_fraction`) |
|---|---|---|---|---|
| reference (both modes) | 1.0 → 1.0 | unchanged | 0.0188 → 0.0190 | 0.977 → 0.977 |
| distortion, no correction | 1.0 → 1.0 | 0.0925 → 0.0941 | 1.324 → 1.355 | 0.964 → 0.985 |
| distortion, external / self-undistort | 1.0 → 1.0 | ±0.001 | ±0.015 | — |
| blur3 | 0.972 → 0.988 | 0.438 → 0.444 | 0.143 → 0.143 | 0.785 → 0.776 |
| rect coded | 1.0 → 1.0 | 0.0524 → 0.0522 | 0.0147 → 0.0150 | 0.727 → 0.938 |
| hex Extended (1239 cells) | 1.0 → 1.0 | 0.0554 → 0.0556 | 0.0126 → 0.0123 | 0.804 → 0.807 |

Plain target-matrix modes: identical outputs.

Stress suite (32 rendered cases × 3 images: partial views, occlusion, tilt
45–60°, radial distortion without a camera model, 1–6 visible markers, blur
3 px), correct IDs summed: 16,814 → 17,452. Largest changes (ID recall):
rect tilt 45° 0.856 → 0.987, hex tilt 45° 0.777 → 0.891, Extended tilt 45°
0.886 → 0.922, hex distortion 0.3 0.730 → 0.884, rect distortion 0.3
0.730 → 0.866; losses: rect distortion 0.15 0.945 → 0.925 (geometric verify
removes more peripheral markers), hex corner precision 0.984 → 0.978 (a correct
ID on a border-cut ring 8.2 px off the 8 px gate). Rect tilt 60° is broken on
both paths (see follow-ups).

Real images (rtv3d, 102 tiles, total identified markers):

| Strategy | Old | Grid-first, no ID correction | Final (ID correction in fallback) |
|---|---|---|---|
| A (20–56 px) | 4,477 | 4,577 | 4,576 |
| B (14–66 px) | 3,696 | 3,844 | 3,913 |
| D (per-strip prior) | 2,702 | 2,821 | 2,867 |

On every tile, no ring carries different IDs on the old and the grid-first
path. All large real-image losses of grid-first alone were fallbacks (no grid
labeled); ID correction inside the fallback recovers them.

Timing (single-pass detect, median of 5): finalize 222.8 → 86.6 ms on the
1239-cell Extended board (total 492.7 → 357.6 ms), 8.2 → 4.0 ms on the
203-cell reference board, unchanged on rect coded.

## Tried and rejected

- **Decode against the target's IDs only** (codebook restricted to the IDs
  the target prints). Off-board decodes vanish and the raw error rate drops
  (blur3 49.6% → 28.2%, Extended 26.2% → 13.5%), but noisy words that used to
  land on an off-board ID — a free error signal — become plausible on-board
  wrong IDs: stress wrong IDs rect tilt 60° 12 → 89 (old path) and hex tilt 60°
  2 → 48 (grid-first); 16–17 cross-path ID conflicts on rtv3d where there were
  0; old-path blur3 recall 0.972 → 0.953.
- **Multi-component labeling + per-component anchoring**
  (`detect_grid_all`): identical outputs on every benchmark, stress case and
  rtv3d strategy — real-tile components are 4–9 rings that fail the frame
  RANSAC or carry < 2 agreeing decodes.

## Upstream defects (projective-grid 0.14.4)

1. `estimate_projective` (`geometry/mod.rs`) is an f32 DLT with h33 = 1 and no
   normalization. On exact projected centres of the 1239-cell perspective board
   it leaves a 3.3 px median residual; with the default 2 px label gate only
   92/1239 correct labels survive. Mitigated here (label gate = ringgrid's
   RANSAC threshold).
2. Hex `label_components` (`topological/hex.rs`) discards a whole connected
   component on the first parallelogram-completion label conflict. Under
   perspective, a two-point hole is enough: exact centres of the same board
   label 1239/1239, removing two neighboring points labels 0. Real fit pools
   have holes, so labeling is chaotic (484 vs 29 labeled rings for a 1%
   change of the input). This is what still sends real views to the fallback.

## Consequences and follow-ups

- Delete `detector/id_correction/` (and `IdCorrectionConfig`) once defect 2
  is fixed upstream and the fallback no longer needs it; until then it runs
  only in the fallback, hex coded targets only.
- The global filter accepts a 6-of-61-inlier homography at extreme tilt (rect
  60°): a few perfect-distance misdecodes agree on a wrong homography and
  completion fills cells along it. Needs a minimum inlier fraction.
- Strategy C / very wide single-pass scale prior (8–220 px): the proposal
  stage yields 1–4 proposals once the maximum diameter exceeds ~110 px.
- Raw distance-0, margin-1 decodes are frequently wrong; any consumer of raw
  decodes needs a geometric check.
- The grid-first path leans on the join/adopt steps because the labeler
  returns small components; once defect 2 is fixed, re-measure whether the
  growth-then-sweep double completion is still needed (it currently adds
  ~2% of the markers on rtv3d A).
