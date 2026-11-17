# Plain / Rect Target Detection

Plain (uncoded) rings carry no per-marker identity, so the [coded
path](overview.md)'s decode → ID-correction → global-filter cannot label them.
Instead, plain targets are labeled by their **lattice position** and, when the
target carries [origin fiducials](../targets/origin-fiducials.md), anchored to an
**absolute board frame**. This page describes that path end to end.

The plain path shares the front half of the pipeline — [proposal](proposal.md),
[outer estimate](outer-estimate.md), [outer fit](outer-fit.md), [inner
estimate](inner-estimate.md), [dedup](dedup.md), and [projective
center](projective-center.md) — with the coded path. It diverges in *finalize*:
where the coded path decodes IDs, the plain path assigns grid coordinates. Which
lattice × coding combination runs which path is summarized in the [target
composition matrix](../targets/target-model.md#composition-matrix--how-each-combination-detects).

## Finalize stages (plain path)

Orchestrated by `pipeline/finalize/grid.rs` (the back half every target runs; coded
targets anchor by decoded IDs instead of dots) when `advanced.use_global_filter` is `true`
(otherwise the shared no-homography passthrough runs and markers stay unlabeled):

| Order | Stage | Description |
|-------|-------|-------------|
| 1 | Projective center | Correct fit centers once per marker (shared with coded) |
| 2 | [Grid assignment](#grid-assignment) | Label ring centers with lattice coordinates; fit the frame homography |
| 3 | [Origin anchor](#origin-resolution) | Resolve the board origin from fiducial dots (when present) |
| 4 | [Completion](#completion) | Coordinate-keyed fits at missing cells; grows the labeled patch when unanchored, then retries the origin anchor |
| 5 | Final H refit | Refit the frame/board homography over all labeled markers |
| 6 | [Geometric verify](overview.md#geometric-verification) | The same lattice-consistency gate as the coded path |

## Grid assignment

`pipeline/assign.rs` (`assign_plain_grid`) turns the finite ring centers into
lattice-labeled correspondences:

1. Collect finite ring centers as point features (needs ≥ 4).
2. Build a `projective_grid::detect_grid` request carrying the lattice kind and,
   for rect, the `(cols, rows)` grid dimensions.
3. Call `detect_grid` for **topological** labeling only (its facade is `f32`).
4. Canonicalize the labels with `canonicalize_entries` (both lattices) so a given
   physical layout always yields the same coordinate assignment. `detect_grid`
   labels are defined only up to the lattice's full symmetry group — reflections
   included (hex labels come out mirrored) — so canonicalization picks, from the
   whole dihedral group, the transform that makes the labeling
   **orientation-preserving** (the image-frame cross product of the `+u` and
   `+v` steps has the sign of the lattice model basis, i.e. the labels relate to
   the board by a rotation + translation, never a reflection) and, among those,
   best aligns `+u` with image `+x` and `+v` with image `+y`. When no symmetry
   yields both a `+u` and a `+v` neighbor pair (e.g. a patch of two non-adjacent
   rows) the handedness cannot be measured and the labels are left as
   `detect_grid` produced them — possibly mirrored; origin resolution below
   still recovers the correct handedness in that case.
5. Refit the frame homography in `f64` with ringgrid's `fit_homography_ransac`
   over the labeled correspondences, keeping only homography inliers — mirroring
   the coded [global filter](projective-center.md#global-filter).

The result is a set of markers labeled in a **canonical relative frame**. If
fewer than four centers survive, grid assignment returns `None` and the markers
pass through unlabeled.

## Origin resolution

When the target carries [origin fiducials](../targets/origin-fiducials.md) and at
least four markers are labeled, `pipeline/anchor.rs` (`resolve_origin`) upgrades
the relative labeling to an **absolute** board frame:

1. **Fit the patch homography once.** A DLT from relative-frame lattice
   millimetres to image pixels over every labeled ring. A placement only
   re-expresses board millimetres by a lattice isometry, under which the
   normalized DLT is equivariant, so one fit serves every placement.
2. **Enumerate placements by anchor cell.** The dot triad surrounds board cell
   `(0, 0)`, so a placement is a lattice symmetry plus the relative-frame cell
   that carries the triad. That anchor cell is searched over the labeled patch's
   coordinate bounds expanded by one lattice step, over the lattice's full
   dihedral group (rotations **and** reflections): the count is set by the
   patch, not by the board, so a small window of a large board costs the same as
   a small board. A placement must put every labeled ring on a board cell.
3. **Gate on orientation.** A placement's board→image map is the patch
   homography composed with the placement, so it is orientation-reversing
   exactly when its symmetry's handedness differs from the patch homography's.
   This physical check is what excludes reflections: when the relative frame is
   orientation-preserving (the normal case) every reflected placement fails it,
   and when canonicalization could not fix the handedness the reflected
   placements are the ones that pass.
4. **Score by dot darkness.** Project each fiducial dot through the patch
   homography and measure normalized `(background − dot)` contrast with a
   distortion-aware disk-vs-annulus sample. A placement's score is its *weakest*
   dot (the min over dots); a placement with a dot off-image is unscorable, not
   dark.
5. **Accept the winner** only if its contrast ≥ `MIN_DOT_CONTRAST = 0.10` **and**
   its margin over every other valid placement ≥ `MIN_MARGIN = 0.05`. This
   "verify at predicted positions" test is deliberately conservative: a wrong
   origin is worse than an unresolved one. A dark speck that forms a second
   triad with two of the dots makes the view ambiguous (unresolved); a speck
   anywhere else cannot complete a triad.

Dots are verified only next to labeled rings, where the patch homography is
measured. The labeler returns one connected component, which can stop short of
the dots; an unresolved plain view therefore anchors once more after
[completion](#completion) has grown the patch over the visible board.

On success, each marker's `grid_coord` is remapped through the winning
coordinate map to absolute board cells, `board_xy_mm` is filled from the target's
`cell_xy_mm`, and the working homography becomes the anchored board→image
homography.

## Completion

Plain completion (`complete_plain_with_h`) is coordinate-keyed and
lattice-generic. It fits conservative markers at cells that the topological
labeler missed:

- **Anchored:** complete across the whole board.
- **Unanchored:** grow the labeled patch's bounding box by one lattice ring per
  round, recovering the neighbors `detect_grid` dropped without assuming a board
  extent.

New completion markers receive their own projective-center correction.

## Frame contract

`enforce_plain_frame_contract` makes the output frame explicit and honest
(`pipeline/result.rs`, [`BoardFrame`](../output-types/detection-result.md)):

- **Anchored** → [`BoardFrame::Absolute`]: `grid_coord` is in board cells and
  `board_xy_mm` is populated.
- **Unanchored** → [`BoardFrame::RelativeCanonical`]: `grid_coord` is in the
  canonical relative frame and **all `board_xy_mm` are cleared to `None`** — a
  wrong millimeter position is worse than none.

Coded targets always report `board_frame = absolute` (IDs are globally unique);
the relative/absolute distinction only arises for plain targets.

**Source:** `pipeline/finalize/grid.rs`, `pipeline/finalize/mod.rs`,
`pipeline/assign.rs`, `pipeline/anchor.rs`, `pipeline/result.rs`,
`target/fiducials.rs`
