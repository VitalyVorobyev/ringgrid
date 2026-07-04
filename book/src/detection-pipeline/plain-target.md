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

Orchestrated by `pipeline/finalize/plain.rs`, dispatched from `finalize/mod.rs`
when the target is not coded and `advanced.use_global_filter` is `true`
(otherwise the shared no-homography passthrough runs and markers stay unlabeled):

| Order | Stage | Description |
|-------|-------|-------------|
| 1 | Projective center | Correct fit centers once per marker (shared with coded) |
| 2 | [Grid assignment](#grid-assignment) | Label ring centers with lattice coordinates; fit the frame homography |
| 3 | [Origin anchor](#origin-resolution) | Resolve the board origin from fiducial dots (when present) |
| 4 | [Completion](#completion) | Coordinate-keyed fits at missing cells; grows the labeled patch when unanchored |
| 5 | Final H refit | Refit the frame/board homography over all labeled markers |
| 6 | [Geometric verify](overview.md#geometric-verification) | The same lattice-consistency gate as the coded path |

## Grid assignment

`pipeline/assign.rs` (`assign_plain_grid`) turns the finite ring centers into
lattice-labeled correspondences:

1. Collect finite ring centers as point features (needs ≥ 4).
2. Build a `projective_grid::detect_grid` request carrying the lattice kind and,
   for rect, the `(cols, rows)` grid dimensions.
3. Call `detect_grid` for **topological** labeling only (its facade is `f32`).
4. Canonicalize the labels — `grid.normalize()` for a square lattice,
   `canonicalize_hex_entries` for hex — so a given physical layout always yields
   the same coordinate assignment.
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

1. **Enumerate coordinate maps.** Consider every `(rotation × translation)` that
   embeds the labeled patch into the board's cell set. Only rotations with
   determinant `+1` are allowed — reflections are excluded because the fiducial
   arrangement (and the whole target) is not mirror-symmetric. The candidate set
   is capped at `MAX_CANDIDATES = 512`.
2. **Fit and gate each candidate.** For each map, fit a board→image homography by
   DLT and reject it if the Jacobian determinant is ≤ 0 at the patch center
   (a folded / reflected mapping).
3. **Score by dot darkness.** Project each fiducial dot through the candidate
   homography and measure normalized `(background − dot)` contrast with a
   distortion-aware disk-vs-annulus sample. A candidate's score is its *weakest*
   dot (the min over dots).
4. **Accept the winner** only if its contrast ≥ `MIN_DOT_CONTRAST = 0.10` **and**
   its margin over the runner-up ≥ `MIN_MARGIN = 0.05`. This "verify at predicted
   positions" test is deliberately conservative: a wrong origin is worse than an
   unresolved one.

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

**Source:** `pipeline/finalize/plain.rs`, `pipeline/finalize/mod.rs`,
`pipeline/assign.rs`, `pipeline/anchor.rs`, `pipeline/result.rs`,
`target/fiducials.rs`
