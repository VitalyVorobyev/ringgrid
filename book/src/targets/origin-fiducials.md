# Origin Fiducials

Plain (uncoded) markers carry no identity, so a plain target's labeling is only
known up to the lattice's rotational symmetry and a lattice translation. Origin
fiducials — small dark filled dots printed in the gaps between markers — pin the
board origin and orientation so outputs can be reported in absolute board
millimeters. This page covers how the dot pattern is validated at construction
time and how the detector resolves the origin from it.

## The dots

`OriginFiducials { dot_radius_mm, dots_mm }` lists dark disks in board-frame
millimeters (the same frame as cell centers). The ISRA preset uses three dots in
an L near the board center. Dots serve two jobs:

1. **Break the lattice symmetry** so exactly one orientation is consistent with
   them.
2. **Be verifiable in the image** — dark against the white background at a
   predictable place.

## Symmetry validation (construction time)

`TargetLayout::new` rejects a fiducial set that does not break **every**
rotational symmetry of the cell lattice. Candidate symmetries come from the
lattice family (90° steps for square, 60° steps for hex) and each is verified
numerically against the actual finite cell positions — so finite-patch effects
are handled exactly (a non-square rect patch only admits a 180° half-turn, a
square patch admits all of 90/180/270). If the dot pattern maps onto itself
under any surviving rotation, construction fails with
`FiducialsRotationallySymmetric`.

Validation also enforces **clearance**: no dot may fall within
`outer_draw_radius_mm + dot_radius_mm` of any marker center, otherwise both
rendering and dot detection would be ill-defined (`DotOverlapsMarker`).

### Why only rotations, not reflections

An opaque planar target viewed by a camera always images through an
**orientation-preserving** homography (positive Jacobian determinant). A
reflected labeling would require an orientation-reversing map, which is
physically impossible for a printed target. So reflections can never cause a
labeling ambiguity, and the dot pattern only has to break rotational symmetry —
not the full dihedral symmetry group. This is the same reason the origin
resolver enumerates rotations only.

## Origin resolution (detection time)

`pipeline::anchor::resolve_origin` runs in the plain finalize path after grid
labeling. Grid labeling produces `grid_coord`s in a **canonical relative frame**;
resolution decides whether they can be remapped to absolute board cells.

The algorithm is *verify-at-predicted-positions* — no separate dot detector:

1. **Enumerate candidates.** For each lattice rotation (determinant +1 only) and
   each lattice translation that embeds the whole labeled patch into the board
   cell set, form a `(rotation × translation)` coordinate map. The candidate set
   is capped at 512; a patch too small on a large board explodes the translation
   count and is declined rather than guessed.
2. **Fit and screen each candidate.** Fit a board→image homography by DLT over
   the labeled correspondences, and reject it if its Jacobian determinant is
   non-positive at the patch center (orientation-reversing — physically
   impossible).
3. **Score dot darkness.** Project each dot through the candidate homography and
   measure the normalized `(background − dot)` intensity contrast: a dark disk at
   the predicted center against a clear background annulus around it. A candidate
   whose dots fall off-image or sub-pixel is *unscorable* (declined), not dark.
   The candidate's score is its weakest dot's contrast.
4. **Accept the winner** only if it clears an **absolute contrast threshold**
   (`0.10`) **and** beats the runner-up by a **margin** (`0.05`). Otherwise the
   origin stays unresolved.

Resolution needs at least four labeled markers to fit a stable homography.

## `board_frame` and what callers see

The outcome is reported on `DetectionResult.board_frame`
(`BoardFrame::origin_resolved()` is the convenience predicate):

| `board_frame` | Meaning | `grid_coord` | `board_xy_mm` |
|---|---|---|---|
| `absolute` | origin resolved (or a coded target) | absolute board cell | present (mm) |
| `relative_canonical` | plain target, origin unresolved | canonical relative frame | **absent** |
| `None` | no grid assignment took place | — | — |

- **Resolved.** Relative labels are remapped to absolute board cells, the frame
  homography is replaced by the anchored board→image homography, and every
  marker's `board_xy_mm` is set from its board cell.
- **Unresolved.** Labels stay in the canonical relative frame (non-negative,
  `+u` roughly along image `+x`), the homography stays in that relative frame,
  and `board_xy_mm` is cleared to `None`. This is a deliberate precision-first
  contract: **a wrong millimeter position is worse than none**, so an ambiguous
  or unverifiable origin never emits absolute coordinates.

A target with **no** fiducials always stays `relative_canonical` for plain
markers. Coded targets are always `absolute` — decoded IDs anchor them directly,
without needing dots.

**Source:** `crates/ringgrid/src/target/fiducials.rs`,
`crates/ringgrid/src/pipeline/anchor.rs`,
`crates/ringgrid/src/pipeline/result.rs`
