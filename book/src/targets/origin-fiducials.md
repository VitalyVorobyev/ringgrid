# Origin Fiducials

Plain (uncoded) markers carry no identity, so a plain target's labeling is only
known up to the lattice's rotational symmetry and a lattice translation. Origin
fiducials — small dark filled dots printed in the gaps between markers — pin the
board origin and orientation so outputs can be reported in absolute board
millimeters. This page covers where the dots go, what is validated at
construction time, and how the detector resolves the origin from them.

## Where the dots go

`OriginFiducials { dot_radius_mm }` stores the dot **size and nothing else**.
Positions are *derived* from the lattice — an asymmetric **L of three lattice
gaps anchored at cell `(0, 0)`**, which both lattices place near the board
center:

| Lattice | Triad |
|---|---|
| **rect** | the gap at `(+pitch/2, +pitch/2)` from cell `(0, 0)`, plus that gap's neighbours one pitch in `-x` and one pitch in `+y` |
| **hex** | the three adjacent triangle holes around cell `(0, 0)` at 30°, 90° and 150° (clearance `pitch`) |

Rect coordinates are centered (a 24-wide board runs `-11..=12`), so cell
`(0, 0)` exists and is central on every board — the same convention the hex
axial lattice already used. Read the resulting positions back with
`TargetLayout::fiducial_dots_mm()`.

**Why derived rather than stored.** Absolute coordinates go stale: change the
board's rows, columns, or pitch and a stored triad silently describes points
that no longer sit in the lattice gaps — or lie off the marker field entirely,
where the anchoring homography must extrapolate to reach them. Deriving them
makes that unrepresentable. (Schema v5 stored `dots_mm`; see
[Target JSON](target-json-v6.md#migrating-from-v5).)

Two properties make this placement the right one:

1. **Anchorable** — the dots sit inside the densely-labeled interior, so the
   board→image homography *interpolates* to them rather than extrapolating to a
   corner it may not have labeled (hex grid labeling in particular has limited
   boundary recall).
2. **Orientation-resolving** — an L of three gaps is not invariant under any
   lattice rotation, so exactly one orientation is consistent with it.

## Validation (construction time)

Because the triad is derived, whole classes of bad input are unrepresentable: no
missing dots, no non-finite coordinates, no rotationally symmetric pattern. What
`TargetLayout::new` still checks is the part you choose, plus the board:

- **Clearance** — no dot may fall within `outer_draw_radius_mm + dot_radius_mm`
  of any marker center, otherwise both rendering and dot detection are
  ill-defined (`DotOverlapsMarker`). This is what bounds the dot size.
- **Board size** — a board too small to hold the triad inside its marker field
  is rejected (`OriginDotsOutsideBoard`); rect boards need at least 3 columns
  and 4 rows.
- **Symmetry** — the derived pattern is re-checked against every rotational
  symmetry of the actual finite cell set (90° steps for square, 60° for hex,
  each verified numerically, so a non-square rect patch admitting only a 180°
  half-turn is handled exactly). Structurally guaranteed by the L, but asserted
  so a future placement change cannot silently ship an unanchorable target.

`fiducials = "auto"` derives the dot size too, targeting a legible `~0.1 × pitch`
shrunk to fit tight gaps — the path that requires no choices at all.

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
