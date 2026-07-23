# ADR-021: Origin Fiducials Are Derived From the Lattice, Not Stored

- **Status:** accepted
- **Date:** 2026-07-23
- **Author role:** pipeline-architect
- **Supersedes:** none (refines ADR-017's fiducial model)

## Context

ADR-017 introduced `OriginFiducials { dot_radius_mm, dots_mm }`, where `dots_mm`
lists dot centers in absolute board millimeters. Absolute coordinates make the
fiducials **independent state that can disagree with the geometry they belong
to**:

- Change a target's rows, columns, or pitch and the stored triad no longer sits
  in the lattice gaps. Nothing rejects this. Reproduced: the `rect_24x24`
  preset's triad (`[161,161] [147,161] [161,175]` mm) copied onto a 12×12 /
  14 mm board — whose cells span only 0–154 mm — constructs successfully, with
  every dot off the marker field, where the anchoring homography must
  extrapolate to reach them.
- Authoring is hostile. Placing dots by hand means computing gap centers in
  board millimeters and satisfying clearance plus rotational-asymmetry
  validation. The published docs' answer was to copy the preset's magic numbers.
- The numbers carry no information. Both triads in the codebase were already
  "three of the four lattice gaps around one cell" — auto placement anchored at
  0-based cell `(11,11)`, the frozen preset at `(11,12)`, differing only in
  which gap is omitted. `tools/gen_synth_rect.py` had independently open-coded
  the same L.

The rect lattice compounded it: hex used centered axial coordinates (cell
`(0, 0)` central) while rect was 0-based from a corner, so there was no shared
cell to anchor a placement rule to.

## Decision

**1. Center rect lattice coordinates.** Each rect axis runs
`-(n-1)/2 ..= n-1-(n-1)/2`, so cell `(0, 0)` always exists and is central —
matching hex. Generation *order* is unchanged (row-major from the physical
top-left), so board millimeters, the `[0,0]` mm anchor, and sequential coded-ID
assignment are untouched. Only integer labels move.

**2. Derive dot positions; store only the size.** `OriginFiducials` becomes
`{ dot_radius_mm }`. Positions come from `origin_dot_positions_mm(lattice)` — an
asymmetric L of three lattice gaps anchored at cell `(0, 0)`:

- **rect:** the gap at `(+pitch/2, +pitch/2)` from cell `(0, 0)`, plus that
  gap's neighbours one pitch in `-x` and one pitch in `+y`;
- **hex:** the three adjacent triangle holes around cell `(0, 0)` at 30°, 90°,
  150°.

`TargetLayout` caches the derived positions alongside its cell cache, so the
detector's per-candidate projection cost is unchanged.

**3. Schema v6.** `fiducials` carries `dot_radius_mm` only. v5 files load, but
their stored `dots_mm` are **compared against** the derived positions; a
mismatch is `LegacyDotsMismatch`, not a silent override.

## Consequences

**The printed board is preserved.** The rect rule reproduces the frozen
`rect_24x24` triad exactly — verified against the literal historical
coordinates in `rect_24x24_derived_dots_match_the_printed_board`. Physical
boards already printed from that preset keep anchoring. The hex rule is a no-op:
the centroid-nearest cell it previously searched for *is* cell `(0, 0)` on every
geometry tested, so hex output is bit-identical while the placement becomes
deterministic rather than data-dependent.

**Whole classes of invalid input become unrepresentable.** Missing dots,
non-finite coordinates, and rotationally symmetric patterns can no longer be
expressed, so `EmptyFiducialDots`, `NonFiniteDot`, and (in practice)
`FiducialsRotationallySymmetric` are gone from the reachable surface. What
remains validated is what the caller still chooses: a dot radius that would
overlap a marker, and a board too small to hold the triad inside its marker
field (`OriginDotsOutsideBoard`; rect needs ≥3 columns, ≥4 rows).

**Breaking, and deliberately loud.** Rect `grid_coord` values change, so stored
rect ground truth must be regenerated. Rect targets generated with
`fiducials = "auto"` by 0.10.x fail to load rather than detecting against dots
that are not where the board has them — the same precision-first contract the
origin resolver already applies (`ADR-017`: a wrong millimeter position is worse
than none). See [migration notes](../migrations/0.10-to-0.11.md).

**One rule, one place.** `origin_dot_positions_mm` is the single source of
truth; the constructors, `with_auto_fiducials`, the `rect_24x24` preset, the
recipe's `fiducials = "auto"`, rendering, and the origin resolver all route
through it. The Python mirror in `tools/gen_synth_rect.py` is held to it by
`tools/tests/test_gen_synth_rect.py`, which compares against the library across
six geometries.

## Alternatives considered

- **Store an anchor cell + radius** (`{ anchor: [11, 12], dot_radius_mm }`).
  Dimension-independent, but still asks the author for a coordinate, and the
  anchor that reproduces the printed board (`(11,12)`) is not symmetric —
  mapping it to `(0,0)` needs per-axis offsets (`-(cols-1)/2` on columns,
  `-rows/2` on rows) that are an artifact of 24×24, not a rule.
- **Keep `dots_mm` as an escape hatch for arbitrary patterns.** No caller
  needed one: both existing triads fit the derived rule. Keeping it would have
  preserved exactly the staleness this ADR exists to remove.
- **Leave rect 0-based and anchor on the board centroid gap.** Works, but leaves
  the hex/rect asymmetry in `grid_coord` and describes placement in terms of a
  centroid search rather than an addressable cell.
