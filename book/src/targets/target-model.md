# The Compositional Target Model

A ringgrid target is described at runtime by a `TargetLayout`. Before 0.8 the
only target was a hex lattice of 16-sector coded rings, modeled by the flat
`BoardLayout` type (removed in 0.9). `TargetLayout` generalizes that into four
**orthogonal** aspects that compose freely:

```text
TargetLayout  =  lattice  ×  ring geometry  ×  coding  ×  optional fiducials
```

- **Lattice** (`LatticeGeometry`) — how marker cells are arranged: `Hex` or
  `Rect`.
- **Ring geometry** (`RingGeometry`) — the outer/inner radii shared by every
  marker.
- **Coding** (`MarkerCoding`) — whether markers carry a decodable identity
  (`Coded16`) or are plain annuli (`Plain`).
- **Fiducials** (`OriginFiducials`, optional) — filled dots that anchor origin
  and orientation for targets whose markers do not encode identity.

Each aspect is a small value type; `TargetLayout::new` composes and validates
them. Geometry cannot be mutated in place — the derived cell cache (positions,
ID↔coordinate lookups) would silently desync — so construction always goes
through `new`, a preset, or a JSON loader.

## Lattice geometry

| Variant | Fields | Nearest-neighbor spacing |
|---|---|---|
| `Hex(HexGeometry)` | `rows`, `long_row_cols`, `pitch_mm` | `pitch_mm × √3` |
| `Rect(RectGeometry)` | `rows`, `cols`, `pitch_mm` | `pitch_mm` |

The hex lattice uses axial rows that alternate between long and short rows;
`long_row_cols` sets the long-row width. The first generated cell is normalized
to board position `[0, 0]` mm, and generation order (top row first, left to
right) is load-bearing — sequential IDs derive from it. Cell coordinates are
axial `(q, r)` for hex and `(col, row)` for rect, both carried as
`projective_grid::Coord { u, v }`.

## Ring geometry

`RingGeometry { outer_radius_mm, inner_radius_mm }` is shared by every marker.
For `Coded16` markers these are the *centerline* radii of the stroked outer and
inner rings; for `Plain` markers they bound the filled annulus directly. The
outermost *drawn* radius differs accordingly: a stroked ring overshoots its
centerline by half the stroke width, while a plain annulus does not.

## Coding

| Variant | Shape | Identity |
|---|---|---|
| `Coded16(CodedRingSpec)` | two stroked rings with a 16-sector code band between them | codebook ID (decoded per marker) |
| `Plain` | a single filled annulus | none — cells are keyed by lattice coordinate |

`CodedRingSpec` carries the `ring_width_mm` stroke and an optional
`id_assignment` (see [ID Assignment Optimization](../id-optimization.md)). Coded
targets are capped at the embedded codebook size (893 codewords); a lattice with
more cells than that is rejected for `Coded16` but valid as `Plain`.

## Fiducials

`OriginFiducials { dot_radius_mm, dots_mm }` are dark filled dots printed in the
lattice gaps. They exist to resolve the board origin and orientation for
**plain** targets, whose markers carry no identity. Coded targets do not need
them — decoded IDs already anchor every marker to a physical cell. See
[Origin Fiducials](origin-fiducials.md) for the validation and anchoring rules.

## Composition matrix — how each combination detects

Every built-in lattice × coding combination detects end-to-end, but the
identity-bearing stages differ. The coded path decodes IDs and labels markers by
codebook lookup; the plain path labels markers by their lattice position.

| Lattice | Coding | Labeling path | ID correction | Output frame |
|---|---|---|---|---|
| Hex | Coded16 | decode → global filter → completion | hex-neighbor BFS consensus | Absolute |
| Rect | Coded16 | decode → global filter → completion | — (global filter + geometric verify only) | Absolute |
| Hex | Plain | `detect_grid` labeling → completion | — | Absolute if fiducials resolve, else RelativeCanonical |
| Rect | Plain | `detect_grid` labeling → completion | — | Absolute if fiducials resolve, else RelativeCanonical |

Key points:

- **Coded targets** run the classic decode-anchored pipeline (see the
  [Detection Pipeline](../detection-pipeline/overview.md)). Every decoded ID maps
  to a physical board cell, so outputs are always in the absolute board frame.
- **[ID correction](../detection-pipeline/id-correction.md)** is a
  hex-neighbor BFS consensus — its algorithmic domain — so it runs only for hex
  coded targets. Rect coded targets rely on the global RANSAC homography filter
  plus geometric verification instead.
- **Plain targets** skip decoding entirely. Fitted ring centers are labeled with
  lattice coordinates by `projective_grid::detect_grid` (labeling only; the
  frame homography is refit in `f64` by ringgrid's RANSAC), then completion grows
  the labeled patch. See [the plain-target path](../detection-pipeline/overview.md#plain-target-path).
- Plain outputs are in a canonical **relative** frame unless the target carries
  origin fiducials that resolve the origin; see
  [Origin Fiducials](origin-fiducials.md).

## Presets

Three presets cover the common cases:

| Preset | Lattice | Coding | Cells | Notes |
|---|---|---|---|---|
| `TargetLayout::default_hex()` | 15-row hex, 8 mm pitch | Coded16 | 203 | the classic 200 mm ringgrid board |
| `TargetLayout::coded_hex(...)` | hex (caller geometry) | Coded16 | — | coded hex from direct geometry arguments |
| `TargetLayout::rect_24x24()` | 24×24 rect, 14 mm pitch | Plain | 576 | 24×24 plain target with three Ø2.8 mm origin dots |

`default_hex()` is geometry-identical to the classic pre-0.9 hex board, so
existing hex-coded workflows are unchanged.

## Construction

```rust
use ringgrid::{
    TargetLayout, LatticeGeometry, RectGeometry, RingGeometry,
    MarkerCoding, OriginFiducials,
};

// A preset
let hex = TargetLayout::default_hex();
let rect = TargetLayout::rect_24x24();

// A custom plain rect target with origin dots
let target = TargetLayout::new(
    "my_rect",
    LatticeGeometry::Rect(RectGeometry { rows: 12, cols: 12, pitch_mm: 14.0 }),
    RingGeometry { outer_radius_mm: 5.6, inner_radius_mm: 2.8 },
    MarkerCoding::Plain,
    Some(OriginFiducials {
        dot_radius_mm: 1.4,
        dots_mm: vec![[77.0, 77.0], [63.0, 77.0]],
    }),
).expect("valid target");
```

The `Detector` / `DetectConfig` constructors take `impl Into<TargetLayout>`, so
a `TargetLayout` (or anything convertible into one) can be passed directly.

## Validation

`TargetLayout::new` rejects illegal targets up front:

- non-finite or non-positive pitch, radii, or ring width;
- `inner_radius_mm >= outer_radius_mm`, or a non-positive code-band gap for
  coded markers;
- a drawn marker diameter that reaches or exceeds the minimum center spacing
  (markers would touch);
- more cells than the codebook can encode, or an out-of-range / duplicate entry
  in `id_assignment` (coded targets);
- fiducial dots that overlap a marker's drawn extent, or a dot pattern that
  fails to break every rotational symmetry of the lattice
  (see [Origin Fiducials](origin-fiducials.md)).

Legacy v4 `board_spec.json` files still load unchanged: `TargetLayout::from_json_*`
auto-migrates the v4 schema to the canonical v5 spec.

**Source:** `crates/ringgrid/src/target/` (`layout.rs`, `lattice.rs`, `ring.rs`,
`fiducials.rs`)
