# Hex Lattice Layout

Ringgrid markers are arranged on a hexagonal lattice, which provides denser
packing than a rectangular grid and ensures that each marker has six equidistant
neighbors. The lattice geometry is parametrized by three values -- rows, columns,
and pitch -- and marker positions are computed at runtime from these parameters
rather than stored as explicit coordinate lists.

## Lattice parameters

The hex lattice is fully defined by three parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rows` | 15 | Number of marker rows |
| `long_row_cols` | 14 | Number of markers in a long row |
| `pitch_mm` | 8.0 mm | Center-to-center distance between adjacent markers |

Rows alternate between **long rows** (with `long_row_cols` markers) and
**short rows** (with `long_row_cols - 1` markers). This staggering is what
produces the hexagonal packing pattern.

For the default board (15 rows, 14 long-row columns), the total marker count
is:

```
8 long rows * 14 + 7 short rows * 13 = 112 + 91 = 203 markers
```

## Axial coordinate system

Each marker position on the lattice is identified by a pair of **axial
coordinates** `(q, r)`, following the standard hex grid convention:

- **r** is the row index, centered around zero. For a board with 15 rows,
  `r` ranges from -7 to +7.
- **q** is the column index within each row, also centered around zero. The
  range of `q` depends on the row length.

Axial coordinates are integers and provide a natural addressing scheme for
hex grids. They are stored as optional fields on each `BoardMarker` for
diagnostic and visualization purposes.

## Cartesian conversion

The conversion from axial coordinates `(q, r)` to Cartesian positions in
millimeters uses the standard hex-to-Cartesian transform:

```
x = pitch * (sqrt(3) * q + sqrt(3)/2 * r)
y = pitch * (3/2 * r)
```

In Rust, this is implemented as:

```rust
fn hex_axial_to_xy_mm(q: i32, r: i32, pitch_mm: f32) -> [f32; 2] {
    let qf = q as f64;
    let rf = r as f64;
    let pitch = pitch_mm as f64;
    let x = pitch * (f64::sqrt(3.0) * qf + 0.5 * f64::sqrt(3.0) * rf);
    let y = pitch * (1.5 * rf);
    [x as f32, y as f32]
}
```

The computation is performed in `f64` to avoid accumulation of rounding errors
across large boards, then truncated to `f32` for the final coordinates.

After generation, all marker positions are translated so that the first marker
(top-left corner) sits at the origin `(0, 0)`.

<!-- TODO: Diagram showing hex lattice with coordinate labels -->

## Nearest-neighbor distance

On this hex lattice, the nearest-neighbor distance between adjacent marker
centers is:

```
d_nn = pitch * sqrt(3) ≈ 8.0 * 1.732 ≈ 13.86 mm
```

This distance determines the minimum clearance between markers and constrains
the maximum allowed marker diameter (see
[Ring Structure](ring-structure.md#design-constraints)).

## The `BoardLayout` type

The `BoardLayout` struct is the runtime representation of a calibration target.
It holds the lattice parameters, marker radii, and a generated list of all
marker positions:

```rust
pub struct BoardLayout {
    pub name: String,
    pub pitch_mm: f32,
    pub rows: usize,
    pub long_row_cols: usize,
    pub marker_outer_radius_mm: f32,
    pub marker_inner_radius_mm: f32,
    pub markers: Vec<BoardMarker>,
    // internal: fast ID -> index lookup
}
```

Key methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `default()` | `BoardLayout` | Default 15x14 board with 203 markers |
| `from_json_file(path)` | `Result<BoardLayout>` | Load from a JSON spec file |
| `xy_mm(id)` | `Option<[f32; 2]>` | Look up Cartesian position by marker ID |
| `n_markers()` | `usize` | Total number of markers |
| `marker_ids()` | `Iterator<usize>` | Iterate over all marker IDs |
| `marker_bounds_mm()` | `Option<([f32;2], [f32;2])>` | Axis-aligned bounding box |
| `marker_span_mm()` | `Option<[f32; 2]>` | Width and height of the marker field |

`BoardLayout` maintains an internal `HashMap<usize, usize>` for O(1) lookup
of marker positions by ID, built automatically during construction.

## The `BoardMarker` type

Each marker on the board is represented by:

```rust
pub struct BoardMarker {
    pub id: usize,
    pub xy_mm: [f32; 2],
    pub q: Option<i16>,
    pub r: Option<i16>,
}
```

The `id` field is the marker's codebook index (0 through 892 for the default
board). Markers are assigned IDs sequentially in row-major order during
generation. The `q` and `r` fields store the axial hex coordinates.

## JSON schema

Board layouts are specified in JSON files using the `ringgrid.target.v3`
schema. The schema is deliberately **parametric**: it contains only the
lattice parameters, and marker positions are generated at runtime. This
avoids the maintenance burden and potential inconsistencies of storing
per-marker coordinate lists.

Example JSON for the default board:

```json
{
    "schema": "ringgrid.target.v3",
    "name": "ringgrid_200mm_hex",
    "pitch_mm": 8.0,
    "rows": 15,
    "long_row_cols": 14,
    "marker_outer_radius_mm": 4.8,
    "marker_inner_radius_mm": 3.2
}
```

Schema fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema` | string | Must be `"ringgrid.target.v3"` |
| `name` | string | Human-readable target name |
| `pitch_mm` | float | Center-to-center marker spacing |
| `rows` | int | Number of rows in the lattice |
| `long_row_cols` | int | Markers per long row |
| `marker_outer_radius_mm` | float | Outer ring radius |
| `marker_inner_radius_mm` | float | Inner ring radius |

The loader enforces strict validation via `#[serde(deny_unknown_fields)]`:
any extra fields (such as legacy `origin_mm`, `board_size_mm`, or explicit
`markers` lists) cause a parse error. This prevents silent use of outdated
board specifications.

## Validation rules

The `BoardLayout` loader validates several geometric constraints:

1. **Positive dimensions**: `pitch_mm`, `marker_outer_radius_mm`, and
   `marker_inner_radius_mm` must all be finite and positive.
2. **Inner < outer**: The inner radius must be strictly less than the outer
   radius.
3. **Non-overlapping markers**: The marker outer diameter must be smaller than
   the nearest-neighbor distance (`pitch * sqrt(3)`).
4. **Sufficient columns**: When `rows > 1`, `long_row_cols` must be at least 2
   (to allow short rows with `long_row_cols - 1 >= 1` markers).

## Board generation

Board specification files are generated by the Python utility
`tools/gen_board_spec.py`:

```bash
python3 tools/gen_board_spec.py \
    --pitch_mm 8.0 \
    --rows 15 \
    --long_row_cols 14 \
    --board_mm 200.0 \
    --json_out tools/board/board_spec.json
```

The generated JSON file is then loaded at runtime by the detector via
`BoardLayout::from_json_file()`, or the `BoardLayout::default()` constructor
can be used without any file for the standard 15x14 board.
