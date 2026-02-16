# Target Generation

ringgrid calibration boards are hex-lattice grids of coded ring markers. The board
geometry is described by a compact parametric JSON file -- marker positions are not
stored explicitly but generated at runtime from a small set of parameters.

## Target JSON Schema

Board specifications use the `ringgrid.target.v3` schema. A minimal example:

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

### Field Reference

| Field | Type | Description |
|---|---|---|
| `schema` | string | Must be `"ringgrid.target.v3"`. Older schemas are rejected. |
| `name` | string | Human-readable board name (must not be empty). |
| `pitch_mm` | float | Center-to-center distance between adjacent markers in millimeters. Must be finite and positive. |
| `rows` | int | Number of hex lattice rows. Must be at least 1. |
| `long_row_cols` | int | Number of columns in the longer (even) rows. Must be at least 2 when `rows > 1`. |
| `marker_outer_radius_mm` | float | Outer ring radius in millimeters (visible dark ring outer edge). Must be positive and less than half the minimum center spacing. |
| `marker_inner_radius_mm` | float | Inner ring radius in millimeters (inner edge of the dark ring). Must be positive and strictly less than `marker_outer_radius_mm`. |

The schema enforces `deny_unknown_fields` -- legacy fields such as `origin_mm`,
`board_size_mm`, or explicit `markers` arrays will cause a parse error.

### How Marker Positions Are Generated

Marker positions are computed at runtime from `(rows, long_row_cols, pitch_mm)` using
hex axial coordinates. The hex lattice alternates between long rows (with
`long_row_cols` markers) and short rows (with `long_row_cols - 1` markers). Each
marker receives:

- **`id`** -- sequential integer starting at 0, assigned in row-major order.
- **`xy_mm`** -- physical position in millimeters, computed from axial coordinates
  `(q, r)` via the standard hex-to-Cartesian mapping.
- **`q`, `r`** -- axial hex coordinates for the marker's lattice position.

The coordinate origin is normalized so that marker 0 (top-left corner of the lattice)
sits at `(0.0, 0.0)`.

For the default 15-row, 14-long-column board, this produces 203 markers spanning
approximately 90 x 168 mm.

## Loading a Board in Rust

Load a board specification from a JSON file:

```rust
use std::path::Path;
use ringgrid::BoardLayout;

let board = BoardLayout::from_json_file(Path::new("board_spec.json"))?;
println!("Board '{}' has {} markers", board.name, board.n_markers());
```

Use the built-in default board (equivalent to the standard 200 mm hex target):

```rust
use ringgrid::BoardLayout;

let board = BoardLayout::default();
// 203 markers, 8.0 mm pitch, 15 rows, 14 long-row columns
```

### Useful `BoardLayout` Methods

| Method | Returns | Description |
|---|---|---|
| `n_markers()` | `usize` | Total number of markers on the board. |
| `xy_mm(id)` | `Option<[f32; 2]>` | Board-space position of a marker by ID. |
| `marker_ids()` | `impl Iterator<Item = usize>` | Iterator over all marker IDs. |
| `max_marker_id()` | `usize` | Largest marker ID present. |
| `marker_bounds_mm()` | `Option<([f32; 2], [f32; 2])>` | Axis-aligned bounding box of marker centers. |
| `marker_span_mm()` | `Option<[f32; 2]>` | Width and height of the marker extent. |
| `marker_outer_radius_mm()` | `f32` | Outer ring radius in mm. |
| `marker_inner_radius_mm()` | `f32` | Inner ring radius in mm. |

## Generating a Board Specification

The `tools/gen_board_spec.py` script produces board JSON from command-line parameters:

```bash
python3 tools/gen_board_spec.py \
    --pitch_mm 8.0 \
    --rows 15 \
    --long_row_cols 14 \
    --board_mm 200.0 \
    --json_out board_spec.json
```

### Generator Arguments

| Argument | Default | Description |
|---|---|---|
| `--pitch_mm` | 8.0 | Center-to-center marker spacing. |
| `--rows` | 15 | Number of hex lattice rows. |
| `--long_row_cols` | 14 | Columns in the long rows. |
| `--board_mm` | 200.0 | Used only to derive the board name when `--name` is not set. |
| `--name` | auto | Board name (defaults to `ringgrid_{board_mm}mm_hex`). |
| `--marker_outer_radius_mm` | `pitch_mm * 0.6` | Outer ring radius. |
| `--marker_inner_radius_mm` | `pitch_mm * 0.4` | Inner ring radius. |
| `--json_out` | `tools/board/board_spec.json` | Output file path. |

When marker radii are not specified explicitly, they default to 60% and 40% of the
pitch respectively.

### Custom Board Examples

A denser board with smaller pitch:

```bash
python3 tools/gen_board_spec.py \
    --pitch_mm 5.0 \
    --rows 20 \
    --long_row_cols 18 \
    --name "dense_5mm_hex" \
    --json_out dense_board.json
```

A small test board:

```bash
python3 tools/gen_board_spec.py \
    --pitch_mm 12.0 \
    --rows 5 \
    --long_row_cols 6 \
    --name "small_test" \
    --json_out small_board.json
```

## Validation Rules

The Rust loader enforces these constraints at parse time:

- `schema` must be exactly `"ringgrid.target.v3"`.
- `name` must not be empty or whitespace-only.
- `pitch_mm` must be finite and positive.
- `rows` must be at least 1.
- `long_row_cols` must be at least 1 (at least 2 when `rows > 1`).
- `marker_outer_radius_mm` must be finite and positive.
- `marker_inner_radius_mm` must be finite, positive, and strictly less than `marker_outer_radius_mm`.
- The marker outer diameter must be smaller than the minimum center spacing
  (`pitch_mm * sqrt(3)`), ensuring markers do not overlap.

## Physical Target Rendering

The JSON board specification describes the logical geometry of the calibration target.
To produce a printable physical target (SVG, PNG, DXF, or similar), the JSON is
consumed by separate rendering tools outside the ringgrid crate scope. The ringgrid
library itself only reads the specification for detection purposes.

## Source Files

- Board layout loading and generation: `crates/ringgrid/src/board_layout.rs`
- Board specification generator: `tools/gen_board_spec.py`
