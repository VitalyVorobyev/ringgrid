# Target Generation

This chapter documents the complete workflow for generating:

- board configuration JSON (`board_spec.json`)
- printable vector target (`.svg`)
- printable raster target (`.png`)

for physical calibration boards.

## Overview

There are two supported generation paths:

1. `tools/gen_synth.py` (recommended fast path)
   - can emit `board_spec.json`, `.svg`, and `.png` in one run
   - can also generate synthetic images/ground truth if needed
2. `tools/gen_board_spec.py` (JSON-only)
   - emits only `board_spec.json`
   - use when you want an explicit parametric board spec without synth assets

## Prerequisites

From repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install numpy
```

## Fastest Workflow (One Command)

Generate JSON + SVG + PNG without synthetic images:

```bash
./.venv/bin/python tools/gen_synth.py \
  --out_dir tools/out/target_print_200mm \
  --n_images 0 \
  --board_mm 200 \
  --pitch_mm 8 \
  --print \
  --print_dpi 600 \
  --print_margin_mm 5 \
  --print_basename target_print
```

`--print` enables both `--print_svg` and `--print_png`.

Outputs:

- `tools/out/target_print_200mm/board_spec.json`
- `tools/out/target_print_200mm/target_print.svg`
- `tools/out/target_print_200mm/target_print.png`

## Step-by-Step Custom Workflow

### Step 1: Generate `board_spec.json`

```bash
python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 \
  --rows 15 \
  --long_row_cols 14 \
  --board_mm 200.0 \
  --json_out tools/board/board_spec.json
```

### Step 2: Generate printable SVG and PNG for matching geometry

```bash
./.venv/bin/python tools/gen_synth.py \
  --out_dir tools/out/target_print_custom \
  --n_images 0 \
  --board_mm 200 \
  --pitch_mm 8 \
  --print_svg \
  --print_png \
  --print_dpi 600 \
  --print_margin_mm 5 \
  --print_basename target_print
```

### Step 3: Use the generated JSON in detection

```bash
cargo run -- detect \
  --target tools/out/target_print_custom/board_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_print_custom/detect.json
```

## Configuration Reference

### `tools/gen_synth.py` geometry options

| Argument | Default | Description |
|---|---|---|
| `--out_dir` | `tools/out/synth_002` | Output directory for all artifacts. |
| `--n_images` | `3` | Number of synthetic rendered images (`0` is valid for print-only runs). |
| `--board_mm` | `200.0` | Board side size used for lattice generation and print canvas. |
| `--pitch_mm` | `8.0` | Marker center-to-center spacing in mm. |
| `--n_markers` | `None` | Optional marker count cap for generated lattice. |
| `--codebook` | `tools/codebook.json` | Codebook JSON used for marker coding. |

### `tools/gen_synth.py` print-output options

| Argument | Default | Description |
|---|---|---|
| `--print` | `false` | Emit both printable SVG and PNG. |
| `--print_svg` | `false` | Emit printable SVG file. |
| `--print_png` | `false` | Emit printable PNG file. |
| `--print_dxf` | `false` | Emit DXF target output. |
| `--print_dpi` | `600.0` | Raster DPI for PNG export (also embedded in PNG metadata). |
| `--print_margin_mm` | `0.0` | Extra white margin around the board in print outputs. |
| `--print_basename` | `target_print` | Base filename for print outputs (without extension). |

Notes:

- SVG is resolution-independent and preferred for professional printing.
- PNG size is derived from `board_mm`, `print_margin_mm`, and `print_dpi`.
- `gen_synth.py` always writes a matching `board_spec.json` to `--out_dir`.

### `tools/gen_board_spec.py` options

| Argument | Default | Description |
|---|---|---|
| `--pitch_mm` | `8.0` | Marker spacing in mm. |
| `--rows` | `15` | Number of hex lattice rows. |
| `--long_row_cols` | `14` | Number of markers in long rows. |
| `--board_mm` | `200.0` | Used for default board name when `--name` is omitted. |
| `--name` | auto | Board name. Default: `ringgrid_{board_mm}mm_hex`. |
| `--marker_outer_radius_mm` | `pitch_mm * 0.6` | Outer ring radius. |
| `--marker_inner_radius_mm` | `pitch_mm * 0.4` | Inner ring radius. |
| `--json_out` | `tools/board/board_spec.json` | Output JSON path. |

## Board JSON Schema (`ringgrid.target.v3`)

Minimal example:

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

Field reference:

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Must be `ringgrid.target.v3`. |
| `name` | string | Human-readable board name. |
| `pitch_mm` | float | Marker center spacing in mm. |
| `rows` | int | Number of lattice rows. |
| `long_row_cols` | int | Marker count for long rows. |
| `marker_outer_radius_mm` | float | Outer ring radius in mm. |
| `marker_inner_radius_mm` | float | Inner ring radius in mm (`0 < inner < outer`). |

## Validation Rules

The Rust loader validates:

- `schema == "ringgrid.target.v3"`
- finite positive `pitch_mm`
- `rows >= 1`
- `long_row_cols >= 1` (`>= 2` when `rows > 1`)
- finite positive radii with `marker_inner_radius_mm < marker_outer_radius_mm`
- marker outer diameter smaller than minimum center spacing

## Quick Validation in Rust

```rust
use ringgrid::BoardLayout;
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("board_spec.json"))?;
println!("{} markers on '{}'", board.n_markers(), board.name);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Practical Print Guidance

- Prefer SVG for final print jobs.
- Keep printer scaling at 100% (no fit-to-page).
- Use `print_margin_mm` if your printer clips near page edges.
- Archive the exact `board_spec.json` that was printed and use that same JSON during detection.

## Related Chapters

- [Fast Start](fast-start.md)
- [CLI Guide](cli-guide.md)
- [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
