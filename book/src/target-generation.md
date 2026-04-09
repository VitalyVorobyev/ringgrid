# Target Generation

This chapter documents the complete workflow for generating:

- board configuration JSON (`board_spec.json`)
- printable vector target (`.svg`)
- printable raster target (`.png`)

for physical calibration boards.

## Overview

There are three equivalent canonical target-generation paths:

1. Rust CLI: `ringgrid gen-target`
   - emits `board_spec.json`, `.svg`, and `.png` in one run
   - best when you want a pure-Rust command-line workflow
2. Python script: `tools/gen_target.py`
   - emits the same `board_spec.json`, `.svg`, and `.png` set
   - best when you are already using the repo's Python tooling
3. Rust API: `BoardLayout` + `write_json_file` / `write_target_svg` / `write_target_png`
   - emits the same canonical artifacts from application code
   - best when target generation is embedded in a Rust program

All three paths use the same Rust target-generation engine. For the same geometry
and print options, they generate the same canonical board JSON, SVG, and PNG.

Additional specialized repo helpers remain available:

- `tools/gen_synth.py` for synth images + ground truth + optional print files
- `tools/gen_board_spec.py` for JSON-only board-spec generation

## Prerequisites

From repository root:

For the Rust CLI path:

```bash
cargo build -p ringgrid-cli
```

For the dedicated Python script path:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip maturin
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

For `gen_synth.py` synthetic generation:

```bash
./.venv/bin/python -m pip install numpy
```

For the Rust API path, add `ringgrid` to your Rust project dependencies.

## Equivalent One-Command Workflows

Rust CLI:

```bash
cargo run -p ringgrid-cli -- gen-target \
  --out_dir tools/out/target_print_200mm \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

Python script:

```bash
./.venv/bin/python tools/gen_target.py \
  --out_dir tools/out/target_print_200mm \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

Outputs:

- `tools/out/target_print_200mm/board_spec.json`
- `tools/out/target_print_200mm/target_print.svg`
- `tools/out/target_print_200mm/target_print.png`

Rust API:

```rust,no_run
use ringgrid::{BoardLayout, PngTargetOptions, SvgTargetOptions};
use std::path::Path;

let board = BoardLayout::with_name("ringgrid_200mm_hex", 8.0, 15, 14, 4.8, 3.2, 1.152).unwrap();

board
    .write_json_file(Path::new("tools/out/target_print_200mm/board_spec.json"))
    .unwrap();
board
    .write_target_svg(
        Path::new("tools/out/target_print_200mm/target_print.svg"),
        &SvgTargetOptions {
            margin_mm: 5.0,
            include_scale_bar: true,
        },
    )
    .unwrap();
board
    .write_target_png(
        Path::new("tools/out/target_print_200mm/target_print.png"),
        &PngTargetOptions {
            dpi: 600.0,
            margin_mm: 5.0,
            include_scale_bar: true,
        },
    )
    .unwrap();
```

## Combined Synth + Print Workflow

Use `tools/gen_synth.py` when you want print outputs and synthetic images from
one command:

```bash
./.venv/bin/python tools/gen_synth.py \
  --out_dir tools/out/target_print_with_synth \
  --n_images 3 \
  --board_mm 200 \
  --pitch_mm 8 \
  --print \
  --print_dpi 600 \
  --print_margin_mm 5 \
  --print_basename target_print
```

## Detection From The Generated Board

All three canonical generation paths above emit the same `board_spec.json` schema.
Use that generated JSON directly in detection:

```bash
cargo run -- detect \
  --target tools/out/target_print_custom/board_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_print_custom/detect.json
```

## Configuration Reference

### Shared `ringgrid gen-target` / `tools/gen_target.py` options

The Rust CLI intentionally mirrors the dedicated Python script's geometry and
print-output arguments. The Rust CLI accepts the underscore names shown below;
hyphenated aliases such as `--pitch-mm` are also accepted.

| Argument | Default | Description |
|---|---|---|
| `--pitch_mm` | required | Marker spacing in mm. |
| `--rows` | required | Number of hex lattice rows. |
| `--long_row_cols` | required | Number of markers in long rows. |
| `--marker_outer_radius_mm` | required | Outer ring radius in mm. |
| `--marker_inner_radius_mm` | required | Inner ring radius in mm. |
| `--marker_ring_width_mm` | required | Full printed ring width in mm for both inner and outer dark rings. |
| `--name` | auto | Optional board name. Omitted uses deterministic geometry-derived naming. |
| `--out_dir` | `tools/out/target` | Output directory for `board_spec.json`, SVG, and PNG. |
| `--basename` | `target_print` | Base filename for SVG/PNG outputs. |
| `--dpi` | `300.0` | Raster DPI for PNG export (also embedded in PNG metadata). |
| `--margin_mm` | `0.0` | Extra white margin around the board in print outputs. |
| `--no-scale-bar` | `false` | Omit the default scale bar from SVG/PNG outputs. |

### Equivalent Rust API mapping

| Rust API surface | Equivalent shared option |
|---|---|
| `BoardLayout::new(...)` / `BoardLayout::with_name(...)` | geometry (`pitch_mm`, `rows`, `long_row_cols`, radii, ring width, optional name) |
| `write_json_file(path)` | `board_spec.json` output |
| `SvgTargetOptions { margin_mm, include_scale_bar }` | `--margin_mm`, `--no-scale-bar` |
| `PngTargetOptions { dpi, margin_mm, include_scale_bar }` | `--dpi`, `--margin_mm`, `--no-scale-bar` |

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

- `gen_target.py` always writes `board_spec.json`, `<basename>.svg`, and `<basename>.png` together.
- SVG is resolution-independent and preferred for professional printing.
- PNG size is derived from the board geometry, requested margin, and DPI.
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
| `--marker_ring_width_mm` | `marker_outer_radius_mm * 0.24` | Full dark-ring width. |
| `--json_out` | `tools/board/board_spec.json` | Output JSON path. |

## Board JSON Schema (`ringgrid.target.v4`)

Minimal example:

```json
{
  "schema": "ringgrid.target.v4",
  "name": "ringgrid_200mm_hex",
  "pitch_mm": 8.0,
  "rows": 15,
  "long_row_cols": 14,
  "marker_outer_radius_mm": 4.8,
  "marker_inner_radius_mm": 3.2,
  "marker_ring_width_mm": 1.152
}
```

Field reference:

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Must be `ringgrid.target.v4`. |
| `name` | string | Human-readable board name. |
| `pitch_mm` | float | Marker center spacing in mm. |
| `rows` | int | Number of lattice rows. |
| `long_row_cols` | int | Marker count for long rows. |
| `marker_outer_radius_mm` | float | Outer ring radius in mm. |
| `marker_inner_radius_mm` | float | Inner ring radius in mm (`0 < inner < outer`). |
| `marker_ring_width_mm` | float | Full printed width of the dark inner and outer rings in mm. |

## Validation Rules

The Rust loader validates:

- `schema == "ringgrid.target.v4"`
- finite positive `pitch_mm`
- `rows >= 1`
- `long_row_cols >= 1` (`>= 2` when `rows > 1`)
- finite positive radii and ring width with `marker_inner_radius_mm < marker_outer_radius_mm`
- a positive code-band gap between the inner and outer ring strokes
- printed marker diameter (including ring stroke width) smaller than minimum center spacing

## Quick Validation in Rust

```rust
use ringgrid::BoardLayout;
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("board_spec.json"))?;
println!("{} markers on '{}'", board.n_markers(), board.name);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Optimized ID Assignment

Generated boards assign codebook IDs to marker positions sequentially (0, 1,
2, ...). For production use, it is recommended to post-process the board spec
with the ID assignment optimizer, which reassigns IDs to maximize the cyclic
Hamming distance between hex-adjacent markers. This makes the ID correction
stage more robust against decode errors.

```bash
.venv/bin/python tools/optimize_id_assignment.py --board board_spec.json --out board_spec_optimized.json
```

Two pre-optimized reference boards are included at `tools/board/board_spec_optimized.json`
and `tools/board/board_spec_extended_opt.json`. See [ID Assignment Optimization](id-optimization.md)
for details, tradeoffs, and usage instructions.

## Practical Print Guidance

- Prefer SVG for final print jobs.
- Keep printer scaling at 100% (no fit-to-page).
- Use `print_margin_mm` if your printer clips near page edges.
- Archive the exact `board_spec.json` that was printed and use that same JSON during detection.

## Related Chapters

- [Fast Start](fast-start.md)
- [CLI Guide](cli-guide.md)
- [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
