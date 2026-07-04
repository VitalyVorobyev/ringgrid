# Target Generation

This chapter documents the complete workflow for generating:

- the canonical target spec JSON (`target_spec.json`, schema `ringgrid.target.v5`)
- a printable vector target (`.svg`)
- a printable raster target (`.png`)
- a 2D CAD target for laser/CNC fabrication (`.dxf`, millimeters)

for physical calibration targets.

## Overview

Two Rust paths cover the full [compositional target model](targets/target-model.md)
— hex or rect lattices, coded or plain markers, and optional origin fiducials:

1. **Rust CLI: `ringgrid gen-target`** — a subcommand family (`hex`, `rect`,
   `preset`, `from-spec`) that emits `target_spec.json`, `.svg`, `.png`, and
   `.dxf` in one run. Best for a pure-Rust command-line workflow.
2. **Rust API: `TargetLayout` + writers** — `write_json_file`,
   `write_target_svg`, `write_target_png`, and `write_target_dxf` emit the same
   artifacts from application code. Best when target generation is embedded in a
   Rust program. (The Python `TargetLayout` exposes the matching
   `write_svg` / `write_png` / `write_dxf`.)

Both paths use the same Rust rendering engine, so identical geometry and print
options produce identical JSON, SVG, PNG, and DXF.

A few older Python helpers remain, but they predate the compositional model and
generate **hex coded** targets only, writing the legacy flat
`ringgrid.target.v4` `board_spec.json` (loaders migrate v4 to v5 automatically):

- `tools/gen_board_spec.py` — JSON-only hex board spec.
- `tools/gen_target.py` — hex board spec plus printable SVG/PNG.
- `tools/gen_synth.py` — synthetic camera images with ground truth, plus
  optional print files.

For rectangular, plain, or fiducial-bearing targets, use the Rust CLI/API or the
typed Python `TargetLayout` (`write_svg` / `write_png` / `write_dxf`).

## Prerequisites

From the repository root.

Rust CLI:

```bash
cargo build -p ringgrid-cli
```

Rust API — add `ringgrid` to your project dependencies.

Python helpers:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip maturin numpy
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

## Rust CLI: `ringgrid gen-target`

Each subcommand builds a `TargetLayout` and writes `target_spec.json`,
`<basename>.svg`, `<basename>.png`, and `<basename>.dxf` to `--out_dir`.

| Subcommand | Target |
|---|---|
| `hex` | Hex lattice of 16-sector coded rings (the classic ringgrid target). |
| `rect` | Rectangular lattice of plain (uncoded) rings, optionally with origin dots. |
| `preset` | A built-in preset: `default-hex` or `rect24x24`. |
| `from-spec` | Render from an existing target spec JSON (v5, or legacy v4). |

**Shared output flags** (accepted by every subcommand):

| Flag | Default | Description |
|---|---|---|
| `--out_dir <path>` | `tools/out/target` | Output directory for `target_spec.json`, SVG, PNG, and DXF. |
| `--basename <string>` | `target_print` | Base filename for SVG/PNG/DXF outputs. |
| `--dpi <f>` | `300.0` | PNG raster DPI (also embedded in PNG metadata). |
| `--margin_mm <mm>` | `0.0` | Extra white margin around the target in print outputs. |
| `--no-scale-bar` | `false` | Omit the default scale bar from SVG/PNG outputs. |

Underscore flag names are primary; hyphenated aliases such as `--out-dir` and
`--margin-mm` are also accepted.

### `hex` — coded hex target

| Flag | Default | Description |
|---|---|---|
| `--pitch_mm <mm>` | required | Marker center spacing in mm. |
| `--rows <n>` | required | Number of hex lattice rows. |
| `--long_row_cols <n>` | required | Number of markers in long rows. |
| `--marker_outer_radius_mm <mm>` | required | Outer ring centerline radius in mm. |
| `--marker_inner_radius_mm <mm>` | required | Inner ring centerline radius in mm. |
| `--marker_ring_width_mm <mm>` | required | Ring stroke width in mm. |
| `--name <string>` | auto | Optional name. Omitted uses a deterministic geometry-derived name. |

```bash
cargo run -p ringgrid-cli -- gen-target hex \
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

- `tools/out/target_print_200mm/target_spec.json`
- `tools/out/target_print_200mm/target_print.svg`
- `tools/out/target_print_200mm/target_print.png`

### `rect` — plain rectangular target

Plain rings are filled annuli with no code band, so there is no ring-width flag.
Add origin dots with `--dot_mm` / `--dot_radius_mm` to anchor the board frame.

| Flag | Default | Description |
|---|---|---|
| `--pitch_mm <mm>` | required | Marker center spacing in mm. |
| `--rows <n>` | required | Number of rows. |
| `--cols <n>` | required | Number of columns. |
| `--marker_outer_radius_mm <mm>` | required | Outer annulus radius in mm. |
| `--marker_inner_radius_mm <mm>` | required | Inner annulus radius in mm. |
| `--dot_mm <x,y>` | none | Origin fiducial dot center in board mm. Repeat per dot; requires `--dot_radius_mm`. |
| `--dot_radius_mm <mm>` | none | Origin fiducial dot radius in mm; requires `--dot_mm`. |
| `--name <string>` | auto | Optional name. Omitted uses a deterministic geometry-derived name. |

```bash
cargo run -p ringgrid-cli -- gen-target rect \
  --out_dir tools/out/target_rect \
  --pitch_mm 14 \
  --rows 24 \
  --cols 24 \
  --marker_outer_radius_mm 5.6 \
  --marker_inner_radius_mm 2.8 \
  --dot_radius_mm 1.4 \
  --dot_mm 161,161 --dot_mm 147,161 --dot_mm 161,175
```

The dot pattern must break every rotational symmetry of the lattice; see
[Fiducial dots](#fiducial-dots) below and [Origin Fiducials](targets/origin-fiducials.md).

### `preset` — built-in targets

```bash
# Classic 15-row coded hex board (200 mm)
cargo run -p ringgrid-cli -- gen-target preset default-hex --out_dir tools/out/target

# 24×24 plain rect target with three origin dots
cargo run -p ringgrid-cli -- gen-target preset rect24x24 --out_dir tools/out/target_rect
```

### `from-spec` — render an existing spec

Re-render (and upgrade to v5) any target spec JSON, whether v5 or legacy v4:

```bash
cargo run -p ringgrid-cli -- gen-target from-spec \
  --spec tools/board/board_spec.json \
  --out_dir tools/out/target_from_spec
```

## Rust API: `TargetLayout`

Construct a `TargetLayout`, then call the writers. `write_json_file` emits v5
JSON; `write_target_svg` / `write_target_png` take `SvgTargetOptions` /
`PngTargetOptions`.

Coded hex from direct geometry:

```rust,no_run
use ringgrid::{TargetLayout, PngTargetOptions, SvgTargetOptions};
use std::path::Path;

// `coded_hex` derives a deterministic geometry-based name.
let target = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).unwrap();

target
    .write_json_file(Path::new("tools/out/target/target_spec.json"))
    .unwrap();
target
    .write_target_svg(
        Path::new("tools/out/target/target_print.svg"),
        &SvgTargetOptions { margin_mm: 5.0, include_scale_bar: true },
    )
    .unwrap();
target
    .write_target_png(
        Path::new("tools/out/target/target_print.png"),
        &PngTargetOptions { dpi: 600.0, margin_mm: 5.0, include_scale_bar: true },
    )
    .unwrap();
```

Plain rect target with origin dots, built with `TargetLayout::new`:

```rust,no_run
use ringgrid::{
    TargetLayout, LatticeGeometry, RectGeometry, RingGeometry, MarkerCoding,
    OriginFiducials, PngTargetOptions, SvgTargetOptions,
};
use std::path::Path;

let target = TargetLayout::new(
    "my_rect_12x12",
    LatticeGeometry::Rect(RectGeometry { rows: 12, cols: 12, pitch_mm: 14.0 }),
    RingGeometry { outer_radius_mm: 5.6, inner_radius_mm: 2.8 },
    MarkerCoding::Plain,
    Some(OriginFiducials {
        dot_radius_mm: 1.4,
        dots_mm: vec![[77.0, 77.0], [63.0, 77.0]],
    }),
)
.unwrap();

target
    .write_target_svg(
        Path::new("tools/out/target_rect/target_print.svg"),
        &SvgTargetOptions::default(),
    )
    .unwrap();
```

The presets are one call each:

```rust
use ringgrid::TargetLayout;

let hex = TargetLayout::default_hex();       // classic coded hex
let rect = TargetLayout::rect_24x24();  // plain rect target with origin dots
```

See the [Compositional Target Model](targets/target-model.md) for the full
construction and validation rules.

## Fiducial dots

Plain (uncoded) targets carry no per-marker identity, so they use dark filled
dots to resolve the board origin and orientation. In the API these are
`OriginFiducials { dot_radius_mm, dots_mm }` (dot centers in board millimeters);
on the CLI they are `--dot_radius_mm` and one `--dot_mm x,y` per dot.

Two rules are validated at construction time:

- dots must not overlap any marker's drawn extent, and
- the dot pattern must break **every** rotational symmetry of the lattice, so a
  detector can recover the board orientation uniquely.

Coded targets do not need fiducials — decoded IDs already anchor every marker to
a physical cell. See [Origin Fiducials](targets/origin-fiducials.md) for the
anchoring and validation details.

## Detection from the generated target

Every generation path above emits a `target_spec.json` that the detector reads
directly:

```bash
cargo run -- detect \
  --target tools/out/target_print_200mm/target_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_print_200mm/detect.json
```

## Target JSON schema

The canonical schema is [`ringgrid.target.v5`](targets/target-json-v5.md), a
compositional document with `lattice`, `marker`, `coding`, and optional
`fiducials` sections. The pre-0.8 flat `ringgrid.target.v4` schema is still
accepted on input and migrated on load; writers always emit v5. The full field
reference, both annotated examples (coded hex and plain rect), and the v4
auto-migration rules live in [Target JSON (schema v5)](targets/target-json-v5.md)
— this page does not duplicate them.

## Optimized ID Assignment

Coded targets assign codebook IDs to marker positions sequentially (0, 1,
2, …). For production use, post-process the target spec with the ID assignment
optimizer, which reassigns IDs to maximize the cyclic Hamming distance between
hex-adjacent markers, making the ID correction stage more robust against decode
errors:

```bash
.venv/bin/python tools/optimize_id_assignment.py --board board_spec.json --out board_spec_optimized.json
```

Two pre-optimized reference boards ship at `tools/board/board_spec_optimized.json`
and `tools/board/board_spec_extended_opt.json`. See
[ID Assignment Optimization](id-optimization.md) for details, tradeoffs, and
usage.

## Combined Synth + Print Workflow

Use `tools/gen_synth.py` when you want synthetic images and print outputs from
one command (hex coded targets only):

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

## Practical Print Guidance

- Prefer SVG for final print jobs; it is resolution-independent.
- Keep printer scaling at 100% (no fit-to-page).
- Use a print margin if your printer clips near page edges.
- Archive the exact `target_spec.json` that was printed and use that same JSON
  during detection.

## Related Chapters

- [The Compositional Target Model](targets/target-model.md)
- [Target JSON (schema v5)](targets/target-json-v5.md)
- [Origin Fiducials](targets/origin-fiducials.md)
- [Fast Start](fast-start.md)
- [CLI Guide](cli-guide.md)
- [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
