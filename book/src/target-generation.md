# Target Generation

This chapter documents the complete workflow for generating:

- the canonical target spec JSON (`target_spec.json`, schema `ringgrid.target.v5`)
- a printable vector target (`.svg`)
- a printable raster target (`.png`)
- a 2D CAD target for laser/CNC fabrication (`.dxf`, millimeters)

for physical calibration targets.

## Overview

Two paths cover the full [compositional target model](targets/target-model.md)
— hex or rect lattices, coded or plain markers, and optional origin fiducials:

1. **Published CLI: `ringgrid gen <recipe>`** — reads a small TOML/JSON *recipe*
   and emits `target_spec.json`, `.svg`, `.png`, and `.dxf` in one run. Best for
   a command-line workflow with no repository checkout.
2. **Rust API: `TargetLayout` + writers** — `write_json_file`,
   `write_target_svg`, `write_target_png`, and `write_target_dxf` emit the same
   artifacts from application code. Best when target generation is embedded in a
   Rust program. (The Python `TargetLayout` exposes the matching
   `write_svg` / `write_png` / `write_dxf`.)

Both paths use the same Rust rendering engine, so identical geometry and print
options produce identical JSON, SVG, PNG, and DXF.

## Install

```bash
# Command-line tool (provides the `ringgrid` binary)
cargo install ringgrid --features cli

# Rust library
cargo add ringgrid

# Python bindings
pip install ringgrid
```

## Recipes

A **recipe** is the authoring format the CLI lowers to a `TargetLayout`. It is a
small TOML file (JSON is also accepted):

```toml
name = "lab_hex_coded"
coding = "coded"          # "coded" | "plain"
fiducials = "none"        # "none" | "auto" | { dot_radius_mm = .., dots_mm = [..] }

[lattice]
kind = "hex"              # "hex" (rows, long_row_cols, pitch_mm)
rows = 15                 #   or "rect" (rows, cols, pitch_mm)
long_row_cols = 14
pitch_mm = 8.0

[marker]
outer_radius_mm = 4.8
inner_radius_mm = 3.2
ring_width_mm = 1.152     # required only for coding = "coded"

[render]
dpi = 300
margin_mm = 5.0
formats = ["json", "svg", "png", "dxf"]
```

The top-level scalar keys (`name`, `coding`, `fiducials`) **must** appear before
the `[lattice]` / `[marker]` / `[render]` tables — a TOML requirement.

### The target matrix

Recipes cover the six valid combinations of `{hex, rect}` × `{coded, plain}` ×
`{origin dots, no dots}`:

| Lattice | Coding | Fiducials | Example recipe | Identity comes from |
|---|---|---|---|---|
| hex | coded | none | `hex_coded` | decoded IDs (absolute frame) |
| rect | coded | none | `rect_coded` | decoded IDs (absolute frame) |
| hex | plain | `auto` dots | `hex_plain_dots` | origin dots (absolute frame) |
| rect | plain | `auto` dots | `rect_plain_dots` | origin dots (absolute frame) |
| hex | plain | none | `hex_plain_nodots` | detecting the **complete** board (relative frame) |
| rect | plain | none | `rect_plain_nodots` | detecting the **complete** board (relative frame) |

The one excluded combination is `coding = "coded"` with `fiducials = "auto"` or
an explicit dot table — it is rejected, because coded markers already carry
identity and cannot use origin dots. `fiducials = "auto"` auto-places a valid,
rotation-asymmetric origin-dot triad near the board center.

Plain targets **without** dots (`hex_plain_nodots`, `rect_plain_nodots`) are
labeled only up to the lattice symmetry, so they report success via
`result.board_complete` (or `ringgrid detect --strict` / the
`require_complete_board` config) — the whole board must be detected, and outputs
stay in a `RelativeCanonical` frame. Plain-with-dots and coded targets resolve an
`Absolute` frame.

### Built-in example recipes

Every combination above ships inside the `ringgrid` binary, so no repository
checkout is needed:

```bash
# List the built-in recipe names
ringgrid example --list

# Print one to stdout
ringgrid example --name hex_coded

# Write one to a file to edit and feed to `gen`
ringgrid example --name rect_plain_dots --out rect_plain_dots.toml
```

## Published CLI: `ringgrid gen`

`gen` reads a recipe and writes `target_spec.json` plus `<basename>.svg`,
`<basename>.png`, and `<basename>.dxf` to `--out`.

```bash
ringgrid gen hex_coded.toml --out ./out/target_print_200mm
```

Outputs:

- `./out/target_print_200mm/target_spec.json`
- `./out/target_print_200mm/target_print.svg`
- `./out/target_print_200mm/target_print.png`
- `./out/target_print_200mm/target_print.dxf`

| Flag | Default | Description |
|---|---|---|
| `<recipe>` | required | Recipe file (`.toml` or `.json`) — positional argument. |
| `--out <dir>` | `out` | Output directory (created if absent). |
| `--basename <name>` | `target_print` | Base filename for the SVG/PNG/DXF outputs. |
| `--name <n>` | recipe value | Override the target name. |
| `--pitch-mm <x>` | recipe value | Override the lattice pitch (mm). |
| `--dpi <x>` | recipe value | Override the PNG resolution (dpi). |
| `--margin-mm <x>` | recipe value | Override the print margin (mm). |
| `--formats <list>` | recipe value | Override the emitted formats (comma-separated: `json,svg,png,dxf`). |

CLI flags override the corresponding recipe fields, so a single recipe can seed
several print runs at different pitches or DPIs.

## Rust API: `TargetLayout`

Construct a `TargetLayout`, then call the writers. `write_json_file` emits v5
JSON; `write_target_svg` / `write_target_png` take `SvgTargetOptions` /
`PngTargetOptions`; `write_target_dxf` writes millimeter CAD geometry.

Coded hex from direct geometry:

```rust,no_run
use ringgrid::{TargetLayout, PngTargetOptions, SvgTargetOptions};
use std::path::Path;

// `coded_hex` derives a deterministic geometry-based name.
let target = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).unwrap();

target
    .write_json_file(Path::new("./out/target/target_spec.json"))
    .unwrap();
target
    .write_target_svg(
        Path::new("./out/target/target_print.svg"),
        &SvgTargetOptions { margin_mm: 5.0, include_scale_bar: true },
    )
    .unwrap();
target
    .write_target_png(
        Path::new("./out/target/target_print.png"),
        &PngTargetOptions { dpi: 600.0, margin_mm: 5.0, include_scale_bar: true },
    )
    .unwrap();
target
    .write_target_dxf(Path::new("./out/target/target_print.dxf"))
    .unwrap();
```

Plain rect target with **auto-placed** origin dots — the same triad the
`fiducials = "auto"` recipe field produces:

```rust,no_run
use ringgrid::{
    TargetLayout, LatticeGeometry, RectGeometry, RingGeometry, MarkerCoding,
    SvgTargetOptions,
};
use std::path::Path;

let target = TargetLayout::with_auto_fiducials(
    "my_rect_12x12",
    LatticeGeometry::Rect(RectGeometry { rows: 12, cols: 12, pitch_mm: 14.0 }),
    RingGeometry { outer_radius_mm: 5.6, inner_radius_mm: 2.8 },
    MarkerCoding::Plain,
)
.unwrap();

target
    .write_target_svg(
        Path::new("./out/target_rect/target_print.svg"),
        &SvgTargetOptions::default(),
    )
    .unwrap();
```

To place dots at explicit coordinates instead, use `TargetLayout::new` with an
`Some(OriginFiducials { dot_radius_mm, dots_mm })`.

The presets are one call each:

```rust
use ringgrid::TargetLayout;

let hex = TargetLayout::default_hex();  // classic coded hex
let rect = TargetLayout::rect_24x24();  // plain rect target with origin dots
```

See the [Compositional Target Model](targets/target-model.md) for the full
construction and validation rules.

## Fiducial dots

Plain (uncoded) targets carry no per-marker identity, so they use dark filled
dots to resolve the board origin and orientation. In a recipe these are the
`fiducials` field (`"auto"`, or an explicit `{ dot_radius_mm, dots_mm }` table);
in the Rust API they are `OriginFiducials { dot_radius_mm, dots_mm }` (dot
centers in board millimeters).

Two rules are validated at construction time:

- dots must not overlap any marker's drawn extent, and
- the dot pattern must break **every** rotational symmetry of the lattice, so a
  detector can recover the board orientation uniquely.

`fiducials = "auto"` satisfies both automatically. Coded targets do not need
fiducials — decoded IDs already anchor every marker to a physical cell, and
`coding = "coded"` with fiducials is rejected. See
[Origin Fiducials](targets/origin-fiducials.md) for the anchoring and validation
details.

## Detection from the generated target

Every generation path above emits a `target_spec.json` that the detector reads
directly:

```bash
ringgrid detect \
  --target ./out/target_print_200mm/target_spec.json \
  --image path/to/photo.png \
  --out ./out/target_print_200mm/detect.json
```

`ringgrid detect --target` also accepts a recipe directly, so you can detect
against a recipe without generating the spec first.

## Target JSON schema

The canonical schema is [`ringgrid.target.v5`](targets/target-json-v5.md), a
compositional document with `lattice`, `marker`, `coding`, and optional
`fiducials` sections. The pre-0.8 flat `ringgrid.target.v4` schema is still
accepted on input and migrated on load; writers always emit v5. The full field
reference, both annotated examples (coded hex and plain rect), and the v4
auto-migration rules live in [Target JSON (schema v5)](targets/target-json-v5.md)
— this page does not duplicate them.

## Practical Print Guidance

- Prefer SVG for final print jobs; it is resolution-independent.
- Keep printer scaling at 100% (no fit-to-page).
- Use a print margin if your printer clips near page edges.
- Archive the exact `target_spec.json` that was printed and use that same JSON
  during detection.

> **Developing ringgrid.** Coded targets assign codebook IDs sequentially. A
> maintainer-only optimizer (`tools/optimize_id_assignment.py`) reassigns IDs so
> hex-adjacent markers have maximally dissimilar codewords, hardening the ID
> correction stage. It, and the synthetic image + print pipeline
> (`tools/gen_synth.py`), require a repository checkout and the in-repo Python
> tooling. See [ID Assignment Optimization](id-optimization.md) and
> [Development](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/development.md).

## Related Chapters

- [The Compositional Target Model](targets/target-model.md)
- [Target JSON (schema v5)](targets/target-json-v5.md)
- [Origin Fiducials](targets/origin-fiducials.md)
- [Fast Start](fast-start.md)
- [CLI Guide](cli-guide.md)
- [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
