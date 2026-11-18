# Target Generation

This chapter documents the complete workflow for generating:

- the canonical target spec JSON (`target_spec.json`, schema `ringgrid.target.v6`)
- a printable vector target (`.svg`)
- a printable raster target (`.png`)
- a 2D CAD target for laser/CNC fabrication (`.dxf`, millimeters)

for physical calibration targets. To show a target on a display instead of
printing it, see [Screen targets](#screen-targets).

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
coding = { kind = "coded" }  # or { kind = "plain" }
fiducials = "none"        # "none" | "auto" | { dot_radius_mm = .. }

[lattice]
kind = "hex"              # "hex" (rows, long_row_cols, pitch_mm)
rows = 15                 #   or "rect" (rows, cols, pitch_mm)
long_row_cols = 14
pitch_mm = 8.0

[marker]
outer_radius_mm = 4.8
inner_radius_mm = 3.2
ring_width_mm = 1.152     # required only for coded markers

[render]
dpi = 300
formats = ["json", "svg", "png", "dxf"]

[render.page]             # omit for a square page fitted to the content
margin_mm = 5.0
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

The one excluded combination is coded markers (`coding = { kind = "coded" }`)
with any fiducials — it is
rejected, because coded markers already carry identity and cannot use origin
dots. `fiducials = "auto"` derives both the placement and a legible dot size;
`{ dot_radius_mm = .. }` keeps the derived placement with a size you choose.
Dot *positions* are never authored — see [Origin Fiducials](targets/origin-fiducials.md).

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

Construct a `TargetLayout`, then call the writers. `write_json_file` emits v6
JSON; `write_target_svg` / `write_target_png` take one `TargetRenderOptions`
(page size / orientation / margin, scale bar, PNG dpi); `write_target_dxf`
writes millimeter CAD geometry.

Coded hex from direct geometry:

```rust,no_run
use ringgrid::{PageSpec, TargetLayout, TargetRenderOptions};
use std::path::Path;

// `coded_hex` derives a deterministic geometry-based name.
let target = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).unwrap();
let options = TargetRenderOptions::default()
    .with_page(PageSpec::default().with_margin_mm(5.0))
    .with_png_dpi(600.0);

target
    .write_json_file(Path::new("./out/target/target_spec.json"))
    .unwrap();
target
    .write_target_svg(Path::new("./out/target/target_print.svg"), &options)
    .unwrap();
target
    .write_target_png(Path::new("./out/target/target_print.png"), &options)
    .unwrap();
target
    .write_target_dxf(Path::new("./out/target/target_print.dxf"))
    .unwrap();
```

Plain rect target with **auto-placed** origin dots — the same triad the
`fiducials = "auto"` recipe field produces:

```rust,no_run
use ringgrid::{OriginDots, TargetLayout, TargetRenderOptions};
use std::path::Path;

let target = TargetLayout::plain_rect(14.0, 12, 12, 5.6, 2.8, OriginDots::Auto)
    .unwrap()
    .with_name("my_rect_12x12")
    .unwrap();

target
    .write_target_svg(
        Path::new("./out/target_rect/target_print.svg"),
        &TargetRenderOptions::default(),
    )
    .unwrap();
```

### One constructor per matrix row

Each row of [the target matrix](#the-target-matrix) is a single call taking
plain scalars — millimeters and counts, no geometry types to import:

| Combination | Call |
|---|---|
| hex + coded | `TargetLayout::coded_hex(pitch_mm, rows, long_row_cols, outer_r_mm, inner_r_mm, ring_width_mm)` |
| rect + coded | `TargetLayout::coded_rect(pitch_mm, rows, cols, outer_r_mm, inner_r_mm, ring_width_mm)` |
| hex + plain + dots | `TargetLayout::plain_hex(pitch_mm, rows, long_row_cols, outer_r_mm, inner_r_mm, OriginDots::Auto)` |
| hex + plain + no dots | `TargetLayout::plain_hex(…, OriginDots::None)` |
| rect + plain + dots | `TargetLayout::plain_rect(pitch_mm, rows, cols, outer_r_mm, inner_r_mm, OriginDots::Auto)` |
| rect + plain + no dots | `TargetLayout::plain_rect(…, OriginDots::None)` |

Each derives a deterministic geometry-based name, so identical geometry always
yields an identical spec; `with_name` attaches a human-readable one. All six
return `Result<TargetLayout, TargetValidationError>` — invalid geometry (markers
larger than the pitch, or packed too tightly to leave a gap for a dot) fails at
construction, not at print time. Coded targets take no `OriginDots` argument at
all: decoded IDs already anchor the board, and coded-plus-dots is rejected.

For a lattice or coding combination the four constructors don't cover, use
`TargetLayout::with_auto_fiducials` (auto dots) or `TargetLayout::new` with
`Some(OriginFiducials { dot_radius_mm })` to pick the dot size yourself. Dot
positions are derived either way — see
[Origin Fiducials](targets/origin-fiducials.md).

The two frozen presets are one call each:

```rust
use ringgrid::TargetLayout;

let hex = TargetLayout::default_hex();  // classic coded hex
let rect = TargetLayout::rect_24x24();  // plain rect target with origin dots
```

A complete, runnable program that renders all six combinations is
[`examples/generate_targets.rs`](https://github.com/VitalyVorobyev/ringgrid/blob/main/crates/ringgrid/examples/generate_targets.rs).

See the [Compositional Target Model](targets/target-model.md) for the full
construction and validation rules.

## Fiducial dots

Plain (uncoded) targets carry no per-marker identity, so they use dark filled
dots to resolve the board origin and orientation. In a recipe these are the
`fiducials` field (`"auto"`, or `{ dot_radius_mm = .. }` for a chosen size);
in the Rust API they are `OriginFiducials { dot_radius_mm }` — the size, and
nothing else. Positions are an L of three lattice gaps around cell `(0, 0)`,
derived from the lattice, so they track the geometry and can never be copied
onto a board they do not fit. Read them back with
`TargetLayout::fiducial_dots_mm()`.

The derived pattern breaks every rotational symmetry of the lattice by
construction; validation additionally rejects a dot radius large enough to
overlap a marker, and a board too small to hold the triad inside its marker
field. Coded targets do not need
fiducials — decoded IDs already anchor every marker to a physical cell, and
coded markers with fiducials are rejected. See
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

The canonical schema is [`ringgrid.target.v6`](targets/target-json-v6.md), a
compositional document with `lattice`, `marker`, `coding`, and optional
`fiducials` sections; it is the only schema the loaders accept. The full field
reference, both annotated examples (coded hex and plain rect), and the notes on
converting older files live in [Target JSON (schema v6)](targets/target-json-v6.md)
— this page does not duplicate them.

## Printing

`gen` reports the board's physical size and whether it fits a standard paper
size — worth reading before a print run, since the default 24×24 rect target is
**343 mm square**, larger than A3:

```console
target: rect_plain_dots — rect 24x24, pitch 14.0 mm, plain, 3 origin dots
markers: 576
print size: 343.2 x 343.2 mm (exceeds A3 297x420 mm — use a plotter or tile the print)
png: 4054 x 4054 px @ 300 dpi
```

From code, `TargetLayout::page_size_mm(&options)` returns the same figure as
`[width_mm, height_mm]` before anything is rendered.

Printing at anything other than 100 % scale silently corrupts every millimeter
the detector reports, and nothing downstream can detect it. Sizing the target
for your camera, printing at true scale, checking the printed scale bar with a
ruler, mounting flat, and archiving the spec are covered in
[Print & Verify](targets/print-and-verify.md).

> **Developing ringgrid.** Coded targets assign codebook IDs sequentially. A
> maintainer-only optimizer (`tools/optimize_id_assignment.py`) reassigns IDs so
> hex-adjacent markers have maximally dissimilar codewords, hardening the ID
> correction stage. It, and the synthetic image + print pipeline
> (`tools/gen_synth.py`), require a repository checkout and the in-repo Python
> tooling. See [ID Assignment Optimization](id-optimization.md) and
> [Development](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/development.md).

## Screen targets

A target can also be shown on a display — an e-reader, a monitor, a tablet —
instead of being printed. `ringgrid gen-screen` (and
`TargetLayout::render_fitted_screen_target` in every binding) produces a PNG
made for one specific screen: one image pixel per display pixel, gray levels
at every edge, and the largest grid that fits at the marker size you ask for.

### Why not show the print PNG

The print PNG from `ringgrid gen` is the wrong input for a display. Its
renderer tests each pixel at one point and writes 0 or 255, so it is
**binary**, and the point it tests is the pixel's corner, so the drawn
pattern sits **half a pixel off** the layout (a common shift, harmless on its
own). Rendered at a display's native resolution it is exactly the `binary`
mode of the [displayed-target study](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/studies/displayed-target.md).

That study simulated an e-reader at **166 dpi (0.1530 mm per display
pixel)** with the default coded hex marker (10.752 mm, 70.3 display px
across), seen at 20–56 camera px per marker. The error that matters for
calibration is the *marker-to-marker* part of the
[photometric](detection-pipeline/photometric-center.md) center error: the
part that is repeatable per marker and is not a common board shift. At 32
camera px per marker and blur σ 0.8 camera px:

| display raster | decoded % | photometric RMS, camera px | marker-to-marker, µm on the board |
|---|---:|---:|---:|
| print | 92.8 | 0.007 | 0.5 |
| binary | 91.5 | 0.017 | 5.0 |
| 4 gray levels (area coverage) | 92.8 | 0.010 | 2.0 |
| 16 gray levels (area coverage) | 93.4 | 0.008 | 0.6 |
| 256 gray levels (area coverage) | 92.7 | 0.008 | 0.5 |

A wider-band variant at 34 camera px per marker gives the same picture: the
per-marker repeatable error is 0.008–0.018 camera px (2.5–5.7 µm) for a
binary display at 0.4–0.6 camera px per display pixel, and 0.002 px for 16
gray levels, against 0.003 px for print.

- **Gray output removes the display term.** With area coverage at 16 or more
  levels the centers match print, from about 30 display px per marker up. A
  binary display needs 140–200 display px per marker to get within about
  1–2 µm of print.
- **The display barely affects decoding.** Gray output moves decode rates by
  2 percentage points or less. A binary display matters only below about 45
  display px per marker: at 20 display px it decodes 70–78 % of markers and
  41–46 % of those decodes are wrong, while 16-level area coverage still
  matches print there.
- **Snapping markers to whole pixels is worse, not better.** Moving each
  center to the nearest pixel center costs 62.6 µm marker-to-marker against
  the layout, which cannot represent the moved positions.
- **Decode failures on real displayed targets come from camera blur against
  the code band,** not from the display. The default band is 0.448 mm, 1/24
  of the drawn diameter; once the camera blur reaches σ ≈ 1.5–2 camera px at
  a band 0.8–1.3 camera px wide, most of the band's contrast is gone.

Projective centers (0.11–0.29 camera px RMS at blur 0.8–1.5) are dominated
by their own bias and do not see the display at all; the gain shows up with
`CircleRefinementMethod::Photometric`. The study is a simulation: it does not
model e-ink pixel gaps, the panel's gray response, glare, or what the reader
software does to the image. See its *Limits* section.

### How the fit works

`gen-screen` takes a **template** target (a recipe or a `target_spec.json`)
and a screen, and builds a new target for that screen.

1. **Scale.** The template is scaled uniformly so its drawn outer diameter
   (outer ring plus its stroke) equals `marker_diameter_mm`. Lattice pitch,
   ring radii, code-band width and origin-dot radius scale together, so the
   template's proportions, lattice kind (hex or rect), coding and codebook are
   kept.
2. **Search.** The usable area is the screen minus `margin_px` on each edge,
   converted to millimeters with `pixel_pitch_mm`. Every grid of at least
   2 × 2 whose drawn content (rings and origin dots) fits that area is a
   candidate. The fit keeps the one with the most markers; ties go to the
   grid whose emptier axis is filled more, then to more rows. A coded grid
   never has more markers than its codebook (893 base, 2180 extended).
3. **IDs.** The fitted grid gets sequential codebook IDs. An explicit
   `id_assignment` in the template is kept only when the fit lands on the
   template's own rows and columns; otherwise the command fails rather than
   silently invent a different assignment.
4. **Placement.** The content's bounding box is centered on the screen. Marker
   centers keep their exact lattice positions, usually at fractional pixel
   coordinates; nothing is snapped.
5. **Rasterization.** Each display pixel gets the ink fraction of its own
   area, from 8 × 8 samples, written as an 8-bit gray level (65 distinct
   values). Code-sector boundaries are integrated the same way as ring edges.
   This is the area-coverage mode of the study, and because coverage is taken
   over the pixel's area it has no half-pixel offset. A panel with fewer gray
   levels quantizes the PNG; the study found 16 levels indistinguishable from
   256.

The result is three files:

| file | contents |
|---|---|
| `target_screen.png` | `width_px` × `height_px`, 8-bit grayscale, for 1:1 display |
| `target_spec.json` | the fitted target (`ringgrid.target.v6`) in millimeters — pass it to the detector |
| `screen.json` | the screen options, the fit (`rows`, `columns`, `marker_count`), the realized `drawn_marker_diameter_mm` / `_px`, `code_band_width_px` (coded only) and `board_origin_px` |

`screen.json` maps the board frame onto the screen. A board point
`(x_mm, y_mm)` from `target_spec.json` is displayed at

```text
x_px = x_mm / pixel_pitch_mm + board_origin_px[0]
y_px = y_mm / pixel_pitch_mm + board_origin_px[1]
```

where integer pixel coordinates are pixel **centers**.

### How to use it

1. **Get the pixel pitch.** `pixel_pitch_mm` is the physical size of one
   display pixel: `25.4 / ppi` from the panel's nominal density, or better,
   the active area's measured width in millimeters divided by its width in
   pixels. The target spec's millimeters are only as right as this number —
   a 1 % pitch error is a 1 % scale error, as with a mis-scaled print — so
   measure it when it matters.
2. **Give the native resolution in the orientation you will use.**
   `--width-px` / `--height-px` are the panel's native pixels; for a portrait
   display, pass the portrait dimensions.
3. **Choose the marker diameter.** `gen-screen` prints the realized marker
   diameter and code-band width in display pixels. Keep the marker at about
   **30 display px or more** across, which with the default proportions puts
   the code band at **1.2 display px or more**; the study found that enough for
   decoding and centers to match print with 4 or more gray levels. A display
   that shows only black and white needs at least 45 display px per marker for
   decoding and 140–200 for print-grade centers. Then check the camera side:
   the code band should stay well above the camera blur. If the band is about
   1 camera px wide and blur is 1.5–2 camera px, use a larger marker or a
   template with a wider band (see the study's recommended recipe).
4. **Display it 1:1.** Show `target_screen.png` with one image pixel on one
   display pixel: no viewer scaling, no fit-to-screen, no dithering, no
   thresholding. Any resampling changes both the pattern and its physical
   size. A macro photograph of the screen tells you whether the viewer honors
   the gray levels.
5. **Detect against `target_spec.json`.** It is a normal target spec; detection
   is unchanged. Use `CircleRefinementMethod::Photometric` (CLI
   `--refine photometric`) when you want the center accuracy the gray raster
   preserves.

#### CLI

For an InkPad Lite e-reader (825 × 1200 px, nominal 150 ppi, so `25.4 / 150`
mm per pixel) with the default coded hex marker at its print size:

```bash
ringgrid example --name hex_coded --out hex_coded.toml
ringgrid gen-screen hex_coded.toml \
  --width-px 825 --height-px 1200 --pixel-pitch-mm 0.16933333333333334 \
  --marker-diameter-mm 10.752 --out out/inkpad-lite
```

```console
wrote out/inkpad-lite/target_screen.png
wrote out/inkpad-lite/target_spec.json
wrote out/inkpad-lite/screen.json
152 markers (16 rows, up to 10 columns), diameter 10.7520 mm / 63.50 px
code band 2.65 display px wide
```

The 16 rows alternate 10 and 9 rings. `--marker-diameter-mm 7.168` divides
marker size and spacing by 1.5 and fits 363 markers (25 rows of 15 and 14) at
42.33 px, with a 1.76 px code band. See the [CLI Guide](cli-guide.md#ringgrid-gen-screen--fit-a-target-to-a-screen)
for every flag.

#### Rust

```rust,no_run
use ringgrid::{ScreenTargetOptions, TargetLayout};

let template = TargetLayout::default_hex();
let options = ScreenTargetOptions::new(825, 1200, 25.4 / 150.0, 10.752).with_margin_px(8);
let screen = template.render_fitted_screen_target(&options)?;

std::fs::write("target_screen.png", &screen.png_bytes)?;
std::fs::write("target_spec.json", &screen.json_text)?;

// Where the first marker is drawn, in screen pixels.
let target = TargetLayout::from_json_str(&screen.json_text)?;
let [x_mm, y_mm] = target.cells()[0].xy_mm;
let origin = screen.metadata.board_origin_px;
let x_px = f64::from(x_mm) / options.pixel_pitch_mm + origin[0];
let y_px = f64::from(y_mm) / options.pixel_pitch_mm + origin[1];
# Ok::<(), Box<dyn std::error::Error>>(())
```

`ScreenTargetOptions` is serde-serializable (`margin_px` defaults to 8), and
failures are typed `ScreenTargetError`s: `NoFit` when not even a 2 × 2 grid
fits, `ExplicitIds` for the ID case above.

#### Python

```python
from pathlib import Path
import ringgrid

template = ringgrid.TargetLayout.default_hex()
screen = template.render_fitted_screen_target({
    "width_px": 825,
    "height_px": 1200,
    "pixel_pitch_mm": 25.4 / 150,
    "marker_diameter_mm": 10.752,
    # "margin_px": 8,
})
Path("target_screen.png").write_bytes(screen.png_bytes)
Path("target_spec.json").write_text(screen.json_text)
print(screen.metadata["marker_count"], screen.metadata["code_band_width_px"])
```

#### JavaScript / WebAssembly

```js
import init, { default_board_json, render_fitted_screen_target_json } from "@vitavision/ringgrid";

await init();
const screen = render_fitted_screen_target_json(
  default_board_json(),
  JSON.stringify({ width_px: 825, height_px: 1200, pixel_pitch_mm: 25.4 / 150, marker_diameter_mm: 10.752 }),
);
// screen.png_bytes: Uint8Array, screen.json_text: string, screen.metadata: object
```

#### Review app

The repository's local review app (`tools/review/`) has a **Generate screen
target** dialog with an InkPad Lite preset (825 × 1200, `25.4 / 150` mm,
10.752 mm markers). It downloads the same three files. See
`tools/review/README.md`.

## Related Chapters

- [The Compositional Target Model](targets/target-model.md)
- [Target JSON (schema v6)](targets/target-json-v6.md)
- [Origin Fiducials](targets/origin-fiducials.md)
- [Print & Verify](targets/print-and-verify.md)
- [Photometric Center Refinement](detection-pipeline/photometric-center.md)
- [Fast Start](fast-start.md)
- [CLI Guide](cli-guide.md)
- [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
