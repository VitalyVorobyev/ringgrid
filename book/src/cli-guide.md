# CLI Guide

The published `ringgrid` command-line tool generates calibration targets from a
recipe and detects them in images. Install it with:

```bash
cargo install ringgrid --features cli
```

This produces a `ringgrid` binary on your `PATH`. (Library users add the crate
with `cargo add ringgrid`; Python users `pip install ringgrid`.)

The binary has four subcommands:

```text
ringgrid gen     <recipe.toml>  --out DIR         # target artifacts
ringgrid detect  --image P --target T --out J     # one image
ringgrid batch   --images DIR --target T --out-dir D
ringgrid example --list | --name NAME [--out FILE]
```

## Recipes

`gen` (and `detect`/`batch`, which also accept one) read a **recipe** — a small
TOML or JSON file describing the target. The CLI lowers the recipe to a
`TargetLayout` and renders it.

```toml
name = "lab_hex_coded"
coding = "coded"          # "coded" | "plain"
fiducials = "none"        # "none" | "auto" | { dot_radius_mm = .. }

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

The top-level scalar keys (`name`, `coding`, `fiducials`) **must** precede the
`[lattice]` / `[marker]` / `[render]` tables — a TOML requirement. Recipes cover
the six valid combinations of `{hex, rect}` × `{coded, plain}` ×
`{origin dots, no dots}`; the one excluded combination is `coding = "coded"`
with fiducials, which is rejected (coded markers already carry identity). See
[Target Generation](target-generation.md) for every field and all six examples.

### `ringgrid example` — list or emit built-in recipes

Built-in recipes ship inside the binary, so you never need a repository
checkout.

```bash
# List the built-in recipe names
ringgrid example --list

# Print a recipe to stdout
ringgrid example --name hex_coded

# Write a recipe to a file to edit and feed to `gen`
ringgrid example --name rect_plain_dots --out rect_plain_dots.toml
```

The available names are `hex_coded`, `rect_coded`, `hex_plain_dots`,
`hex_plain_nodots`, `rect_plain_dots`, and `rect_plain_nodots`.

## `ringgrid gen` — generate target artifacts

Reads a recipe and writes `target_spec.json` (schema `ringgrid.target.v5`) plus
the printable `<basename>.svg`, `.png`, and `.dxf` to the output directory.

```bash
ringgrid gen hex_coded.toml --out ./out/target
```

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

CLI flags override the corresponding recipe fields.

## `ringgrid detect` — detect markers in an image

Loads an image, runs the detection pipeline against a target, and writes the
result JSON.

```bash
ringgrid detect \
    --image photo.png \
    --target target_spec.json \
    --out result.json
```

| Flag | Default | Description |
|---|---|---|
| `--image <path>` | required | Input image file. |
| `--target <path>` | required | Target spec (`target_spec.json`) **or** a recipe (`.toml`/`.json`). |
| `--out <path>` | stdout | Output JSON path. When omitted, the result JSON is printed to stdout. |
| `--marker-diameter <px>` | auto | Approximate marker outer diameter (px) for focused single-pass detection. |
| `--config <path>` | none | Detection-config overlay (`.json`/`.toml`) — see [Configuration](configuration/detect-config.md). |
| `--strict` | false | Require the complete board: fail unless every cell is detected. |

`--strict` maps onto the same `require_complete_board` gate that plain targets
without origin dots rely on (their identity comes from detecting the whole
board). Fine-grained detection behavior — scale prior, RANSAC thresholds,
completion gates, center refinement — is set through the `--config` overlay
rather than dedicated flags; see [Configuration](configuration/detect-config.md).

## `ringgrid batch` — detect across a directory

Runs detection on every image in a directory, writing one `<stem>.json` per
image plus an aggregate `summary.json`.

```bash
ringgrid batch \
    --images ./captures \
    --target target_spec.json \
    --out-dir ./out/batch
```

| Flag | Default | Description |
|---|---|---|
| `--images <dir>` | required | Directory of input images. |
| `--target <path>` | required | Target spec or recipe. |
| `--out-dir <dir>` | required | Directory for per-image `<stem>.json` results (created if absent). |
| `--summary <path>` | `<out-dir>/summary.json` | Aggregate summary path. |
| `--marker-diameter <px>` | auto | Approximate marker outer diameter (px). |
| `--config <path>` | none | Detection-config overlay (`.json`/`.toml`). |
| `--strict` | false | Require the complete board on every image. |

The `summary.json` records, per image, the marker count, decoded count, and
`board_complete` flag.

## Logging

ringgrid uses the `tracing` crate for structured logging. Control verbosity with
the `RUST_LOG` environment variable:

```bash
# Default level (info) -- shows summary statistics
ringgrid detect --image photo.png --target target_spec.json --out result.json

# Debug level -- shows per-stage diagnostics
RUST_LOG=debug ringgrid detect --image photo.png --target target_spec.json --out result.json

# Trace level -- shows detailed per-marker information
RUST_LOG=trace ringgrid detect --image photo.png --target target_spec.json --out result.json
```

At the default `info` level, the detector logs image dimensions, the loaded
target, detected and decoded marker counts, homography statistics, and the
output path.

## Output Format

`ringgrid detect` writes the serialized `DetectionResult` fields at the top
level:

- `detected_markers`
- `center_frame`
- `homography_frame`
- `image_size`
- optional `homography` and `self_undistort`
- a nested `diagnostics` object carrying per-marker algorithm internals
  (`diagnostics.markers`) and homography RANSAC statistics (`diagnostics.ransac`)

The full file shape, nested marker fields, and frame semantics are documented in
[Output Format](./output-format.md).

## Adaptive scale

Adaptive multi-scale detection is exposed through the Rust and Python libraries
(not the published CLI, which uses the regular config-driven flow):

- `Detector::detect_adaptive`
- `Detector::detect_adaptive_with_hint`
- `Detector::detect_multiscale`

The Python bindings expose the same concepts on `ringgrid.Detector`. See
[Adaptive Scale Detection](detection-modes/adaptive-scale.md).

> **Developing ringgrid.** The repository also ships an in-repo development
> binary, `ringgrid-dev`, with maintainer-only subcommands (`codebook-info`,
> `board-info`, `decode-test`, and the legacy `gen-target` family) and a
> repository checkout is required. Run it with
> `cargo run -p ringgrid-cli --bin ringgrid-dev -- <subcommand>`. See
> [Development](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/development.md).

## Source Files

- Published binary: `crates/ringgrid/src/bin/ringgrid.rs`
- CLI support (recipes, artifacts, detect): `crates/ringgrid/src/cli/`
