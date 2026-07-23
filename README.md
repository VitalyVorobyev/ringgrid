[![CI](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/ci.yml)
[![Publish Rust crates (crates.io)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-crates.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-crates.yml)
[![Publish Docs](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-docs.yml)
[![Release Python Package to PyPI](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/release-pypi.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/release-pypi.yml)
[![Security Audit](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/audit.yml)

# ringgrid

`ringgrid` is a pure-Rust detector for dense ring calibration targets on hex or
rectangular lattices. It fits marker ellipses with subpixel precision, optionally
decodes 16-sector ID codes, estimates the board-to-image homography, and returns
a small structured result — no OpenCV bindings, all image processing in Rust.

## At a Glance

- Subpixel ring-marker detection via direct ellipse fitting and projective center correction
- Two ways to identify markers: **coded** 16-sector rings (decode to stable IDs) or **plain** rings anchored by origin dots or a complete-board layout
- Hex or rectangular lattices, from one compositional `TargetLayout`
- Optional camera-distortion handling and self-undistort estimation
- One library, many surfaces: a Rust crate, a Python package, C/C++ bindings, a WebAssembly package, and a CLI

## Visual Overview

`ringgrid` detects two target families from one `TargetLayout` model — a
**coded hex** board (16-sector rings decode to stable IDs) and a **plain rect**
board (uncoded rings, grid-labeled and anchored by origin dots). Each pair below
shows the printable target and a detection overlay (green = fitted ellipses).

| Coded hex — decoded IDs | Plain rect — origin-anchored |
|---|---|
| ![Coded hex target print](docs/assets/target_print.png) | ![Plain rect target print](docs/assets/rect_target_print.png) |
| ![Coded hex detection overlay](docs/assets/det_overlay_0002.png) | ![Plain rect detection overlay](docs/assets/rect_det_overlay.png) |

## Install

```bash
cargo add ringgrid                       # Rust library
cargo install ringgrid --features cli    # `ringgrid` CLI (target gen + detection)
pip install ringgrid                     # Python package
npm install @vitavision/ringgrid         # JavaScript / WebAssembly
vcpkg install ringgrid                   # C / C++ (or CMake from source)
```

The library has a clean dependency graph by default; the CLI is behind the
`cli` feature so it never bloats library builds. There is also an in-browser
[WASM demo](https://vitalyvorobyev.github.io/ringgrid/demo/) — no install needed.

## Quick Start (Rust)

Point a `Detector` at a target and hand it a grayscale image:

```rust,no_run
use ringgrid::{Detector, TargetLayout};
use std::path::Path;

// Load a target spec (produced by `ringgrid gen`, see below), or build one
// in code with TargetLayout::coded_hex(...) / ::rect_24x24() / ::new(...).
let target = TargetLayout::from_json_file(Path::new("target_spec.json")).unwrap();
let image = image::open("photo.png").unwrap().to_luma8();

let detector = Detector::new(target);
let result = detector.detect(&image).unwrap();

for m in &result.detected_markers {
    match m.id {
        Some(id) => println!("marker {id} at ({:.1}, {:.1})", m.center[0], m.center[1]),
        None => println!("cell {:?} at ({:.1}, {:.1})", m.grid_coord, m.center[0], m.center[1]),
    }
}
```

When the marker diameter is roughly known, `Detector::with_marker_diameter_hint(target, px)`
skips scale probing. For scenes with a wide marker-size range, use
`detector.detect_adaptive(&image)`.

### Detection config

`DetectConfig` holds the durable choices; construct it from a target and tune
fields as needed:

```rust,no_run
# use ringgrid::{DetectConfig, Detector, MarkerScalePrior, TargetLayout};
let mut config = DetectConfig::from_target(TargetLayout::rect_24x24());
config.marker_scale = MarkerScalePrior::from_nominal_diameter_px(32.0); // scale prior
config.require_complete_board = true;   // plain targets: fail unless every cell is found
config.self_undistort.enable = true;    // estimate & correct lens distortion
let detector = Detector::with_config(config);
```

The most-used knobs are `marker_scale`, `circle_refinement`, `self_undistort`,
and `require_complete_board`; per-stage tuning lives under `config.advanced`.
See the [Configuration guide](https://vitalyvorobyev.github.io/ringgrid/book/configuration/detect-config.html).

### Detection result

`detect` returns a slim [`DetectionResult`]:

| Field | Meaning |
|---|---|
| `detected_markers` | each with `center` (image px), `grid_coord`, optional decoded `id`, optional `board_xy_mm`, and inner/outer `ellipse_*` |
| `homography` | board-to-image 3×3, when enough markers anchor it |
| `board_frame` | `Absolute` (origin resolved via codes or dots) or `RelativeCanonical` (labeled up to lattice symmetry) |
| `board_complete` | `Some(true/false)` when a board was labeled — the success signal for plain, no-dots targets |
| `image_size`, `center_frame`, `homography_frame`, `self_undistort` | frame metadata + optional distortion model |

Per-marker fit/decode metrics and RANSAC stats are an opt-in channel — call
`detector.detect_with_diagnostics(&image)`. The result serializes to JSON via
`serde`; the exact shape is in the [Detection Output Format](https://vitalyvorobyev.github.io/ringgrid/book/output-format.html).

## Generate a target

Author a small recipe once (TOML or JSON) and render printable artifacts +
the canonical `target_spec.json` with the CLI:

```bash
ringgrid example --name rect_plain_dots > my_target.toml  # start from a built-in
ringgrid gen my_target.toml --out ./out                   # writes SVG, PNG, DXF, JSON
```

```toml
# my_target.toml — plain rings on a square lattice, anchored by origin dots
name = "lab_rect"
coding = "plain"       # coded | plain
fiducials = "auto"     # none | auto | explicit dot table

[lattice]
kind = "rect"          # rect | hex
rows = 24
cols = 24
pitch_mm = 14.0

[marker]
outer_radius_mm = 5.6
inner_radius_mm = 2.8

[render]
dpi = 600
formats = ["json", "svg", "png", "dxf"]
```

`gen` reports the board's physical size and whether it fits a paper size, so you
find out before the print run that the defaults above make a 343 mm square board.
Any recipe field can be overridden on the command line (`--pitch-mm`, `--dpi`, …),
and `ringgrid example --list` shows the built-in recipes — one per valid target
combination, ready to copy and adapt.

Prefer code? The same target is one call, then the writers:

```rust,no_run
use ringgrid::{OriginDots, TargetLayout};
use std::path::Path;

let target = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)?
    .with_name("lab_rect")?;
target.write_json_file(Path::new("./out/target_spec.json"))?;
target.write_target_svg(Path::new("./out/target_print.svg"), &Default::default())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Then [print and verify it](https://vitalyvorobyev.github.io/ringgrid/book/targets/print-and-verify.html)
— printing at anything other than 100 % scale is the quiet way to get wrong
millimeters out of a correct detector.

## The target matrix

Markers must be identifiable. That comes from **codes**, from **origin dots**, or
from detecting the **complete board**. All six combinations below are supported;
only *coded + dots* is excluded, because codes already resolve identity and
orientation, making dots redundant.

| Lattice | Coding | Origin dots | How the board is anchored | One call | Recipe |
|---|---|---|---|---|---|
| hex  | coded | — | decoded IDs → absolute frame | `TargetLayout::coded_hex(pitch, rows, long_row_cols, outer_r, inner_r, ring_w)` | `hex_coded` |
| rect | coded | — | decoded IDs → absolute frame | `TargetLayout::coded_rect(pitch, rows, cols, outer_r, inner_r, ring_w)` | `rect_coded` |
| hex  | plain | ✓ | origin dots → absolute frame | `TargetLayout::plain_hex(pitch, rows, long_row_cols, outer_r, inner_r, OriginDots::Auto)` | `hex_plain_dots` |
| hex  | plain | — | complete board → relative frame (`board_complete`) | `TargetLayout::plain_hex(…, OriginDots::None)` | `hex_plain_nodots` |
| rect | plain | ✓ | origin dots → absolute frame | `TargetLayout::plain_rect(pitch, rows, cols, outer_r, inner_r, OriginDots::Auto)` | `rect_plain_dots` |
| rect | plain | — | complete board → relative frame (`board_complete`) | `TargetLayout::plain_rect(…, OriginDots::None)` | `rect_plain_nodots` |

Every constructor derives a deterministic geometry-based name; add your own with
`.with_name("lab_bench")`. `TargetLayout::default_hex()` and
`TargetLayout::rect_24x24()` are frozen presets of rows 1 and 5. Python mirrors
all four constructors (`TargetLayout.plain_rect(14.0, 24, 24, 5.6, 2.8, dots=True)`).

For plain, no-dots targets there is no way to fix the origin, so a run is
"successful" only when the **whole board** is detected — gate on
`result.board_complete` (or set `require_complete_board`). Origin dots are placed
automatically (`fiducials = "auto"`) so the pattern always resolves orientation.

## Interfaces

- **Rust** — the core library. See [`crates/ringgrid/README.md`](crates/ringgrid/README.md) and the [API reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/).
- **Python** — `pip install ringgrid`. See [`crates/ringgrid-py/README.md`](crates/ringgrid-py/README.md).
- **C / C++** — `vcpkg install ringgrid` (or CMake from source); a stable C ABI + header-only C++ wrapper. See [`crates/ringgrid-c/README.md`](crates/ringgrid-c/README.md).
- **CLI** — `cargo install ringgrid --features cli`; `ringgrid gen | detect | batch | example`. See the [CLI Guide](https://vitalyvorobyev.github.io/ringgrid/book/cli-guide.html).
- **WASM / npm** — `npm install @vitavision/ringgrid`; in-browser detection. Try the [live demo](https://vitalyvorobyev.github.io/ringgrid/demo/).

## Documentation

The full user guide — marker design, pipeline stages, math, configuration, and
target generation — is the [mdBook](https://vitalyvorobyev.github.io/ringgrid/book/).

Working on ringgrid itself (building from source, synthetic evaluation, generated
assets, benchmarks)? See [`docs/development.md`](docs/development.md). Pre-1.0
upgrade notes live in [`docs/migrations/`](docs/migrations/).

## Diligence Statement

This project is developed with AI coding assistants (`Codex` and `Claude Code`) as
implementation tools. Not every code path is manually line-reviewed by a human
before merge. The project author validates algorithmic behavior and numerical
results and enforces quality gates before release.
