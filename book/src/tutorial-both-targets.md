# Tutorial: Both Targets, End to End

This tutorial walks the full arc — **generate → detect → interpret** — for the
two headline targets ringgrid supports:

- a **coded hex** target (16-sector rings, decoded to globally unique IDs), and
- a **plain rect** target (uncoded rings on a rectangular lattice, labeled by
  lattice coordinate and — with origin dots — anchored to absolute board mm).

Each generation step writes the same four artifacts: the canonical
`target_spec.json`, a printable `.svg` and `.png`, and a `.dxf` (2D CAD in
millimeters) for laser/CNC fabrication. See [Target
Generation](target-generation.md) for every recipe field and the [Compositional
Target Model](targets/target-model.md) for the geometry.

These two targets are `hex_coded` and `rect_plain_dots` — two of the six
built-in example recipes. All six combinations of `{hex, rect}` × `{coded, plain}`
× `{origin dots, no dots}` are available (see the
[target matrix](target-generation.md#the-target-matrix)); origin-dot anchoring
now works for hex plain targets too.

Install the CLI once:

```bash
cargo install ringgrid --features cli
```

---

## Part A — Coded hex target

### 1. Generate

Grab the built-in recipe and render it:

```bash
ringgrid example --name hex_coded --out hex_coded.toml
ringgrid gen hex_coded.toml --out ./out/hex
```

The equivalent Rust and Python (both write `target_spec.json` + `.svg`/`.png`/`.dxf`):

```rust
use ringgrid::TargetLayout;
let hex = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)?;
hex.write_json_file("./out/hex/target_spec.json".as_ref())?;
hex.write_target_svg("./out/hex/target_print.svg".as_ref(), &Default::default())?;
hex.write_target_png("./out/hex/target_print.png".as_ref(), &Default::default())?;
hex.write_target_dxf("./out/hex/target_print.dxf".as_ref())?;
```

```python
import ringgrid
hex = ringgrid.TargetLayout.coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)
hex.write_svg("./out/hex/target_print.svg")
hex.write_png("./out/hex/target_print.png", dpi=600.0)
hex.write_dxf("./out/hex/target_print.dxf")
```

### 2. Detect

```bash
ringgrid detect \
  --target ./out/hex/target_spec.json \
  --image path/to/hex_photo.png \
  --out ./out/hex/detect.json
```

### 3. Interpret

Coded markers decode to a unique `id`; IDs anchor an absolute board frame:

```json
{
  "board_frame": "absolute",
  "detected_markers": [
    { "id": 42, "grid_coord": [3, -1], "center": [812.4, 655.1], "board_xy_mm": [24.0, 41.6] }
  ],
  "homography": [ /* 3x3 board→image */ ]
}
```

- `id` — codebook index (0–892), globally unique on the board.
- `center` — sub-pixel marker center in **image** pixels.
- `board_xy_mm` — the marker's known board position (absolute).
- `board_frame` is always `absolute` for coded targets.

---

## Part B — Plain rect target

### 1. Generate

The built-in `rect_plain_dots` recipe is a 24×24 plain rect target with an
auto-placed origin-dot triad (the same target as the `rect_24x24` preset):

```bash
ringgrid example --name rect_plain_dots --out rect_plain_dots.toml
ringgrid gen rect_plain_dots.toml --out ./out/rect
```

`gen` reports what it built — note the board is **343 mm square**, larger than
A3, so plan the print accordingly (see [Print & Verify](targets/print-and-verify.md)):

```text
target: rect_plain_dots — rect 24x24, pitch 14.0 mm, plain, 3 origin dots
markers: 576
print size: 343.2 x 343.2 mm (exceeds A3 297x420 mm — use a plotter or tile the print)
png: 4054 x 4054 px @ 300 dpi
```

The equivalent Rust and Python. `OriginDots::Auto` / `dots=True` place the
origin-dot triad in the lattice gaps for you — dot positions are computed from
the lattice and validated (clear of every marker, breaking every rotational
symmetry), so you never write dot coordinates by hand:

```rust
use ringgrid::{OriginDots, TargetLayout};
let rect = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)?
    .with_name("rect_plain_dots")?;
rect.write_json_file("./out/rect/target_spec.json".as_ref())?;
rect.write_target_svg("./out/rect/target_print.svg".as_ref(), &Default::default())?;
rect.write_target_png("./out/rect/target_print.png".as_ref(), &Default::default())?;
rect.write_target_dxf("./out/rect/target_print.dxf".as_ref())?;
```

```python
import ringgrid
rect = ringgrid.TargetLayout.plain_rect(14.0, 24, 24, 5.6, 2.8, dots=True)
rect.write_svg("./out/rect/target_print.svg")
rect.write_png("./out/rect/target_print.png")
rect.write_dxf("./out/rect/target_print.dxf")
```

For a target *without* dots, pass `OriginDots::None` / `dots=False` — then the
board is labeled only up to lattice symmetry and you must detect all of it
(`--strict`, or gate on `board_complete`).

### 2. Detect

Detection is the same command — the target JSON tells the detector which path to
run:

```bash
ringgrid detect \
  --target ./out/rect/target_spec.json \
  --image path/to/rect_photo.png \
  --out ./out/rect/detect.json
```

### 3. Interpret

Plain markers carry **no `id`** — they are keyed by `grid_coord`. Whether
positions are absolute depends on origin resolution (see [Plain / Rect Target
Detection](detection-pipeline/plain-target.md)):

```json
{
  "board_frame": "absolute",
  "detected_markers": [
    { "id": null, "grid_coord": [-11, -11], "center": [120.3, 133.7], "board_xy_mm": [0.0, 0.0] },
    { "id": null, "grid_coord": [-10, -11], "center": [176.9, 133.5], "board_xy_mm": [14.0, 0.0] }
  ]
}
```

- `id` is `null`; use `grid_coord` as the marker key. Rect coordinates are
  centered, so this 24×24 board runs `-11..=12` per axis: `[-11, -11]` is the
  corner cell at `[0, 0]` mm, and `[0, 0]` is the central cell the origin dots
  surround.
- **`board_frame: absolute`** — the origin dots were resolved, so `grid_coord`
  is in board cells and `board_xy_mm` is populated.
- **`board_frame: relative_canonical`** — no origin was resolved (target has no
  dots, or they were not visible). `grid_coord` is in a canonical *relative*
  frame and every `board_xy_mm` is `null`. For a plain target without dots
  (`rect_plain_nodots`), pass `ringgrid detect --strict` to require the complete
  board. A wrong millimeter position is worse than none.

---

## Recap

| Step | Coded hex | Plain rect |
|---|---|---|
| Recipe | `hex_coded` | `rect_plain_dots` |
| Generate (CLI) | `ringgrid gen hex_coded.toml …` | `ringgrid gen rect_plain_dots.toml …` |
| Generate (Rust) | `TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)` | `TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)` |
| Marker key | `id` (0–892) | `grid_coord` |
| Frame | always `absolute` | `absolute` (dots resolved) or `relative_canonical` |
| Artifacts | `.json` `.svg` `.png` `.dxf` | `.json` `.svg` `.png` `.dxf` |

Between generating and detecting comes the step that most often goes wrong
silently: [Print & Verify](targets/print-and-verify.md).

Where to go next: the full result schema in [Detection Output
Format](output-format.md), the plain-path algorithm in [Plain / Rect Target
Detection](detection-pipeline/plain-target.md), and frame semantics in
[Coordinate Frames](coordinate-frames.md).
