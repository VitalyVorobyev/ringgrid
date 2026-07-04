# Tutorial: Both Targets, End to End

This tutorial walks the full arc — **generate → detect → interpret** — for the
two target families ringgrid supports:

- a **coded hex** target (16-sector rings, decoded to globally unique IDs), and
- a **plain rect** target (uncoded rings on a rectangular lattice, labeled by
  lattice coordinate and — with origin dots — anchored to absolute board mm).

Each generation step writes the same four artifacts: the canonical
`target_spec.json`, a printable `.svg` and `.png`, and a `.dxf` (2D CAD in
millimeters) for laser/CNC fabrication. See [Target
Generation](target-generation.md) for every flag and the [Compositional Target
Model](targets/target-model.md) for the geometry.

Build the CLI once:

```bash
cargo build -p ringgrid-cli
```

---

## Part A — Coded hex target

### 1. Generate

Rust CLI:

```bash
cargo run -p ringgrid-cli -- gen-target hex \
  --out_dir tools/out/hex \
  --pitch_mm 8 --rows 15 --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --dpi 600 --margin_mm 5
```

The equivalent Rust and Python (both write `target_spec.json` + `.svg`/`.png`/`.dxf`):

```rust
use ringgrid::TargetLayout;
let hex = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)?;
hex.write_json_file("tools/out/hex/target_spec.json".as_ref())?;
hex.write_target_svg("tools/out/hex/target_print.svg".as_ref(), &Default::default())?;
hex.write_target_png("tools/out/hex/target_print.png".as_ref(), &Default::default())?;
hex.write_target_dxf("tools/out/hex/target_print.dxf".as_ref())?;
```

```python
import ringgrid
hex = ringgrid.TargetLayout.coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)
hex.write_svg("tools/out/hex/target_print.svg")
hex.write_png("tools/out/hex/target_print.png", dpi=600.0)
hex.write_dxf("tools/out/hex/target_print.dxf")
```

### 2. Detect

```bash
cargo run -- detect \
  --target tools/out/hex/target_spec.json \
  --image path/to/hex_photo.png \
  --out tools/out/hex/detect.json
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

The bundled `rect24x24` preset is a 24×24 plain rect target with three origin
dots:

```bash
cargo run -p ringgrid-cli -- gen-target preset rect24x24 --out_dir tools/out/rect
```

Or build a custom plain rect with the `rect` subcommand (add `--dot_mm x,y`
`--dot_radius_mm r` for origin dots), or in code:

```python
import ringgrid
rect = ringgrid.TargetLayout.rect_24x24()
rect.write_svg("tools/out/rect/target_print.svg")
rect.write_png("tools/out/rect/target_print.png")
rect.write_dxf("tools/out/rect/target_print.dxf")
```

### 2. Detect

Detection is the same command — the target JSON tells the detector which path to
run:

```bash
cargo run -- detect \
  --target tools/out/rect/target_spec.json \
  --image path/to/rect_photo.png \
  --out tools/out/rect/detect.json
```

### 3. Interpret

Plain markers carry **no `id`** — they are keyed by `grid_coord`. Whether
positions are absolute depends on origin resolution (see [Plain / Rect Target
Detection](detection-pipeline/plain-target.md)):

```json
{
  "board_frame": "absolute",
  "detected_markers": [
    { "id": null, "grid_coord": [0, 0], "center": [120.3, 133.7], "board_xy_mm": [0.0, 0.0] },
    { "id": null, "grid_coord": [1, 0], "center": [176.9, 133.5], "board_xy_mm": [14.0, 0.0] }
  ]
}
```

- `id` is `null`; use `grid_coord` (`[col, row]` for rect) as the marker key.
- **`board_frame: absolute`** — the origin dots were resolved, so `grid_coord`
  is in board cells and `board_xy_mm` is populated.
- **`board_frame: relative_canonical`** — no origin was resolved (target has no
  dots, or they were not visible). `grid_coord` is in a canonical *relative*
  frame and every `board_xy_mm` is `null`. A wrong millimeter position is worse
  than none.

---

## Recap

| Step | Coded hex | Plain rect |
|---|---|---|
| Generate | `gen-target hex …` | `gen-target preset rect24x24` |
| Marker key | `id` (0–892) | `grid_coord` |
| Frame | always `absolute` | `absolute` (dots resolved) or `relative_canonical` |
| Artifacts | `.json` `.svg` `.png` `.dxf` | `.json` `.svg` `.png` `.dxf` |

Where to go next: the full result schema in [Detection Output
Format](output-format.md), the plain-path algorithm in [Plain / Rect Target
Detection](detection-pipeline/plain-target.md), and frame semantics in
[Coordinate Frames](coordinate-frames.md).
