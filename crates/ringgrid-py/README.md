# ringgrid (Python)

Python bindings for the `ringgrid` detector (PyO3 + maturin).

## Install

From PyPI:

```bash
pip install ringgrid
```

With plotting helpers:

```bash
pip install "ringgrid[viz]"
```

From source (repository checkout):

```bash
pip install maturin
maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

## Fast Start: Generate `board_spec.json` + Printable SVG/PNG

Installed-package target generation is available directly from `import ringgrid`:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.from_geometry(
    8.0,
    15,
    14,
    4.8,
    3.2,
    name="ringgrid_200mm_hex",
)

board.to_spec_json(Path("board_spec.json"))
board.write_svg(Path("target_print.svg"), margin_mm=5.0)
board.write_png(Path("target_print.png"), dpi=600.0, margin_mm=5.0)
```

Key knobs:

| API | What it controls | Typical value |
|---|---|---|
| `BoardLayout.from_geometry(...)` | Board geometry (`pitch_mm`, `rows`, `long_row_cols`, radii) | `8.0`, `15`, `14`, `4.8`, `3.2` |
| `name=` | Optional explicit board name; omitted uses deterministic geometry-derived name | `"ringgrid_200mm_hex"` |
| `write_svg(..., margin_mm=...)` | Extra white border around the printable page | `3-10` |
| `write_png(..., dpi=...)` | PNG raster resolution and embedded print metadata | `300` or `600` |
| `write_png(..., include_scale_bar=...)` | Include or omit the default scale bar | `True` |

Outputs:

- `board_spec.json`
- `target_print.svg`
- `target_print.png`

Load this board in Python:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.from_json_file(Path("tools/out/target_faststart/board_spec.json"))
cfg = ringgrid.DetectConfig(board)
detector = ringgrid.Detector(cfg)
# Convenience defaults: detector = ringgrid.Detector.from_board(board)
```

If you are working from a repository checkout and also need synthetic images or
ground truth, the repo tools under `tools/` still provide the combined
generation/evaluation workflow. The installed package target-generation API is
for board JSON + printable SVG/PNG only.

Complete target-generation tutorial and full flag reference:
- https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html

## Features

- Native `BoardLayout` target generation for canonical spec JSON + printable SVG/PNG
- Native `Detector` API with NumPy input support
- Full `DetectionResult` model objects with JSON round-trips
- Optional plotting helpers in `ringgrid.viz` (`pip install ringgrid[viz]`)

## Input Rules

- `Detector.detect(...)` accepts:
  - `np.ndarray` with `dtype=uint8` and shape `(H, W)` (grayscale)
  - `np.ndarray` with `dtype=uint8` and shape `(H, W, 3|4)` (RGB/RGBA, auto-converted to grayscale)
  - image file path (`str` or `pathlib.Path`)
- Other dtypes/shapes raise `TypeError`.

## Adaptive Detection

Use adaptive detection when marker diameter varies substantially across the
image (near/far perspective, zoom changes, mixed target scales).

### Which Method Should I Use?

| Situation | Recommended call | Why |
|---|---|---|
| You do not know marker size in advance | `detector.detect_adaptive(image)` | Probes scale and auto-selects tiers |
| You know approximate marker diameter (px) | `detector.detect_adaptive(image, nominal_diameter_px=d)` | Skips probe and uses focused two-tier bracket around `d` |
| You need fixed/reproducible tier policy | `detector.detect_multiscale(image, tiers)` | Full explicit control over tiers |
| Marker size range is tight and runtime is priority | `detector.detect(image)` | Single-pass (fastest) |

Canonical adaptive entry point is:
- `Detector.detect_adaptive(image, nominal_diameter_px: float | None = None)`

Compatibility alias (deprecated, still supported):
- `Detector.detect_adaptive_with_hint(image, nominal_diameter_px=...)`

Tier objects:
- `ScaleTier(diameter_min_px, diameter_max_px)`
- `ScaleTiers([...])`
- Presets: `ScaleTiers.four_tier_wide()`, `ScaleTiers.two_tier_standard()`
- Single-pass equivalent: `ScaleTiers.single(MarkerScalePrior(...))`

### Practical Recipes

Unknown scene scale:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.default()
detector = ringgrid.Detector.from_board(board)
image = Path("testdata/target_3_split_00.png")

result = detector.detect_adaptive(image)
```

Known nominal diameter (for example, ~32 px):

```python
result = detector.detect_adaptive(image, nominal_diameter_px=32.0)
```

Inspect tiers used by adaptive logic (debug/repro):

```python
tiers = detector.adaptive_tiers(image, nominal_diameter_px=32.0)
for tier in tiers.tiers:
    print(tier.diameter_min_px, tier.diameter_max_px)

# Re-run exactly those tiers
result = detector.detect_multiscale(image, tiers)
```

## Examples

Run from repository root:

```bash
python crates/ringgrid-py/examples/basic_detect.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_py.json

python crates/ringgrid-py/examples/detect_with_camera.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_cam_py.json

python crates/ringgrid-py/examples/detect_adaptive.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_adaptive_py.json

python crates/ringgrid-py/examples/detect_multiscale.py \
  --image testdata/target_3_split_00.png \
  --tiers four_tier_wide \
  --out testdata/target_3_split_00_det_multiscale_py.json
```

Plotting example:

```bash
python crates/ringgrid-py/examples/plot_detection.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_overlay_py.png
```
