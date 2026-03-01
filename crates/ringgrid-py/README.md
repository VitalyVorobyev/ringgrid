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

Target-generation scripts live in the repository root (`tools/`). If you only
installed from PyPI, clone the repository first to access these scripts.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install numpy

./.venv/bin/python tools/gen_synth.py \
  --out_dir tools/out/target_faststart \
  --n_images 0 \
  --board_mm 200 \
  --pitch_mm 8 \
  --print \
  --print_dpi 600 \
  --print_margin_mm 5 \
  --print_basename target_print
```

Key knobs:

| Flag | What it controls | Typical value |
|---|---|---|
| `--board_mm` | Physical board side length (mm) | `200` |
| `--pitch_mm` | Marker spacing (mm) | `8` |
| `--n_images` | Number of synthetic images (`0` for print-only) | `0` |
| `--print_dpi` | PNG raster resolution | `300` or `600` |
| `--print_margin_mm` | Extra white border | `3-10` |
| `--print_basename` | Output file basename | `target_print` |

Outputs:

- `tools/out/target_faststart/board_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`

Load this board in Python:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.from_json_file(Path("tools/out/target_faststart/board_spec.json"))
cfg = ringgrid.DetectConfig(board)
detector = ringgrid.Detector(board, cfg)
```

Complete target-generation tutorial and full flag reference:
- https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html

## Features

- Native `Detector` API with NumPy input support
- Full `DetectionResult` model objects with JSON round-trips
- Optional plotting helpers in `ringgrid.viz` (`pip install ringgrid[viz]`)

## Input Rules

- `Detector.detect(...)` accepts:
  - `np.ndarray` with `dtype=uint8` and shape `(H, W)` (grayscale)
  - `np.ndarray` with `dtype=uint8` and shape `(H, W, 3|4)` (RGB/RGBA, auto-converted to grayscale)
  - image file path (`str` or `pathlib.Path`)
- Other dtypes/shapes raise `TypeError`.

## Adaptive Scale Note

Adaptive multi-scale entry points are currently exposed in the Rust API
(`detect_adaptive`, `detect_adaptive_with_hint`, `detect_multiscale`).
Python bindings currently provide `detect` and `detect_with_mapper`.

## Examples

Run from repository root:

```bash
python crates/ringgrid-py/examples/basic_detect.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_py.json

python crates/ringgrid-py/examples/detect_with_camera.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_cam_py.json
```

Plotting example:

```bash
python crates/ringgrid-py/examples/plot_detection.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_overlay_py.png
```
