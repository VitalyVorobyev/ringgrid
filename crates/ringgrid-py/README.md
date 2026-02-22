# ringgrid (Python)

Python bindings for the `ringgrid` detector, powered by PyO3 + maturin.

## Install (from source)

```bash
pip install maturin
maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

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

## Examples

Run from repository root after `maturin develop`:

```bash
python crates/ringgrid-py/examples/basic_detect.py \
  --image data/target_3_split_00.png \
  --out data/target_3_split_00_det_py.json

python crates/ringgrid-py/examples/detect_with_camera.py \
  --image data/target_3_split_00.png \
  --out data/target_3_split_00_det_cam_py.json
```

Plotting example (requires matplotlib extra):

```bash
pip install -e crates/ringgrid-py[viz]
python crates/ringgrid-py/examples/plot_detection.py \
  --image data/target_3_split_00.png \
  --out data/target_3_split_00_overlay_py.png
```
