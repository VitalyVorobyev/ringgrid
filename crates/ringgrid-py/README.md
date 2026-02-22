# ringgrid (Python)

Python bindings for the `ringgrid` detector, powered by PyO3 + maturin.

## Install (from source)

```bash
pip install maturin
maturin build -m crates/ringgrid-py/Cargo.toml --release
```

## Features

- Native `Detector` API with NumPy input support
- Full `DetectionResult` model objects with JSON round-trips
- Optional plotting helpers in `ringgrid.viz` (`pip install ringgrid[viz]`)
