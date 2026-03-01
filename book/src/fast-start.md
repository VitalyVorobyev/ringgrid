# Fast Start

This section gets you from zero to:

- `board_spec.json` (target config used by the detector)
- printable `target_print.svg`
- printable `target_print.png`

in one command.

## 1. Install minimal tooling

From repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install numpy
```

`tools/gen_synth.py` only needs NumPy for target generation.

## 2. Generate target JSON + SVG + PNG

```bash
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

`--print` enables both `--print_svg` and `--print_png`.

## 3. Output files

After the command finishes, you will have:

- `tools/out/target_faststart/board_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`

## 4. Detect against this board

```bash
cargo run -- detect \
  --target tools/out/target_faststart/board_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_faststart/detect.json
```

## 5. Scale handling

- Start with default detection first (`Detector::detect` or CLI `detect`).
- For scenes with very small and very large markers in the same image, use adaptive multi-scale APIs:
  - `Detector::detect_adaptive`
  - `Detector::detect_adaptive_with_hint`
  - `Detector::detect_multiscale`

See [Adaptive Scale Detection](detection-modes/adaptive-scale.md).

## Next Reads

- Full configuration and flag reference: [Target Generation](target-generation.md)
- CLI usage and detection flags: [CLI Guide](cli-guide.md)
- Adaptive scale details: [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
