# Fast Start

This section gets you from zero to:

- `board_spec.json` (target config used by the detector)
- printable `target_print.svg`
- printable `target_print.png`

in one command.

## 1. Generate target JSON + SVG + PNG

Choose one of the three equivalent target-generation paths.

Rust CLI:

```bash
cargo run -p ringgrid-cli -- gen-target \
  --out_dir tools/out/target_faststart \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

Python script (same geometry, same output files):

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip maturin
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
./.venv/bin/python tools/gen_target.py \
  --out_dir tools/out/target_faststart \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

Rust API:

- Use `BoardLayout::new` / `BoardLayout::with_name` with `write_json_file`,
  `write_target_svg`, and `write_target_png` when target generation happens
  inside a Rust application instead of from the terminal.

## 2. Output files

After the command finishes, you will have:

- `tools/out/target_faststart/board_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`

If you also need synthetic camera renders and ground truth, use
`tools/gen_synth.py` instead of the dedicated Rust CLI or Python target-generator path.

## 3. Detect against this board

```bash
cargo run -- detect \
  --target tools/out/target_faststart/board_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_faststart/detect.json
```

`detect.json` contains the final marker list, coordinate-frame metadata,
optional homography/RANSAC statistics, and optional mapper diagnostics. See
[Detection Output Format](output-format.md).

## 4. Scale handling

- Start with default detection first (`Detector::detect` or CLI `detect`).
- For scenes with very small and very large markers in the same image, use adaptive multi-scale APIs:
  - `Detector::detect_adaptive`
  - `Detector::detect_adaptive_with_hint`
  - `Detector::detect_multiscale`

See [Adaptive Scale Detection](detection-modes/adaptive-scale.md).

## Next Reads

- Full configuration and flag reference: [Target Generation](target-generation.md)
- CLI usage and detection flags: [CLI Guide](cli-guide.md)
- Detection JSON schema: [Detection Output Format](output-format.md)
- Adaptive scale details: [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
