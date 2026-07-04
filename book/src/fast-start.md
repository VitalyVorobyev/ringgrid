# Fast Start

This section gets you from zero to:

- `target_spec.json` (target config used by the detector)
- printable `target_print.svg`
- printable `target_print.png`

in one command.

## 1. Generate target JSON + SVG + PNG

The canonical path is the Rust CLI `gen-target` subcommand family. For the
classic hex coded board:

```bash
cargo run -p ringgrid-cli -- gen-target hex \
  --out_dir tools/out/target_faststart \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --dpi 600 \
  --margin_mm 5
```

Other paths (the `TargetLayout` Rust API, the `tools/gen_target.py` Python
script, and the `rect` / `preset` / `from-spec` subcommands for plain and
rectangular targets) are covered in [Target Generation](target-generation.md).

## 2. Output files

After the command finishes, you will have:

- `tools/out/target_faststart/target_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`

If you also need synthetic camera renders and ground truth, use
`tools/gen_synth.py` instead of the target-generator path.

## 3. Detect against this board

```bash
cargo run -- detect \
  --target tools/out/target_faststart/target_spec.json \
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
