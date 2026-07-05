# Fast Start

This section gets you from zero to:

- `target_spec.json` (target config used by the detector)
- printable `target_print.svg`
- printable `target_print.png`
- fabrication-ready `target_print.dxf`

in three commands, using the published `ringgrid` binary.

## 0. Install

```bash
cargo install ringgrid --features cli
```

This puts a `ringgrid` binary on your `PATH`. (Library users run
`cargo add ringgrid`; Python users `pip install ringgrid`.)

## 1. Get a recipe

A *recipe* is the small TOML (or JSON) file that describes the target you want.
Start from a built-in example — the classic hex coded board:

```bash
ringgrid example --name hex_coded --out hex_coded.toml
```

Run `ringgrid example --list` to see all built-in recipes (the six valid
combinations of `{hex, rect}` × `{coded, plain}` × `{origin dots, no dots}`).

## 2. Generate target JSON + SVG + PNG + DXF

```bash
ringgrid gen hex_coded.toml --out ./out/target_faststart
```

Other paths (the `TargetLayout` Rust API, custom recipes, and the plain /
rectangular target families) are covered in
[Target Generation](target-generation.md).

## 3. Output files

After the command finishes, you will have:

- `./out/target_faststart/target_spec.json`
- `./out/target_faststart/target_print.svg`
- `./out/target_faststart/target_print.png`
- `./out/target_faststart/target_print.dxf`

## 4. Detect against this board

```bash
ringgrid detect \
  --target ./out/target_faststart/target_spec.json \
  --image path/to/photo.png \
  --out ./out/target_faststart/detect.json
```

`detect.json` contains the final marker list, coordinate-frame metadata,
optional homography/RANSAC statistics, and optional mapper diagnostics. See
[Detection Output Format](output-format.md). Omit `--out` to print the JSON to
stdout instead.

> **Developing ringgrid.** If you also need synthetic camera renders and
> ground truth for benchmarking, those live in the in-repo Python tooling
> (`tools/gen_synth.py`) and require a repository checkout. See
> [Development](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/development.md).

## 5. Scale handling

- Start with default detection first (`Detector::detect`, or CLI `detect`).
- For scenes with very small and very large markers in the same image, use the
  adaptive multi-scale APIs (exposed via the Rust and Python libraries):
  - `Detector::detect_adaptive`
  - `Detector::detect_adaptive_with_hint`
  - `Detector::detect_multiscale`

See [Adaptive Scale Detection](detection-modes/adaptive-scale.md).

## Next Reads

- Full configuration and recipe reference: [Target Generation](target-generation.md)
- CLI usage and detection flags: [CLI Guide](cli-guide.md)
- Detection JSON schema: [Detection Output Format](output-format.md)
- Adaptive scale details: [Adaptive Scale Detection](detection-modes/adaptive-scale.md)
