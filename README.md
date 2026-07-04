[![CI](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/ci.yml)
[![Publish Rust crates (crates.io)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-crates.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-crates.yml)
[![Publish Docs](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-docs.yml)
[![Release Python Package to PyPI](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/release-pypi.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/release-pypi.yml)
[![Security Audit](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/audit.yml)

# ringgrid

`ringgrid` is a pure-Rust detector for dense ring calibration targets on hex or rectangular lattices. It detects markers with subpixel precision, decodes stable baseline IDs for 16-sector coded rings from the shipped 893-codeword profile, estimates homography, and can generate printable target artifacts without OpenCV bindings.

## At a Glance

- Subpixel ring-marker detection using direct ellipse fitting and projective center correction
- Stable shipped `base` profile (`893` IDs, minimum cyclic Hamming distance `2`) plus opt-in `extended`
- Rust library, CLI workflow, and Python bindings in one workspace
- Compositional `TargetLayout`: hex or rect lattices, coded (16-sector) or plain rings, and optional origin-dot fiducials (legacy v4 `board_spec.json` files still load via auto-migration; the deprecated `BoardLayout` type was removed in 0.9 — see the [migration guide](https://vitalyvorobyev.github.io/ringgrid/book/migration-0.8.html))
- Canonical `target_spec.json` (schema v5) plus printable SVG/PNG target generation

Pipeline at a glance: proposals -> local fit/decode -> dedup -> projective center -> `id_correction` -> optional global filter -> optional completion -> final homography refit.

## Visual Overview

`ringgrid` detects two target families from one `TargetLayout` model — a
**coded hex** board (16-sector rings decode to stable IDs) and a **plain rect**
board (uncoded rings, grid-labeled and anchored by origin dots). Each pair below
shows the printable target and a detection overlay (green = fitted ellipses).

| Coded hex — decoded IDs | Plain rect — origin-anchored |
|---|---|
| ![Coded hex target print](docs/assets/target_print.png) | ![Plain rect target print](docs/assets/rect_target_print.png) |
| ![Coded hex detection overlay](docs/assets/det_overlay_0002.png) | ![Plain rect detection overlay](docs/assets/rect_det_overlay.png) |

## Quick Links

| I want to... | Start here |
|---|---|
| Print a target and run first detection from this repo | [Quick Start](#quick-start-from-the-repo) |
| Try detection in the browser, no install | [Live Demo](https://vitalyvorobyev.github.io/ringgrid/demo/) |
| Read the full user guide | [mdBook User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) |
| Use the CLI | [CLI Guide](https://vitalyvorobyev.github.io/ringgrid/book/cli-guide.html) |
| Understand `detect.json` | [Detection Output Format](https://vitalyvorobyev.github.io/ringgrid/book/output-format.html) |
| Use the Rust crate | [crates/ringgrid/README.md](crates/ringgrid/README.md) |
| Use the Python package | [crates/ringgrid-py/README.md](crates/ringgrid-py/README.md) |
| Work on the repo itself | [docs/development.md](docs/development.md) |
| Inspect scoring and benchmark context | [docs/performance.md](docs/performance.md) |

## Quick Start From the Repo

### 1. Generate `target_spec.json` plus printable SVG/PNG/DXF

Choose one of the three target-generation paths. `gen-target` is a subcommand
family (`hex`, `rect`, `preset`, `from-spec`); the classic hex coded board lives
under `hex`. Every path also emits a DXF (2D CAD, millimeters) for laser/CNC
fabrication.

Rust CLI:

```bash
cargo run -p ringgrid-cli -- gen-target hex \
  --out_dir tools/out/target_faststart \
  --pitch_mm 8 \
  --rows 15 \
  --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 \
  --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --name ringgrid_200mm_hex \
  --dpi 600 \
  --margin_mm 5
```

Python (typed `TargetLayout` API — same geometry, same artifact set):

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip maturin
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

```python
from pathlib import Path
import ringgrid

out = Path("tools/out/target_faststart")
out.mkdir(parents=True, exist_ok=True)
target = ringgrid.TargetLayout.coded_hex(
    pitch_mm=8.0, rows=15, long_row_cols=14,
    outer_radius_mm=4.8, inner_radius_mm=3.2, ring_width_mm=1.152,
)
(out / "target_spec.json").write_text(target.to_spec_json())
target.write_svg(out / "target_print.svg", margin_mm=5.0)
target.write_png(out / "target_print.png", dpi=600.0, margin_mm=5.0)
target.write_dxf(out / "target_print.dxf")
```

Rust API:

- Use [`TargetLayout::coded_hex(...)`](crates/ringgrid/README.md) (or the compositional `TargetLayout::new(...)`) plus `write_json_file`, `write_target_svg`, `write_target_png`, and `write_target_dxf` when generation is part of your application code.

Generated files (identical across the Rust CLI, Rust API, and Python paths; all write a v5 `target_spec.json`):

- `tools/out/target_faststart/target_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`
- `tools/out/target_faststart/target_print.dxf`

### 2. Run detection

```bash
cargo run -- detect \
  --target tools/out/target_faststart/target_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_faststart/detect.json
```

The written `detect.json` contains the final `detected_markers` list plus image
size, coordinate-frame metadata, optional `homography` / `ransac`, and optional
diagnostic blocks such as `self_undistort`, CLI `camera`, and proposal data.
The full schema is documented in [Book: Detection Output Format](https://vitalyvorobyev.github.io/ringgrid/book/output-format.html).

### 3. Optional synthetic eval loop

Install the extra Python deps used by the synth/eval/viz tools before running this loop:

```bash
./.venv/bin/python -m pip install numpy matplotlib
```

```bash
./.venv/bin/python tools/gen_synth.py --out_dir tools/out/synth_001 --n_images 1 --blur_px 1.0

cargo run -- detect \
  --image tools/out/synth_001/img_0000.png \
  --out tools/out/synth_001/det_0000.json

./.venv/bin/python tools/score_detect.py \
  --gt tools/out/synth_001/gt_0000.json \
  --pred tools/out/synth_001/det_0000.json \
  --gate 8.0 \
  --out tools/out/synth_001/score_0000.json
```

If you want the full generate -> detect -> score loop in one command, use `./.venv/bin/python tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_run`.

## Choose an Interface

### CLI

Use `ringgrid gen-target <hex|rect|preset|from-spec>` / `ringgrid detect` or `cargo run -- gen-target ...` / `cargo run -- detect ...` when you want file-oriented workflows over printable targets, images, and JSON outputs. The full flag reference is in the [CLI Guide](https://vitalyvorobyev.github.io/ringgrid/book/cli-guide.html).

### Rust crate

The core detector lives in [`crates/ringgrid/README.md`](crates/ringgrid/README.md). That README covers Rust-library usage, Rust target-generation APIs, adaptive detection modes, and camera-model integration in more detail than this front page should.

### Python package

The Python bindings live in [`crates/ringgrid-py/README.md`](crates/ringgrid-py/README.md). Use them when you want installed-package target generation, Python-side detector configuration, or plotting helpers.

## Detection Output

The detector output is centered on `detected_markers`. Each marker can contain a
decoded `id`, its lattice `grid_coord`, `board_xy_mm`, image-space `center`,
optional `center_mapped`, and ellipse fits. Per-marker fit/decode metrics and
the detection `source` live in the opt-in `diagnostics` channel.

At the top level, you always get `image_size`, `center_frame`, and
`homography_frame`, plus `board_frame` for grid-labeled runs. When enough valid
IDs exist, you also get the board-to-image `homography` (RANSAC stats live in the
`diagnostics` channel). Self-undistort runs add `self_undistort`. CLI runs with camera input
echo the top-level `camera`, and `--include-proposals` adds proposal
diagnostics. Full reference: [Book: Detection Output Format](https://vitalyvorobyev.github.io/ringgrid/book/output-format.html).

## Documentation Map

- [Live Demo](https://vitalyvorobyev.github.io/ringgrid/demo/) - in-browser WASM detection over sample and uploaded images, no install required
- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) - full mdBook covering marker design, pipeline stages, math, configuration, target generation, and usage
- [Book: Fast Start](https://vitalyvorobyev.github.io/ringgrid/book/fast-start.html) - repo-oriented first-run path for target generation and detection
- [Book: Detection Output Format](https://vitalyvorobyev.github.io/ringgrid/book/output-format.html) - exact `detect.json` structure, marker fields, and CLI-only wrapper fields
- [Book: Target Generation](https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html) - JSON/SVG/PNG generation details and flags
- [Book: Proposal Diagnostics](https://vitalyvorobyev.github.io/ringgrid/book/detection-modes/proposal-diagnostics.html) - proposal-only API, accumulator heatmap, and tuning workflow
- [Book: Adaptive Scale Detection](https://vitalyvorobyev.github.io/ringgrid/book/detection-modes/adaptive-scale.html) - multi-scale detection modes and tier selection
- [Rust API Reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/) - rustdoc for the public Rust surface
- [Rust crate README](crates/ringgrid/README.md) - crate-level Rust examples and API-oriented guidance
- [Python package README](crates/ringgrid-py/README.md) - installed-package usage and Python `DetectConfig` field guide
- [Development Guide](docs/development.md) - repo layout, contributor workflows, generated assets, and validation commands
- [Performance & Evaluation](docs/performance.md) - scoring semantics, benchmark commands, and published snapshot tables
- [Proposal Performance Analysis](docs/proposal-performance-analysis.md) - proposal-stage hotspot analysis and alternative algorithms
- [Tuning Guide](docs/tuning-guide.md) - symptom-to-config tuning notes for difficult image conditions

## Diligence Statement

This project is developed with AI coding assistants (`Codex` and `Claude Code`) as implementation tools. Not every code path is manually line-reviewed by a human before merge. The project author validates algorithmic behavior and numerical results and enforces quality gates before release.
