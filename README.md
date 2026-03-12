[![CI](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/ci.yml)
[![Publish Rust crates (crates.io)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-crates.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-crates.yml)
[![Publish Docs](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/publish-docs.yml)
[![Release Python Package to PyPI](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/release-pypi.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/release-pypi.yml)
[![Security Audit](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/ringgrid/actions/workflows/audit.yml)

# ringgrid

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice. It detects markers with subpixel precision, decodes stable baseline IDs from the shipped 893-codeword profile, estimates homography, and can generate printable target artifacts without OpenCV bindings.

## At a Glance

- Subpixel ring-marker detection using direct ellipse fitting and projective center correction
- Stable shipped `base` profile (`893` IDs, minimum cyclic Hamming distance `2`) plus opt-in `extended`
- Rust library, CLI workflow, and Python bindings in one workspace
- Canonical `board_spec.json` plus printable SVG/PNG target generation

Pipeline at a glance: proposals -> local fit/decode -> dedup -> projective center -> `id_correction` -> optional global filter -> optional completion -> final homography refit.

## Visual Overview

Target print example:

![Ringgrid target print](docs/assets/target_print.png)

Detection overlay example:

![Detection overlay example](docs/assets/det_overlay_0002.png)

## Quick Links

| I want to... | Start here |
|---|---|
| Print a target and run first detection from this repo | [Quick Start](#quick-start-from-the-repo) |
| Read the full user guide | [mdBook User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) |
| Use the CLI | [CLI Guide](https://vitalyvorobyev.github.io/ringgrid/book/cli-guide.html) |
| Use the Rust crate | [crates/ringgrid/README.md](crates/ringgrid/README.md) |
| Use the Python package | [crates/ringgrid-py/README.md](crates/ringgrid-py/README.md) |
| Work on the repo itself | [docs/development.md](docs/development.md) |
| Inspect scoring and benchmark context | [docs/performance.md](docs/performance.md) |

## Quick Start From the Repo

### 1. Generate `board_spec.json` plus printable SVG/PNG

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

Python script (same geometry, same artifact set):

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

- Use [`BoardLayout::new` / `BoardLayout::with_name`](crates/ringgrid/README.md) plus `write_json_file`, `write_target_svg`, and `write_target_png` when generation is part of your application code.

Generated files:

- `tools/out/target_faststart/board_spec.json`
- `tools/out/target_faststart/target_print.svg`
- `tools/out/target_faststart/target_print.png`

### 2. Run detection

```bash
cargo run -- detect \
  --target tools/out/target_faststart/board_spec.json \
  --image path/to/photo.png \
  --out tools/out/target_faststart/detect.json
```

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

Use `ringgrid gen-target` / `ringgrid detect` or `cargo run -- gen-target ...` / `cargo run -- detect ...` when you want file-oriented workflows over printable targets, images, and JSON outputs. The full flag reference is in the [CLI Guide](https://vitalyvorobyev.github.io/ringgrid/book/cli-guide.html).

### Rust crate

The core detector lives in [`crates/ringgrid/README.md`](crates/ringgrid/README.md). That README covers Rust-library usage, Rust target-generation APIs, adaptive detection modes, and camera-model integration in more detail than this front page should.

### Python package

The Python bindings live in [`crates/ringgrid-py/README.md`](crates/ringgrid-py/README.md). Use them when you want installed-package target generation, Python-side detector configuration, or plotting helpers.

## Documentation Map

- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) - full mdBook covering marker design, pipeline stages, math, configuration, target generation, and usage
- [Book: Fast Start](https://vitalyvorobyev.github.io/ringgrid/book/fast-start.html) - repo-oriented first-run path for target generation and detection
- [Book: Target Generation](https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html) - JSON/SVG/PNG generation details and flags
- [Book: Adaptive Scale Detection](https://vitalyvorobyev.github.io/ringgrid/book/detection-modes/adaptive-scale.html) - multi-scale detection modes and tier selection
- [Rust API Reference](https://vitalyvorobyev.github.io/ringgrid/ringgrid/) - rustdoc for the public Rust surface
- [Rust crate README](crates/ringgrid/README.md) - crate-level Rust examples and API-oriented guidance
- [Python package README](crates/ringgrid-py/README.md) - installed-package usage and Python `DetectConfig` field guide
- [Development Guide](docs/development.md) - repo layout, contributor workflows, generated assets, and validation commands
- [Performance & Evaluation](docs/performance.md) - scoring semantics, benchmark commands, and published snapshot tables
- [Tuning Guide](docs/tuning-guide.md) - symptom-to-config tuning notes for difficult image conditions

## Diligence Statement

This project is developed with AI coding assistants (`Codex` and `Claude Code`) as implementation tools. Not every code path is manually line-reviewed by a human before merge. The project author validates algorithmic behavior and numerical results and enforces quality gates before release.
