# Development Guide

This page collects contributor-facing and repo-maintainer material that was moved out of the root `README.md`.

## Local Setup

### Rust

```bash
cargo build
cargo test
```

### Python tooling

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip maturin numpy matplotlib
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

### Docs

```bash
mdbook build book
```

## Repository Map

```text
crates/
  ringgrid/
    src/
      lib.rs       # public re-exports
      api.rs       # Detector facade
      pipeline/    # single-pass / multi-pass orchestration
      detector/    # proposal, fit, decode, dedup, filter, completion
      ring/        # radial sampling and projective center logic
      marker/      # codebook, decode, marker spec
      homography/  # DLT, RANSAC, refit utilities
      conic/       # ellipse types, fitting, RANSAC, eigenvalue solver
      pixelmap/    # camera models, PixelMapper, self-undistort
    examples/      # concise library usage examples
  ringgrid-cli/    # clap-based CLI binary
  ringgrid-py/     # Python bindings and package README
tools/
  gen_target.py        # board_spec.json + SVG + PNG generation
  gen_synth.py         # synthetic dataset generator
  run_synth_eval.py    # generate -> detect -> score
  score_detect.py      # scoring utility
  viz_detect.py        # detection overlay renderer
docs/
  module_structure.md
  pipeline_analysis.md
  performance.md
```

For deeper ownership and layering notes, see [module_structure.md](module_structure.md) and [pipeline_analysis.md](pipeline_analysis.md).

## Common Workflows

### Run Rust examples

```bash
cargo run -p ringgrid --example basic_detect -- \
  crates/ringgrid/examples/target.json tools/out/synth_001/img_0000.png

cargo run -p ringgrid --example detect_with_camera -- \
  crates/ringgrid/examples/target.json tools/out/synth_001/img_0000.png

cargo run -p ringgrid --example detect_with_self_undistort -- \
  crates/ringgrid/examples/target.json tools/out/synth_001/img_0000.png
```

For fuller Rust-library usage and detection-mode examples, use [`../crates/ringgrid/README.md`](../crates/ringgrid/README.md). For Python-side usage, use [`../crates/ringgrid-py/README.md`](../crates/ringgrid-py/README.md).

### Regenerate shipped/generated assets

The committed codebook artifacts live in `tools/codebook.json` and `crates/ringgrid/src/marker/codebook.rs`. Regenerate them only via the generators:

```bash
python3 tools/gen_codebook.py \
  --n 893 --seed 1 \
  --out_json tools/codebook.json \
  --out_rs crates/ringgrid/src/marker/codebook.rs

python3 tools/gen_board_spec.py \
  --pitch_mm 8.0 \
  --rows 15 --long_row_cols 14 \
  --board_mm 200.0 \
  --json_out tools/board/board_spec.json
```

Rebuild after regeneration:

```bash
cargo build --release
```

### Run validation checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --all-features --no-deps
cargo test --doc --workspace
mdbook build book
./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check
./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release
./.venv/bin/python -m pytest crates/ringgrid-py/tests -q
```

## More References

- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) for end-user workflows, theory, and detailed configuration notes
- [Performance & Evaluation](performance.md) for scoring semantics and benchmark commands
- [Tuning Guide](tuning-guide.md) for symptom-driven config adjustments
- [docs/workflows/](workflows) for the repo’s task and handoff workflows
