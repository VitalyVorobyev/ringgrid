# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

`ringgrid` is a pure-Rust detector for dense coded ring calibration targets on a hex lattice. It detects markers, decodes 16-sector IDs, estimates homography, and exports structured JSON. No OpenCV bindings — all image processing is in Rust. The goal is to provide a polished external Rust library API.

## Workspace Structure

Cargo workspace (edition 2024, MSRV 1.91) with two crate members plus excluded bindings:
- `crates/ringgrid/` — detection algorithms, math primitives, result types
- `crates/ringgrid-cli/` — CLI binary (`ringgrid`) with clap-based argument parsing
- `crates/ringgrid-py/` — PyO3 Python bindings (excluded from workspace, built via maturin)
- `crates/ringgrid-wasm/` — WebAssembly bindings (excluded from workspace, built via wasm-pack)
- `crates/ringgrid-c/` — C/C++ ABI (excluded from workspace, cbindgen header + C++ RAII wrapper, CMake package + vcpkg overlay port)
- `tools/` — Python utilities for synthetic data generation, evaluation, scoring, and visualization
- `py/` — Python scripts for rtv3d benchmarking and overlay visualization

Key external dependency: `radsym` crate provides the proposal-stage center detection (fused RSD with Scharr gradient). Python package management uses `uv` (virtualenv at `.venv`).

## Engineering principles & critical review

Hold every change to a high design bar, and actively keep this workspace
from drifting back into undisciplined, copy-paste "agentic slop." Apply
these principles by default — not only when explicitly asked:

- **SOLID** — one responsibility per type/module; extend behavior through
  the detector / refiner / orientation traits, not by editing parallel
  match arms; depend on trait abstractions (`DenseDetector`, the refiner
  traits), not concretions.
- **DRY / single source of truth** — one canonical definition per concept.
  Lower config to core params in exactly one place; never let a
  threshold's or parameter's meaning be duplicated or diverge across
  crates.
- **KISS & YAGNI** — choose the simplest design that meets the actual
  requirement; do not add config knobs, enum variants, or abstraction
  layers for hypothetical future needs.
- **Make illegal states unrepresentable** — push invariants into the type
  system (enum-with-payload over a discriminator + parallel fields;
  `Option` / newtypes over magic sentinels) so misuse fails to compile.
- **Least astonishment & orthogonality** — APIs do what their names say;
  keep independent concerns independent (orientation is a cross-cutting
  stage, not a sub-mode of one detector).
- **Minimal, honest public surface** — expose only what callers need; keep
  diagnostics and internals out of the stable API (see *Public surface
  hygiene*).

**Be critical of every proposal — including the user's.** Treat a request
as the start of a design discussion, not an instruction to implement
verbatim. Before coding, review it against the principles above and the
existing architecture; if it would introduce duplication, leak internals,
add a dominated alternative, widen the public surface needlessly, or
otherwise degrade the design, **say so and offer the cleaner alternative**
rather than building it as-stated. When the better design needs a breaking
change and the crate is pre-1.0, prefer the better design (see *Decisive
cleanup*). Every change should leave the workspace's design and style
better than it found them.

## Module Layout

```
crates/ringgrid/src/
├── lib.rs              # Re-exports only (public API: root + diagnostics/codebook tiers)
├── api.rs              # Detector facade + propose_with_* free functions
├── bin/ringgrid.rs     # Published CLI (feature `cli`): `gen` + `detect`
├── cli/                # Internal CLI support (#[doc(hidden)], feature `cli`)
│   ├── recipe.rs       # TargetRecipe/RenderRecipe authoring schema (TOML/JSON)
│   ├── artifacts.rs    # write_target_artifacts — renders the bundle, writes files
│   └── detect.rs       # Shared detect/load helpers
├── target/             # Compositional target model (lattice × ring × coding × fiducials)
│   ├── layout.rs       # TargetLayout + presets (default_hex, coded_hex, rect_24x24)
│   ├── lattice.rs      # LatticeGeometry (Hex | Rect) + cell generation
│   ├── ring.rs         # RingGeometry, MarkerCoding (Coded16 | Plain), CodebookProfile
│   ├── page.rs         # PageSpec/PageSize/PageOrientation (printed sheet)
│   ├── fiducials.rs    # OriginFiducials (validation: rotation-asymmetry, clearance)
│   ├── schema.rs       # JSON v6 (the only supported schema)
│   └── error.rs        # TargetValidationError, TargetLoadError
├── target_generation.rs # SVG/PNG/DXF rendering + TargetRenderOptions/TargetArtifacts
├── proposal/           # Center proposal generation (delegates to radsym)
│   ├── mod.rs          # Adapter: radsym fused RSD → ringgrid Proposal types
│   ├── config.rs       # ProposalConfig (translated to radsym RsdConfig)
│   └── tests.rs        # Behavioral tests
├── detector/           # Per-marker detection primitives
│   ├── config/         # DetectConfig + sub-configs
│   │   ├── mod.rs      # DetectConfig, AdvancedDetectConfig, prior derivation
│   │   ├── fit.rs      # InnerFitConfig, OuterFitConfig
│   │   ├── scale.rs    # MarkerScalePrior, ScaleTier(s), ProposalDownscale
│   │   └── stages.rs   # Completion/ProjectiveCenter/IdCorrection/Recovery configs
│   ├── outer_fit/      # RANSAC ellipse fitting (Fitzgibbon direct LS)
│   │   ├── mod.rs, sampling.rs, scoring.rs, solver.rs
│   ├── inner_fit.rs    # Inner ring ellipse fit
│   ├── inner_as_outer_recovery.rs # Recover inner ring from outer-fit failures
│   ├── marker_build.rs # DetectedMarker, FitMetrics structs + builder
│   ├── dedup.rs        # Spatial + ID-based deduplication
│   ├── global_filter.rs# RANSAC homography filter
│   ├── completion.rs   # Conservative fits at missing cells (Id- or Coord-keyed)
│   ├── center_correction.rs # Projective center application
│   └── id_correction/  # BFS hex-neighbor ID consensus (fallback only)
│       ├── mod.rs, engine.rs, workspace.rs, types.rs, index.rs
│       ├── local.rs, homography.rs, vote.rs, math.rs
│       ├── bootstrap.rs, consistency.rs, cleanup.rs, diagnostics.rs
├── ring/               # Ring-level sampling & estimation primitives
│   ├── outer_estimate.rs    # Radius hypotheses via radial profile peaks
│   ├── inner_estimate.rs    # Inner ring scale estimation
│   ├── radial_profile.rs    # Radial intensity profile sampling
│   ├── radial_estimator.rs  # Radial profile estimator
│   ├── edge_sample.rs       # Edge point sampling (distortion-aware)
│   └── projective_center.rs # Unbiased center from inner+outer conics
├── marker/             # Code decoding & marker specification
│   ├── decode.rs       # 16-sector code sampling → codebook match
│   ├── codec.rs        # Codebook matching logic
│   ├── codebook.rs     # Generated codeword tables: base (893) + extended (2180)
│   └── marker_spec.rs  # MarkerSpec type
├── pipeline/           # Detection pipeline orchestration
│   ├── mod.rs          # Module glue, re-exports
│   ├── run.rs          # Top-level orchestrator + two-pass/self-undistort/multiscale
│   ├── fit_decode.rs   # Proposals → fit → decode → dedup
│   ├── finalize/       # Pipeline back half
│   │   ├── mod.rs      # run/premerge/postmerge entry points
│   │   ├── grid.rs     # Grid-first back half (every target): label → anchor → identify → complete → H → verify
│   │   ├── fallback.rs # No grid: ID correction + decode-anchored global filter (coded), pass-through (plain)
│   │   └── common.rs   # Final H, frame mapping, correspondence sync, geometric verify
│   ├── assign.rs       # Grid labeling (projective_grid::detect_grid) + f64 frame H
│   ├── anchor.rs       # Origin resolution: decoded-ID vote (coded) or origin dots (plain)
│   ├── geometric_verify.rs # Local/global geometric consistency checks
│   ├── photometric.rs  # Last stage: photometric center refinement + final-H refit (once per call)
│   ├── scale_probe.rs  # Ring angular-variance sweep → dominant radius estimates
│   ├── axis_ratio_filter.rs # Ellipse axis-ratio gating
│   ├── stats.rs        # Pipeline counters
│   ├── time_compat.rs  # wasm-safe Instant
│   ├── prelude.rs      # Common imports for pipeline modules
│   └── result.rs       # DetectionResult, DetectedMarker, BoardFrame, diagnostics
├── homography/         # Homography estimation & utilities
│   ├── core.rs         # DLT + RANSAC, RansacStats
│   ├── utils.rs        # Refit, reprojection error, matrix conversion
│   └── correspondence.rs # Point correspondence matching
├── conic/              # Ellipse/conic math
│   ├── types.rs        # Ellipse type
│   ├── fit.rs          # Direct ellipse fitting (Fitzgibbon)
│   ├── ransac.rs       # RANSAC ellipse fitting
│   └── eigen.rs        # Generalized eigenvalue solver
├── pixelmap/           # Camera, distortion, pixel mapping
│   ├── cameramodel.rs  # CameraModel, CameraIntrinsics
│   ├── distortion.rs   # RadialTangentialDistortion, DivisionModel
│   └── self_undistort/ # Self-undistort estimation
│       ├── mod.rs, config.rs, result.rs
│       ├── estimator.rs, objective.rs, optimizer.rs, policy.rs
```

## Detection Pipeline Architecture

### Single-pass stages (in order)

1. **Proposal** (`proposal/mod.rs`) — radsym fused RSD (Scharr gradient + magnitude voting + NMS) → candidate centers
2. **Outer Estimate** (`ring/outer_estimate.rs`) — radius hypotheses via radial profile peaks
3. **Outer Fit** (`detector/outer_fit/`) — RANSAC ellipse fitting (Fitzgibbon direct LS)
4. **Decode** (`marker/decode.rs`) — 16-sector code sampling → codebook match (base 893 / extended 2180, selected by the target)
5. **Inner Fit** (`detector/inner_fit.rs`) — inner ring ellipse fit
6. **Dedup** (`detector/dedup.rs`) — spatial + ID-based deduplication
7. **Projective Center** (`detector/center_correction.rs`) — correct fit-decode marker centers (once per marker)
8. **Grid Labeling** (`pipeline/assign.rs`) — label every fitted ring, decoded or not, with lattice coords (relative frame)
9. **Anchor + Identify** (`pipeline/anchor.rs`, `finalize/grid.rs`) — decoded IDs vote for the board placement (coded) or origin dots verify it (plain); each labeled ring takes its cell's ID (a contradicting decode is overridden)
10. **Join + Completion** (`finalize/grid.rs`, `detector/completion.rs`) — decoded dropped rings join via RANSAC; completion grows from the patch, adopting dropped rings before refitting missing cells
11. **Final H Refit + Geometric Verify** (`finalize/common.rs`, `pipeline/geometric_verify.rs`)

Fallback (`finalize/fallback.rs`) when no grid is labeled (coded: and anchored): hex ID correction (`detector/id_correction/`) → global RANSAC filter over decodes → completion; with fewer than 6 consistent decodes, markers without H. Stages 1–6 in `pipeline/fit_decode.rs`, 7–11 in `pipeline/finalize/`, sequencing in `pipeline/run.rs`.

### Multi-scale / adaptive pipeline

For multi-scale detection (`detect_multiscale`, `detect_adaptive`, `detect_adaptive_with_hint`), the pipeline is split at the finalize boundary:

- **`finalize_premerge`** — runs stage 7 (projective center) only. Called once per tier.
- **`merge_multiscale_markers`** (`detector/dedup.rs`) — size-consistency-aware NMS across all tier outputs. Prefers markers whose outer radius matches the neighborhood median (k=6 hex-lattice neighbors); confidence breaks ties.
- **`finalize_postmerge`** — runs stages 8–11 (and the fallback) exactly once on the merged pool.

Scale probe (`pipeline/scale_probe.rs`): ring angular-variance sweep at top-K gradient proposals over 20 geometric radius candidates (4–110 px). High variance at a radius indicates the code band (alternating bright/dark sectors). Code-band midpoint ≈ 0.8× outer ring radius; results feed `ScaleTiers::from_detected_radii`.

Default `MarkerScalePrior` is **[14, 66] px**.

## Public API

All detection goes through the `Detector` struct (`api.rs`). Standalone proposal functions are also available:
- `find_ellipse_centers`, `find_ellipse_centers_with_heatmap` — low-level proposal generation
- `propose_with_marker_scale`, `propose_with_heatmap_and_marker_scale` — board-aware proposal helpers

Key public types (root tier):
- `Detector`, `DetectError` — entry point
- `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod` — configuration
- `ScaleTier`, `ScaleTiers` — multi-scale tier configuration (presets: `four_tier_wide`, `two_tier_standard`, `single`)
- `DetectionResult`, `DetectedMarker`, `BoardFrame`, `DetectionFrame` — stable results
- `TargetLayout` + `LatticeGeometry`/`RingGeometry`/`MarkerCoding`/`OriginFiducials` — target geometry (JSON is `ringgrid.target.v6` only: v4/v5 files are rejected with an "unsupported schema" error, convert them by loading with 0.15 and re-saving; the `BoardLayout`/`BoardMarker` types were removed in 0.9). `TARGET_SCHEMA_VERSION` is the one definition of the emitted schema string.
- `TargetRenderOptions`, `TargetArtifacts`, `PageSpec`/`PageSize`/`PageOrientation`, `TargetGenerationError` — target rendering. `render_target_artifacts` returns all four formats in memory and works without the `std` feature, so every binding shares one contract.
- `CodebookProfile` — `Base` (893) or `Extended` (2180). Lives on `CodedRingSpec`, i.e. on the **target**: rendering draws from it and `DetectConfig::with_target` derives `advanced.decode.codebook_profile` from it, so a printed target can never be decoded against a different table.
- `CameraModel`, `CameraIntrinsics`, `PixelMapper` — camera/distortion
- `Ellipse` — conic geometry
- `Proposal`, `ProposalConfig`, `ProposalResult` — proposal types

Tiered modules (deliberately not at the root):
- `ringgrid::diagnostics` — opt-in channel from `detect_with_diagnostics`: `DetectionDiagnostics`, `MarkerDiagnostics`, `FitMetrics`, `DecodeMetrics`, `DetectionSource`, `InnerFitReason`, `InnerFitStatus`, `RansacStats`, `StageTimings`
- `ringgrid::codebook` — codebook inspection: `CodebookInfo`, `CodewordMatch`, `codebook_info`, `decode_word`

## Build & Development Commands

```bash
# Build
cargo build --release

# Tests
cargo test --workspace

# Lint & format
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Python bindings (uses uv for package management)
# VIRTUAL_ENV must point at .venv so maturin installs into it, not system Python.
cd crates/ringgrid-py && VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --release && cd ../..

# WASM bindings
wasm-pack build crates/ringgrid-wasm --target web --release

# JSON Schemas (schemas/*.json, shipped in the npm package) — regenerate after
# touching any config / target-spec / render-options type or its doc comments.
# CI runs the --check form.
cargo xtask emit-schemas [--check]

# Run detector
cargo run -p ringgrid-cli --bin ringgrid-dev -- detect --image <path> --target <path> --out <path> --marker-diameter 32.0

# End-to-end synthetic eval (generate → detect → score)
.venv/bin/python tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_run

# Score a single detection
.venv/bin/python tools/score_detect.py --gt <gt.json> --pred <det.json> --gate 8.0 --out <score.json>

# Regenerate embedded codebook/board constants (don't hand-edit these)
.venv/bin/python tools/gen_codebook.py --n 893 --seed 1 --out_json tools/codebook.json --out_rs crates/ringgrid/src/marker/codebook.rs
# tools/board/board_spec.json is the `hex_coded` example recipe rendered with `--name ringgrid_200mm_hex`:
cargo run -p ringgrid --features cli --bin ringgrid -- example --name hex_coded --out /tmp/board.toml
cargo run -p ringgrid --features cli --bin ringgrid -- gen /tmp/board.toml --out tools/board --formats json --name ringgrid_200mm_hex
mv tools/board/target_spec.json tools/board/board_spec.json
cargo build  # rebuild after regenerating
```

## Key Math Modules

- `conic/` — ellipse/conic types (`types.rs`), direct fitting (`fit.rs`), generalized eigenvalue solver (`eigen.rs`), RANSAC (`ransac.rs`)
- `homography/` — DLT with Hartley normalization + RANSAC (`core.rs`), refit & reprojection utilities (`utils.rs`)
- `ring/projective_center.rs` — unbiased center recovery from inner+outer conics

## Center Correction Strategies

Single-choice selector `DetectConfig::circle_refinement` (`CircleRefinementMethod`): `None` | `ProjectiveCenter` (default) | `Photometric`

CLI (both `ringgrid` and `ringgrid-dev`): `--refine {none,projective-center,photometric}`; when absent the config/default applies. `ringgrid-dev` keeps `--circle-refine-method` as an alias (tools use it).

Projective center correction is applied once per marker: before grid labeling for fit-decode markers, and after completion for newly refitted markers only. Logic is in `pipeline/finalize/`.

`Photometric` runs on top of projective-center correction: a per-marker fit of the blurred printed pattern (`ring/photometric_center.rs`, stage wrapper `detector/photometric_refine.rs`). It is opt-in because it costs milliseconds per marker, and it is the pipeline's last stage (`pipeline/photometric.rs`): every public entry point in `pipeline/run.rs` runs it exactly once, on the finished result (after adaptive candidate selection, never per pass/tier/candidate), then refits the final homography and its `RansacStats` from the refined centers via finalize's `phase_final_h`. Its research-only API (per-pixel residuals, covariance diagnostics) is behind the non-default `study` cargo feature.

## Camera / Distortion Support

Optional radial-tangential distortion model (`pixelmap/`). When camera intrinsics are provided:
- Two-pass detection: pass-1 without mapper, pass-2 with mapper using pass-1 seeds
- Final centers mapped back to image space; ellipse/homography in working frame
- `PixelMapper` trait allows custom distortion adapters

Self-undistort mode estimates a division-model distortion correction from detected ellipse edge points, then re-runs detection with the estimated mapper.

## Versioning

The workspace version is defined once in `Cargo.toml` under `[workspace.package]`.
`ringgrid` and `ringgrid-cli` inherit it via `version.workspace = true`.

The three binding crates (`ringgrid-py`, `ringgrid-wasm`, `ringgrid-c`) are
`exclude`d from the workspace and each declares its own `[workspace]` table, so
they **cannot** use `version.workspace = true` — cargo inheritance is only
available to workspace members, and rewriting them to inherit breaks
`cargo metadata` outright. Their package fields are deliberate literals.

When bumping the version, update **seven** locations:
1. `Cargo.toml` — `[workspace.package] version` (single source of truth for workspace crates)
2. `crates/ringgrid-cli/Cargo.toml` — `ringgrid` dependency version
3. `crates/ringgrid-py/Cargo.toml` — `version` field + `ringgrid` dependency version
4. `crates/ringgrid-py/pyproject.toml` — `project.version` field
5. `crates/ringgrid-wasm/Cargo.toml` — `version` field + `ringgrid` dependency version
6. `crates/ringgrid-c/Cargo.toml` — `version` field + `ringgrid` dependency version
7. `crates/ringgrid-c/` packaging: `CMakeLists.txt` `project(... VERSION ...)` **and**
   `vcpkg/vcpkg.json` `version` — neither is covered by a CI guard, so they
   silently drift (they sat at 0.10.1 through the 0.11.0 release)

**Release step for the vcpkg port:** `vcpkg.json`'s version is what the
portfile's released-tarball mode fetches as `v${VERSION}`, so between the bump
and the git tag that mode cannot work. `portfile.cmake` therefore carries
`SHA512 0` between a bump and its tag — it fails and prints the expected hash,
rather than failing on a stale hash from an older release. **After tagging,
replace it with the real hash** (`shasum -a 512` of the
`github.com/VitalyVorobyev/ringgrid/archive/v<version>.tar.gz` tarball). CI is
unaffected either way — both vcpkg jobs build from the local checkout via
`RINGGRID_SOURCE_DIR`. `publish-crates.yml` now guards `CMakeLists.txt` and
`vcpkg.json` against the tag, so those two can no longer drift silently.

CI workflows (`.github/workflows/publish-crates.yml`, `release-pypi.yml`) verify
version consistency between the git tag and the Cargo/pyproject files using
`tomllib` (`publish-crates.yml` guards `ringgrid` + `ringgrid-c`;
`release-pypi.yml` guards `ringgrid` + both `ringgrid-py` files + `ringgrid-c`;
`release-npm.yml` checks the wasm crate inline). The CI scripts also resolve
`version.workspace = true` by falling back to the workspace root, for the two
workspace member crates that do inherit.

After bumping, run `cargo update --workspace` at the root **and** for each of
the three binding crates so their sibling `Cargo.lock` files stay in sync.

**No `exr` in the graph (do not re-add it).** `image` is declared once at the
workspace root with `default-features = false` and an explicit codec set
(`png`, `jpeg`, `tiff`, `bmp`), and `imageproc` with `default-features = false`
(its `default` pulls `image/default` back in). That keeps `exr`, `ravif`/`rav1e`
and the rest of the codec stack out of every artifact — the published library,
the wasm cdylib and the C shared library alike. Restoring `image`'s default
features re-links an AVIF encoder into all of them and reintroduces `exr`, whose
`pulp` dependency breaks the MSVC link of `ringgrid-c`'s test binary under fat
LTO (`LNK1276: invalid directive`) and turns the Windows C/C++ ABI job red.

Check it with `cargo tree --edges normal -i exr`, which must print nothing.
The root `Cargo.lock` still lists `exr`/`ravif` as entries — a lockfile records
the union of the graph over all feature selections, and `imageproc` declares an
optional `image/default` path — but no build enables them, and
`cargo clean -p exr` finds no artifacts. The three binding lockfiles do not
list them at all.

## Feature Flags (ringgrid crate)

- `std` (default) — file I/O only: `from_json_file`, `write_json_file`, `write_target_svg`/`_png`/`_dxf`. Disable for WASM targets.
- WASM crate uses `default-features = false` to exclude `std`. Rendering is unaffected: `render_target_svg`/`_png`/`_dxf` and `render_target_artifacts` (PNG bytes included) work with the feature off, because `png` is a plain dependency — `image`'s png codec links it anyway, so it costs nothing.
- `cli` — the published `ringgrid` binary (`gen` + `detect`). Off by default.
- `study` — research-only photometric study API (`photometric_residuals`, `Detector::detect_*_with_photometric_residuals`). Off by default, not exposed by the bindings or schemas; `ringgrid-cli` enables it for its examples through a dev-dependency only, so `ringgrid-dev` is built without it.

## Decision-Making Rules

- **Never draw conclusions without reproducible evidence.** Every claim about correctness, performance, or behavior must be backed by a test, benchmark, or concrete reproducing example. "I believe this works" is not acceptable — run the code, show the output.
- **When in doubt, stop and ask.** If the task is ambiguous, the expected behavior is unclear, or there are multiple reasonable interpretations, do not guess. Ask the user for clarification before proceeding.
- **Do not speculate about root causes.** When debugging, reproduce the issue first, then form a hypothesis, then verify it with a test. Do not propose fixes based on reading code alone — run the failing case and confirm the diagnosis.

## Conventions

- Algorithms go in `ringgrid`; CLI/file I/O in `ringgrid-cli`
- External JSON uses `serde` structs (see `DetectionResult` in `pipeline/result.rs`)
- Never introduce OpenCV bindings
- `marker/codebook.rs` is generated; board target is runtime JSON (`tools/board/board_spec.json`) — regenerate via the generators / `ringgrid gen`, never hand-edit generated Rust
- Logging via `tracing` crate; control with `RUST_LOG=debug|info|trace`
- `lib.rs` is purely re-exports; type definitions live at their construction sites
- Keep one source of truth for shared configs/defaults: if a config is used by multiple stages, define it once and reuse it directly.
- Avoid mirrored structs (`*Params` vs `*Config`) that carry the same fields/defaults; consolidate into one type unless domains are genuinely different.
- Avoid duplicated thresholds/gating logic across modules; centralize constants and semantics.
- If a translation layer is unavoidable, document why it exists and ensure only one side owns defaults.
- Python package management uses `uv`. Virtual environment at `.venv`. Use `.venv/bin/python` and `.venv/bin/uv pip install` for all Python operations.

## CI / checks

GitHub Actions workflows in `.github/workflows/`:
- `ci.yml` — fmt, clippy, test, WASM build smoke test on push/PR; also the release
  workflow's `cargo test --workspace --all-targets --all-features --release` on ubuntu
  and macOS, which runs the per-platform golden snapshots (`tests/golden_entry_points.rs`)
- `publish-crates.yml` — publish ringgrid to crates.io on tag (ringgrid-cli is intentionally unpublished, installable from source)
- `release-pypi.yml` — build and publish Python wheels to PyPI on tag
- `release-npm.yml` — build and publish WASM npm package on tag (copies `schemas/` into the package via `tools/ci/package_schemas.sh`)
- `release.yml` — create GitHub release
- `publish-docs.yml` — build and deploy mdBook docs
- `audit.yml` — dependency audit

Run locally:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Regression Gate

The `/regression-gate` skill runs the full regression suite (invoke after significant changes):

1. Build release + Python bindings
2. Run 3 synthetic benchmarks + 1 real-world benchmark (rtv3d, local-only):
   - `bash tools/run_reference_benchmark.sh`
   - `bash tools/run_distortion_benchmark.sh`
   - `bash tools/run_blur2_benchmark.sh`
   - `.venv/bin/python tools/run_rtv3d_eval.py` (skipped if `data/rtv3d` absent)
3. Validate against `tools/ci/regression_baseline.json`
4. Run `tools/ci/maintainability_guardrails.py`

Baselines: `tools/ci/regression_baseline.json` (synthetic + rtv3d thresholds), `tools/ci/maintainability_baseline.json` (function size, dead code, doc coverage).
