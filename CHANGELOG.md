# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Changes on `main` since `v0.5.0`.

### Added

**Rust CLI target generation**
- Added `ringgrid gen-target` to generate canonical `board_spec.json` plus
  printable SVG/PNG artifacts directly from board geometry arguments.
- The Rust CLI path is now documented as equivalent to the Rust API writers and
  the repo-local `tools/gen_target.py` workflow for the same geometry/print
  settings.
- Added `gentarget.sh` as a repo helper for common target-generation inputs.

**Proposal diagnostics as public API**
- Extracted proposal generation into a standalone public `ringgrid::proposal`
  module with `find_ellipse_centers(...)` and
  `find_ellipse_centers_with_heatmap(...)`.
- Added detector-backed Rust helpers `Detector::propose(...)` and
  `Detector::propose_with_heatmap(...)`, plus size-aware free functions that
  derive proposal tuning from board geometry and marker scale hints.
- Added Python proposal diagnostics surfaces:
  `Detector.propose(...)`, `Detector.propose_with_heatmap(...)`,
  module-level `ringgrid.propose(...)`, `ringgrid.propose_with_heatmap(...)`,
  and the `ProposalResult` payload with proposals plus accumulator heatmap.
- Added `ringgrid.viz.plot_proposal_diagnostics(...)` and the repo-local
  `tools/plot_proposal.py` workflow for overlaying proposal peaks and heatmaps.
- Added CLI output support for proposal diagnostics via
  `ringgrid detect --include-proposals`, which writes top-level
  `proposal_frame`, `proposal_count`, and `proposals` fields.

**Performance instrumentation**
- Added a dedicated `detect_fixture` benchmark and checked-in detection fixture
  data for proposal-stage performance work and reproducible measurements.

### Changed

**Proposal stage architecture**
- Refactored the old monolithic proposal implementation out of
  `detector/proposal.rs` into a dedicated `proposal/` module split across
  `config`, `gradient`, `voting`, `nms`, and focused tests.
- This separation makes proposal generation usable as a standalone stage without
  routing through the full detection pipeline and clarifies the ownership
  boundary between pass-1 center finding and later fit/decode stages.

**Proposal-stage tuning and throughput**
- Optimized the proposal voting hot path and added benchmark coverage for the
  dominant proposal-stage workload.
- Added `ProposalDownscale` to `DetectConfig` so pass-1 proposals can run on a
  downscaled image while all downstream fitting and decoding stays at full
  resolution.
- Surfaced proposal downscale and related proposal diagnostics controls through
  the CLI/config plumbing, making proposal-cost tuning explicit instead of
  implicit.

**Documentation and interface framing**
- Reworked the root `README.md` into a user-oriented entry point that routes
  readers to the CLI guide, Rust crate docs, Python package docs, output-format
  reference, and performance notes.
- Expanded the Rust and Python READMEs so target generation, proposal-only
  diagnostics, and output-contract references are documented consistently across
  interfaces.

### Documentation

- Added a dedicated mdBook page for the detection output schema, covering
  `DetectionResult`, top-level frame metadata, optional homography/RANSAC,
  CLI-only `camera`, and CLI-only proposal diagnostics fields.
- Added a dedicated mdBook page for proposal diagnostics, including standalone
  Rust/Python proposal APIs, the accumulator heatmap contract, and the plotting
  workflow.
- Extended the CLI guide, fast-start docs, and target-generation docs to cover
  `ringgrid gen-target`, printable artifact generation parity, and the
  `--include-proposals` detection-output option.
- Added `docs/proposal-performance-analysis.md` to capture the current proposal
  hotspot profile, the rationale for the existing algorithm, and the most
  credible future alternatives.

### CI

- Modernized GitHub Actions workflows to current stable major versions for core
  checkout, Python setup, artifact upload/download, and GitHub release steps.
- Replaced hand-rolled tool installation in CI with maintained installer
  actions, added tighter default permissions and job timeouts, and added
  Dependabot updates for GitHub Actions dependencies.

## [0.5.0] — 2026-03-12

### Added

**Rust target generation**
- `BoardLayout` now exposes file-oriented target generation for canonical `ringgrid.target.v4`
  JSON plus printable SVG/PNG output from the Rust crate.
- Added deterministic JSON/SVG/PNG fixtures and integration coverage for the new generator
  surface.

**Python target generation**
- `ringgrid-py` now exposes installed-package target-generation helpers on `BoardLayout`:
  `to_spec_json()`, `write_svg()`, and `write_png()`.
- Added dedicated repo CLI `tools/gen_target.py` for print-target generation without going
  through the synth pipeline.

**Codebook profiles**
- Added opt-in `extended` codebook profile while preserving shipped `base` IDs `0..892`.
- Surfaced profile selection through CLI helpers, Rust/Python config, and regenerated
  codebook artifacts/docs.

**CLI calibration loading**
- `ringgrid detect --calibration <file.json>` now loads a Brown-Conrady `CameraModel`
  from JSON as an additive alternative to inline `--cam-*` flags.
- The CLI accepts either the direct serde camera-model shape or detector-output JSON with
  a top-level `camera` wrapper.

**Rust CLI target generation**
- Added `ringgrid gen-target` to generate canonical `board_spec.json` plus
  printable SVG/PNG directly from the Rust CLI.
- The Rust CLI generation command matches the dedicated `tools/gen_target.py`
  geometry/print options and writes the same artifact set.

### Changed

**Dependency upgrades**
- `rand` upgraded to `0.10`.
- `criterion` upgraded to `0.8`.
- `nalgebra` upgraded to `0.34`.

**`rand` API migration**
- Replaced deprecated/removed `Rng::gen_range(...)` calls with
  `RngExt::random_range(...)` across library code, tests, and benchmarks.
- Kept deterministic behavior intact by preserving all explicit `StdRng::seed_from_u64(...)`
  seeds in test and benchmark fixtures.

**`criterion` API migration**
- Replaced deprecated `criterion::black_box(...)` with `std::hint::black_box(...)`
  in benchmark code.
- Bench harness remains compatible with `criterion_group!` / `criterion_main!`.

**Tooling compatibility**
- Updated benchmark/test support code to compile cleanly with `-D warnings`
  under the new dependency set.

**Python config performance**
- `ringgrid-py` `DetectConfig` now caches resolved JSON snapshots for hot getter/setter paths
  while preserving `to_dict()` output and overlay semantics.

**Completion and fit robustness**
- Completion now seeds missing IDs from a local board-mm affine fit when 3-4 nearby decoded
  neighbors are available, falling back to the global homography otherwise.
- Outer-edge sampling now drops rays whose recovered radius differs from the expected outer
  radius by more than 40%, screening inner-ring contamination before ellipse fitting.
- Final marker cleanup now clears IDs whose inner/outer mean-axis ratio deviates strongly from
  the global fit-decoded median, catching residual inner-as-outer failures missed earlier in
  the pipeline.

**Documentation and onboarding**
- Refactored the root `README.md` into a user-first landing page with quickstart and interface
  routing.
- Split maintainer-focused material into `docs/development.md` and scoring/benchmark guidance
  into `docs/performance.md`.
- Expanded `crates/ringgrid-py/README.md` with a full Python-facing `DetectConfig` field guide.
- Reconciled codebook docs and decision records with shipped baseline invariants and the new
  `extended` profile contract.

**Target-generation documentation**
- Reworked README and mdBook target-generation guidance so Rust API, Rust CLI,
  and Python script workflows are documented as equivalent paths over the same
  target-generation engine.
- Canonical target JSON moved to `ringgrid.target.v4` with explicit
  `marker_ring_width_mm`; the runtime loader no longer accepts `v3`.

### Fixed

- `hotpaths` benchmark build now passes with latest `rand`/`criterion` APIs.
- `tools/gen_codebook.py --base_json ...` now preserves source-seed provenance in emitted JSON
  and Rust metadata.
- Legacy Python decode-config payloads missing `codebook_profile` now default to `"base"`.
- The `extended` codebook appendix excludes new complement-collision exact matches under
  inverted polarity.

## [0.4.0] — 2026-03-01

### Added

**Detection provenance**
- `DetectionSource` enum (`FitDecoded | Completion | SeededPass`) on `DetectedMarker`
  — tags the pipeline stage that produced each marker; serialised as `"fit_decoded"`,
  `"completion"`, `"seeded_pass"` in JSON output.

**Fit diagnostics in `FitMetrics`**
- `h_reproj_err_px: Option<f32>` — reprojection error (pixels) between the marker center
  and its board position projected through the final homography; populated for all markers
  with a valid decoded ID after the final H refit.
- `radii_std_outer_px: Option<f32>` — standard deviation of per-ray outer radii; high
  values (> 30 % of the mean) indicate inner/outer edge contamination.

**Completion quality gates**
- `CompletionParams::max_radii_std_ratio` (default `0.35`) — rejects completion candidates
  whose outer-radii coefficient of variation exceeds the threshold; prevents contaminated
  fits where some rays sample the inner ring.
- `CompletionStats::n_decode_mismatch` — counts how many completion attempts produced a
  valid fit that decoded to a different ID than the expected board position.

**Confidence refinement**
- `DetectConfig::h_reproj_confidence_alpha` (default `0.2`) — after annotating
  `h_reproj_err_px`, applies a soft penalty: `confidence *= 1 / (1 + alpha × err)`.
  A 5 px reprojection error reduces confidence by ≈ 50 %. Set to `0.0` to disable.

**Documentation**
- `docs/` — adaptive-scale detection guide and fast-start usage guide.

**Python API**
- Adaptive tier introspection: `ScaleTiers.tiers` property exposes the configured
  `ScaleTier` list from Python.
- Simplified `detect_adaptive` call: keyword-argument defaults mirror the Rust API
  without requiring explicit tier construction for common cases.

---

## [0.3.0] — 2026-03-01

### Added

**Adaptive multi-scale detection**
- `Detector::detect_adaptive(image, config)` — automatically estimates dominant marker
  radius via a ring angular-variance sweep (`scale_probe`), selects the appropriate
  scale tier, and runs the full pipeline.
- `Detector::detect_adaptive_with_hint(image, config, hint)` — same as above with a
  user-supplied radius hint bypassing the probe.
- `Detector::detect_multiscale(image, tiers, config)` — runs one fit-decode pass per
  `ScaleTier` and merges results with size-consistency-aware NMS.
- `ScaleTier` — single-scale configuration (min/max diameter in px).
- `ScaleTiers` — ordered collection of tiers; built-in presets:
  - `four_tier_wide` — `[8–24, 20–60, 50–130, 110–220]` px (widest coverage)
  - `two_tier_standard` — `[14–42, 36–100]` px
  - `single` — single tier matching `MarkerScalePrior`
  - `from_detected_radii` — derives tiers from a pass-1 radius histogram

**Scale probe**
- `pipeline/scale_probe.rs` — ring angular-variance sweep over 20 geometric radius
  candidates (4–110 px) at top-K gradient proposals; code-band midpoint at ≈ 0.8×
  outer radius; results feed `ScaleTiers::from_detected_radii`.

**Multi-scale pipeline split**
- `finalize_premerge` — projective-center correction per tier; returns
  `Vec<DetectedMarker>` without global filter or completion.
- `merge_multiscale_markers` — size-consistency-aware NMS across all tier outputs;
  prefers markers whose outer radius matches the k = 6 hex-lattice neighbor median;
  confidence breaks ties.
- `finalize_postmerge` — global filter + H-guided completion + final H refit, run
  exactly once on the merged pool.

**Python**
- Python 3.14 support.

### Changed

- `MarkerScalePrior` default updated from `[20, 56]` to `[14, 66]` px,
  yielding +11.4 % on the rtv3d benchmark dataset.

---

## [0.2.5] — 2026-02-28

### Fixed

- CI: pinned manylinux Python interpreter for maturin wheel builds.
- CI: removed unsupported `replace()` call from PyPI release workflow.

---

## [0.2.2] — 2026-02-28

### Changed

- Python docstrings polished with structured parameter and return documentation.
- Example image paths updated to reference `testdata/` directory.

---

## [0.2.1] — 2026-02-22

### Fixed

- CI: maturin `develop` command now runs inside a virtual environment.
- CI: miscellaneous Python release pipeline fixes.

---

## [0.2.0] — 2026-02-22

### Added

- **Python bindings** via PyO3 and maturin; published to PyPI as `ringgrid`.
- `Detector`, `DetectConfig`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, and
  `DetectionResult` exposed to Python with full docstrings.

---

## [0.1.0] — 2026-02-22

Initial release. Full detection pipeline for dense coded ring calibration targets on a
hex lattice.

### Detection pipeline

- **Proposal** — Scharr gradient voting + NMS for candidate center extraction.
- **Outer estimate** — radial intensity profile peak-finding for radius hypotheses.
- **Outer fit** — Fitzgibbon direct ellipse fitting + RANSAC (Sampson distance inlier
  metric).
- **Decode** — 16-sector ring code sampling → match against 893-codeword codebook with
  cyclic rotation search and Hamming margin.
- **Inner fit** — inner ring ellipse from radial edge samples; configurable quality gates.
- **Dedup** — spatial NMS + per-ID deduplication.
- **Projective center** — unbiased center recovery from inner + outer conic pair.
- **Structural ID correction** — BFS hex-neighborhood consensus with local 2-D affine
  fitting; corrects swapped IDs and clears unverifiable detections.
- **Global filter** — RANSAC homography (DLT + Hartley normalization) from decoded
  markers; outlier rejection gate in pixels.
- **H-guided completion** — conservative fits at board positions projected by H but not
  yet detected; optional perfect-decode gate for high-distortion setups.
- **Final H refit** — homography updated from the full corrected marker pool.

### Result types

- `DetectedMarker` — center (image px), optional inner/outer ellipses, optional edge
  point arrays, fit metrics, decode metrics, board-space coordinates (`board_xy_mm`).
- `FitMetrics` — RANSAC inlier ratios, RMS Sampson residuals, angular gap, inner fit
  status and rejection reason, neighbor radius ratio, inner theta consistency.
- `DecodeMetrics` — observed word, best ID, best rotation, Hamming distance, margin,
  decode confidence.
- `DetectionResult` — detected markers, homography, RANSAC stats, image size.

### Configuration

- `DetectConfig` — unified config struct; all sub-configs (proposal, edge sampling,
  decode, inner/outer fit, projective center, completion, homography RANSAC,
  self-undistort, ID correction) accessible as named fields.
- `MarkerScalePrior` — min/max diameter range in working-frame pixels.
- `BoardLayout` — hex-lattice geometry loader from JSON.
- `MarkerSpec` — inner/outer radius ratio and sector geometry.

### Camera / distortion support

- `CameraModel` + `CameraIntrinsics` — optional radial-tangential distortion model.
- Two-pass detection: pass-1 without mapper; pass-2 with mapper using pass-1 seeds.
- `SelfUndistortConfig` — estimates a single-parameter division-model distortion from
  detected ellipse edge points; re-runs detection with the estimated mapper.

### Infrastructure

- Pure Rust — no OpenCV bindings.
- `tracing`-based structured logging (`RUST_LOG=debug|info|trace`).
- Typed reject-reason enums throughout the pipeline (no stringly-typed errors).
- Baseline-locked maintainability guardrails enforced in CI.
- CLI binary (`ringgrid detect`) with `--image`, `--out`, `--marker-diameter`,
  `--circle-refine-method`, and camera calibration flags.
