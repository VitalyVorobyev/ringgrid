# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — v0.4.0

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
