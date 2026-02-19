# Backlog

## Status Values
- `todo` — not started
- `in-progress` — actively being worked
- `blocked` — waiting on something
- `done` — completed

## Priority Values
- `P0` — blocking release or correctness
- `P1` — next up
- `P2` — planned
- `P3` — someday

---

## Active Sprint

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| — | — | — | — | — | — | — |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| ALGO-003 | todo | P1 | algo | Add DetectionSource flag to DetectedMarker | infra | Enum `FitDecoded\|Completion`; set at construction sites in `pipeline/fit_decode.rs` and `detector/completion.rs`; serialize to JSON; enables separating FP sources in analysis |
| ALGO-004 | todo | P1 | algo | Expose per-marker H-reprojection error in DetectedMarker output | algo | Add `h_reproj_err_px: Option<f32>` to `DetectedMarker`; populate in `pipeline/finalize.rs` after final H refit using `homography_reprojection_error` already in `homography/core.rs:68-83`; serialize to JSON |
| ALGO-005 | todo | P1 | algo | Add outer-radii scatter gate to completion | algo | Compute `std_dev(edge.outer_radii)` after `compute_candidate_quality`; add `max_radii_std_ratio: f32 = 0.35` to `CompletionParams`; reject with new `CompletionGateRejectReason::RadiiScatterTooHigh`; catches inner/outer edge contamination |
| ALGO-006 | todo | P1 | algo | Incorporate H-reproj-error into final confidence (soft penalty) | algo | After computing `h_reproj_err_px` per marker in finalize, apply `confidence *= 1/(1 + alpha * reproj_err)` (alpha ≈ 0.2); add `h_reproj_confidence_alpha: f32` to config; most impactful confidence fix for geometrically inconsistent markers |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| ALGO-007 | todo | P2 | algo | Expose outer_radii std dev in FitMetrics | algo | Add `radii_std_outer_px: Option<f32>` to `FitMetrics` in `detector/marker_build.rs`; compute `std_dev(outer_radii)` from `EdgeSampleResult.outer_radii`; high std dev (>30% of mean) fingerprints inner/outer edge contamination |
| ALGO-008 | todo | P2 | algo | Log and count completion decode mismatches in CompletionStats | algo | Add `n_decode_mismatch: usize` to `CompletionStats`; increment when `check_decode_gate` returns `Some`; promote existing `tracing::debug!` to `tracing::info!` |
| ALGO-009 | todo | P2 | algo | Local affine seed prediction for completion | algo | For each missing board ID, find 3-4 nearest decoded neighbors in board-mm space, fit local affine (2×6 least-squares, ≥3 correspondences), use affine-projected position as seed instead of global H; fall back to global H if fewer than 3 neighbors; absorbs non-radial Scheimpflug residuals |
| INFRA-008 | todo | P2 | infra | CLI flag to load Brown-Conrady calibration JSON as mapper | infra | Add `--calibration <file.json>` to ringgrid-cli; deserialize `RadialTangentialDistortion` from JSON; construct `CameraModel`-based `PixelMapper`; `RadialTangentialDistortion` struct already exists in `pixelmap/distortion.rs` |
| ALGO-010 | todo | P3 | algo | Pre-screen contaminated outer rays before RANSAC | algo | In outer edge collection, filter rays where `\|r_ray - r_expected\| > 0.4 * r_expected` before RANSAC; discards rays that landed on wrong ring, complementing RANSAC's Sampson-distance inlier gate |
| ALGO-011 | todo | P3 | algo | Enforce inner/outer axis ratio consistency as post-filter | algo | After all markers collected, compute global median `(inner.mean_axis / outer.mean_axis)` from fit-decoded markers; flag or reject markers where ratio deviates >25%; catches cases where "outer" fit anchored to inner ring |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| ALGO-012 | 2026-02-19 | algo | Fix confidence formula ceiling and harden decode gates | Fixed `conf_margin = margin / 3.0` → `/ CODEBOOK_MIN_CYCLIC_DIST` in `marker/codec.rs` (perfect decode now scores 1.0); raised `DEFAULT_MIN_DECODE_CONFIDENCE` 0.15 → 0.30; added `min_decode_margin: u8 = 1` hard gate with `MarginTooLow` reject reason in `marker/decode.rs`; added `miss_confidence_factor: f32 = 0.7` (inner-fit miss penalty) to `InnerFitConfig`; added RMS Sampson soft penalty `1/(1+rms)` in `pipeline/fit_decode.rs`; added `require_perfect_decode` gate to `CompletionParams` and `--complete-require-perfect-decode` CLI flag |
| INFRA-007 | 2026-02-19 | infra | Add maintainability guardrails to CI | Accepted: added CI guardrail runner (`tools/ci/maintainability_guardrails.py`) with baseline policy (`tools/ci/maintainability_baseline.json`) enforcing no new dead-code allowances in hot modules, no growth/new oversized hotspot functions (threshold 120, baseline-locked), and rustdoc missing-docs warning non-regression; wired static + rustdoc guardrails into `.github/workflows/ci.yml`; reported fmt/clippy/tests and guardrail checks pass |
| ALGO-002 | 2026-02-19 | algo | Decompose projective-center solver into testable stages | Accepted: decomposed `ring_center_projective_with_debug` in `ring/projective_center.rs` into explicit stages (conic preparation, eigenvalue separation, projective point candidate generation, candidate scoring, and best-candidate selection) while preserving solver policy and output semantics; reported fmt/clippy/tests pass and synth eval aggregate identical to baseline |
| BUG-002 | 2026-02-19 | bug | Make seed proposal selection confidence-ordered and deterministic | Accepted: `DetectionResult::seed_proposals` now ranks seeds explicitly by confidence with deterministic tie-breaking (decoded status/id, center coordinates, source index), applies truncation post-ranking, and filters non-finite centers; added regression tests and reported clippy/tests pass with synth eval aggregate identical to baseline |
| INFRA-006 | 2026-02-19 | infra | Split outer-fit responsibilities and remove hardcoded solver knobs | Accepted: decomposed outer-fit into focused modules (`outer_fit/mod.rs`, `sampling.rs`, `solver.rs`, `scoring.rs`), removed local hardcoded solver literals, added shared `OuterFitConfig` in `DetectConfig` with behavior-preserving defaults, preserved completion-path policy behavior and reject semantics; reported fmt/clippy/tests pass and synth eval deltas at zero versus baseline |
| INFRA-005 | 2026-02-19 | infra | Harden BoardLayout invariants and loading errors | Accepted: made `BoardLayout` marker storage private with read-only accessors, removed public index-repair footgun, added typed `BoardLayoutValidationError`/`BoardLayoutLoadError`, and propagated typed loader errors through detector target-file constructors; reported fmt/clippy/tests pass and synth eval deltas at zero versus baseline |
| INFRA-004 | 2026-02-17 | infra | Deduplicate homography correspondence/stats utilities | Accepted: added shared internal homography correspondence/stat utilities in `homography/correspondence.rs`; rewired `global_filter`, `homography::utils`, and self-undistort objective path; preserved behavior with explicit frame semantics and duplicate-ID policy controls; reported fmt/clippy/tests pass and synth eval deltas at zero versus baseline |
| BUG-001 | 2026-02-17 | bug | Fix decode config drift and expose hidden thresholds | Accepted: aligned `DecodeConfig` rustdoc with runtime defaults via shared constants; promoted decode hidden thresholds to explicit config (`min_decode_contrast`, `threshold_max_iters`, `threshold_convergence_eps`) with serde-safe defaults; added deterministic threshold-loop and compatibility tests; reported fmt/clippy/tests pass and synth eval deltas within gate |
| INFRA-003 | 2026-02-17 | infra | Replace stringly reject reasons with typed enums | Accepted: replaced string-based reject/failure channels in `decode`/`inner_fit`/`outer_fit`/`completion`/`fit_decode` with typed enums + structured context; aggregation now keyed by typed reason codes; reported `fmt`/`clippy -D warnings`/`test` pass and blur=3 eval delta is zero versus baseline |
| ALGO-001 | 2026-02-16 | algo | Unify duplicated radial-estimator core (inner/outer) | Accepted: introduced shared `ring::radial_estimator` core and rewired both `inner_estimate` and `outer_estimate`; maintainability objective met with no material accuracy/perf regression in validation artifacts; local `fmt`/`clippy -D warnings`/`test` passed |
| INFRA-002 | 2026-02-16 | infra | Decompose self-undistort into focused modules | Completed: split `pixelmap::self_undistort` into focused modules (`config`, `result`, `objective`, `optimizer`, `policy`, `estimator`, tests), preserved public entrypoint surface, and passed fmt/clippy/tests + required blur3/reference/distortion validation scripts |
