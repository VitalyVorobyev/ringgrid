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

## ID Model
- Backlog ids (`INFRA-011`, `ALGO-014`, `DOCS-003`) are the stable planning ids used in this file.
- Workflow handoff ids (`TASK-012-...`) are execution-trace ids used under `docs/handoffs/`.
- Handoff reports should record both ids when the work came from the backlog.

---

## Active Sprint

_None currently assigned._

## Up Next

_None currently assigned._

## Backlog

_None currently assigned._

## API / Interface Tracking

- Rust API backlog direction: add file-oriented JSON/SVG/PNG target-generation API in `ringgrid` crate using direct board geometry args.
- Python API backlog direction: expose the same target-generation capability in `ringgrid-py` package surface (installed-package usable).
- `DetectConfig` backlog direction: internal Python caching/refactor only; no intentional public behavior changes.
- Codebook backlog direction: default profile unchanged; optional extension profile additive and opt-in.
- Proposal module direction: standalone `proposal/` module with no ringgrid-type dependencies in core API; `ProposalConfig` with unified `min_distance`; `find_ellipse_centers()` entry points; `ProposalResult` with `heatmap` field.

## Acceptance Scenarios (Attached to Tasks)

## Locked Defaults

- ID convention: keep existing `ALGO`/`INFRA` streams and add `DOCS-*` for docs-focused backlog work.
- Target-generation outputs in first milestone: JSON + SVG + PNG.
- API style for target generation: file-oriented for both Rust and Python.
- Codebook extension policy: stable base + optional extension, 16-bit only, target extension size = max feasible.
- Root README policy: user-quickstart-first with links to separate developer/performance docs.

## Historical Notes

- `FEAT-001` and `PERF-001` through `PERF-005` were completed before the current `docs/backlog.md`
  structure was normalized. Canonical closure notes remain in `docs/sessions/` and should be
  treated as done historical work, not open backlog debt.

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| INFRA-015 | 2026-03-22 | infra | Update public API surface and Python bindings for proposal refactor | Updated `lib.rs` re-exports for `proposal` module; `ringgrid-py` bindings renamed `ProposalDiagnostics` → `ProposalResult`, `nms_accumulator` → `heatmap`; added `edge_thinning`/`min_distance`/`proposal_downscale` to Python surface; added `--proposal-downscale` CLI flag; backward-compat `ProposalDiagnostics` alias kept. |
| PERF-006 | 2026-03-22 | perf | Optional proposal-stage image downscaling | Added `ProposalDownscale` enum (`Auto`/`Off`/`Factor(u32)`) on `DetectConfig`; auto factor from `floor(d_min/20)` clamped `[1,4]`; downscale before proposals in `pipeline/run.rs`, upscale coordinates after; downstream stages at full resolution; CLI `--proposal-downscale` flag. |
| ALGO-016 | 2026-03-22 | algo | Canny-style edge thinning for proposal gradient NMS | Added gradient-direction NMS in `proposal/gradient.rs` before strong-edge collection; 4-direction quantization without `atan2` using integer `mag_sq`; `edge_thinning: bool` config flag (default `true` in standalone API); reduces strong-edge count, proportional voting speedup. |
| ALGO-015 | 2026-03-22 | algo | Extract proposal module with standalone ellipse-center API | Moved `detector/proposal.rs` → `proposal/` module (`mod.rs`, `config.rs`, `gradient.rs`, `voting.rs`, `nms.rs`, `tests.rs`); unified `nms_radius` + `min_seed_distance_px` into `min_distance`; entry points `find_ellipse_centers()` / `find_ellipse_centers_with_heatmap()`; renamed `ProposalDiagnostics` → `ProposalResult` with `heatmap` field. |
| INFRA-013 | 2026-03-11 | infra | Finalize `v0.5.0` release readiness and publication | Closed after implementing `INFRA-008` + `ALGO-009/010/011`, aligning the Python package metadata on `0.5.0`, updating the changelog/docs, rerunning the full release baseline (`fmt`, `clippy -D warnings`, workspace tests, rustdoc, doctests, `mdbook`, typing-artifact check, `maturin develop`, Python tests), and cutting the local `v0.5.0` release tag. |
| INFRA-008 | 2026-03-11 | infra | CLI flag to load Brown-Conrady calibration JSON as mapper | Added `--calibration <file.json>` as an additive alternative to inline `--cam-*`; accepts direct `CameraModel` JSON or detector-output wrapper JSON, rejects mixed camera inputs, and documents the JSON path in the CLI guide, tuning guide, and external-mapper docs. |
| ALGO-011 | 2026-03-11 | algo | Enforce inner/outer axis ratio consistency as post-filter | Added a final consistency pass that clears marker IDs whose inner/outer mean-axis ratio deviates more than 25% from the global fit-decoded median, catching residual inner-as-outer failures after collection; added finalize regression tests. |
| ALGO-010 | 2026-03-11 | algo | Pre-screen contaminated outer rays before RANSAC | Outer-edge sampling now rejects rays whose recovered radius differs from the expected outer radius by more than 40% before ellipse fitting, screening inner-ring contamination earlier in the pipeline; added synthetic sampling regression coverage. |
| ALGO-009 | 2026-03-11 | algo | Local affine seed for completion (share affine from ALGO-013) | Completion now fits a local board-mm affine from the 3-4 nearest decoded neighbors and uses that projected seed before falling back to the global homography, improving missing-ID recovery under local projective distortion; added completion seed/fallback unit tests. |
| DOCS-002 | 2026-03-11 | docs | Main README user-first refactor + split dev/perf notes | Accepted via `TASK-016`: refactored the root `README.md` into a user-first landing page centered on install, quickstart, interface routing, and documentation navigation; moved maintainer-focused material into `docs/development.md`; moved scoring/benchmark detail into `docs/performance.md`; added the missing synth/eval Python dependency note for the optional repo workflow; reviewer approval reproduced the full Rust/doc/mdBook/Python validation baseline. |
| DOCS-001 | 2026-03-11 | docs | ringgrid-py README: complete DetectConfig field guide | Accepted via `TASK-015`: expanded `crates/ringgrid-py/README.md` with a full Python-facing `DetectConfig` field guide covering sections, top-level controls, aliases, derived defaults, and tuning notes; exposed `DecodeConfig.codebook_profile` through the typed Python/stub surface with backward-compatible defaulting for legacy payloads; added decode-parity and README surface-drift regression coverage; reviewer approval reproduced the full Rust/doc/Python validation baseline. |
| ALGO-014 | 2026-03-11 | algo | Optional extended codebook mode beyond 893, stable base IDs | Accepted via `TASK-014`: added explicit `base`/`extended` codebook profiles while preserving shipped IDs `0..892`, threaded profile selection through decode and profile-local exact-match gates, regenerated artifacts/docs/CLI for the approved additive `extended` profile (`2180` total, `1287` appended), fixed loaded-baseline seed provenance when `--base_json` is used, added Rust/Python regression coverage for inverted-polarity stability and generator provenance, and reviewer approval reproduced the full local CI baseline plus the former polarity and mismatched-seed blocker repros. |
| INFRA-012 | 2026-03-10 | infra | Create dedicated tools/gen_target.py | Accepted via `TASK-013`: added thin repo-root `tools/gen_target.py` over the shipped Python target-generation API, added deterministic subprocess parity/error coverage in `tools/tests/test_gen_target.py`, updated README/mdbook guidance to route print-only users to the dedicated tool while preserving `gen_synth.py` for synth workflows, and reviewer approval reproduced the dedicated tool tests, Rust target-generation integration tests, `mdbook build`, and invalid-DPI CLI failure behavior. |
| INFRA-011 | 2026-03-10 | infra | Expose target generation in ringgrid-py (file-oriented) | Accepted via `TASK-012`: exposed installed-package Python target-generation APIs on `BoardLayout` (`to_spec_json`, `write_svg`, `write_png`) over the Rust engine, refreshed mutable spec fields before emitting outputs, added JSON/SVG/PNG parity plus mutation-regression coverage, and reviewer approval reproduced the typing-artifact check and focused Python target-generation tests. |
| DOCS-003 | 2026-03-09 | docs | Reconcile codebook docs/invariants with shipped artifacts | Accepted via `TASK-011`: aligned `README`, `book`, and `docs/decisions` with shipped codebook artifacts (`n=893`, `bits=16`, `min cyclic Hamming=2`, `seed=1`), corrected the `codebook-info` example and generated Rust artifact path, added provenance/reproducibility notes, and reviewer approved with only optional broader DEC-011 boundary cleanup follow-up. |
| INFRA-010 | 2026-03-08 | infra | Add Rust target-generation API in ringgrid crate (file-oriented) | Accepted: added additive `BoardLayout`-centered Rust target-generation API in `ringgrid` for canonical `ringgrid.target.v3` JSON plus printable SVG/PNG output, added compact committed JSON/SVG/PNG parity fixtures and integration coverage, and fixed `write_target_png` to always emit PNG bytes with DPI-preserving `pHYs` metadata; reviewer approval reproduced via `cargo test -p ringgrid --test target_generation` and `cargo test -p ringgrid` |
| INFRA-009 | 2026-03-08 | infra | Eliminate full JSON snapshot churn in ringgrid-py DetectConfig | Accepted: added lazy resolved-config caching in `ringgrid-py` `DetectConfig`, kept native refresh for whole-section/marker-scale updates, added cache/parity regression tests plus `benchmark_detect_config.py`, and verified hot getter/setter paths exceed the `>=4x` target (`~5.7x` to `24.7x`) while preserving `to_dict()`/overlay semantics |
| ALGO-003/004/005/006/007/008 | 2026-03-01 | algo | Detection provenance, fit diagnostics, and confidence refinement | `DetectionSource` enum (FitDecoded\|Completion\|SeededPass) on `DetectedMarker`; `h_reproj_err_px` and `radii_std_outer_px` in `FitMetrics`; outer-radii scatter gate in completion (max_radii_std_ratio=0.35); H-reproj confidence soft-penalty (alpha=0.2, `confidence *= 1/(1 + alpha * err)`); `n_decode_mismatch` in `CompletionStats` |
| ALGO-013 | 2026-02-20 | algo | Structural ID verification and correction | `detector/id_correction.rs` (new); `IdCorrectionConfig` added to `detector/config.rs` + `DetectConfig`; stage wired in `pipeline/finalize.rs` after projective center, before global filter; `--id-correct` CLI flag; `"id_correction"` JSON config section; 7 unit tests covering hex neighbors, affine fit, pitch estimation, wrong-ID correction |
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
