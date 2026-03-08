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
| INFRA-010 | todo | P0 | infra | Add Rust target-generation API in ringgrid crate (file-oriented) | infra | Add file-oriented Rust API in `ringgrid` for target generation from direct board geometry args (`pitch_mm`, `rows`, `long_row_cols`, `marker_outer_radius_mm`, `marker_inner_radius_mm`) and write JSON/SVG/PNG; DXF deferred. Acceptance: output parity tests against current `tools/gen_synth.py` geometry semantics. |
| DOCS-003 | todo | P0 | docs | Reconcile codebook docs/invariants with shipped artifacts | docs | Align docs (`README`, `book`, `docs/decisions`) with shipped codebook facts (`n=893`, current min cyclic distance), and document generation provenance/reproducibility expectations. Acceptance: consistency pass across docs and constants in repo. |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| INFRA-011 | todo | P1 | infra | Expose target generation in ringgrid-py (file-oriented) | infra | Add Python API wrappers over Rust target-generation API and export them in package surface; must work from installed package (not repo-tools-only). Acceptance: JSON/SVG/PNG parity tests for identical board args. |
| INFRA-012 | todo | P1 | infra | Create dedicated tools/gen_target.py | infra | Add thin dedicated script with direct board args (`pitch_mm`, `rows`, `long_row_cols`, `marker_outer_radius_mm`, `marker_inner_radius_mm`) that emits JSON/SVG/PNG via shared generation API. Acceptance: script outputs match Rust/Python generation for same inputs. |
| ALGO-014 | todo | P1 | algo | Optional extended codebook mode beyond 893, stable base IDs | algo | Keep default 16-bit base profile stable (IDs `0..892` unchanged), define additive opt-in extension profile, and compute max-feasible extension size under 16-bit constraints. Acceptance: compatibility tests prove default decode identity and extension-mode isolation. |
| DOCS-001 | todo | P1 | docs | ringgrid-py README: complete DetectConfig field guide | docs | Expand `crates/ringgrid-py/README.md` with full `DetectConfig` coverage (all sections/parameters), defaults, practical ranges, and tuning hints. Acceptance: README field guide aligns with current Python/Rust config surface. |
| DOCS-002 | todo | P1 | docs | Main README user-first refactor + split dev/perf notes | docs | Refocus root `README.md` on workspace users (install/quickstart/usage), move deep developer/performance material into dedicated docs, and add links from README. Acceptance: user-first README with clear links to developer/performance docs. |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| ALGO-009 | todo | P2 | algo | Local affine seed for completion (share affine from ALGO-013) | algo | For each missing board ID, reuse `fit_local_affine` from `detector/id_correction.rs`; find 3-4 nearest decoded neighbors in board-mm space, use affine-projected position as seed; fall back to global H with fewer than 3 neighbors; absorbs non-radial Scheimpflug residuals |
| INFRA-008 | todo | P2 | infra | CLI flag to load Brown-Conrady calibration JSON as mapper | infra | Add `--calibration <file.json>` to ringgrid-cli; deserialize `RadialTangentialDistortion` from JSON; construct `CameraModel`-based `PixelMapper`; `RadialTangentialDistortion` struct already exists in `pixelmap/distortion.rs` |
| ALGO-010 | todo | P3 | algo | Pre-screen contaminated outer rays before RANSAC | algo | In outer edge collection, filter rays where `\|r_ray - r_expected\| > 0.4 * r_expected` before RANSAC; discards rays that landed on wrong ring, complementing RANSAC's Sampson-distance inlier gate |
| ALGO-011 | todo | P3 | algo | Enforce inner/outer axis ratio consistency as post-filter | algo | After all markers collected, compute global median `(inner.mean_axis / outer.mean_axis)` from fit-decoded markers; flag or reject markers where ratio deviates >25%; catches cases where "outer" fit anchored to inner ring |

## API / Interface Tracking

- Rust API backlog direction: add file-oriented JSON/SVG/PNG target-generation API in `ringgrid` crate using direct board geometry args.
- Python API backlog direction: expose the same target-generation capability in `ringgrid-py` package surface (installed-package usable).
- `DetectConfig` backlog direction: internal Python caching/refactor only; no intentional public behavior changes.
- Codebook backlog direction: default profile unchanged; optional extension profile additive and opt-in.

## Acceptance Scenarios (Attached to Tasks)

- `INFRA-009`: microbenchmark for `DetectConfig` hot getters/setters must show at least 4x improvement; overlay and `to_dict()` parity must hold.
- `INFRA-010`/`INFRA-011`/`INFRA-012`: JSON/SVG/PNG generation parity tests against current `gen_synth.py` geometry semantics for identical board args.
- `ALGO-014`: compatibility tests show IDs `0..892` decode identically in default mode; extension profile remains explicitly opt-in.
- `DOCS-001`/`DOCS-002`/`DOCS-003`: documentation consistency pass must match shipped constants and supported API surfaces.

## Locked Defaults

- ID convention: keep existing `ALGO`/`INFRA` streams and add `DOCS-*` for docs-focused backlog work.
- Target-generation outputs in first milestone: JSON + SVG + PNG.
- API style for target generation: file-oriented for both Rust and Python.
- Codebook extension policy: stable base + optional extension, 16-bit only, target extension size = max feasible.
- Root README policy: user-quickstart-first with links to separate developer/performance docs.

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
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
