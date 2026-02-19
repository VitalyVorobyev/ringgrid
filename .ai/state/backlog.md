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
| BUG-002 | in-progress | P2 | bug | Make seed proposal selection confidence-ordered and deterministic | Algorithm Engineer | Dispatched for bug-fix triage/implementation; make `DetectionResult::seed_proposals` explicitly confidence-ranked with deterministic tie-breaking before `max_seeds` truncation |

## Up Next

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| ALGO-002 | todo | P2 | algo | Decompose projective-center solver into testable stages | Algorithm Engineer | `ring_center_projective_with_debug` in `crates/ringgrid/src/ring/projective_center.rs` is a large mixed-responsibility routine; split candidate generation/scoring/selection for clarity and safer evolution |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| INFRA-007 | todo | P2 | infra | Add maintainability guardrails to CI | Pipeline Architect | Add lint/doc complexity policy (function-size hotspots, rustdoc coverage, forbid new `allow(dead_code)` in hot modules) to prevent quality regressions |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| INFRA-006 | 2026-02-19 | infra | Split outer-fit responsibilities and remove hardcoded solver knobs | Accepted: decomposed outer-fit into focused modules (`outer_fit/mod.rs`, `sampling.rs`, `solver.rs`, `scoring.rs`), removed local hardcoded solver literals, added shared `OuterFitConfig` in `DetectConfig` with behavior-preserving defaults, preserved completion-path policy behavior and reject semantics; reported fmt/clippy/tests pass and synth eval deltas at zero versus baseline |
| INFRA-005 | 2026-02-19 | infra | Harden BoardLayout invariants and loading errors | Accepted: made `BoardLayout` marker storage private with read-only accessors, removed public index-repair footgun, added typed `BoardLayoutValidationError`/`BoardLayoutLoadError`, and propagated typed loader errors through detector target-file constructors; reported fmt/clippy/tests pass and synth eval deltas at zero versus baseline |
| INFRA-004 | 2026-02-17 | infra | Deduplicate homography correspondence/stats utilities | Accepted: added shared internal homography correspondence/stat utilities in `homography/correspondence.rs`; rewired `global_filter`, `homography::utils`, and self-undistort objective path; preserved behavior with explicit frame semantics and duplicate-ID policy controls; reported fmt/clippy/tests pass and synth eval deltas at zero versus baseline |
| BUG-001 | 2026-02-17 | bug | Fix decode config drift and expose hidden thresholds | Accepted: aligned `DecodeConfig` rustdoc with runtime defaults via shared constants; promoted decode hidden thresholds to explicit config (`min_decode_contrast`, `threshold_max_iters`, `threshold_convergence_eps`) with serde-safe defaults; added deterministic threshold-loop and compatibility tests; reported fmt/clippy/tests pass and synth eval deltas within gate |
| INFRA-003 | 2026-02-17 | infra | Replace stringly reject reasons with typed enums | Accepted: replaced string-based reject/failure channels in `decode`/`inner_fit`/`outer_fit`/`completion`/`fit_decode` with typed enums + structured context; aggregation now keyed by typed reason codes; reported `fmt`/`clippy -D warnings`/`test` pass and blur=3 eval delta is zero versus baseline |
| ALGO-001 | 2026-02-16 | algo | Unify duplicated radial-estimator core (inner/outer) | Accepted: introduced shared `ring::radial_estimator` core and rewired both `inner_estimate` and `outer_estimate`; maintainability objective met with no material accuracy/perf regression in validation artifacts; local `fmt`/`clippy -D warnings`/`test` passed |
| INFRA-002 | 2026-02-16 | infra | Decompose self-undistort into focused modules | Completed: split `pixelmap::self_undistort` into focused modules (`config`, `result`, `objective`, `optimizer`, `policy`, `estimator`, tests), preserved public entrypoint surface, and passed fmt/clippy/tests + required blur3/reference/distortion validation scripts |
| PERF-003 | 2026-02-16 | perf | Standardize perf validation suite (blur=3 batch + reference/distortion scripts) | Completed: canonical runbook `.ai/workflows/perf-validation-suite-runbook.md` (session snapshot retained), blur gate shell wrapper `tools/run_blur3_benchmark.sh`, standardized report template `.ai/templates/accuracy-report.md`, PERF handoff contract updates, and dry-run report `.ai/state/sessions/2026-02-16-PERF-003-dry-run-accuracy-report.md` |
| PERF-005 | 2026-02-16 | perf | Optimize inner-fit hotspot group | Completed: `inner_fit_64r_96t_nomapper` `68.051 us -> 59.358 us` (`-12.77%`), `inner_fit_64r_96t_mapper` `93.909 us -> 82.433 us` (`-12.22%`); validation gates passed |
| PERF-004 | 2026-02-16 | perf | Optimize outer-fit/outer-estimate hotspot group | Completed: `outer_estimate_64r_48t_nomapper` `34.365 us -> 16.996 us` (`-50.54%`), `outer_estimate_64r_48t_mapper` `40.082 us -> 23.310 us` (`-41.85%`); validation gates passed |
