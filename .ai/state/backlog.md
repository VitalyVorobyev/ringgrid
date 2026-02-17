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
| — | — | — | — | — | — | — |

## Backlog

| ID | Status | Priority | Type | Title | Role | Notes |
|----|--------|----------|------|-------|------|-------|
| INFRA-004 | todo | P2 | infra | Deduplicate homography correspondence/stats utilities | Pipeline Architect | Similar correspondence and stats logic exists in `detector/global_filter.rs`, `homography/utils.rs`, and `pixelmap/self_undistort.rs`; consolidate into one internal utility layer |
| INFRA-005 | todo | P2 | infra | Harden BoardLayout invariants and loading errors | Pipeline Architect | `BoardLayout` exposes mutable `markers` while requiring manual `build_index()` sync; enforce invariant-safe mutation API and typed load/validation errors |
| INFRA-006 | todo | P2 | infra | Split outer-fit responsibilities and remove hardcoded solver knobs | Pipeline Architect | `crates/ringgrid/src/detector/outer_fit.rs` mixes sampling, fitting, decode, and scoring; contains local hardcoded RANSAC config instead of shared config source |
| BUG-002 | todo | P2 | bug | Make seed proposal selection confidence-ordered and deterministic | Pipeline Architect | `DetectionResult::seed_proposals` currently takes first markers in iteration order; rank by confidence and define tie-breaking for stable pass-2 seeds |
| ALGO-002 | todo | P2 | algo | Decompose projective-center solver into testable stages | Algorithm Engineer | `ring_center_projective_with_debug` in `crates/ringgrid/src/ring/projective_center.rs` is a large mixed-responsibility routine; split candidate generation/scoring/selection for clarity and safer evolution |
| INFRA-007 | todo | P2 | infra | Add maintainability guardrails to CI | Pipeline Architect | Add lint/doc complexity policy (function-size hotspots, rustdoc coverage, forbid new `allow(dead_code)` in hot modules) to prevent quality regressions |

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| BUG-001 | 2026-02-17 | bug | Fix decode config drift and expose hidden thresholds | Accepted: aligned `DecodeConfig` rustdoc with runtime defaults via shared constants; promoted decode hidden thresholds to explicit config (`min_decode_contrast`, `threshold_max_iters`, `threshold_convergence_eps`) with serde-safe defaults; added deterministic threshold-loop and compatibility tests; reported fmt/clippy/tests pass and synth eval deltas within gate |
| INFRA-003 | 2026-02-17 | infra | Replace stringly reject reasons with typed enums | Accepted: replaced string-based reject/failure channels in `decode`/`inner_fit`/`outer_fit`/`completion`/`fit_decode` with typed enums + structured context; aggregation now keyed by typed reason codes; reported `fmt`/`clippy -D warnings`/`test` pass and blur=3 eval delta is zero versus baseline |
| ALGO-001 | 2026-02-16 | algo | Unify duplicated radial-estimator core (inner/outer) | Accepted: introduced shared `ring::radial_estimator` core and rewired both `inner_estimate` and `outer_estimate`; maintainability objective met with no material accuracy/perf regression in validation artifacts; local `fmt`/`clippy -D warnings`/`test` passed |
| INFRA-002 | 2026-02-16 | infra | Decompose self-undistort into focused modules | Completed: split `pixelmap::self_undistort` into focused modules (`config`, `result`, `objective`, `optimizer`, `policy`, `estimator`, tests), preserved public entrypoint surface, and passed fmt/clippy/tests + required blur3/reference/distortion validation scripts |
| PERF-003 | 2026-02-16 | perf | Standardize perf validation suite (blur=3 batch + reference/distortion scripts) | Completed: canonical runbook `.ai/workflows/perf-validation-suite-runbook.md` (session snapshot retained), blur gate shell wrapper `tools/run_blur3_benchmark.sh`, standardized report template `.ai/templates/accuracy-report.md`, PERF handoff contract updates, and dry-run report `.ai/state/sessions/2026-02-16-PERF-003-dry-run-accuracy-report.md` |
| PERF-005 | 2026-02-16 | perf | Optimize inner-fit hotspot group | Completed: `inner_fit_64r_96t_nomapper` `68.051 us -> 59.358 us` (`-12.77%`), `inner_fit_64r_96t_mapper` `93.909 us -> 82.433 us` (`-12.22%`); validation gates passed |
| PERF-004 | 2026-02-16 | perf | Optimize outer-fit/outer-estimate hotspot group | Completed: `outer_estimate_64r_48t_nomapper` `34.365 us -> 16.996 us` (`-50.54%`), `outer_estimate_64r_48t_mapper` `40.082 us -> 23.310 us` (`-41.85%`); validation gates passed |
| PERF-002 | 2026-02-16 | perf | Optimize proposal hotspot (`detector::proposal::find_proposals`) | Accepted: `proposal_1280x1024` `42.404 ms → 33.612 ms` (`-20.73%`), `proposal_1920x1080` `60.614 ms → 48.357 ms` (`-20.22%`); required validation gates passed |
| PERF-001 | 2026-02-16 | perf | Establish comprehensive performance tracing baseline and benchmark harness | Completed baseline report + hotspot ranking + follow-up plan. Allocation counts documented via proxy (RSS) due `xctrace`/SIP tooling limits |
| FEAT-001 | 2026-02-16 | feature | Normalize marker center API and simplify finalize flow | Closed by human decision: marginal regression accepted for now; follow-up perf/accuracy work tracked in PERF tasks |
| INFRA-001 | 2026-02-15 | infra | Set up `.ai/` agent workflow | Bootstrap ADR-001 |
