# Session: Project Lead Code Quality Audit

- **Date:** 2026-02-16
- **Scope:** `crates/ringgrid/src` (library crate only)
- **Goal:** Identify maintainability/robustness weak points and convert findings into backlog tasks
- **Execution constraints:** No test/benchmark runs in this phase (Project Lead audit pass)

## Findings (Ordered by Maintainability Risk)

1. `self_undistort` is monolithic and mixes multiple responsibilities.
- Evidence:
- `crates/ringgrid/src/pixelmap/self_undistort.rs:120` objective function implementation.
- `crates/ringgrid/src/pixelmap/self_undistort.rs:168` optimizer implementation.
- `crates/ringgrid/src/pixelmap/self_undistort.rs:207` homography validation implementation.
- `crates/ringgrid/src/pixelmap/self_undistort.rs:283` end-to-end orchestration and policy gating.
- Impact:
- Hard to reason about correctness boundaries.
- High change risk; policy tweaks and numerical changes are tightly coupled.

2. Inner/outer radial estimators duplicate core algorithmic flow.
- Evidence:
- `crates/ringgrid/src/ring/inner_estimate.rs:97` to `crates/ringgrid/src/ring/inner_estimate.rs:160` radial sampling and derivative collection loop.
- `crates/ringgrid/src/ring/outer_estimate.rs:141` to `crates/ringgrid/src/ring/outer_estimate.rs:180` similar radial sampling/derivative loop.
- Polarity candidate + aggregate + select pattern appears in both modules (`inner_estimate.rs:190`, `outer_estimate.rs:203`).
- Impact:
- Divergent behavior risk when only one estimator is changed.
- Review/test burden is higher than necessary.

3. Failure/reject channels are mostly stringly typed.
- Evidence:
- `crates/ringgrid/src/detector/inner_fit.rs:30` (`Option<String>` reason).
- `crates/ringgrid/src/detector/outer_fit.rs:20` (`Result<..., String>`).
- `crates/ringgrid/src/detector/completion.rs:84` (`Result<(), String>` gates).
- `crates/ringgrid/src/marker/decode.rs:108` (`reject_reason: Option<String>`).
- Impact:
- Brittle cross-stage handling and low tooling support.
- Harder to enforce stable semantics in logs/JSON/debugging.

4. `outer_fit` has mixed concerns and hidden tuning constants.
- Evidence:
- Hardcoded RANSAC config in `crates/ringgrid/src/detector/outer_fit.rs:22` to `crates/ringgrid/src/detector/outer_fit.rs:27`.
- Same file owns sampling, fitting, decode coupling, and candidate scoring (`outer_fit.rs:313`, `outer_fit.rs:341`, `outer_fit.rs:346`).
- Impact:
- SRP violation; harder to tune and validate independently.
- Config drift risk (local constants vs shared config).

5. Homography correspondence/stats logic is duplicated across modules.
- Evidence:
- `crates/ringgrid/src/detector/global_filter.rs:17` to `crates/ringgrid/src/detector/global_filter.rs:31` correspondence collection.
- `crates/ringgrid/src/homography/utils.rs:15` to `crates/ringgrid/src/homography/utils.rs:25` similar collection logic.
- `crates/ringgrid/src/pixelmap/self_undistort.rs:212` to `crates/ringgrid/src/pixelmap/self_undistort.rs:245` parallel correspondence path.
- Impact:
- Multiple places to maintain frame conventions and filtering rules.
- Regression risk when one codepath evolves independently.

6. `BoardLayout` allows invariant violations via public mutable fields.
- Evidence:
- Public fields in `crates/ringgrid/src/board_layout.rs:53` to `crates/ringgrid/src/board_layout.rs:64`.
- Manual index sync requirement `build_index()` in `crates/ringgrid/src/board_layout.rs:79`.
- Lookup depends on private index in `crates/ringgrid/src/board_layout.rs:91`.
- Impact:
- Easy to create stale `id_to_idx` state after mutation.
- Hidden correctness failures in downstream stages.

7. Public-facing error surface is not strongly typed end-to-end.
- Evidence:
- `BoardLayout::from_json_file` returns `Box<dyn Error>` in `crates/ringgrid/src/board_layout.rs:147`.
- `Detector::from_target_json_file*` also returns `Box<dyn Error>` in `crates/ringgrid/src/api.rs:64`.
- Impact:
- Harder API contracts for library users.
- Reduced diagnosability and error matching in callers.

8. Decode config docs and implementation have drift.
- Evidence:
- Doc comments claim defaults different from implementation in `crates/ringgrid/src/marker/decode.rs:43` to `crates/ringgrid/src/marker/decode.rs:55` vs `decode.rs:60` to `decode.rs:64`.
- Additional hidden thresholds are hardcoded (`decode.rs:202`, `decode.rs:222`, `decode.rs:238`).
- Impact:
- Surprise for users tuning decode behavior.
- Weak source-of-truth quality for critical thresholding logic.

9. Seed proposal selection is order-dependent and not explicitly ranked.
- Evidence:
- `crates/ringgrid/src/pipeline/result.rs:61` to `crates/ringgrid/src/pipeline/result.rs:64` uses iteration order + `take(max)`.
- Impact:
- Non-explicit two-pass seed policy; reproducibility and quality may vary with marker order.

10. Projective-center solver core is dense and difficult to evolve safely.
- Evidence:
- `crates/ringgrid/src/ring/projective_center.rs:163` to `crates/ringgrid/src/ring/projective_center.rs:310` contains candidate generation, scoring, and selection in one large function.
- Impact:
- High cognitive load and change risk for math-sensitive code.

## Backlog Tasks Added

- `INFRA-002` Decompose self-undistort into focused modules.
- `ALGO-001` Unify duplicated radial-estimator core (inner/outer).
- `INFRA-003` Replace stringly reject reasons with typed enums.
- `BUG-001` Fix decode config drift and expose hidden thresholds.
- `INFRA-004` Deduplicate homography correspondence/stats utilities.
- `INFRA-005` Harden `BoardLayout` invariants and loading errors.
- `INFRA-006` Split outer-fit responsibilities and remove hardcoded solver knobs.
- `BUG-002` Make seed proposal selection confidence-ordered and deterministic.
- `ALGO-002` Decompose projective-center solver into testable stages.
- `INFRA-007` Add maintainability guardrails to CI.

## ADR Candidates

1. Reject reason representation policy:
- Internal typed enums only, or typed enums with stable serialized string codes?

2. `BoardLayout` mutability contract:
- Keep mutable public struct with explicit reindex API, or make internals private and enforce invariant-safe mutation methods only?

3. Self-undistort objective strategy:
- Keep dual objective paths (homography-first fallback), or split into explicit strategy modes with deterministic selection policy?

## Project-Level Suggestions (Beyond Refactor Tasks)

1. Add fuzz/property tests for geometry and decode boundaries.
- Targets: conic fitting, homography estimation, decode thresholding, mapper round-trip constraints.

2. Add API stability policy for v1.
- Explicit deprecation window and changelog tags for config/result schema changes.

3. Add reproducibility and observability standards.
- Require seed logging and stage-level counters in structured output for hard-to-reproduce regressions.

4. Add release-readiness checklist.
- Include docs coverage, example parity, backward-compat JSON checks, and benchmark regression thresholds.
