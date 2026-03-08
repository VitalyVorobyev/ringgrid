# Handoff: Project Lead → Algorithm Engineer

- **Task:** ALGO-001: Unify Duplicated Radial-Estimator Core (Inner/Outer)
- **Date:** 2026-02-16
- **Branch:** code_quality

## Work Completed

- Reviewed and accepted `INFRA-002` self-undistort modularization.
- Promoted `ALGO-001` to Active Sprint (`in-progress`).
- Created task spec:
  - `.ai/state/sessions/2026-02-16-ALGO-001-spec.md`

## Key Findings

- `ring::inner_estimate` and `ring::outer_estimate` currently duplicate major parts of radial estimator machinery (sampling, derivative, aggregation, polarity candidates, coverage gating).
- This duplication is a maintainability and drift risk that should be removed with a shared core and explicit per-stage policies.
- From project policy: API breaks are acceptable in `v0.1.x` when they materially improve architecture and maintainability.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `ALGO-001` into Active Sprint (`in-progress`) |
| `.ai/state/sessions/2026-02-16-ALGO-001-spec.md` | Added full algorithm task specification |
| `.ai/state/sessions/2026-02-16-ALGO-001-lead-algorithm-engineer.md` | Added dispatch handoff |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| Validation contract | `.ai/workflows/perf-validation-suite-runbook.md` |
| Center-mean guardrail | investigate/escalate if `> +0.01 px` vs baseline |
| Homography guardrail | investigate/escalate if self/vs-GT mean delta `> +0.02 px` |

## Performance State

| Benchmark | Result |
|-----------|--------|
| Focus | algorithm/maintainability refactor with behavior preservation |
| Constraint | no material estimator-path regressions; no new hot-loop allocation churn |

## Open Questions

- What is the cleanest shared API for radial estimation without over-generalizing?
- Which stage-specific behaviors must remain independent (e.g., outer hypotheses vs inner single-peak gating)?
- Can current configs remain stable, or should they be normalized into a common estimator policy type?

## Recommended Next Steps

1. Write a short design note for shared radial-estimator core boundaries and per-stage policy hooks.
2. Implement shared core under `ring/` and migrate both inner/outer estimators to use it.
3. Add regression tests proving behavioral parity (or explicitly documented improvements).
4. Run quality + validation gates and hand off with before/after evidence.
5. If a hard-to-reverse API/config choice is introduced, add an ADR.

## Blocking Issues

None.
