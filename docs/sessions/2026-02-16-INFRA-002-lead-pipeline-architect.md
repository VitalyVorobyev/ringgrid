# Handoff: Project Lead → Pipeline Architect

- **Task:** INFRA-002: Decompose Self-Undistort into Focused Modules
- **Date:** 2026-02-16
- **Branch:** code_quality

## Work Completed

- Completed code-quality audit for `crates/ringgrid/src` and identified self-undistort modularity as the top maintainability priority.
- Added task spec:
  - `.ai/state/sessions/2026-02-16-INFRA-002-spec.md`
- Promoted `INFRA-002` to Active Sprint (`in-progress`) in backlog.
- Captured detailed evidence in:
  - `.ai/state/sessions/2026-02-16-lead-code-quality-audit.md`

## Key Findings

- `crates/ringgrid/src/pixelmap/self_undistort.rs` currently combines objective modeling, optimizer/search, homography validation, and orchestration in one module.
- Current structure increases coupling and slows safe iteration.
- This task is architecture-first; runtime behavior should remain stable unless explicitly improved.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `INFRA-002` into Active Sprint (`in-progress`) |
| `.ai/state/sessions/2026-02-16-INFRA-002-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-16-INFRA-002-lead-pipeline-architect.md` | Added dispatch handoff |
| `.ai/state/sessions/2026-02-16-lead-code-quality-audit.md` | Added audit evidence and rationale |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| Source of truth | `.ai/workflows/perf-validation-suite-runbook.md` |
| Required gate policy | center-mean regression investigate/escalate if `> +0.01 px` |
| Validation expectation | run all three standard PERF/accuracy gates before closure |

## Performance State

| Benchmark | Result |
|-----------|--------|
| Focus | maintainability refactor, not direct optimization |
| Constraint | no material end-to-end regression on standard benchmarks unless justified |

## Open Questions

- Should self-undistort become explicit strategy modes with typed policy objects?
- Should we keep one public `estimate_self_undistort` entrypoint or introduce a smaller composable API?
- Which API breaks produce the largest simplification for `v0.1.x`?

## Recommended Next Steps

1. Produce a decomposition design for self-undistort with explicit module boundaries and data contracts.
2. Decide API shape intentionally; API-breaking changes are allowed at `v0.1.x` when they materially improve maintainability.
3. Implement decomposition incrementally with behavior-preserving tests at each step.
4. Run required quality/validation gates and document baseline vs after artifacts.
5. If design choices are significant or hard to reverse, author an ADR in `.ai/state/decisions/`.

## Blocking Issues

None.
