# Handoff: Project Lead → Pipeline Architect

- **Task:** INFRA-006: Split Outer-Fit Responsibilities and Remove Hardcoded Solver Knobs
- **Date:** 2026-02-19
- **Branch:** main

## Work Completed

- Promoted `INFRA-006` to Active Sprint (`in-progress`) in backlog.
- Added and finalized task specification:
  - `.ai/state/sessions/2026-02-19-INFRA-006-spec.md`
- Reconfirmed current hotspot and maintainability issues in `detector/outer_fit.rs`:
  - mixed responsibilities (sampling, fitting, decode/scoring, completion policy),
  - local hardcoded outer-fit RANSAC knobs not sourced from shared config.

## Key Findings

- `outer_fit.rs` currently carries both policy orchestration and low-level mechanics, increasing maintenance risk.
- Solver defaults for outer-fit are locally hardcoded and drift-prone against centralized config strategy.
- Completion entrypoint behavior should remain policy-specific while reusing the same underlying outer-fit primitives.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `INFRA-006` into Active Sprint (`in-progress`); set `BUG-002` as Up Next |
| `.ai/state/sessions/2026-02-19-INFRA-006-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-19-INFRA-006-lead-pipeline-architect.md` | Added dispatch handoff |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| Center error (mean) | not measured in dispatch phase |
| Center error (p50) | not measured in dispatch phase |
| Center error (p95) | not measured in dispatch phase |
| Decode success rate | not measured in dispatch phase |
| Homography self-error (mean) | not measured in dispatch phase |
| Homography vs-GT error (mean) | not measured in dispatch phase |

## Performance State

| Benchmark | Result |
|-----------|--------|
| Focus | maintainability/config-contract refactor |
| Constraint | no material runtime regression from baseline |

## Open Questions

- What is the cleanest split boundary for outer-fit responsibilities without introducing adapter-layer duplication?
- Should outer-fit RANSAC knobs reuse existing shared config shape directly or use a dedicated outer-fit config block under `DetectConfig`?
- Which completion-only behaviors remain policy-layer knobs vs shared core behavior?

## Recommended Next Steps

1. Design a small internal module split for outer-fit responsibilities and define call boundaries.
2. Move outer-fit solver knobs into shared config/default source and update all call sites.
3. Preserve reject reasons and add behavior-parity tests for baseline/completion entrypoints.
4. Run quality/eval gates and hand off with baseline vs after evidence.
5. If config surface changes materially, include explicit migration notes (and ADR if needed).

## Blocking Issues

None.
