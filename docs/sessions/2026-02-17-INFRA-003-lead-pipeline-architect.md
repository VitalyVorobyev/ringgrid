# Handoff: Project Lead → Pipeline Architect

- **Task:** INFRA-003: Replace Stringly Reject Reasons with Typed Enums
- **Date:** 2026-02-17
- **Branch:** code_quality

## Work Completed

- Prioritized `INFRA-003` as the next active maintainability task from the backlog.
- Added and finalized task spec:
  - `.ai/state/sessions/2026-02-17-INFRA-003-spec.md`
- Updated backlog to mark `INFRA-003` as `in-progress` in Active Sprint.
- Reconfirmed stringly reject surfaces in scoped modules:
  - `detector/inner_fit.rs`
  - `detector/outer_fit.rs`
  - `detector/completion.rs`
  - `marker/decode.rs`
  - `pipeline/fit_decode.rs`

## Key Findings

- Reject/failure semantics are currently represented with free-form strings and mixed conventions (`Option<String>`, `Result<_, String>`, inline formatted text).
- Some reasons include embedded dynamic numeric payloads, which makes aggregation and stability difficult (for example summary buckets in `pipeline/fit_decode.rs`).
- This is a cross-cutting maintainability issue and should be solved with typed reason enums and explicit serialization/formatting policy.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `INFRA-003` into Active Sprint (`in-progress`) |
| `.ai/state/sessions/2026-02-17-INFRA-003-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-17-INFRA-003-lead-pipeline-architect.md` | Added dispatch handoff |

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
| Focus | maintainability/type-safety refactor, not direct optimization |
| Constraint | no material regression beyond guardrails in task spec |

## Open Questions

- Should this use one shared cross-stage reject enum or stage-local enums with a typed wrapper at pipeline boundaries?
- What serialization form should be canonical for diagnostics/logging (for example stable snake_case code strings vs tagged enum objects)?
- Do we keep numeric context in structured fields per reject variant, or emit a compact code plus supplemental diagnostics payload?
- Does this decision require an ADR before implementation, given cross-cutting impact on diagnostics contracts?

## Recommended Next Steps

1. Produce a short architecture note for reason enum topology and diagnostics serialization policy.
2. Decide ADR need early; if required, author before implementation begins.
3. Implement incrementally in scoped modules and update aggregation in `pipeline/fit_decode.rs`.
4. Add/extend tests for reason mapping and serialization behavior.
5. Run required quality and evaluation gates and hand off back with measured deltas.

## Blocking Issues

None.
