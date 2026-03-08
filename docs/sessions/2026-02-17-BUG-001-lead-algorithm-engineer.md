# Handoff: Project Lead → Algorithm Engineer

- **Task:** BUG-001: Fix Decode Config Drift and Expose Hidden Thresholds
- **Date:** 2026-02-17
- **Branch:** code_quality

## Work Completed

- Promoted `BUG-001` to Active Sprint (`in-progress`) in backlog.
- Added and finalized task specification:
  - `.ai/state/sessions/2026-02-17-BUG-001-spec.md`
- Reconfirmed live defect context in `crates/ringgrid/src/marker/decode.rs`:
  - rustdoc defaults drift from `DecodeConfig::default()` values,
  - decode still relies on hidden constants for minimum contrast and threshold iteration/convergence controls.

## Key Findings

- Current `DecodeConfig` docs describe defaults that do not match runtime behavior.
- Decode gate/threshold behavior includes hidden tuning knobs (`MIN_DECODE_CONTRAST`, fixed k-means iteration count, fixed epsilon), creating a source-of-truth ambiguity.
- This is a correctness/maintainability bug in user-facing configuration semantics, not a performance-driven task.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `BUG-001` into Active Sprint (`in-progress`) |
| `.ai/state/sessions/2026-02-17-BUG-001-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-17-BUG-001-lead-algorithm-engineer.md` | Added dispatch handoff |

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
| Focus | correctness/config-contract bug fix |
| Constraint | no material regression from baseline decode-involved paths |

## Open Questions

- Should hidden decode thresholds be promoted to public `DecodeConfig` fields, or remain fixed invariants with explicit docs/tests?
- If new config fields are added, what serde/default migration behavior should be guaranteed?
- Does this change require a conditional Pipeline Architect API check handoff before close-out?

## Recommended Next Steps

1. Reproduce and quantify the doc/default drift with a focused unit test in `marker/decode.rs`.
2. Choose one policy for hidden constants (promote to config or codify as invariant) and implement consistently.
3. Add regression tests for default values, low-contrast gate, and thresholding convergence guards.
4. Run required quality/eval gates and report accuracy deltas.
5. If public config surface changes, hand off to Pipeline Architect for API check before returning to Project Lead.

## Blocking Issues

None.
