# Handoff: Project Lead → Algorithm Engineer

- **Task:** BUG-002: Make Seed Proposal Selection Confidence-Ordered and Deterministic
- **Date:** 2026-02-19
- **Branch:** main

## Work Completed

- Promoted `BUG-002` to Active Sprint (`in-progress`) in backlog.
- Added and finalized task specification:
  - `.ai/state/sessions/2026-02-19-BUG-002-spec.md`
- Reconfirmed defect location and behavior:
  - `crates/ringgrid/src/pipeline/result.rs` currently applies `take(max)` in iteration order before any explicit ranking.
  - pass-2 orchestration consumes these seeds via `crates/ringgrid/src/pipeline/run.rs`.

## Key Findings

- Current seed selection policy is implicit and order-dependent.
- With `max_seeds` enabled, higher-confidence markers can be excluded if they appear later in `detected_markers`.
- Deterministic tie-breaking for equal-confidence seeds is not explicitly defined in current seed extraction logic.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `BUG-002` into Active Sprint (`in-progress`); promoted `ALGO-002` to Up Next |
| `.ai/state/sessions/2026-02-19-BUG-002-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-19-BUG-002-lead-algorithm-engineer.md` | Added dispatch handoff |

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
| Focus | correctness/determinism in pass-2 seed policy |
| Constraint | no material runtime regression in seed selection path |

## Open Questions

- What deterministic tie-break key is simplest and most robust for equal-confidence markers?
- Should tie-break prefer decoded IDs (if present), or remain purely geometric/score-based?
- Are there any call sites that implicitly rely on existing iteration-order behavior?

## Recommended Next Steps

1. Add a focused regression test in `pipeline/result.rs` reproducing order-dependent behavior under `max_seeds`.
2. Implement explicit confidence-first ranking with deterministic tie-breaking for `seed_proposals`.
3. Add tests for permuted-input determinism, truncation semantics, and non-finite center filtering.
4. Run required quality/eval gates and report metric deltas.
5. Hand off to Pipeline Architect only if API/docs semantics require explicit public-contract review.

## Blocking Issues

None.
