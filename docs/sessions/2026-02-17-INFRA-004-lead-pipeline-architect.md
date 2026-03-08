# Handoff: Project Lead → Pipeline Architect

- **Task:** INFRA-004: Deduplicate Homography Correspondence/Stats Utilities
- **Date:** 2026-02-17
- **Branch:** code_quality

## Work Completed

- Promoted `INFRA-004` to Active Sprint (`in-progress`) in backlog.
- Added and finalized task specification:
  - `.ai/state/sessions/2026-02-17-INFRA-004-spec.md`
- Revalidated duplication surface for correspondence/stat logic in:
  - `crates/ringgrid/src/detector/global_filter.rs`
  - `crates/ringgrid/src/homography/utils.rs`
  - `crates/ringgrid/src/pixelmap/self_undistort/objective.rs`

## Key Findings

- Homography correspondence assembly and reprojection-stat computation logic is repeated with local variations across multiple modules.
- Drift risk is non-trivial: filtering semantics, duplicate handling, and stat summaries can diverge silently when one path changes.
- Scope is architecture/internal refactor; target outcome is shared internal utility layer with explicit frame semantics and behavior preservation.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `INFRA-004` into Active Sprint (`in-progress`); set `INFRA-005` as Up Next |
| `.ai/state/sessions/2026-02-17-INFRA-004-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-17-INFRA-004-lead-pipeline-architect.md` | Added dispatch handoff |

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
| Focus | maintainability/refactor with behavior preservation |
| Constraint | no material end-to-end regression beyond guardrails in task spec |

## Open Questions

- Where should shared utilities live (`homography/utils.rs` extension vs dedicated internal submodule)?
- How should duplicate-ID correspondence policy be normalized across call sites (confidence-first, first-seen, or caller-provided policy)?
- Which utility boundaries need explicit frame labels to avoid working/image-frame ambiguity?
- Does any chosen normalization require an ADR due to cross-cutting semantics?

## Recommended Next Steps

1. Propose a concise internal utility API for correspondence extraction + stat computation.
2. Migrate `global_filter`, `homography::utils`, and self-undistort homography-self-error path to the shared utilities.
3. Add focused tests for shared utilities and call-site behavior parity.
4. Run required quality/eval gates and hand off with baseline vs after evidence.
5. If a hard-to-reverse semantic normalization is introduced, capture it in ADR.

## Blocking Issues

None.
