# Handoff: Project Lead → Pipeline Architect

- **Task:** INFRA-005: Harden BoardLayout Invariants and Loading Errors
- **Date:** 2026-02-19
- **Branch:** code_quality

## Work Completed

- Promoted `INFRA-005` to Active Sprint (`in-progress`) in backlog.
- Added and finalized task specification:
  - `.ai/state/sessions/2026-02-19-INFRA-005-spec.md`
- Reconfirmed current risk surface in code:
  - `BoardLayout` exposes mutable `markers` while relying on internal `id_to_idx`.
  - `build_index()` is required after external mutation.
  - Board-loading and detector convenience constructors use weakly typed error boundaries.

## Key Findings

- Invariant safety risk: public mutable markers can desynchronize board lookup behavior if reindexing is missed.
- Error-contract risk: loader failures are not strongly typed end-to-end, limiting actionable diagnostics.
- Task is architecture/API quality focused and likely requires intentional API surface cleanup.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Moved `INFRA-005` into Active Sprint (`in-progress`); set `INFRA-006` as Up Next |
| `.ai/state/sessions/2026-02-19-INFRA-005-spec.md` | Added full task spec |
| `.ai/state/sessions/2026-02-19-INFRA-005-lead-pipeline-architect.md` | Added dispatch handoff |

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
| Focus | invariants + typed error contracts |
| Constraint | no material runtime regression from baseline |

## Open Questions

- What mutability contract should `BoardLayout` expose (fully private markers + controlled mutators, or limited public access with automatic reindexing guarantees)?
- How should typed load/validation errors be modeled and propagated through `Detector::from_target_json_file*`?
- Is an ADR required for public API changes to BoardLayout mutation and loader error signatures?

## Recommended Next Steps

1. Produce a compact API design for invariant-safe `BoardLayout` mutation and typed load errors.
2. Implement typed board-load errors and coherent propagation through detector constructors.
3. Remove or constrain manual `build_index()` usage so stale index states are impossible via public API.
4. Add tests for invalid layout/load cases and invariant integrity after mutation operations.
5. Run quality/eval gates and hand off with behavior/regression evidence.

## Blocking Issues

None.
