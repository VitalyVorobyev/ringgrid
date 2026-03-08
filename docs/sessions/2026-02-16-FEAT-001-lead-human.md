# Handoff: Project Lead → Human

- **Task:** FEAT-001: Normalize Marker Center API and Simplify Finalize Flow
- **Date:** 2026-02-16
- **Branch:** release

## Closure Decision

- FEAT-001 is closed and moved to Done by explicit human decision.
- Validation found a marginal regression on the deterministic blur=3.0 batch:
  - recall `0.9468` vs baseline `0.949`
  - center error mean `0.3155 px` vs baseline `0.278 px` (+`0.0375 px`)
- This regression is accepted for now; follow-up performance/accuracy improvements are tracked under PERF tasks.

## Notes

- FEAT-001 API/contract checks passed and CI gates passed in validation.
- Worst fixtures remain documented for future triage:
  - `tools/out/eval_feat001_validation/det/score_0004.json`
  - `tools/out/eval_feat001_validation/det/score_0009.json`
