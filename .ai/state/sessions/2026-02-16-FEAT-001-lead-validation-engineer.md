# Handoff: Project Lead → Validation Engineer

- **Task:** FEAT-001: Normalize Marker Center API and Simplify Finalize Flow
- **Date:** 2026-02-16
- **Branch:** release

## Work Completed

- Reviewed task spec: `.ai/state/sessions/2026-02-15-FEAT-001-spec.md`.
- Reviewed implementation commit `6523322` and confirmed scope alignment with FEAT-001 acceptance criteria.
- Updated backlog active role to Validation Engineer for formal verification and close-out readiness.

## Key Findings

- FEAT-001 implementation appears to be present across API, pipeline, CLI, docs, and Python eval/scoring scripts.
- No post-implementation workflow handoff note exists yet, so validation evidence is currently missing from `state/sessions/`.
- Repo has unrelated working tree changes (`README.md`, `.github/workflows/publish-docs.yml`, `book/`) that should remain untouched during validation handoff prep.

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/backlog.md` | Updated FEAT-001 active role to Validation Engineer and noted pending validation gate |
| `.ai/state/sessions/2026-02-16-FEAT-001-lead-validation-engineer.md` | Added dispatch handoff for validation phase |

## Test Results

- **cargo test:** not run (dispatch phase)
- **cargo clippy:** not run (dispatch phase)
- **cargo fmt:** not run (dispatch phase)

## Accuracy State

| Metric | Value |
|--------|-------|
| Center error (mean) | not measured |
| Center error (p50) | not measured |
| Center error (p95) | not measured |
| Decode success rate | not measured |
| Homography reproj error | not measured |

## Performance State

| Benchmark | Result |
|-----------|--------|
| n/a | not measured |

## Open Questions

- Does synthetic eval confirm no material center/homography regression versus FEAT-001 baseline expectations?
- Are any perf-sensitive paths measurably regressed enough to require conditional Performance Engineer phase?

## Recommended Next Steps

1. Run CI checks:
   - `cargo fmt --all`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`
2. Run synthetic eval for FEAT-001 validation:
   - `python3 tools/run_synth_eval.py --n 10 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_feat001_validation`
3. Verify contract-level outputs:
   - `DetectedMarker.center` is image-space in all modes.
   - `center_mapped` semantics are correct.
   - `center_projective*` fields are absent from outputs.
   - `DetectionResult.center_frame` and `homography_frame` are present and correct.
4. Write handoff note to Pipeline Architect with full results and explicit pass/fail against FEAT-001 acceptance criteria.

## Blocking Issues

None.

---

# Handoff: Validation Engineer → Algorithm Engineer

- **Task:** FEAT-001: Normalize Marker Center API and Simplify Finalize Flow
- **Date:** 2026-02-16
- **Branch:** release

## Work Completed

- Ran CI-quality checks to completion:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
- Ran CI synth-smoke equivalent:
  - `cargo build --workspace`
  - `./.venv/bin/python tools/run_synth_eval.py --n 1 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_feat001_smoke`
- Ran deterministic regression batch:
  - `./.venv/bin/python tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_feat001_validation`
- Ran reference anchor benchmark:
  - `./.venv/bin/python tools/run_reference_benchmark.py --out_dir tools/out/reference_validation --modes projective_center --corrections none`
- Verified FEAT-001 output contract on produced detections:
  - `DetectionResult.center_frame` and `DetectionResult.homography_frame` present.
  - `DetectedMarker.center` serialized in image frame.
  - `DetectedMarker.center_mapped` present when mapper/camera path is active.
  - Legacy `center_projective*` fields absent.

## Key Findings

- **Rust CI gates:** all pass.
- **Synth smoke (`n=1`, blur `1.0`)** is clean:
  - precision `1.000`, recall `1.000`, center mean `0.0727 px`.
- **Regression batch (`n=10`, blur `3.0`)** shows measurable center-error regression:
  - precision `1.000`, recall `0.9468`, center mean `0.3155 px`.
  - baseline reference: recall `0.949`, center mean `0.278 px`.
  - delta: recall `-0.0022`, center mean `+0.0375 px`.
- Regression alert threshold is exceeded (`+0.0375 px > 0.01 px`).
- Worst recall fixtures in this batch:
  - `tools/out/eval_feat001_validation/det/score_0004.json` (recall `0.8966`, miss `21`)
  - `tools/out/eval_feat001_validation/det/score_0009.json` (recall `0.9064`, miss `19`)

## Files Changed

| File | Change |
|------|--------|
| `.ai/state/sessions/2026-02-16-FEAT-001-lead-validation-engineer.md` | Added validation results and Algorithm Engineer handoff |

## Test Results

- **cargo fmt --all --check:** pass
- **cargo clippy --workspace --all-targets --all-features -- -D warnings:** pass
- **cargo test --workspace --all-features:** pass
- **cargo build --workspace:** pass

## Accuracy State

| Metric | Value |
|--------|-------|
| Precision (10-image blur=3.0 batch) | 1.000 |
| Recall (10-image blur=3.0 batch) | 0.9468 |
| Center error mean (10-image blur=3.0 batch) | 0.3155 px |
| Center error p50 (avg per-image) | 0.2918 px |
| Center error p95 (avg per-image) | 0.6765 px |
| Center error max (worst image) | 1.3013 px |
| Homography self-error mean | 0.2788 px |
| Homography error vs GT mean | 0.1416 px |

Reference anchor (projective-center, clean synth):

| Metric | Value |
|--------|-------|
| Precision | 1.000 |
| Recall | 1.000 |
| Center error mean | 0.0528 px |
| Homography self-error mean | 0.0496 px |
| Homography error vs GT mean | 0.0191 px |

## Performance State

| Benchmark | Result |
|-----------|--------|
| Runtime/throughput profiling | not measured (validation phase only) |

## New Fixtures Added

- None in this phase (validation-only run; no code changes made).

## Python Tooling Changes

- None.
- Note: system `python3` lacked `numpy`; validation used `./.venv/bin/python` with existing minimal deps (`numpy`, `matplotlib`).

## Open Questions

- Which pipeline stage is driving the blur=3.0 center-error increase (+0.0375 px mean vs baseline): proposal/fit/decode quality drift or later-stage finalize/refine/completion behavior?

## Recommended Next Steps

1. Reproduce and triage on `img_0004` and `img_0009` in `tools/out/eval_feat001_validation/synth/`.
2. Use debug overlays and per-stage dumps to localize where misses/offsets are introduced.
3. Propose targeted algorithm changes, then rerun the same deterministic `n=10`, blur=3.0 batch and check delta against the `>0.01 px` alert threshold.

## Blocking Issues

None.
