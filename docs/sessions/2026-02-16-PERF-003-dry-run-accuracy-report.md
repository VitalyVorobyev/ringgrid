# Accuracy Report: PERF-003 Dry Run (PERF-004 -> PERF-005)

- **Date:** 2026-02-16
- **Workflow:** `.ai/workflows/performance-optimization.md` (Phase 3)
- **Runbook:** `.ai/workflows/perf-validation-suite-runbook.md`
- **Baseline label:** perf004
- **After label:** perf005
- **Baseline commit:** n/a (artifact comparison run)
- **After commit:** n/a (artifact comparison run)

## Commands

| Gate | Baseline command | After command |
|------|------------------|---------------|
| Blur-3 synth eval (`n=10`) | `bash tools/run_blur3_benchmark.sh && rm -rf tools/out/eval_perf004_blur3 && cp -R tools/out/eval_blur3_post_pipeline tools/out/eval_perf004_blur3` | `bash tools/run_blur3_benchmark.sh && rm -rf tools/out/eval_perf005_blur3 && cp -R tools/out/eval_blur3_post_pipeline tools/out/eval_perf005_blur3` |
| Reference benchmark | `bash tools/run_reference_benchmark.sh` | `bash tools/run_reference_benchmark.sh` |
| Distortion benchmark | `bash tools/run_distortion_benchmark.sh` | `bash tools/run_distortion_benchmark.sh` |

## Artifact Paths

| Gate | Baseline artifact | After artifact |
|------|-------------------|----------------|
| Blur-3 synth eval | `tools/out/eval_perf004_blur3/det/aggregate.json` | `tools/out/eval_perf005_blur3/det/aggregate.json` |
| Reference benchmark | `.ai/state/sessions/2026-02-16-PERF-004-performance-handoff.md` | `.ai/state/sessions/2026-02-16-PERF-005-performance-handoff.md` |
| Distortion benchmark | `.ai/state/sessions/2026-02-16-PERF-004-performance-handoff.md` | `.ai/state/sessions/2026-02-16-PERF-005-performance-handoff.md` |

Note: Gate B/C values are taken from the PERF-004/PERF-005 handoff summaries because both runs wrote to the same script output paths.

## Gate A: Blur-3 Synth Eval Metrics

| Metric | Baseline | After | Delta | Status |
|--------|----------|-------|-------|--------|
| Precision | 1.000000 | 1.000000 | +0.000000 | pass |
| Recall | 0.945813 | 0.947783 | +0.001970 | pass |
| F1 | 0.972152 | 0.973192 | +0.001040 | pass |
| Decode success rate | 1.000000 | 1.000000 | +0.000000 | pass |
| Center mean (px) | 0.316741 | 0.314815 | -0.001926 | pass |
| Center p50 (px, avg per-image) | 0.294316 | 0.287943 | -0.006373 | pass |
| Center p95 (px, avg per-image) | 0.677076 | 0.679549 | +0.002473 | pass |
| Center max (px, worst image) | 1.299109 | 1.292663 | -0.006446 | pass |
| Homography self mean (px) | 0.276979 | 0.275513 | -0.001466 | pass |
| Homography self p95 (px, avg per-image) | 0.616430 | 0.620739 | +0.004308 | pass |
| Homography vs-GT mean (px) | 0.145697 | 0.144393 | -0.001304 | pass |
| Homography vs-GT p95 (px, avg per-image) | 0.240680 | 0.234580 | -0.006101 | pass |

## Gate A: Frame Consistency Check

| Field set | Baseline | After | Status |
|-----------|----------|-------|--------|
| `center_gt_frame / pred_center_frame / homography_self_error.eval_frame / homography_error_vs_gt.gt_frame / homography_error_vs_gt.pred_h_frame` | `image / image / image / image / image` | `image / image / image / image / image` | pass |

## Gate B: Reference Benchmark Summary

| Mode | Metric | Baseline | After | Delta | Status |
|------|--------|----------|-------|-------|--------|
| `none__none` | Precision | 1.000 | 1.000 | +0.000 | pass |
| `none__none` | Recall | 1.000 | 1.000 | +0.000 | pass |
| `none__none` | Center mean (px) | 0.0743 | 0.0743 | +0.0000 | pass |
| `none__none` | Homography self mean (px) | n/a | n/a | n/a | n/a |
| `none__none` | Homography vs-GT mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__none` | Precision | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__none` | Recall | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__none` | Center mean (px) | 0.0528 | 0.0529 | +0.0001 | pass |
| `projective_center__none` | Homography self mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__none` | Homography vs-GT mean (px) | n/a | n/a | n/a | n/a |

## Gate C: Distortion Benchmark Summary

| Mode | Metric | Baseline | After | Delta | Status |
|------|--------|----------|-------|-------|--------|
| `projective_center__none` | Precision | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__none` | Recall | 0.9787 | 0.9770 | -0.0017 | monitor |
| `projective_center__none` | Center mean (px) | 0.1065 | 0.0948 | -0.0117 | pass |
| `projective_center__none` | Homography self mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__none` | Homography vs-GT mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__external` | Precision | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__external` | Recall | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__external` | Center mean (px) | 0.0798 | 0.0767 | -0.0031 | pass |
| `projective_center__external` | Homography self mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__external` | Homography vs-GT mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__self_undistort` | Precision | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__self_undistort` | Recall | 1.000 | 1.000 | +0.000 | pass |
| `projective_center__self_undistort` | Center mean (px) | 0.0801 | 0.0778 | -0.0023 | pass |
| `projective_center__self_undistort` | Homography self mean (px) | n/a | n/a | n/a | n/a |
| `projective_center__self_undistort` | Homography vs-GT mean (px) | n/a | n/a | n/a | n/a |

## Threshold Evaluation

| Rule | Result | Status |
|------|--------|--------|
| Blur-3 center mean delta `<= +0.01 px` | `-0.001926 px` | pass |
| Blur-3 homography self mean delta `<= +0.02 px` | `-0.001466 px` | pass |
| Blur-3 homography vs-GT mean delta `<= +0.02 px` | `-0.001304 px` | pass |
| Reference benchmark precision/recall deltas recorded | yes | pass |
| Distortion benchmark precision/recall deltas recorded | yes | pass |

## Verdict

- **Overall:** Pass
- **Escalate to Algorithm Engineer:** no
- **Escalate to Pipeline Architect:** no
- **Notes:** Dry-run shows the standardized format is usable for all three required PERF validation gates, including threshold calls and artifact provenance.
