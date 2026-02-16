# Accuracy Report: [Task ID]

- **Date:** YYYY-MM-DD
- **Baseline commit:** [hash]
- **Change commit:** [hash]
- **Eval command:** `python3 tools/run_synth_eval.py [args]`
- **Challenging eval command:** `python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 [args]`
- **Reference benchmark command:** `bash tools/run_reference_benchmark.sh`
- **Distortion benchmark command:** `bash tools/run_distortion_benchmark.sh`

## Summary

[One-line: improved / regressed / neutral]

## Detection Metrics

| Metric | Baseline | After Change | Delta |
|--------|----------|-------------|-------|
| Precision | | | |
| Recall | | | |
| F1 | | | |
| Markers detected (mean) | | | |

## Center Error (px)

| Statistic | Baseline | After Change | Delta |
|-----------|----------|-------------|-------|
| Mean | | | |
| Median (p50) | | | |
| p95 | | | |
| Max | | | |

## Decode Metrics

| Metric | Baseline | After Change | Delta |
|--------|----------|-------------|-------|
| Decode success rate | | | |
| Hamming distance (mean) | | | |

## Homography

| Metric | Baseline | After Change | Delta |
|--------|----------|-------------|-------|
| Reprojection error (mean px) | | | |
| Inlier ratio | | | |

## Reference Benchmark (Script Output Summary)

| Metric | Baseline | After Change | Delta |
|--------|----------|-------------|-------|
| Precision | | | |
| Recall | | | |
| Center error mean (px) | | | |
| Homography self-error mean (px) | | | |

## Distortion Benchmark (Script Output Summary)

| Mode/Correction | Baseline | After Change | Delta |
|-----------------|----------|-------------|-------|
| projective_center + none | | | |
| projective_center + external | | | |
| projective_center + self_undistort | | | |

## Eval Conditions

- Synthetic images: [count, resolution]
- Blur: [px]
- Noise sigma: [value]
- Marker diameter: [px]
- Gate distance: [px]
- Distortion benchmark params: [fx/fy/cx/cy and k1..k3/p1/p2 values used]

## Verdict

[Pass / Fail / Conditional-pass with explanation]

Threshold: flag any mean center error increase > 0.01 px.
