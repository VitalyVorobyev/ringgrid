# Accuracy Report: [Task ID]

- **Date:** YYYY-MM-DD
- **Baseline commit:** [hash]
- **Change commit:** [hash]
- **Eval command:** `python3 tools/run_synth_eval.py [args]`

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

## Eval Conditions

- Synthetic images: [count, resolution]
- Blur: [px]
- Noise sigma: [value]
- Marker diameter: [px]
- Gate distance: [px]

## Verdict

[Pass / Fail / Conditional-pass with explanation]

Threshold: flag any mean center error increase > 0.01 px.
