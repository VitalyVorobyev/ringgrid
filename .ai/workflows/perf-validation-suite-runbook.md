# PERF Validation Suite Runbook

- **Canonical owner:** Validation Engineer
- **Primary workflow:** `.ai/workflows/performance-optimization.md` (Phase 3)
- **Canonical template:** `.ai/templates/accuracy-report.md`

## Purpose

Run the same three validation gates for every PERF task and report baseline vs after deltas in a consistent format.

## Preconditions

1. Build and quality baseline:
   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace --all-features
   ```
2. Python env and deps available:
   - `./.venv/bin/python` (or compatible venv python)
   - `numpy`, `matplotlib`
3. Choose labels:
   - `BASELINE_LABEL` (for pre-change run)
   - `AFTER_LABEL` (for post-change run)

## Gate A: Blur-3 Synthetic Eval (`n=10`)

### Commands

```bash
# Baseline snapshot
bash tools/run_blur3_benchmark.sh
rm -rf tools/out/eval_${BASELINE_LABEL}_blur3
cp -R tools/out/eval_blur3_post_pipeline tools/out/eval_${BASELINE_LABEL}_blur3

# After-change snapshot
bash tools/run_blur3_benchmark.sh
rm -rf tools/out/eval_${AFTER_LABEL}_blur3
cp -R tools/out/eval_blur3_post_pipeline tools/out/eval_${AFTER_LABEL}_blur3
```

Canonical script:
- `tools/run_blur3_benchmark.sh`

### Required output artifacts

- `tools/out/eval_${BASELINE_LABEL}_blur3/det/aggregate.json`
- `tools/out/eval_${AFTER_LABEL}_blur3/det/aggregate.json`
- `tools/out/eval_${BASELINE_LABEL}_blur3/det/score_*.json` (for frame checks)
- `tools/out/eval_${AFTER_LABEL}_blur3/det/score_*.json` (for frame checks)

### Required metrics

- Precision, recall
- Center error: mean, p50, p95, max
- Homography: self mean/p95, vs-GT mean/p95
- Decode success rate (`n_pred_with_id / n_pred`)

### Frame-invariant check (metrology guardrail)

```bash
for f in tools/out/eval_${AFTER_LABEL}_blur3/det/score_*.json; do
  jq -r '[.center_gt_frame,.pred_center_frame,.homography_self_error.eval_frame,.homography_error_vs_gt.gt_frame,.homography_error_vs_gt.pred_h_frame] | @tsv' "$f"
done | sort -u
```

Expected unique row: `image image image image image`.

## Gate B: Reference Benchmark Script

### Command

```bash
# Baseline snapshot
bash tools/run_reference_benchmark.sh
cp tools/out/reference_benchmark_post_pipeline/summary.json \
  tools/out/reference_benchmark_post_pipeline_${BASELINE_LABEL}.summary.json

# After-change snapshot
bash tools/run_reference_benchmark.sh
cp tools/out/reference_benchmark_post_pipeline/summary.json \
  tools/out/reference_benchmark_post_pipeline_${AFTER_LABEL}.summary.json
```

### Required output artifacts

- `tools/out/reference_benchmark_post_pipeline_${BASELINE_LABEL}.summary.json`
- `tools/out/reference_benchmark_post_pipeline_${AFTER_LABEL}.summary.json`

### Required metrics (per mode)

- Precision, recall
- Avg center mean px
- Avg homography self mean px
- Avg homography vs-GT mean px

Required modes:
- `none__none`
- `projective_center__none`

## Gate C: Distortion Benchmark Script

### Command

```bash
# Baseline snapshot
bash tools/run_distortion_benchmark.sh
cp tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json \
  tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline_${BASELINE_LABEL}.summary.json

# After-change snapshot
bash tools/run_distortion_benchmark.sh
cp tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json \
  tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline_${AFTER_LABEL}.summary.json
```

### Required output artifacts

- `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline_${BASELINE_LABEL}.summary.json`
- `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline_${AFTER_LABEL}.summary.json`

### Required metrics (per correction mode)

- Precision, recall
- Avg center mean px
- Avg homography self mean px
- Avg homography vs-GT mean px

Required correction rows:
- `projective_center__none`
- `projective_center__external`
- `projective_center__self_undistort`

## Pass/Fail and Escalation Gates

1. Fail/escalate if blur-3 center mean delta is `> +0.01 px`.
2. Escalate if blur-3 homography self mean delta or vs-GT mean delta is `> +0.02 px`.
3. Always report precision/recall deltas for all three gates.
4. Escalate to Algorithm Engineer when any gate shows consistent accuracy regression after one rerun.
5. Escalate to Pipeline Architect if frame fields are inconsistent or mixed.

## Reporting Contract

1. Fill `.ai/templates/accuracy-report.md`.
2. Include exact commands used and artifact paths for baseline/after.
3. Attach deltas for all required metrics in the three gates.
4. Use the standardized PERF handoff sections in `.ai/templates/handoff-note.md`.
