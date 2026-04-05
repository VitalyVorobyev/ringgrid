---
name: regression-gate
description: Run regression benchmarks and maintainability guardrails. Rebuilds, runs 3 synthetic benchmarks + 1 real-world benchmark + code quality checks, compares against baselines, auto-fixes maintainability issues, reports benchmark regressions. Invoke after significant code changes or before releases.
---

# Regression Gate

Run the full regression suite: build, benchmark, maintainability check.

## Step 1 — Build and lint

Build the release binary and Python bindings, then run fmt and clippy:

```bash
cargo build --release
cd crates/ringgrid-py && ../../.venv/bin/maturin develop --release && cd ../..
```

If the build fails, stop and report the error. Do not proceed to further steps.

After a successful build, run formatting and lint checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

- **fmt failures**: Run `cargo fmt --all` to auto-fix, then report what changed.
- **clippy failures**: Fix the warnings in code. Do not add `#[allow(...)]` unless truly justified. Report each fix.

## Step 2 — Run benchmarks

Run all 3 synthetic benchmark scripts in sequence using `.venv`:

```bash
bash tools/run_reference_benchmark.sh
bash tools/run_distortion_benchmark.sh
bash tools/run_blur3_benchmark.sh
```

Each script writes results to `tools/out/`. If any script fails, report the error and continue with the remaining benchmarks.

**RTv3D real-world benchmark (local only — private data)**

If the `data/rtv3d` directory exists, also run:

```bash
.venv/bin/python tools/run_rtv3d_eval.py
```

If `data/rtv3d` does not exist, skip this step and report `RTv3D benchmark: SKIPPED (no data)`. This dataset is private and not available in CI.

## Step 3 — Check benchmark results against baselines

Read `tools/ci/regression_baseline.json` for the success criteria.

**For reference and distortion benchmarks** (summary.json format):

For each mode in the baseline, read the corresponding mode from the benchmark output and check:
- `avg_precision >= baseline.min_precision`
- `avg_recall >= baseline.min_recall`
- `avg_center_mean_px <= baseline.max_center_mean_px`
- `avg_h_vs_gt_mean_px <= baseline.max_h_vs_gt_mean_px`

**For blur3 benchmark** (aggregate.json format):

- `avg_precision >= baseline.criteria.min_precision`
- `avg_recall >= baseline.criteria.min_recall`
- `avg_center_error <= baseline.criteria.max_center_error`

**For rtv3d benchmark** (report.json format):

For each strategy in the baseline, read the corresponding strategy from the benchmark output and check:
- `total_decoded >= baseline.min_total_decoded`

Collect all failures into a report table.

## Step 4 — Run maintainability guardrails

```bash
.venv/bin/python tools/ci/maintainability_guardrails.py
```

## Step 5 — Report and fix

### Benchmark regressions (report only)

If any benchmark metric regressed, print a table:

```
BENCHMARK REGRESSIONS:
| Benchmark    | Mode                    | Metric            | Baseline | Actual  | Status |
|--------------|-------------------------|--------------------|----------|---------|--------|
| reference    | projective_center__none | center_mean_px     | <=0.084  | 0.095   | FAIL   |
```

Do NOT auto-fix benchmark regressions. Report them clearly so the user can investigate.

### Maintainability issues (auto-fix)

If the maintainability guardrail fails:

1. **New oversized functions (>120 lines)**: Do NOT add them to a whitelist. Instead, refactor by extracting helper functions to bring each function under 120 lines. The `allowed_oversized_functions` map must remain empty.
2. **New dead_code/too_many_args allows**: Add to the baseline only if truly justified. Report what was added and why.
3. **Rustdoc regressions (missing-docs > 0)**: Add `///` doc comments to all undocumented public items. The `max_warnings` must remain 0.

After fixing, re-run the guardrail to confirm it passes.

### Success

If everything passes, print:

```
REGRESSION GATE: ALL PASSED
  Format (cargo fmt): OK
  Lint (cargo clippy): OK
  Reference benchmark: OK (2 modes)
  Distortion benchmark: OK (3 modes)
  Blur3 benchmark: OK
  RTv3D benchmark: OK (2 strategies)  — or SKIPPED (no data)
  Maintainability: OK
```

## Baseline management

To regenerate baselines after intentional improvements:

1. Run the benchmarks (Step 2)
2. Update `tools/ci/regression_baseline.json` with the new metric values, adding appropriate margins:
   - Precision/recall: use exact value (these should be 1.0 or very close)
   - Center error: actual + 0.03 px margin
   - H vs GT error: actual + 0.03 px margin
   - Blur3 recall: actual - 0.01 margin (allows slight variation)
3. Commit the updated baseline

## Key files

- Baselines: `tools/ci/regression_baseline.json`
- Maintainability: `tools/ci/maintainability_baseline.json`, `tools/ci/maintainability_guardrails.py`
- Benchmark scripts: `tools/run_reference_benchmark.sh`, `tools/run_distortion_benchmark.sh`, `tools/run_blur3_benchmark.sh`, `tools/run_rtv3d_eval.py`
- Benchmark outputs: `tools/out/reference_benchmark_post_pipeline/`, `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/`, `tools/out/eval_blur3_post_pipeline/`, `tools/out/rtv3d_eval/`
