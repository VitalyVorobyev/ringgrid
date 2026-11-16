---
name: regression-gate
description: Run regression benchmarks and maintainability guardrails. Rebuilds, runs 4 synthetic benchmarks (hex-coded reference/distortion/blur3 + the target-matrix benchmark covering every lattice x coding x dots combination and the Extended codebook) + 1 real-world benchmark + code quality checks, compares against baselines, auto-fixes maintainability issues, reports benchmark regressions. Invoke after significant code changes or before releases.
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

Run all 4 synthetic benchmark scripts in sequence using `.venv`:

```bash
bash tools/run_reference_benchmark.sh
bash tools/run_distortion_benchmark.sh
bash tools/run_blur3_benchmark.sh
bash tools/run_target_matrix_benchmark.sh
```

Each script writes results to `tools/out/`. If any script fails, report the error and continue with the remaining benchmarks.

The target-matrix benchmark (~20 s) runs one case per target combination not
covered by the hex-coded benchmarks: `rect_plain_dots`, `rect_plain_nodots`,
`rect_coded`, `hex_plain_dots`, `hex_plain_nodots`, and `hex_coded_extended`
(1239-cell board on the Extended codebook, IDs up to 1238). Images are rendered
like the library's printed target by `tools/gen_synth_target.py` and scored by
`tools/score_detect_target.py`; `--cases <name>...` runs a subset.

**RTv3D real-world benchmark (local only — private data)**

If the `data/rtv3d_0` directory exists (the eval's default `--data-dir`), also run:

```bash
.venv/bin/python tools/run_rtv3d_eval.py
```

If `data/rtv3d_0` does not exist, skip this step and report `RTv3D benchmark: SKIPPED (no data)`. This dataset is private and not available in CI: never commit or copy anything derived from it (tiles, detections, overlays) outside the gitignored `data/` and `tools/out/`. It has 17 usable strips (2 on the eval's skip list) = 102 tiles; strategies A, B and D are gated (`total_decoded`), C (8-220 px single pass) detects nothing by design of the proposal stage.

## Step 3 — Check benchmark results against baselines

```bash
.venv/bin/python tools/ci/check_regression_baseline.py
```

It reads `tools/ci/regression_baseline.json` and checks every benchmark's
output (exit code 1 on any failure, printing each failing check; `--verbose`
prints all checks). Each threshold `min_<metric>` requires `value >= bound`,
`max_<metric>` requires `value <= bound`, where the value is the output's
`<metric>` or `avg_<metric>` — for top-level `criteria` (blur3) or per
`modes` / `strategies` entry. A missing or null value fails. The rtv3d entry is
skipped when `data/rtv3d_0` is absent.

What the gated metrics mean:

- **reference / distortion** (`summary.json`, per mode): precision, recall,
  `center_mean_px`, `h_vs_gt_mean_px`.
- **blur3** (`aggregate.json`): precision, recall, `center_error`.
- **target matrix** (`summary.json`, per case): precision, recall,
  `center_mean_px`, `coord_accuracy` (grid labels vs ground truth: exact in the
  absolute frame, best lattice rotation + translation in the relative frame);
  with dots or coded: `origin_resolution_rate`, `origin_correct_rate`,
  `h_vs_gt_mean_px`; without dots: `origin_resolution_rate <= 0` (resolving an
  origin without fiducials would mean fabricated anchoring); coded:
  `id_accuracy` and `decoded_fraction` (share of markers decoded from their own
  code band rather than filled in by completion); Extended: `ext_recall` /
  `ext_decoded_rate` over IDs beyond the 893-word base table.
  The no-dots modes gate both `coord_accuracy` and
  `coord_accuracy_up_to_mirror` at 1.0: a drop in the first only means the
  relative frame came out mirrored, a drop in both means wrong labels.
- **rtv3d** (`report.json`, per strategy): `total_decoded`.

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
  Target matrix benchmark: OK (6 modes)
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
   - Decode fractions (`decoded_fraction`, `ext_decoded_rate`): actual - 0.01
   - Coordinate / origin / ID accuracy: exact
3. Commit the updated baseline

## Key files

- Baselines: `tools/ci/regression_baseline.json`
- Maintainability: `tools/ci/maintainability_baseline.json`, `tools/ci/maintainability_guardrails.py`
- Baseline check: `tools/ci/check_regression_baseline.py`
- Benchmark scripts: `tools/run_reference_benchmark.sh`, `tools/run_distortion_benchmark.sh`, `tools/run_blur3_benchmark.sh`, `tools/run_target_matrix_benchmark.sh`, `tools/run_rtv3d_eval.py`
- Benchmark outputs: `tools/out/reference_benchmark_post_pipeline/`, `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/`, `tools/out/eval_blur3_post_pipeline/`, `tools/out/target_matrix_benchmark/`, `tools/out/rtv3d_eval/`
