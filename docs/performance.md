# Performance and Evaluation

This page collects the scoring notes, benchmark snapshots, and reference commands that were moved out of the root `README.md`.

## Distortion Correction Modes

ringgrid supports three distortion-correction modes:

| Mode | Requires | Method |
|---|---|---|
| None | — | Single-pass in image coordinates |
| External camera | Calibrated intrinsics + distortion | Two-pass with `CameraModel` implementing `PixelMapper` |
| Self-undistort | Nothing | Estimates a single-parameter division model from detected markers |

Camera model and self-undistort are mutually exclusive. The detailed algorithm notes live in the [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/).

## Synthetic Scoring Metrics

`tools/score_detect.py` reports several geometric metrics. The three main ones are:

- `center_error`: true-positive error between predicted `marker.center` and ground-truth center in the selected frame
- `homography_self_error`: homography self-consistency error between `project(H_est, board_xy_mm)` and the predicted marker center
- `homography_error_vs_gt`: absolute error between the estimated homography and ground-truth projection

Interpretation:

- Lower is better for all three.
- `homography_self_error` can be lower than `center_error`, because it measures consistency against detected centers, not absolute ground-truth center error.
- For cross-run comparisons, score all variants in distorted image space.
- Prefer `--center-gt-key image --homography-gt-key image`.
- Prefer `--pred-center-frame auto --pred-homography-frame auto` so the scorer uses detector-emitted frame metadata.

Distortion-aware eval example:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 3 \
  --blur_px 0.8 \
  --out_dir tools/out/r4_distortion_eval \
  --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --pass_camera_to_detector
```

Self-undistort eval example:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 3 \
  --blur_px 0.8 \
  --out_dir tools/out/r4_self_undistort_eval \
  --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --self_undistort
```

## Benchmark Snapshots

### Distortion Benchmark (Projective-Center, 3 Images)

Source:

- `tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline/summary.json`

Example distorted sample used in this benchmark:

![Distortion benchmark sample](assets/distortion_benchmark_sample.png)

Run command:

```bash
./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0 --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --corrections none external self_undistort \
  --modes projective_center
```

The benchmark script defaults to `cargo run` so the source tree remains the build of record. Use `--use-prebuilt-binary` only when you intentionally want to benchmark an existing binary artifact.

Image-space metric snapshot:

| Correction | Precision | Recall | Center mean (px) | H self mean/p95 (px) | H vs GT mean/p95 (px) |
|---|---:|---:|---:|---:|---:|
| `none` | 1.000 | 0.974 | 0.232 | 1.030 / 2.961 | 1.345 / 3.611 |
| `external` | 1.000 | 1.000 | 0.078 | 0.075 / 0.146 | 0.020 / 0.029 |
| `self_undistort` | 1.000 | 1.000 | 0.078 | 0.210 / 0.426 | 0.193 / 0.408 |

Notes:

- Scripts score in distorted image space for all three correction variants.
- For `self_undistort`, the scoring frame is selected per image from `self_undistort.applied` in detection JSON.
- On this synthetic distortion setup, self-undistort is much better than no correction, but still less accurate than external calibration parameters.

### Reference Benchmark (Clean, 3 Images)

Source:

- `tools/out/reference_benchmark_post_pipeline/summary.json`

Run command:

```bash
./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/reference_benchmark_post_pipeline \
  --n_images 3 \
  --blur_px 0.8 \
  --noise_sigma 0.0 \
  --marker_diameter 32.0 \
  --modes none projective_center
```

| Mode | Center mean (px) | H self mean/p95 (px) | H vs GT mean/p95 (px) |
|---|---:|---:|---:|
| `none` | 0.072 | 0.065 / 0.132 | 0.033 / 0.049 |
| `projective-center` | 0.054 | 0.051 / 0.098 | 0.019 / 0.030 |

### Regression Batch (10 Images)

Source:

- `tools/out/regress_r2_batch/det/aggregate.json`

This set is intentionally harder (`blur_px=3.0`), and markers are visibly weak or blurred.

Example image from this stress set:

![Regression blur=3.0 sample](assets/regression_blur3_sample.png)

Run command:

```bash
./.venv/bin/python tools/run_synth_eval.py \
  --n 10 \
  --blur_px 3.0 \
  --out_dir tools/out/regress_r2_batch \
  --marker_diameter 32.0
```

Snapshot:

| Metric | Value |
|---|---:|
| Images | 10 |
| Avg precision | 1.000 |
| Avg recall | 0.949 |
| Avg TP / image | 192.6 |
| Avg FP / image | 0.0 |
| Avg center error (px) | 0.278 |
| Avg H vs GT error (px) | 0.147 |
| Avg H self error (px) | 0.235 |

## Related Docs

- [docs/tuning-guide.md](tuning-guide.md) for symptom-driven tuning notes
- [docs/proposal-performance-analysis.md](proposal-performance-analysis.md) for proposal-stage cost structure and alternative approaches
- [User Guide](https://vitalyvorobyev.github.io/ringgrid/book/) for pipeline, math, and configuration detail
- [Development Guide](development.md) for repo-maintainer workflows and validation commands
