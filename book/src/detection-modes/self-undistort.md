# Self-Undistort Mode

Self-undistort estimates lens distortion directly from the detected markers,
without requiring camera calibration data. It uses a 1-parameter division model
and decides automatically whether the estimated correction is beneficial.

## When to Use

- You do not have calibrated camera intrinsics or distortion coefficients.
- You suspect the lens has significant barrel or pincushion distortion.
- You want the detector to self-correct without external calibration.

If you have a full camera model, use [External PixelMapper](external-mapper.md)
instead -- it is more accurate for cameras with tangential distortion or higher-order
radial terms.

## Division Model

The self-undistort stage fits a single-parameter division model:

```
x_u = cx + (x_d - cx) / (1 + lambda * r^2)
y_u = cy + (y_d - cy) / (1 + lambda * r^2)
```

where `r^2 = (x_d - cx)^2 + (y_d - cy)^2` and `(cx, cy)` is the image center.

- **Negative lambda** corresponds to barrel distortion (most common in wide-angle
  lenses).
- **Positive lambda** corresponds to pincushion distortion.
- **lambda = 0** is the identity (no correction).

## Pipeline Flow

1. **Baseline detection** -- a standard single-pass detect runs in image
   coordinates, producing markers with inner and outer edge points.
2. **Lambda estimation** -- a golden-section search minimizes a robust objective
   over the configured lambda range. The objective measures how well inner and
   outer ring edges fit ellipses after undistortion (lower residual = better
   conic consistency). When enough decoded markers with board IDs are available,
   a homography-based objective is used instead for higher accuracy.
3. **Accept/reject** -- the estimated lambda is accepted only if:
   - The objective improvement exceeds `improvement_threshold` (relative) and
     `min_abs_improvement` (absolute).
   - The estimated `|lambda|` exceeds `min_lambda_abs`.
   - The solution is not at the edge of the search range (when
     `reject_range_edge` is enabled).
   - Homography validation passes (when enough decoded markers are available).
4. **Second pass** -- if accepted, detection re-runs with the estimated division
   model as a pixel mapper, using pass-1 markers as seeds.

## Enabling Self-Undistort

Self-undistort is disabled by default. Enable it through the config before
creating the detector:

```rust
use ringgrid::{BoardLayout, DetectConfig, Detector, MarkerScalePrior};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json"))?;
let image = image::open("photo.png")?.to_luma8();

let mut cfg = DetectConfig::from_target(board);
cfg.self_undistort.enable = true;
cfg.self_undistort.min_markers = 12;

let detector = Detector::with_config(cfg);
let result = detector.detect(&image);

if let Some(su) = &result.self_undistort {
    println!("Lambda: {:.3e}, applied: {}", su.model.lambda, su.applied);
    println!("Objective: {:.4} -> {:.4}", su.objective_at_zero, su.objective_at_lambda);
}
```

## SelfUndistortResult

When `self_undistort.enable` is `true`, `result.self_undistort` is always `Some`,
even if the correction was not applied. The struct contains:

| Field | Type | Description |
|---|---|---|
| `model` | `DivisionModel` | Estimated division model (lambda, cx, cy) |
| `applied` | `bool` | Whether the model was actually used for re-detection |
| `objective_at_zero` | `f64` | Baseline objective value (lambda = 0) |
| `objective_at_lambda` | `f64` | Objective value at the estimated lambda |
| `n_markers_used` | `usize` | Number of markers used in the estimation |

If `applied` is `false`, the estimated lambda did not meet the acceptance
criteria and detection results are from the baseline (image-coordinate) pass
only.

## Configuration Parameters

The `SelfUndistortConfig` struct controls the estimation behavior:

| Parameter | Default | Description |
|---|---|---|
| `enable` | `false` | Master switch for self-undistort |
| `lambda_range` | `[-8e-7, 8e-7]` | Search range for the lambda parameter |
| `max_evals` | `40` | Maximum objective evaluations for golden-section search |
| `min_markers` | `6` | Minimum markers with both inner+outer edge points |
| `improvement_threshold` | `0.01` | Minimum relative objective improvement |
| `min_abs_improvement` | `1e-4` | Minimum absolute objective improvement |
| `trim_fraction` | `0.1` | Trim fraction for robust aggregation (drop 10% tails) |
| `min_lambda_abs` | `5e-9` | Minimum \|lambda\| to consider non-trivial |
| `reject_range_edge` | `true` | Reject solutions near lambda-range boundaries |
| `range_edge_margin_frac` | `0.02` | Relative margin treated as boundary zone |
| `validation_min_markers` | `24` | Minimum decoded-ID matches for H-validation |
| `validation_abs_improvement_px` | `0.05` | Minimum absolute H-error improvement (px) |
| `validation_rel_improvement` | `0.03` | Minimum relative H-error improvement |

## Coordinate Frames

When the model is applied (`applied == true`):

| Field | Frame |
|---|---|
| `center` | Image (distorted pixel coordinates) |
| `center_mapped` | Working (undistorted via division model) |
| `homography` | Working -> Board |
| `homography_frame` | `DetectionFrame::Working` |

When the model is not applied (`applied == false`), the result is identical to
[simple detection](simple.md) -- all coordinates are in image space and
`center_mapped` is `None`.

## Important Notes

- Self-undistort and external mapper are mutually exclusive. Calling
  `detect_with_mapper` never runs self-undistort, regardless of the config
  setting.
- The division model has only one free parameter (lambda). It cannot model
  tangential distortion or higher-order radial terms. For complex distortion,
  use a calibrated `CameraModel` instead.
- The estimation requires enough detected markers with usable edge points.
  If the baseline pass finds fewer than `min_markers` qualifying markers,
  `result.self_undistort` will be `None`.

## Source Files

- `crates/ringgrid/src/api.rs` -- `Detector::detect` branches on `self_undistort.enable`.
- `crates/ringgrid/src/pixelmap/self_undistort.rs` -- estimation logic, `SelfUndistortConfig`, `SelfUndistortResult`.
- `crates/ringgrid/src/pixelmap/distortion.rs` -- `DivisionModel` type and `PixelMapper` implementation.
- `crates/ringgrid/examples/detect_with_self_undistort.rs` -- complete runnable example.
