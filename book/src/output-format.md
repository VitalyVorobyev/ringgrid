# Detection Output Format

`ringgrid` exposes two closely related JSON shapes:

- the library value `DetectionResult`, serialized directly with `serde_json`
- the CLI output file written by `ringgrid detect --out ...`, which flattens the
  same `DetectionResult` fields at the top level and may add a few CLI-only
  fields

The primary payload is always `detected_markers`. Everything else describes the
image, coordinate frames, optional board homography, and optional diagnostics.

Optional fields are omitted when absent. They are not serialized as `null`.

## Library vs CLI

If you serialize the Rust result directly:

```rust
let json = serde_json::to_string_pretty(&result)?;
```

you get the fields of [`DetectionResult`](output-types/detection-result.md) only.

If you run:

```bash
ringgrid detect --image photo.png --out result.json
```

the written JSON contains those same top-level fields, and may additionally
include:

- `camera` when detection used `--calibration` or inline `--cam-*` parameters
- `proposal_frame`, `proposal_count`, and `proposals` when
  `--include-proposals` is enabled

## Top-Level Fields

| Field | Present when | Meaning |
|---|---|---|
| `detected_markers` | always | Final emitted markers. Each entry is a [`DetectedMarker`](output-types/detected-marker.md). |
| `center_frame` | always | Coordinate frame of each marker `center`. Current contract: always `image`. |
| `homography_frame` | always | Coordinate frame of the `homography` matrix: `image` or `working`. |
| `image_size` | always | Input image dimensions as `[width, height]`. |
| `homography` | when fitted | 3x3 row-major homography mapping board millimeters into `homography_frame`. |
| `ransac` | when homography exists | Quality statistics for the fitted homography. See [`RansacStats`](output-types/ransac-stats.md). |
| `self_undistort` | when self-undistort ran | Estimated division-model correction and whether it was applied. |
| `camera` | CLI only, when camera input was provided | The `CameraModel` used by the two-pass mapper path. |
| `proposal_frame` | CLI only, with `--include-proposals` | Coordinate frame of `proposals`. Currently always `image`. |
| `proposal_count` | CLI only, with `--include-proposals` | Number of serialized proposals. |
| `proposals` | CLI only, with `--include-proposals` | Pass-1 center proposals, each with `x`, `y`, and `score`. |

## What Each Marker Contains

Each entry in `detected_markers` describes one final marker hypothesis after the
full pipeline and post-processing:

| Field | Meaning |
|---|---|
| `id` | Decoded codebook index. Omitted when decoding was rejected or cleared. |
| `board_xy_mm` | Board-space marker location in millimeters for valid decoded IDs. |
| `confidence` | Combined fit/decode confidence in `[0, 1]`. |
| `center` | Marker center in raw image pixels. Always safe to overlay on the original image. |
| `center_mapped` | Working-frame center when a mapper was active. |
| `ellipse_outer`, `ellipse_inner` | Fitted ellipse parameters. With a mapper, ellipse coordinates are in the working frame. |
| `edge_points_outer`, `edge_points_inner` | Raw subpixel edge points retained for diagnostics and downstream analysis. |
| `fit` | Fit-quality metrics such as arc coverage, residuals, angular gaps, and reprojection error. |
| `decode` | Decode-quality metrics such as observed word, best distance, margin, and rotation. |
| `source` | Which pipeline path produced the final marker. |

`source` uses these enum values:

- `fit_decoded`: normal proposal -> fit -> decode path
- `completion`: homography-guided completion stage
- `seeded_pass`: pass-2 seeded re-fit in mapper-based detection

Markers without `id` can still be useful geometrically: they keep `center`,
ellipse fits, and fit-quality metrics, but they do not contribute to homography
estimation.

## Frames and Homography

Two frame fields tell you how to interpret the geometry:

- `center_frame` describes `DetectedMarker.center`
- `homography_frame` describes `homography`

Important contract:

- `center` is always in the original image frame
- `center_mapped` is the undistorted working-frame center when a mapper was active
- `homography` maps board millimeters into the frame named by `homography_frame`

This means:

- use `center` for overlays on the source image
- use `center_mapped` and `homography` together when working in the mapper's
  undistorted frame

See [Coordinate Frames](coordinate-frames.md) for the exact conventions.

## Typical CLI Output

```json
{
  "detected_markers": [
    {
      "id": 42,
      "board_xy_mm": [24.0, 16.0],
      "confidence": 0.95,
      "center": [512.3, 384.7],
      "ellipse_outer": {
        "cx": 512.3,
        "cy": 384.7,
        "a": 16.1,
        "b": 15.8,
        "angle": 0.12
      },
      "fit": {
        "n_angles_total": 64,
        "n_angles_with_both_edges": 58,
        "n_points_outer": 58,
        "n_points_inner": 52,
        "ransac_inlier_ratio_outer": 0.93,
        "rms_residual_outer": 0.31
      },
      "decode": {
        "observed_word": 45231,
        "best_id": 42,
        "best_rotation": 3,
        "best_dist": 0,
        "margin": 5,
        "decode_confidence": 0.95
      },
      "source": "fit_decoded"
    }
  ],
  "center_frame": "image",
  "homography_frame": "image",
  "image_size": [1920, 1080],
  "homography": [
    [3.52, 0.08, 640.1],
    [-0.05, 3.48, 480.3],
    [0.00012, -0.00003, 1.0]
  ],
  "ransac": {
    "n_candidates": 35,
    "n_inliers": 33,
    "threshold_px": 5.0,
    "mean_err_px": 0.72,
    "p95_err_px": 1.45
  }
}
```

When the CLI is run with a camera model or proposal diagnostics, extra top-level
fields are added:

```json
{
  "camera": {
    "intrinsics": { "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 480.0 },
    "distortion": { "k1": -0.15, "k2": 0.05, "p1": 0.001, "p2": -0.001, "k3": 0.0 }
  },
  "proposal_frame": "image",
  "proposal_count": 128,
  "proposals": [
    { "x": 510.2, "y": 381.7, "score": 94.8 }
  ]
}
```

## Detailed Field References

- [DetectionResult](output-types/detection-result.md)
- [DetectedMarker](output-types/detected-marker.md)
- [FitMetrics & DecodeMetrics](output-types/fit-metrics.md)
- [RansacStats](output-types/ransac-stats.md)
- [Coordinate Frames](coordinate-frames.md)
