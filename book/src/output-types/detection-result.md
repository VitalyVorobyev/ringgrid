# DetectionResult

`DetectionResult` is the top-level output returned by `Detector::detect()` and `Detector::detect_with_mapper()`. It contains all detected markers, the fitted board-to-image homography (when available), and metadata describing the coordinate frames used.

**Source:** `crates/ringgrid/src/pipeline/result.rs`

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `detected_markers` | `Vec<DetectedMarker>` | All detected markers in the image. See [DetectedMarker](detected-marker.md). |
| `center_frame` | `DetectionFrame` | Coordinate frame of each marker's `center` field. Current contract: always `Image`. |
| `homography_frame` | `DetectionFrame` | Coordinate frame of the `homography` matrix (`Image` or `Working`). |
| `image_size` | `[u32; 2]` | Image dimensions as `[width, height]`. |
| `homography` | `Option<[[f64; 3]; 3]>` | 3x3 row-major board-to-output-frame homography. Present when 4 or more markers were decoded. |
| `ransac` | `Option<RansacStats>` | RANSAC quality statistics for the homography fit. See [RansacStats](ransac-stats.md). |
| `self_undistort` | `Option<SelfUndistortResult>` | Estimated division-model distortion correction, present when self-undistort mode was used. |

## DetectionFrame

`DetectionFrame` is an enum with two variants:

- **`Image`** -- raw image pixel coordinates.
- **`Working`** -- working-frame coordinates (undistorted pixel space when a `PixelMapper` is active).

## Frame conventions

The values of `center_frame` and `homography_frame` depend on how detection was invoked:

| Detection mode | `center_frame` | `homography_frame` |
|----------------|----------------|---------------------|
| `Detector::detect()` (no mapper) | `Image` | `Image` |
| `Detector::detect_with_mapper()` | `Image` | `Working` |
| Self-undistort (correction not applied) | `Image` | `Image` |
| Self-undistort (correction applied) | `Image` | `Working` |

Marker centers (`DetectedMarker::center`) are always in image-space pixel coordinates, regardless of mapper usage. When a mapper is active, the working-frame center is available in `DetectedMarker::center_mapped`. The homography maps board coordinates to whichever frame `homography_frame` indicates.

## Homography

The `homography` field contains a 3x3 row-major matrix that maps board coordinates (in mm) to the output frame (image or working, as indicated by `homography_frame`). It is computed via RANSAC when at least 4 decoded markers are available.

To project a board point `(bx, by)` through the homography:

```
[u']     [h[0][0]  h[0][1]  h[0][2]]   [bx]
[v'] =   [h[1][0]  h[1][1]  h[1][2]] * [by]
[w ]     [h[2][0]  h[2][1]  h[2][2]]   [1 ]

pixel_x = u' / w
pixel_y = v' / w
```

## Serialization

`DetectionResult` derives `serde::Serialize` and `serde::Deserialize`. Optional fields (`homography`, `ransac`, `self_undistort`) use `#[serde(skip_serializing_if = "Option::is_none")]`, so they are omitted from the JSON output when absent.

## Example JSON

A typical serialized result with a fitted homography:

```json
{
  "detected_markers": [
    {
      "id": 42,
      "confidence": 0.95,
      "center": [512.3, 384.7],
      "ellipse_outer": {
        "cx": 512.3, "cy": 384.7, "a": 16.1, "b": 15.8, "angle": 0.12
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
      }
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

When no homography could be fitted (fewer than 4 decoded markers), the `homography` and `ransac` fields are omitted entirely from the JSON output.
