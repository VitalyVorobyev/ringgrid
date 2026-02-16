# DetectedMarker

`DetectedMarker` represents a single detected ring marker in the image. Each marker carries its decoded ID (when available), pixel-space center, fitted ellipse parameters, and quality metrics.

**Source:** `crates/ringgrid/src/detector/marker_build.rs`

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `Option<usize>` | Codebook index (0--892). `None` if decoding was rejected due to insufficient confidence or Hamming distance. |
| `confidence` | `f32` | Combined detection and decode confidence in `[0, 1]`. |
| `center` | `[f64; 2]` | Marker center in raw image pixel coordinates `[x, y]`. |
| `center_mapped` | `Option<[f64; 2]>` | Marker center in working-frame coordinates. Present only when a `PixelMapper` is active. |
| `ellipse_outer` | `Option<Ellipse>` | Fitted outer ring ellipse parameters. |
| `ellipse_inner` | `Option<Ellipse>` | Fitted inner ring ellipse parameters. Present when inner fitting succeeded. |
| `edge_points_outer` | `Option<Vec<[f64; 2]>>` | Raw sub-pixel outer edge inlier points used for ellipse fitting. |
| `edge_points_inner` | `Option<Vec<[f64; 2]>>` | Raw sub-pixel inner edge inlier points used for ellipse fitting. |
| `fit` | `FitMetrics` | Fit quality metrics. See [FitMetrics](fit-metrics.md). |
| `decode` | `Option<DecodeMetrics>` | Decode quality metrics. Present when decoding was attempted. See [FitMetrics & DecodeMetrics](fit-metrics.md). |

## Center coordinate frames

The `center` field is **always** in raw image pixel coordinates, regardless of whether a `PixelMapper` is active. This ensures that downstream consumers can always overlay detections on the original image without coordinate conversion.

When a mapper is active (e.g., radial distortion correction), `center_mapped` provides the corresponding position in the working frame (undistorted pixel space). The working-frame center is used internally for homography fitting and completion, but the image-space center remains the canonical output.

## Ellipse coordinate frame

The `ellipse_outer` and `ellipse_inner` fields use the `Ellipse` type with five parameters:

| Parameter | Description |
|-----------|-------------|
| `cx`, `cy` | Ellipse center |
| `a` | Semi-major axis length |
| `b` | Semi-minor axis length |
| `angle` | Rotation angle in radians |

When no mapper is active, the ellipse coordinates are in image space. When a mapper is active, ellipses are in the **working frame** (undistorted pixel space), because edge sampling and fitting operate in that frame. This means that `ellipse_outer.cx` may differ from `center[0]` when a mapper is active.

## Markers without decoded IDs

Markers with `id: None` were detected (ellipse fitted successfully) but failed the codebook matching step. Possible reasons include:

- Hamming distance to the nearest codeword exceeded the threshold.
- Decode confidence fell below the minimum.
- Insufficient contrast in the code band.

These markers still have valid `center`, `ellipse_outer`, and `fit` fields. They can be useful for distortion estimation or as candidate positions, but they do not contribute to the homography fit.

## Serialization

`DetectedMarker` derives `serde::Serialize` and `serde::Deserialize`. All `Option` fields use `#[serde(skip_serializing_if = "Option::is_none")]`, so absent fields are omitted from JSON output.

## Example JSON

A fully decoded marker:

```json
{
  "id": 127,
  "confidence": 0.92,
  "center": [800.5, 600.2],
  "ellipse_outer": {
    "cx": 800.5, "cy": 600.2, "a": 16.3, "b": 15.9, "angle": 0.05
  },
  "ellipse_inner": {
    "cx": 800.4, "cy": 600.1, "a": 10.7, "b": 10.4, "angle": 0.06
  },
  "fit": {
    "n_angles_total": 64,
    "n_angles_with_both_edges": 60,
    "n_points_outer": 60,
    "n_points_inner": 55,
    "ransac_inlier_ratio_outer": 0.95,
    "ransac_inlier_ratio_inner": 0.91,
    "rms_residual_outer": 0.28,
    "rms_residual_inner": 0.35
  },
  "decode": {
    "observed_word": 52419,
    "best_id": 127,
    "best_rotation": 7,
    "best_dist": 0,
    "margin": 4,
    "decode_confidence": 0.92
  }
}
```

A marker that was detected but not decoded:

```json
{
  "confidence": 0.3,
  "center": [200.1, 150.8],
  "ellipse_outer": {
    "cx": 200.1, "cy": 150.8, "a": 14.2, "b": 12.1, "angle": 0.78
  },
  "fit": {
    "n_angles_total": 64,
    "n_angles_with_both_edges": 31,
    "n_points_outer": 42,
    "n_points_inner": 0,
    "ransac_inlier_ratio_outer": 0.72,
    "rms_residual_outer": 0.89
  }
}
```
