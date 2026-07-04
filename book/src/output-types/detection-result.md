# DetectionResult & DetectionDiagnostics

`DetectionResult` is the slim, stable top-level output returned by
`Detector::detect()` and `Detector::detect_with_mapper()`. It contains all
detected markers, the fitted board-to-image homography (when available), and
metadata describing the coordinate frames used.

Algorithm internals — per-marker fit/decode metrics, raw edge points, and the
homography RANSAC statistics — are **not** on `DetectionResult`. They live in
the opt-in [`DetectionDiagnostics`](#detectiondiagnostics) channel, returned
alongside the result by `Detector::detect_with_diagnostics()` (and the mapper
variant `detect_with_mapper_diagnostics()`).

`DetectionResult` is `#[non_exhaustive]`: construct it via `DetectionResult::empty`
or `Default` and mutate fields, rather than with a struct literal.

For the full CLI JSON file shape written by `ringgrid detect`, including the
nested `diagnostics` object and optional top-level `camera` and
proposal-diagnostics fields, see [Detection Output Format](../output-format.md).

**Source:** `crates/ringgrid/src/pipeline/result.rs`

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `detected_markers` | `Vec<DetectedMarker>` | All detected markers in the image. See [DetectedMarker](detected-marker.md). |
| `center_frame` | `DetectionFrame` | Coordinate frame of each marker's `center` field. Current contract: always `Image`. |
| `homography_frame` | `DetectionFrame` | Coordinate frame of the `homography` matrix (`Image` or `Working`). |
| `image_size` | `[u32; 2]` | Image dimensions as `[width, height]`. |
| `homography` | `Option<[[f64; 3]; 3]>` | 3x3 row-major board-to-output-frame homography. Present when 4 or more markers were decoded. |
| `board_frame` | `Option<BoardFrame>` | Reference frame of `grid_coord` / `board_xy_mm` / `homography` outputs: `Absolute` or `RelativeCanonical`. `None` when no grid assignment took place (no markers labeled). Coded targets are always `Absolute`; plain targets are `Absolute` only when origin fiducials resolved the board origin. See [Origin Fiducials](../targets/origin-fiducials.md). |
| `self_undistort` | `Option<SelfUndistortResult>` | Estimated division-model distortion correction, present when self-undistort mode was used. |

`DetectionResult` no longer carries a `ransac` field — the RANSAC homography
statistics moved to [`DetectionDiagnostics`](#detectiondiagnostics) in v0.6.
The pipeline-internal `seed_proposals()` method was also removed.

## DetectionDiagnostics

`DetectionDiagnostics` is the opt-in diagnostics counterpart to
`DetectionResult`. It is returned only by `Detector::detect_with_diagnostics()`;
`detect()` returns the slim `DetectionResult` alone.

| Field | Type | Description |
|-------|------|-------------|
| `markers` | `Vec<MarkerDiagnostics>` | Per-marker algorithm internals. Positionally aligned 1:1 with `DetectionResult::detected_markers`: `markers[i]` describes the marker at index `i`. See [MarkerDiagnostics](fit-metrics.md#markerdiagnostics). |
| `ransac` | `Option<RansacStats>` | RANSAC quality statistics for the homography fit. Present when a homography was fitted. See [RansacStats](ransac-stats.md). |

`DetectionDiagnostics` is `#[non_exhaustive]` and derives `serde::Serialize` /
`serde::Deserialize`. In the CLI `detect.json` it appears as the nested
`diagnostics` object.

```rust
let (result, diagnostics) = detector.detect_with_diagnostics(&image);
if let Some(stats) = &diagnostics.ransac {
    println!("homography inliers: {}/{}", stats.n_inliers, stats.n_candidates);
}
```

## DetectionFrame

`DetectionFrame` is an enum with two variants:

- **`Image`** -- raw image pixel coordinates.
- **`Working`** -- working-frame coordinates (undistorted pixel space when a `PixelMapper` is active).

## BoardFrame

`BoardFrame` is an enum with two variants, reported on `board_frame` whenever
grid-labeled outputs (`grid_coord`, `board_xy_mm`, and the homography's source
plane) are available:

- **`Absolute`** -- outputs are absolute board-frame values. Always the case
  for coded targets (decoded IDs anchor markers to physical cells); for plain
  targets, only when origin fiducials resolved the board origin.
- **`RelativeCanonical`** -- origin unresolved: `grid_coord` is in a canonical
  relative frame (non-negative, `+u` roughly along image `+x`); `board_xy_mm`
  is absent on every labeled marker.

`BoardFrame::origin_resolved()` is a convenience predicate equivalent to
`matches!(frame, BoardFrame::Absolute)`. See [Origin Fiducials](../targets/origin-fiducials.md)
for the full resolution algorithm and the per-marker consequences.

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

`DetectionResult` derives `serde::Serialize` and `serde::Deserialize`. Optional fields (`homography`, `board_frame`, `self_undistort`) use `#[serde(skip_serializing_if = "Option::is_none")]`, so they are omitted from the JSON output when absent -- never serialized as `null`. Serializing a `DetectionResult` does **not** include diagnostics — serialize the `DetectionDiagnostics` value separately, or use the CLI, which nests it under `diagnostics`.

## Example JSON

A typical serialized `DetectionResult` with a fitted homography:

```json
{
  "detected_markers": [
    {
      "id": 42,
      "grid_coord": [6, 3],
      "confidence": 0.95,
      "center": [512.3, 384.7],
      "board_xy_mm": [48.0, 24.0],
      "ellipse_outer": {
        "cx": 512.3, "cy": 384.7, "a": 16.1, "b": 15.8, "angle": 0.12
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
  "board_frame": "absolute"
}
```

The matching `DetectionDiagnostics` serializes as:

```json
{
  "markers": [
    {
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
  "ransac": {
    "n_candidates": 35,
    "n_inliers": 33,
    "threshold_px": 5.0,
    "mean_err_px": 0.72,
    "p95_err_px": 1.45
  }
}
```

When no homography could be fitted (fewer than 4 decoded markers), the
`homography` field is omitted from `DetectionResult` and `ransac` is omitted
from `DetectionDiagnostics`.
