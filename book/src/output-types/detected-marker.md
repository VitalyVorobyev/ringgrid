# DetectedMarker

`DetectedMarker` represents a single detected ring marker in the image. It is
the slim, stable primary output: decoded ID (when available), pixel-space
center, board coordinates, and the fitted ellipse parameters.

Algorithm internals — fit metrics, decode metrics, raw edge points, and the
producing pipeline stage — are **not** fields of `DetectedMarker`. They live in
the opt-in [`MarkerDiagnostics`](fit-metrics.md#markerdiagnostics) channel,
obtained via `Detector::detect_with_diagnostics`. Each `MarkerDiagnostics` is
positionally aligned 1:1 with the corresponding `DetectedMarker`.

`DetectedMarker` is `#[non_exhaustive]`: construct it via `Default` and mutate
fields, rather than with a struct literal.

**Source:** `crates/ringgrid/src/pipeline/result.rs`

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `Option<usize>` | Codebook index in the active profile. `None` if decoding was rejected due to insufficient confidence or Hamming distance. |
| `confidence` | `f32` | Combined detection and decode confidence in `[0, 1]`. |
| `center` | `[f64; 2]` | Marker center in raw image pixel coordinates `[x, y]`. |
| `center_mapped` | `Option<[f64; 2]>` | Marker center in working-frame coordinates. Present only when a `PixelMapper` is active. |
| `board_xy_mm` | `Option<[f64; 2]>` | Board-space marker location in millimeters (`BoardLayout::xy_mm` semantics). Present only when `id` is valid for the active board layout. |
| `ellipse_outer` | `Option<Ellipse>` | Fitted outer ring ellipse parameters. |
| `ellipse_inner` | `Option<Ellipse>` | Fitted inner ring ellipse parameters. Present when inner fitting succeeded. |

## Relocated diagnostics fields

The following fields were on `DetectedMarker` before v0.6 and now live on the
paired [`MarkerDiagnostics`](fit-metrics.md#markerdiagnostics) entry:

- `fit` — fit quality metrics. See [FitMetrics](fit-metrics.md).
- `decode` — decode quality metrics. See [DecodeMetrics](fit-metrics.md#decodemetrics).
- `source` — pipeline path that produced the marker.
- `edge_points_outer`, `edge_points_inner` — raw sub-pixel edge inlier points.

Access them by zipping `result.detected_markers` with `diagnostics.markers`:

```rust
let (result, diagnostics) = detector.detect_with_diagnostics(&image);
for (m, d) in result.detected_markers.iter().zip(&diagnostics.markers) {
    println!("{:?} source={:?} pts={}", m.id, d.source, d.fit.n_points_outer);
}
```

## DetectionSource

`MarkerDiagnostics.source` tells you how the marker entered the final result:

- `fit_decoded` -- the normal proposal -> fit -> decode path
- `completion` -- the homography-guided completion stage filled a missing board marker
- `seeded_pass` -- the marker was re-fitted during mapper-based pass-2 detection

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

Markers with `id: None` were detected (ellipse fitted successfully) but failed the codebook matching or structural verification stage. Possible reasons include:

- Hamming distance to the nearest codeword exceeded the threshold.
- Decode confidence fell below the minimum.
- ID contradicted board-local structural consistency in `id_correction`.
- Insufficient contrast in the code band.

These markers still have valid `center` and `ellipse_outer` fields, and their
paired `MarkerDiagnostics` entry keeps the `fit` metrics. They can be useful for
distortion estimation or as candidate positions, but they do not contribute to
the homography fit.

## ID/board consistency contract

Final emitted markers enforce strict ID/layout consistency:

- if `id` is `Some(i)`, then `board_xy_mm` is present and equals the active board layout coordinate of `i`
- if `id` is `None`, then `board_xy_mm` is omitted
- if a decoded ID is not found in the active board layout, it is cleared before output (`id=None`, `board_xy_mm=None`)

## Serialization

`DetectedMarker` derives `serde::Serialize` and `serde::Deserialize`. All `Option` fields use `#[serde(skip_serializing_if = "Option::is_none")]`, so absent fields are omitted from JSON output. The relocated fit/decode/source/edge-point fields serialize on the paired [`MarkerDiagnostics`](fit-metrics.md#markerdiagnostics) value instead.

## Example JSON

A fully decoded marker:

```json
{
  "id": 127,
  "board_xy_mm": [40.0, 24.0],
  "confidence": 0.92,
  "center": [800.5, 600.2],
  "ellipse_outer": {
    "cx": 800.5, "cy": 600.2, "a": 16.3, "b": 15.9, "angle": 0.05
  },
  "ellipse_inner": {
    "cx": 800.4, "cy": 600.1, "a": 10.7, "b": 10.4, "angle": 0.06
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
  }
}
```

The corresponding fit and decode metrics for these markers live in
`diagnostics.markers` — see
[FitMetrics, DecodeMetrics & MarkerDiagnostics](fit-metrics.md).
