# Simple Detection

Simple detection is the most straightforward mode: no camera model, no distortion
correction. The detector runs a single pass directly in image pixel coordinates.

## Pipeline

When you call `detector.detect(&image)` with self-undistort disabled (the default),
the pipeline runs once in raw image coordinates:

1. Gradient voting and NMS produce candidate centers.
2. Outer and inner ellipses are fitted via RANSAC.
3. 16-sector codes are sampled and matched against the 893-codeword codebook.
4. Spatial and ID-based deduplication removes redundant detections.
5. If enough decoded markers exist, a RANSAC homography is fitted.
6. H-guided refinement and completion fill in missing markers.

All geometry stays in image space throughout.

## Coordinate Frames

| Field | Frame |
|---|---|
| `center` | Image (distorted pixel coordinates) |
| `center_mapped` | `None` |
| `homography` | Image -> Board (maps board mm to image pixels) |
| `center_frame` | `DetectionFrame::Image` |
| `homography_frame` | `DetectionFrame::Image` |

## When to Use

- The camera has negligible lens distortion.
- You want the fastest possible results without extra configuration.
- You are working with synthetic or pre-rectified images.
- You plan to handle distortion correction externally.

## Basic Usage

The minimal workflow loads a board layout, opens a grayscale image, and runs
detection:

```rust
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json"))?;
let image = image::open("photo.png")?.to_luma8();

let detector = Detector::new(board);
let result = detector.detect(&image);

for marker in &result.detected_markers {
    if let Some(id) = marker.id {
        println!("Marker {} at ({:.1}, {:.1})", id, marker.center[0], marker.center[1]);
    }
}
```

## Providing a Marker Diameter Hint

If you know the approximate marker diameter in pixels, passing it as a hint
narrows the radius search and speeds up detection:

```rust
let detector = Detector::with_marker_diameter_hint(board, 32.0);
```

## Providing a Scale Range

When marker sizes vary across the image (e.g. perspective foreshortening), you
can specify a min/max diameter range with `MarkerScalePrior`:

```rust
use ringgrid::MarkerScalePrior;

let detector = Detector::with_marker_scale(board, MarkerScalePrior::new(20.0, 56.0));
```

## One-Step Construction from JSON

`Detector::from_target_json_file` loads the board layout and creates the detector
in a single call:

```rust
let detector = Detector::from_target_json_file(Path::new("target.json"))?;
```

## Post-Construction Tuning

After creating the detector you can adjust individual configuration fields through
`config_mut()`. For example, to disable the completion stage:

```rust
let mut detector = Detector::new(board);
detector.config_mut().completion.enable = false;
```

## Serializing Results

`DetectionResult` implements `serde::Serialize`, so you can write it to JSON
directly:

```rust
let json = serde_json::to_string_pretty(&result)?;
std::fs::write("output.json", json)?;
```

The output JSON contains detected markers with their IDs, centers, ellipse
parameters, fit metrics, and (when available) the board-to-image homography.

## Source Files

- `crates/ringgrid/src/api.rs` -- `Detector` struct and all constructor variants.
- `crates/ringgrid/examples/basic_detect.rs` -- complete runnable example.
