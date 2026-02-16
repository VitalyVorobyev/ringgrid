# External PixelMapper

When you have calibrated camera intrinsics and distortion coefficients, you can
supply them to the detector as a `CameraModel`. This enables a two-pass pipeline
that accounts for lens distortion during fitting and decoding, producing more
accurate marker centers and a cleaner homography.

## Two-Pass Pipeline

Calling `detector.detect_with_mapper(&image, &camera)` runs:

1. **Pass 1 (no mapper)** -- standard single-pass detection in image coordinates.
   This produces seed proposals (approximate marker locations).
2. **Pass 2 (with mapper)** -- re-runs fitting and decoding around the pass-1
   seeds, mapping edge samples through the distortion model so that ellipse
   fitting operates in undistorted (working-frame) coordinates.

The two-pass approach avoids the cost of undistorting the entire image while still
giving the fitting stages clean, distortion-free geometry.

## Camera Model

`CameraModel` pairs pinhole intrinsics with Brown-Conrady radial-tangential
distortion:

```rust
use ringgrid::{CameraIntrinsics, CameraModel, RadialTangentialDistortion};

let camera = CameraModel {
    intrinsics: CameraIntrinsics {
        fx: 900.0,       // focal length x (pixels)
        fy: 900.0,       // focal length y (pixels)
        cx: 640.0,       // principal point x (pixels)
        cy: 480.0,       // principal point y (pixels)
    },
    distortion: RadialTangentialDistortion {
        k1: -0.15,       // radial
        k2: 0.05,        // radial
        p1: 0.001,       // tangential
        p2: -0.001,      // tangential
        k3: 0.0,         // radial (6th order)
    },
};
```

The distortion model follows the standard Brown-Conrady convention used by
OpenCV and most calibration toolboxes:

- **k1, k2, k3** -- radial distortion coefficients.
- **p1, p2** -- tangential (decentering) distortion coefficients.

Undistortion is performed iteratively (fixed-point iteration in normalized
coordinates, up to 15 iterations by default with 1e-12 convergence threshold).

## Coordinate Frames

| Field | Frame |
|---|---|
| `center` | Image (distorted pixel coordinates, always) |
| `center_mapped` | Working (undistorted pixel coordinates) |
| `homography` | Working -> Board (maps board mm to undistorted pixels) |
| `center_frame` | `DetectionFrame::Image` |
| `homography_frame` | `DetectionFrame::Working` |

Marker centers are always reported in image space so they can be overlaid on the
original photo. The `center_mapped` field provides the corresponding undistorted
position in the working frame, which is the coordinate system the homography
operates in.

## Full Example

```rust
use ringgrid::{BoardLayout, CameraIntrinsics, CameraModel, Detector, RadialTangentialDistortion};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json"))?;
let image = image::open("photo.png")?.to_luma8();
let (w, h) = image.dimensions();

let camera = CameraModel {
    intrinsics: CameraIntrinsics {
        fx: 900.0, fy: 900.0,
        cx: w as f64 * 0.5,
        cy: h as f64 * 0.5,
    },
    distortion: RadialTangentialDistortion {
        k1: -0.15, k2: 0.05,
        p1: 0.001, p2: -0.001,
        k3: 0.0,
    },
};

let detector = Detector::new(board);
let result = detector.detect_with_mapper(&image, &camera);

for marker in &result.detected_markers {
    // center is always in image (distorted) space
    println!("Image: ({:.1}, {:.1})", marker.center[0], marker.center[1]);
    // center_mapped is in working (undistorted) space
    if let Some(mapped) = marker.center_mapped {
        println!("Working: ({:.1}, {:.1})", mapped[0], mapped[1]);
    }
}
```

## Important Notes

- **Self-undistort is skipped.** When you call `detect_with_mapper`, the
  self-undistort estimation is not run, regardless of the
  `config.self_undistort.enable` setting. The provided mapper takes precedence.
- **result.self_undistort is None.** Since self-undistort does not run, this field
  will always be `None` when using `detect_with_mapper`.
- **Homography maps board to working frame.** The 3x3 homography in
  `result.homography` transforms board coordinates (mm) into undistorted pixel
  coordinates, not raw image pixels. To project into the original image, apply
  the camera distortion model to the working-frame points.

## Source Files

- `crates/ringgrid/src/api.rs` -- `Detector::detect_with_mapper` method.
- `crates/ringgrid/src/pixelmap/cameramodel.rs` -- `CameraModel`, `CameraIntrinsics`, `PixelMapper` implementation.
- `crates/ringgrid/examples/detect_with_camera.rs` -- complete runnable example.
