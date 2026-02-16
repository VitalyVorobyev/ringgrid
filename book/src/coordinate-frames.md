# Coordinate Frames

ringgrid operates in two coordinate frames. Understanding which frame applies to each output field is essential for correct downstream use.

## Image Frame

The **image frame** uses pixel coordinates directly from the input `GrayImage`:

- Origin at the top-left corner of the image
- X increases rightward, Y increases downward
- Units are pixels
- This is the native frame of the detector when no pixel mapper is active

`DetectedMarker.center` is **always** in the image frame, regardless of detection mode. This ensures that center coordinates can always be overlaid directly on the original image.

## Working Frame

The **working frame** is the coordinate system used internally during detection when a pixel mapper is active. It is the *undistorted* (or otherwise transformed) coordinate space defined by the mapper's `image_to_working_pixel` method.

When a mapper is used (either an external `PixelMapper` or the self-undistort division model):

- Edge sampling and ellipse fitting operate in the working frame
- The homography maps from board coordinates (mm) to the working frame
- `DetectedMarker.center_mapped` contains the marker center in the working frame
- `DetectedMarker.center` remains in the image frame (mapped back via `working_to_image_pixel`)

## Frame Metadata in `DetectionResult`

Every `DetectionResult` includes explicit frame metadata so downstream code never has to guess:

```rust
pub struct DetectionResult {
    /// Always `DetectionFrame::Image` — centers are always image-space.
    pub center_frame: DetectionFrame,
    /// `Image` when no mapper is active; `Working` when a mapper was used.
    pub homography_frame: DetectionFrame,
    // ...
}
```

| Detection mode | `center_frame` | `homography_frame` | `center_mapped` present? |
|---|---|---|---|
| Simple (`detect()`) | `Image` | `Image` | No |
| External mapper (`detect_with_mapper()`) | `Image` | `Working` | Yes |
| Self-undistort (applied) | `Image` | `Working` | Yes |
| Self-undistort (not applied) | `Image` | `Image` | No |

## Homography Frame

The `homography` field in `DetectionResult` maps from **board coordinates** (millimeters, as defined in `BoardLayout`) to whichever frame `homography_frame` indicates:

- When `homography_frame == Image`: the homography maps board mm to distorted image pixels.
- When `homography_frame == Working`: the homography maps board mm to undistorted working-frame pixels.

To project a board point to image pixels when a mapper was used, compose the homography with `working_to_image_pixel`:

```rust
// H maps board_mm -> working_frame
let working_xy = project_homography(h, board_xy_mm);
// Map back to image pixels
let image_xy = mapper.working_to_image_pixel(working_xy);
```

## Practical Guidelines

1. **For visualization** (overlaying detections on the original image): use `center` directly — it is always in image coordinates.

2. **For calibration** (computing camera parameters): use `center_mapped` when available, since it is in the undistorted frame where the homography is valid.

3. **For reprojection error**: match the frame of your ground truth to `homography_frame` metadata, or map both to image space for consistent comparison.

4. **When implementing a custom `PixelMapper`**: ensure that `image_to_working_pixel` and `working_to_image_pixel` are consistent inverses. Return `None` for out-of-bounds coordinates.
