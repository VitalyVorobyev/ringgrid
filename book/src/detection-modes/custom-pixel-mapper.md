# Custom PixelMapper

The `PixelMapper` trait abstracts distortion/undistortion so you can plug in any
lens model. The built-in `CameraModel` and `DivisionModel` both implement it,
but you can provide your own implementation for exotic distortion models,
look-up-table corrections, or domain-specific coordinate transforms.

## The PixelMapper Trait

```rust
pub trait PixelMapper {
    /// Map from image (distorted) pixel coordinates to working coordinates.
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]>;

    /// Map from working coordinates back to image (distorted) pixel coordinates.
    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]>;
}
```

- **`image_to_working_pixel`** transforms a distorted image-space point into the
  undistorted working frame. Return `None` if the point cannot be mapped (e.g.
  it falls outside the valid domain of the distortion model).
- **`working_to_image_pixel`** transforms an undistorted working-frame point
  back into distorted image space. Return `None` if the inverse mapping fails.
- The two methods must be approximate inverses of each other. Perfect numerical
  round-tripping is not required, but the error should be small relative to a
  pixel.

Both methods are called during the two-pass pipeline: `working_to_image_pixel`
maps working-frame sample coordinates into the image for pixel lookups, and
`image_to_working_pixel` maps detected edge points into the working frame for
ellipse fitting.

## Implementation Example

A simple radial-only distortion model with a single coefficient:

```rust
use ringgrid::PixelMapper;

struct SimpleRadialMapper {
    cx: f64,
    cy: f64,
    k1: f64,
}

impl PixelMapper for SimpleRadialMapper {
    fn image_to_working_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let dx = p[0] - self.cx;
        let dy = p[1] - self.cy;
        let r2 = dx * dx + dy * dy;
        let scale = 1.0 + self.k1 * r2;
        if scale.abs() < 1e-12 {
            return None;
        }
        Some([self.cx + dx / scale, self.cy + dy / scale])
    }

    fn working_to_image_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let dx = p[0] - self.cx;
        let dy = p[1] - self.cy;
        let r2 = dx * dx + dy * dy;
        let scale = 1.0 + self.k1 * r2;
        Some([self.cx + dx * scale, self.cy + dy * scale])
    }
}
```

## Using a Custom Mapper

Pass your mapper to `detect_with_mapper` just like you would a `CameraModel`:

```rust
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json"))?;
let image = image::open("photo.png")?.to_luma8();
let (w, h) = image.dimensions();

let mapper = SimpleRadialMapper {
    cx: w as f64 * 0.5,
    cy: h as f64 * 0.5,
    k1: -1e-7,
};

let detector = Detector::new(board);
let result = detector.detect_with_mapper(&image, &mapper);

for marker in &result.detected_markers {
    println!("Image: ({:.1}, {:.1})", marker.center[0], marker.center[1]);
    if let Some(mapped) = marker.center_mapped {
        println!("Working: ({:.1}, {:.1})", mapped[0], mapped[1]);
    }
}
```

The coordinate frames are the same as for the
[external mapper](external-mapper.md) mode: `center` is image-space,
`center_mapped` is working-frame, and the homography maps board coordinates to
the working frame.

## Built-In Implementations

Two types in ringgrid already implement `PixelMapper`:

| Type | Description |
|---|---|
| `CameraModel` | Full Brown-Conrady model (k1, k2, k3 radial + p1, p2 tangential) with pinhole intrinsics. Undistortion is iterative. |
| `DivisionModel` | 1-parameter division model (`lambda`). Used internally by self-undistort. Undistortion is closed-form; distortion (inverse) is iterative. |

Both are in the `pixelmap` module and can serve as reference implementations
when writing your own mapper.

## Design Guidelines

When implementing `PixelMapper`:

- **Return `None` for invalid inputs.** If a point is outside the image or the
  distortion formula diverges, return `None` rather than a garbage coordinate.
  The detector will skip that sample gracefully.
- **Keep the methods fast.** They are called per edge-sample point, potentially
  thousands of times per image. Avoid allocations or heavy computation in the
  hot path.
- **Test round-trip accuracy.** Verify that
  `working_to_image_pixel(image_to_working_pixel(p))` returns a value close to
  `p` for points across the image. Sub-pixel accuracy (< 0.01 px error) is
  recommended.

## Important Notes

- Self-undistort is not run when `detect_with_mapper` is called. The provided
  mapper fully replaces any automatic distortion estimation.
- The mapper is used only during the second pass. The first pass always runs in
  raw image coordinates to generate seed proposals.

## Source Files

- `crates/ringgrid/src/pixelmap/mod.rs` -- `PixelMapper` trait definition.
- `crates/ringgrid/src/pixelmap/cameramodel.rs` -- `CameraModel` implements `PixelMapper`.
- `crates/ringgrid/src/pixelmap/distortion.rs` -- `DivisionModel` implements `PixelMapper`.
- `crates/ringgrid/src/api.rs` -- `Detector::detect_with_mapper`.
