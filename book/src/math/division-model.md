# Division Distortion Model

The division model is a single-parameter radial distortion model used by ringgrid's self-undistort mode. Its simplicity makes it suitable for blind estimation from detected markers when no external camera calibration is available.

## The Model

The division model maps distorted (observed) pixel coordinates to undistorted coordinates:

```
x_u = cx + (x_d - cx) / (1 + λ · r²)
y_u = cy + (y_d - cy) / (1 + λ · r²)
```

where:
- `(x_d, y_d)` are distorted image coordinates
- `(x_u, y_u)` are undistorted (working-frame) coordinates
- `(cx, cy)` is the distortion center (assumed at the image center)
- `r² = (x_d - cx)² + (y_d - cy)²` is the squared radial distance from the distortion center
- `λ` is the single distortion parameter

**Sign convention**:
- `λ < 0` → barrel distortion (most common in wide-angle lenses)
- `λ > 0` → pincushion distortion
- `λ = 0` → no distortion (identity mapping)

## Forward and Inverse Mapping

**Forward** (distorted → undistorted): The division model has a closed-form forward mapping, making undistortion cheap to compute.

**Inverse** (undistorted → distorted): There is no closed-form inverse. ringgrid uses an iterative fixed-point method:

```
Initialize: (dx, dy) = (ux, uy)

Repeat up to 20 iterations:
    r² = dx² + dy²
    factor = 1 + λ · r²
    dx_new = ux · factor
    dy_new = uy · factor
    if ||(dx_new, dy_new) - (dx, dy)|| < 1e-12: break
    (dx, dy) = (dx_new, dy_new)

Result: (cx + dx, cy + dy)
```

This converges quickly for typical distortion magnitudes.

## PixelMapper Implementation

`DivisionModel` implements the `PixelMapper` trait:

```rust
impl PixelMapper for DivisionModel {
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
        Some(self.undistort_point(image_xy))  // closed-form forward
    }

    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
        self.distort_point(working_xy)  // iterative inverse
    }
}
```

## Self-Undistort Estimation

When `config.self_undistort.enable = true`, ringgrid estimates the optimal λ from the detected markers:

### Estimation Flow

1. **Baseline detection**: run the standard pipeline (no distortion correction) to detect initial markers
2. **Check prerequisites**: need at least `min_markers` (default 6) markers with both inner and outer edge points
3. **Optimize λ**: search for the λ that minimizes an objective function over a bounded range `[lambda_min, lambda_max]` (default `[-8e-7, 8e-7]`)
4. **Accept/reject**: apply gates to decide if the estimated correction is meaningful
5. **Pass-2 detection**: if accepted, re-run detection with the estimated `DivisionModel` as the pixel mapper

### Objective Function

The optimizer evaluates candidate λ values by:

**Primary objective** (when ≥4 decoded IDs exist): homography self-consistency in the working frame. For each candidate λ, undistort all marker centers, refit a homography, and measure the mean reprojection error.

**Fallback objective** (when fewer decoded IDs exist): conic consistency. For each candidate λ, re-sample edge points in the undistorted frame, refit inner/outer ellipses, and measure the Sampson residuals. Better distortion correction produces more circular (lower-residual) ellipse fits.

### Accept/Reject Gates

The estimated λ is accepted only if:

- The objective at λ is meaningfully better than at λ=0 (identity)
- `|λ|` is non-trivial (not too close to zero)
- λ is not at the boundary of the search range (boundary solutions are unreliable)

The `SelfUndistortResult` struct reports:

| Field | Meaning |
|---|---|
| `model` | The estimated `DivisionModel` (λ, cx, cy) |
| `applied` | Whether the correction was accepted and applied |
| `objective_at_zero` | Objective value with no correction |
| `objective_at_lambda` | Objective value at the estimated λ |
| `n_markers_used` | Number of markers contributing to the estimation |

## Comparison with Brown-Conrady

| Property | Division Model | Brown-Conrady |
|---|---|---|
| Parameters | 1 (λ) | 5 (k1, k2, p1, p2, k3) |
| Requires intrinsics | No (center at image center) | Yes (fx, fy, cx, cy) |
| Used by | Self-undistort mode | `detect_with_mapper` with `CameraModel` |
| Forward mapping | Closed-form | Closed-form |
| Inverse mapping | Iterative | Iterative |
| Accuracy | Captures dominant radial distortion | Full radial + tangential model |

The division model is intentionally simple — it captures the dominant barrel/pincushion distortion with a single parameter, making it robust for blind estimation. For higher accuracy, provide a full camera model via `detect_with_mapper`.

## Configuration

`SelfUndistortConfig`:

| Parameter | Default | Purpose |
|---|---|---|
| `enable` | `false` | Enable self-undistort estimation |
| `lambda_range` | `[-8e-7, 8e-7]` | Search bounds for λ |
| `min_markers` | 6 | Minimum markers with inner+outer edges |

**Source**: `pixelmap/distortion.rs`, `pixelmap/self_undistort.rs`
