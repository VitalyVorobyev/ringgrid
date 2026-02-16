# Outer Ellipse Fit

Once a radius hypothesis is available from the [outer estimation](outer-estimate.md) stage, the detector fits an ellipse to the outer ring boundary using RANSAC with the Fitzgibbon direct least-squares solver.

## Edge Point Sampling

For each proposal center, the detector casts radial rays outward at uniformly spaced angles and locates the intensity transition along each ray. The edge detection uses the Scharr gradient magnitude — each ray is sampled at sub-pixel resolution, and the point of maximum gradient response within the expected radius range is recorded as an edge point.

Key parameters from `EdgeSampleConfig`:

| Parameter | Description |
|---|---|
| `n_rays` | Number of uniformly spaced radial rays (typically 64–128) |
| `r_min` | Minimum search radius in pixels |
| `r_max` | Maximum search radius in pixels |

When a `PixelMapper` is active, edge sampling operates in the working (undistorted) frame. The ray endpoints are mapped from working coordinates to image coordinates for pixel lookups, then the edge points are recorded in working-frame coordinates.

## RANSAC Ellipse Fitting

The edge points are passed to a RANSAC loop that uses the [Fitzgibbon ellipse fitter](../math/fitzgibbon-ellipse.md) as the minimal solver:

1. **Sample**: randomly select 6 edge points (the minimum for Fitzgibbon)
2. **Fit**: compute the direct least-squares ellipse
3. **Score**: count inliers using Sampson distance as the error metric
4. **Iterate**: repeat for `max_iters` iterations, keeping the best model
5. **Refit**: fit a final ellipse from all inliers of the best model

The Sampson distance provides a first-order approximation of the geometric distance from a point to the conic. It is cheaper to compute than the true geometric distance while being a much better approximation than algebraic distance.

See [RANSAC Robust Estimation](../math/ransac.md) and [Fitzgibbon Ellipse Fitting](../math/fitzgibbon-ellipse.md) for mathematical details.

## Validation Gates

After RANSAC fitting, the resulting ellipse must pass several validation checks:

| Gate | Default | Purpose |
|---|---|---|
| Semi-axis bounds | `min_semi_axis` = 3.0, `max_semi_axis` = 15.0 (derived from scale prior) | Reject fits that are too small or too large |
| Aspect ratio | `max_aspect_ratio` = 3.0 | Reject highly elongated fits (likely not a ring) |
| Inlier ratio | Minimum fraction of edge points that are inliers | Reject poor fits |

These bounds are automatically derived from the `MarkerScalePrior` via `apply_marker_scale_prior`.

## Output

A successful outer fit produces an `Ellipse` struct:

```rust
pub struct Ellipse {
    pub cx: f64,    // center x
    pub cy: f64,    // center y
    pub a: f64,     // semi-major axis
    pub b: f64,     // semi-minor axis
    pub angle: f64, // rotation angle (radians)
}
```

The ellipse center `(cx, cy)` serves as the initial marker center estimate. This center is later refined by [projective center correction](projective-center.md) if both inner and outer ellipses are available.

**Source**: `detector/outer_fit.rs`, `ring/edge_sample.rs`, `conic/ransac.rs`
