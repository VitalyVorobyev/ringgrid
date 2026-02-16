# Inner Ellipse Estimation

The inner ellipse fit provides the second conic needed for [projective center correction](projective-center.md). It uses the outer ellipse as a geometric prior to constrain the search region.

## Search Region

The expected inner ring radius is defined by `MarkerSpec.r_inner_expected` — the ratio of the inner edge radius to the outer edge radius in normalized coordinates. For the default marker geometry:

- `r_inner_expected = 0.328 / 0.672 ≈ 0.488`

The search window extends ±`inner_search_halfwidth` (default 0.08) around this expected ratio, scaled by the fitted outer ellipse size.

## Edge Sampling

Edge points for the inner ring are sampled along radial rays from the marker center, looking for intensity transitions within the inner search window. The gradient polarity (`inner_grad_polarity`) constrains which transitions are accepted — by default `LightToDark`, matching the synthetic marker convention where the inner ring boundary transitions from light (inside) to dark (ring band).

The radial profile is aggregated across theta samples using the configured aggregator (median or trimmed mean). A minimum `min_theta_coverage` fraction of rays must produce valid edge detections for the estimate to proceed.

## RANSAC Ellipse Fitting

The inner edge points are fitted with RANSAC using the same Fitzgibbon solver as the outer fit, but with separate configuration via `InnerFitConfig`:

| Parameter | Default | Purpose |
|---|---|---|
| `min_points` | 20 | Minimum edge points to attempt fit |
| `min_inlier_ratio` | 0.5 | Minimum RANSAC inlier fraction |
| `max_rms_residual` | 1.0 px | Maximum RMS Sampson residual |
| `max_center_shift_px` | 12.0 px | Maximum center offset from outer fit |
| `max_ratio_abs_error` | 0.15 | Maximum deviation of recovered scale ratio from radial hint |
| `ransac.max_iters` | 200 | RANSAC iterations |
| `ransac.inlier_threshold` | 1.5 px | Sampson distance inlier threshold |
| `ransac.min_inliers` | 8 | Minimum inlier count |

## Validation

After fitting, the inner ellipse is validated against the outer ellipse:

1. **Center consistency**: the inner ellipse center must be within `max_center_shift_px` of the outer ellipse center
2. **Scale ratio**: the ratio of inner to outer semi-axes must be close to `r_inner_expected` (within `max_ratio_abs_error`)
3. **Fit quality**: RMS residual must be below `max_rms_residual`

If validation fails, the inner ellipse is rejected and projective center correction will not be available for this marker. The marker can still be detected using only the outer ellipse center.

**Source**: `ring/inner_estimate.rs`, `detector/inner_fit.rs`
