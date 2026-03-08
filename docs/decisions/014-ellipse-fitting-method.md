# DEC-014: Ellipse Fitting — Fitzgibbon Direct LS + RANSAC

**Status:** Active
**Date:** 2025

## Decision

Ellipse fitting uses Fitzgibbon's direct least-squares method as the core
estimator, wrapped in RANSAC for robustness.

### Algorithm

1. **Fitzgibbon direct LS** (`fit_ellipse_direct`): solves the constrained
   eigenvalue problem with the ellipse constraint `4AC - B² = 1` via a
   generalized eigensolver. Minimum 6 points.

2. **RANSAC wrapper** (`try_fit_ellipse_ransac`): samples 6-point subsets,
   fits with Fitzgibbon, scores by Sampson distance. Requires ≥ 8 points
   (for meaningful inlier/outlier separation).

### Fallback strategy

- ≥ 8 points → try RANSAC; on failure → fall back to direct fit.
- 6–7 points → direct fit only.
- < 6 points → reject.

### Distance metric

- **Sampson distance** (first-order geometric distance approximation) is used
  for inlier classification and RMS quality reporting.
- `rms_sampson_distance(ellipse, points)` is the standard quality metric.

### Quality gates (outer ellipse)

- Semi-axes within `[min_semi_axis, max_semi_axis]`.
- Aspect ratio ≤ `max_aspect_ratio` (default 3.0).
- `is_valid()` passes (finite, positive axes).
