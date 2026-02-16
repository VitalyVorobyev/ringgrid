# DEC-008: Homography Convention

**Status:** Active
**Date:** 2025

## Decision

The homography `H` maps **board millimeters → working-frame pixels** (or
image pixels when no mapper is used).

### Convention

```
[u, v, w]^T = H * [x_mm, y_mm, 1]^T
pixel = [u/w, v/w]
```

- `H` is 3×3, estimated via DLT with Hartley normalization.
- Stored in `DetectionResult.homography` as `[[f64; 3]; 3]` (row-major).
- `homography_frame` records whether the output is `Image` or `Working`.

### Estimation details

- DLT minimum: 4 point correspondences.
- RANSAC for outlier rejection when ≥ 4 decoded markers are available.
- Inlier threshold and iteration count configured via `RansacHomographyConfig`.
- Final H is refit on all corrected centers if ≥ 10 markers are present,
  kept only if it reduces mean reprojection error.

### `RansacStats` contract

- `n_candidates`: total decoded markers fed to RANSAC.
- `n_inliers`: inliers under threshold.
- `threshold_px`, `mean_err_px`, `p95_err_px`: all in working-frame pixels.

### Conic projection under homography

For conic `Q` in the plane, the image conic is `Q' = H^{-T} Q H^{-1}`.
This is used in projective center tests and must remain consistent.
