# Projective Center & Global Filter

This chapter covers two key stages that work together: projective center correction and global homography filtering.

## Projective Center Correction

Under perspective projection, the center of a fitted ellipse is **biased** — it does not correspond to the true projection of the circle's center. This bias grows with the viewing angle.

ringgrid corrects this bias using the **conic pencil approach**: when both inner and outer ellipses are available for a marker, the two conics constrain a pencil whose degenerate members intersect at the true projected center. See [Projective Center Recovery](../math/projective-center-recovery.md) for the full derivation.

### Three-Pass Application

Projective center correction is applied **three times** during the pipeline:

| Pass | When | Which markers |
|---|---|---|
| 1st | Before global filter | All markers from fit-decode stage |
| 2nd | After H-guided refinement | Markers with newly fitted ellipses |
| 3rd | After completion | Completion-added markers only |

Each pass recomputes the corrected center from the current ellipse pair, since refinement and completion may produce different ellipse fits.

### Configuration

`ProjectiveCenterParams` controls the correction:

| Parameter | Default | Purpose |
|---|---|---|
| `enable` | `true` | Master switch |
| `use_expected_ratio` | `true` | Use `r_inner_expected` as eigenvalue prior |
| `ratio_penalty_weight` | 1.0 | Weight for ratio-prior penalty |
| `max_center_shift_px` | Derived from scale prior | Reject corrections that shift the center too far |
| `max_selected_residual` | `Some(0.25)` | Reject candidates with high geometric residual |
| `min_eig_separation` | `Some(1e-6)` | Reject when eigenvalues are too close (unstable) |

When correction is rejected (gates not met), the original ellipse-fit center is preserved.

## Global Homography Filter

Once enough markers are decoded and center-corrected, the detector estimates a board-to-image homography using RANSAC. This serves two purposes:

1. **Outlier rejection**: markers that are inconsistent with the dominant planar mapping are discarded
2. **Enable downstream stages**: H-guided refinement and completion require a valid homography

### Requirements

The global filter requires:

- At least 4 decoded markers with known board positions (from `BoardLayout`)
- `use_global_filter = true` in `DetectConfig`

When fewer than 4 decoded markers are available, the global filter is skipped and stages 8–13 do not run.

### Algorithm

1. Build correspondences: for each decoded marker, pair its board position `(xy_mm)` with its detected center
2. Run RANSAC homography fitting (see [DLT Homography](../math/dlt-homography.md)):
   - Sample 4 random correspondences
   - Estimate H via DLT with Hartley normalization
   - Count inliers (reprojection error < `inlier_threshold`)
   - Keep the model with most inliers
   - Refit from all inliers
3. Discard outlier markers

### Configuration

`RansacHomographyConfig`:

| Parameter | Default | Purpose |
|---|---|---|
| `max_iters` | 2000 | Maximum RANSAC iterations |
| `inlier_threshold` | 5.0 px | Reprojection error threshold |
| `min_inliers` | 6 | Minimum inliers for a valid model |
| `seed` | 0 | Random seed for reproducibility |

### Output

The global filter produces:

- A fitted homography matrix H (3x3, stored in `DetectionResult.homography`)
- `RansacStats` with inlier counts and error statistics
- A filtered marker set containing only inliers

### Short-Circuit

When `use_global_filter = false`, the entire finalization phase (stages 7–13) is skipped. The detector returns the markers from the fit-decode phase without any homography-based post-processing.

**Source**: `detector/center_correction.rs`, `detector/global_filter.rs`, `homography/core.rs`
