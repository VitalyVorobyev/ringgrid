# RansacStats

`RansacStats` reports the quality of the RANSAC-fitted board-to-image homography. It is present in `DetectionResult::ransac` whenever a homography was successfully estimated (requires at least 4 decoded markers).

**Source:** `crates/ringgrid/src/homography/core.rs`

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_candidates` | `usize` | Total number of decoded marker correspondences fed to RANSAC. |
| `n_inliers` | `usize` | Number of correspondences classified as inliers after the final refit. |
| `threshold_px` | `f64` | Reprojection error threshold (in working-frame pixels) used to classify inliers. |
| `mean_err_px` | `f64` | Mean reprojection error across all inliers (in working-frame pixels). |
| `p95_err_px` | `f64` | 95th percentile reprojection error across inliers (in working-frame pixels). |

## Interpretation

### Mean reprojection error

The `mean_err_px` field is the single most informative quality indicator for the homography fit:

| `mean_err_px` | Assessment |
|---------------|------------|
| < 0.5 px | Excellent -- very precise calibration-grade fit |
| 0.5 -- 1.0 px | Good -- suitable for most applications |
| 1.0 -- 3.0 px | Acceptable -- some noise or mild distortion present |
| 3.0 -- 5.0 px | Marginal -- consider checking for distortion, wrong scale, or occlusion |
| > 5.0 px | Poor -- likely issues with marker scale, large lens distortion, or significant occlusion |

### Inlier ratio

The ratio `n_inliers / n_candidates` indicates how clean the set of decoded markers is:

| Inlier ratio | Assessment |
|--------------|------------|
| > 0.90 | Excellent -- nearly all detections are consistent |
| 0.80 -- 0.90 | Good -- a few outliers filtered |
| 0.60 -- 0.80 | Some markers have incorrect IDs or poor localization |
| < 0.60 | Problematic -- many false decodes or systematic error |

### Tail behavior

When `p95_err_px` is significantly larger than `mean_err_px` (e.g., more than 3x), a small number of inlier markers have notably worse localization than the rest. This can indicate:

- A few markers near the image edge with higher distortion.
- Partially occluded markers that passed the inlier threshold but are not well-localized.
- A mild systematic error (e.g., wrong marker diameter) that affects markers far from the image center more than those near it.

### Threshold

The `threshold_px` field records the reprojection error threshold used during RANSAC. Any correspondence with error below this threshold is classified as an inlier. The default is 5.0 pixels (configurable via `RansacHomographyConfig::inlier_threshold`). After RANSAC selects the best model, the homography is refit using all inliers, and the final `n_inliers`, `mean_err_px`, and `p95_err_px` are recomputed against the refit model.

## Example JSON

```json
{
  "n_candidates": 48,
  "n_inliers": 45,
  "threshold_px": 5.0,
  "mean_err_px": 0.63,
  "p95_err_px": 1.22
}
```

## Absence

When `DetectionResult::ransac` is `None`, no homography was fitted. This happens when fewer than 4 markers were decoded or when RANSAC failed to find enough inliers (controlled by `RansacHomographyConfig::min_inliers`, default 6).
