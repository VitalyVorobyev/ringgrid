# RANSAC Robust Estimation

RANSAC (Random Sample Consensus) is a general framework for fitting models from data contaminated with outliers. ringgrid uses RANSAC in two contexts: ellipse fitting from edge points and homography estimation from marker correspondences.

## The RANSAC Algorithm

Given N data points, a model that requires at least m points to fit, and an inlier threshold ε:

```
best_model = None
best_inlier_count = 0

for iteration in 1..max_iters:
    1. Randomly select m points from the dataset
    2. Fit a model from the m-point minimal sample
    3. For each remaining point, compute the error to the model
    4. Count inliers: points with error < ε
    5. If inlier_count > best_inlier_count:
         best_model = model
         best_inlier_count = inlier_count
    6. (Optional) Early exit if inlier_count / N > 0.9

Final: refit the model from all inliers of best_model
```

The final refit step is critical — the initial model was fit from only m points, but the refit uses all inliers, yielding a more accurate estimate.

## Expected Iterations

The number of iterations needed to find an all-inlier sample with probability p (typically 0.99) depends on the inlier ratio w:

```
k = log(1 - p) / log(1 - w^m)
```

| Inlier ratio w | m = 4 (homography) | m = 6 (ellipse) |
|---|---|---|
| 0.9 | 5 | 8 |
| 0.7 | 16 | 47 |
| 0.5 | 71 | 292 |
| 0.3 | 493 | 5,802 |

ringgrid defaults to 2000 iterations for homography RANSAC and 200–500 for ellipse RANSAC, which is sufficient for typical inlier ratios.

## Ellipse RANSAC

**Minimal sample size**: 6 points (the minimum for Fitzgibbon direct ellipse fit)

**Model fitting**: the [Fitzgibbon algorithm](fitzgibbon-ellipse.md) solves a constrained generalized eigenvalue problem to produce an ellipse in a single algebraic step.

**Error metric**: Sampson distance, a first-order approximation of the geometric (orthogonal) distance from a point to a conic.

For a conic with coefficients `a = [A, B, C, D, E, F]` and a point `(x, y)`:

```
f(x, y) = Ax² + Bxy + Cy² + Dx + Ey + F    (algebraic distance)

∇f = [2Ax + By + D, Bx + 2Cy + E]           (gradient)

d_Sampson = f(x, y) / ||∇f(x, y)||           (Sampson distance)
```

The Sampson distance approximates the signed geometric distance. For inlier classification, the absolute value `|d_Sampson|` is compared against the threshold.

**Configuration** (`RansacConfig`):

| Parameter | Typical value | Purpose |
|---|---|---|
| `max_iters` | 200–500 | Iteration budget |
| `inlier_threshold` | 1.0–2.0 px | Sampson distance threshold |
| `min_inliers` | 8 | Minimum inlier count for acceptance |
| `seed` | Fixed | Reproducible random seed |

**Source**: `conic/ransac.rs`

## Homography RANSAC

**Minimal sample size**: 4 point correspondences

**Model fitting**: [DLT with Hartley normalization](dlt-homography.md)

**Error metric**: reprojection error

```
error = ||project(H, src) - dst||₂
```

where `project(H, [x, y])` computes the projective mapping `H·[x, y, 1]ᵀ` and dehomogenizes.

**Algorithm specifics in ringgrid**:

1. Sample 4 distinct random correspondences
2. Fit H via DLT
3. Count inliers (reprojection error < `inlier_threshold`)
4. Track best model
5. **Early exit** when >90% of points are inliers
6. After all iterations, **refit** from all inliers of the best model
7. Recompute inlier mask with the refit H

**Configuration** (`RansacHomographyConfig`):

| Parameter | Default | Purpose |
|---|---|---|
| `max_iters` | 2000 | Iteration budget |
| `inlier_threshold` | 5.0 px | Reprojection error threshold |
| `min_inliers` | 6 | Minimum inlier count |
| `seed` | 0 | Reproducible random seed |

**Output** (`RansacStats`):

| Field | Meaning |
|---|---|
| `n_candidates` | Total correspondences fed to RANSAC |
| `n_inliers` | Inliers after final refit |
| `threshold_px` | Threshold used |
| `mean_err_px` | Mean inlier reprojection error |
| `p95_err_px` | 95th percentile reprojection error |

**Source**: `homography/core.rs`
