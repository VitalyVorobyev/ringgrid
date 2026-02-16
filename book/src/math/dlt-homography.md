# DLT Homography

The Direct Linear Transform (DLT) estimates a 2D projective transformation (homography) from point correspondences. ringgrid uses DLT to compute the board-to-image mapping from decoded marker positions.

## The Homography Model

A homography H is a 3×3 matrix that maps points between two projective planes:

```
[x']     [h₁₁ h₁₂ h₁₃] [x]
[y'] ~ H·[x, y, 1]ᵀ = [h₂₁ h₂₂ h₂₃] [y]
[w']     [h₃₁ h₃₂ h₃₃] [1]
```

The `~` denotes equality up to scale. The actual image coordinates are obtained by dehomogenizing:

```
x' = h₁₁x + h₁₂y + h₁₃
     ─────────────────────
     h₃₁x + h₃₂y + h₃₃

y' = h₂₁x + h₂₂y + h₂₃
     ─────────────────────
     h₃₁x + h₃₂y + h₃₃
```

H has 8 degrees of freedom (9 entries minus 1 for overall scale). Each point correspondence provides 2 equations, so a minimum of 4 correspondences are needed.

## DLT Construction

From a correspondence `(sx, sy) → (dx, dy)`, cross-multiplying to eliminate the unknown scale gives two linear equations in the 9 entries of h = [h₁₁, h₁₂, ..., h₃₃]ᵀ:

```
Row 2i:    [  0   0   0  | -sx  -sy  -1  | dy·sx  dy·sy  dy ] · h = 0
Row 2i+1:  [ sx  sy   1  |   0    0   0  | -dx·sx -dx·sy -dx] · h = 0
```

Stacking n correspondences produces a 2n × 9 matrix A. The solution h minimizes `||Ah||` subject to `||h|| = 1`.

## Solution via Eigendecomposition

The minimizer of `||Ah||²` subject to `||h|| = 1` is the eigenvector of AᵀA corresponding to its smallest eigenvalue.

ringgrid computes the 9×9 symmetric matrix AᵀA, then uses `SymmetricEigen` to find all eigenvalues and eigenvectors. The eigenvector associated with the smallest eigenvalue is reshaped into the 3×3 homography matrix.

Note: this is mathematically equivalent to taking the last right singular vector from the SVD of A, but computing the 9×9 eigensystem is more efficient than thin-SVD of a 2n×9 matrix.

## Hartley Normalization

Raw point coordinates can have very different scales (e.g., board coordinates in mm vs. image coordinates in pixels), leading to poor numerical conditioning of AᵀA. **Hartley normalization** addresses this:

For each point set (source and destination):

1. Compute the centroid `(cx, cy)` of the point set
2. Compute the mean distance from the centroid
3. Construct a normalizing transform T that:
   - Translates the centroid to the origin
   - Scales so the mean distance from the origin equals √2

```
T = [s  0  -s·cx]
    [0  s  -s·cy]
    [0  0    1  ]

where s = √2 / mean_distance
```

The DLT is then solved in normalized coordinates, and the result is denormalized:

```
H = T_dst⁻¹ · H_normalized · T_src
```

This normalization dramatically improves numerical stability and is essential for accurate results.

## Normalization of H

After denormalization, H is rescaled so that `h₃₃ = 1` (when `|h₃₃|` is not too small). This conventional normalization makes the homography directly usable for projection without an extra scale factor.

## Reprojection Error

The quality of a fitted homography is measured by **reprojection error** — the Euclidean distance between the projected source point and the observed destination point:

```
error_i = ||project(H, src_i) - dst_i||₂
```

In ringgrid, reprojection errors are reported in pixels and used for:

- RANSAC inlier classification
- Homography quality assessment (`RansacStats.mean_err_px`, `p95_err_px`)
- Accept/reject decisions for H refits

**Source**: `homography/core.rs`, `homography/utils.rs`
