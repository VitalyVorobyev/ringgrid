# Projective Center Recovery

This chapter derives the algorithm that recovers the true projected center of a circle from two concentric circle projections, without requiring camera intrinsics.

## The Problem

Under perspective projection, a circle in 3D projects to an ellipse in the image. The center of the projected ellipse is **not** the projection of the circle's 3D center. This projective bias is systematic — it pushes the apparent center away from the image center — and grows with the viewing angle and distance from the optical axis.

For calibration applications where subpixel center accuracy is required, this bias must be corrected.

## Concentric Circles and the Conic Pencil

Consider two concentric circles in the target plane with radii r_inner and r_outer. Under perspective projection, they map to two conics (ellipses) Q_inner and Q_outer in the image plane.

The **conic pencil** spanned by these two conics is the family:

```
Q(λ) = Q_outer - λ · Q_inner
```

where λ is a scalar parameter. Each member of the pencil is a 3×3 symmetric matrix representing a conic.

### Key Insight

The pencil contains **degenerate members** — conics with determinant zero — that factor into pairs of lines. These degenerate conics correspond to eigenvalues λ of the generalized eigenvalue problem:

```
Q_outer · v = λ · Q_inner · v
```

equivalently:

```
(Q_outer · Q_inner⁻¹) · v = λ · v
```

The true projected center lies at the intersection of the line pairs from the degenerate pencil members.

## Algorithm

The algorithm as implemented in ringgrid:

### Step 1: Normalize Conics

Both conic matrices are normalized to unit Frobenius norm. This improves numerical stability for subsequent eigenvalue computation.

### Step 2: Compute Eigenvalues

Form the matrix A = Q_outer · Q_inner⁻¹ and compute its three eigenvalues λ₁, λ₂, λ₃ (which may be complex).

### Step 3: Find Candidate Centers

For each eigenvalue λᵢ, compute the candidate center using two methods:

**Method A** (Wang et al.): Find the null vector u of (A - λᵢI), then compute p = Q_inner⁻¹ · u. The candidate center in image coordinates is `(p₁/p₃, p₂/p₃)`.

**Method B**: Find the null vector of (Q_outer - λᵢ · Q_inner) directly, and dehomogenize.

Both methods are algebraically equivalent but may differ numerically; ringgrid tries both and selects the best.

### Step 4: Score Candidates

Each candidate center p is scored by combining several criteria:

**Geometric residual**: Measures how well p lies on the pole-polar relationship with both conics. Computed as the normalized cross-product of Q₁·p and Q₂·p:

```
residual = ||( Q₁·p ) × ( Q₂·p )|| / (||Q₁·p|| · ||Q₂·p||)
```

A true projective center yields residual ≈ 0.

**Eigenvalue separation**: The gap between λᵢ and its nearest neighbor. Well-separated eigenvalues indicate a stable solution; degenerate (repeated) eigenvalues are numerically unstable.

**Imaginary-part penalty**: Small weights penalize complex eigenvalues and eigenvectors, since the true solution should be real.

**Ratio prior**: When the expected radius ratio k = r_inner/r_outer is known, the eigenvalue should be close to k². The penalty `|λ - k²|` biases selection toward the physically expected solution.

The total score combines these terms:

```
score = residual + w_imag_λ · |Im(λ)| + w_imag_v · ||Im(v)|| + w_ratio · |λ - k²|
```

### Step 5: Select Best Candidate

Candidates are compared by eigenvalue separation first (preferring well-separated eigenvalues), then by score. The candidate with the best combined criterion is selected.

## Gates in the Detection Pipeline

The detector applies additional gates via `ProjectiveCenterParams`:

| Gate | Purpose |
|---|---|
| `max_center_shift_px` | Reject if correction moves center too far from ellipse-fit center |
| `max_selected_residual` | Reject if geometric residual is too high (unreliable solution) |
| `min_eig_separation` | Reject if eigenvalues are nearly degenerate (unstable) |

When any gate rejects the correction, the original ellipse-fit center is preserved.

## Accuracy

On synthetic data with clean conics (no noise), the algorithm recovers the true projected center to machine precision (~1e-8 px). With noisy ellipse fits from real edge points, typical corrections are on the order of 0.01–0.5 px, depending on the perspective distortion.

The algorithm is **scale-invariant**: scaling either conic by a constant does not affect the result.

## References

- Wang, Y., et al. "Projective Correction of Circular Targets." 2019.

**Source**: `ring/projective_center.rs`
