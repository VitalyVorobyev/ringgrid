# Fitzgibbon Ellipse Fitting

This chapter describes the direct least-squares ellipse fitting method used in ringgrid, based on the work of Fitzgibbon, Pilu, and Fisher (1999). The method is the workhorse behind every ellipse fit in the detection pipeline -- from outer ring RANSAC fits to inner ring estimation and completion refits.

**Source**: `crates/ringgrid/src/conic/fit.rs`, `crates/ringgrid/src/conic/types.rs`

## The General Conic Equation

Any conic section (ellipse, hyperbola, parabola, or degenerate line pair) can be described by the implicit equation:

```
A x² + B xy + C y² + D x + E y + F = 0
```

We collect the six coefficients into a vector:

```
a = [A, B, C, D, E, F]ᵀ
```

The conic type is determined by the discriminant of the quadratic part:

- **Ellipse**: `B² - 4AC < 0`
- **Parabola**: `B² - 4AC = 0`
- **Hyperbola**: `B² - 4AC > 0`

The conic equation can also be written in matrix form. Given a point in homogeneous coordinates `x = [x, y, 1]ᵀ`, the conic locus is `xᵀ Q x = 0` where:

```
Q = | A    B/2  D/2 |
    | B/2  C    E/2 |
    | D/2  E/2  F   |
```

This symmetric 3x3 matrix `Q` is the **conic matrix** used extensively in the projective center recovery algorithm (see the [Projective Center Recovery](projective-center-recovery.md) chapter).

## The Design Matrix

Given `n` data points `(x_i, y_i)` for `i = 1, ..., n`, we construct the **design matrix** `D` where each row encodes the conic monomials evaluated at one point:

```
    | x₁²  x₁y₁  y₁²  x₁  y₁  1 |
D = | x₂²  x₂y₂  y₂²  x₂  y₂  1 |
    |  ⋮     ⋮     ⋮    ⋮   ⋮   ⋮ |
    | xₙ²  xₙyₙ  yₙ²  xₙ  yₙ  1 |
```

This is an `n x 6` matrix. For a point lying exactly on the conic, row `i` dotted with the coefficient vector `a` gives zero: `D_i · a = 0`. In the presence of noise, `Da` will not be exactly zero, and the entries of `Da` are the **algebraic distances** of the points to the conic.

## The Constrained Minimization Problem

The fitting objective is to minimize the sum of squared algebraic distances:

```
minimize  ||D a||²  =  aᵀ Dᵀ D a  =  aᵀ S a
```

subject to the constraint that the conic is an ellipse. Without a constraint, the trivial solution `a = 0` minimizes the objective.

Fitzgibbon et al. encode the ellipse constraint `4AC - B² > 0` via the **constraint matrix**:

```
C₁ = | 0   0   2 |
     | 0  -1   0 |
     | 2   0   0 |
```

This matrix acts on the quadratic coefficient sub-vector `a₁ = [A, B, C]ᵀ`. The quadratic form `a₁ᵀ C₁ a₁ = 4AC - B²` is positive if and only if the conic is an ellipse. The constraint is:

```
aᵀ C a = 1
```

where `C` is the 6x6 block-diagonal matrix with `C₁` in the upper-left 3x3 block and zeros elsewhere. This normalizes the scale of `a` and forces the ellipse condition simultaneously.

## The Scatter Matrix and Its Partition

Define the **scatter matrix** (or normal matrix):

```
S = Dᵀ D    (6 x 6, symmetric positive semi-definite)
```

Partition `S` into four 3x3 blocks corresponding to the quadratic terms `[A, B, C]` and the linear terms `[D, E, F]`:

```
S = | S₁₁  S₁₂ |
    | S₂₁  S₂₂ |
```

where:
- `S₁₁` (3x3): quadratic-quadratic cross-products
- `S₁₂` (3x3): quadratic-linear cross-products; `S₂₁ = S₁₂ᵀ`
- `S₂₂` (3x3): linear-linear cross-products

Similarly, partition the coefficient vector:

```
a = | a₁ |    a₁ = [A, B, C]ᵀ
    | a₂ |    a₂ = [D, E, F]ᵀ
```

## Reduction to a 3x3 Eigenvalue Problem

The Lagrangian of the constrained problem is:

```
L(a, λ) = aᵀ S a - λ (aᵀ C a - 1)
```

Setting `∂L/∂a = 0`:

```
S a = λ C a
```

Expanding in blocks:

```
S₁₁ a₁ + S₁₂ a₂ = λ C₁ a₁       ... (1)
S₂₁ a₁ + S₂₂ a₂ = 0               ... (2)
```

Equation (2) gives zero because the constraint matrix `C` has zeros in the lower-right block. Solving (2) for `a₂`:

```
a₂ = -S₂₂⁻¹ S₂₁ a₁
```

Substituting back into (1):

```
(S₁₁ - S₁₂ S₂₂⁻¹ S₂₁) a₁ = λ C₁ a₁
```

Define the **reduced matrix**:

```
M = S₁₁ - S₁₂ S₂₂⁻¹ S₂₁
```

This is the Schur complement of `S₂₂` in `S`. The problem becomes the 3x3 **generalized eigenvalue problem**:

```
M a₁ = λ C₁ a₁
```

Or equivalently, multiplying both sides by `C₁⁻¹`:

```
C₁⁻¹ M a₁ = λ a₁
```

where:

```
C₁⁻¹ = |  0    0   1/2 |
        |  0   -1    0  |
        | 1/2   0    0  |
```

Note that `C₁⁻¹ M` is generally **not symmetric**, so a standard symmetric eigendecomposition cannot be used. In ringgrid, this is solved via real Schur decomposition of the 3x3 matrix `C₁⁻¹ M` (implemented in `conic/eigen.rs`).

## Selecting the Correct Eigenvalue

The 3x3 system produces three eigenvalue-eigenvector pairs. The correct solution is the eigenvector `a₁` whose eigenvalue `λ` satisfies the **positive-definiteness** condition of the ellipse constraint:

```
a₁ᵀ C₁ a₁ = 4AC - B² > 0
```

In other words, we select the eigenpair where the constraint value is positive. Among the three eigenvalues of `C₁⁻¹ M`, exactly one will have this property when a valid ellipse solution exists.

## Recovering the Full Coefficient Vector

Once `a₁ = [A, B, C]ᵀ` is determined, the linear coefficients are recovered by back-substitution:

```
a₂ = -S₂₂⁻¹ S₂₁ a₁
```

This gives the complete conic coefficient vector `a = [A, B, C, D, E, F]ᵀ`.

## Hartley-Style Normalization

Numerical stability is critical. When image coordinates are in the range of hundreds of pixels, the entries of the design matrix `D` span many orders of magnitude (from `x²` terms around `10⁵` to the constant `1`). This makes the scatter matrix `S` ill-conditioned.

Ringgrid applies **Hartley-style normalization** before fitting:

1. Compute the centroid `(m_x, m_y)` of the input points
2. Compute the mean distance `d` of points from the centroid
3. Set scale factor `s = √2 / d`
4. Transform each point: `x' = s(x - m_x)`, `y' = s(y - m_y)`

After this transformation, the points are centered at the origin with mean distance `√2` from the origin. The design matrix entries are all of order `O(1)`, dramatically improving the condition number of `S`.

The normalization is computed in `normalization_params()`:

```rust
let scale = if mean_dist > 1e-15 {
    std::f64::consts::SQRT_2 / mean_dist
} else {
    1.0
};
```

## Denormalization of Conic Coefficients

The fitting is performed in normalized coordinates, producing conic coefficients `[A', B', C', D', E', F']`. These must be mapped back to original image coordinates.

Given the normalization transform `x' = s(x - m_x)`, `y' = s(y - m_y)`, we substitute into the normalized conic equation `A'x'² + B'x'y' + C'y'² + D'x' + E'y' + F' = 0`:

```
A'[s(x - m_x)]² + B'[s(x - m_x)][s(y - m_y)] + C'[s(y - m_y)]²
    + D'[s(x - m_x)] + E'[s(y - m_y)] + F' = 0
```

Expanding and collecting terms by monomial:

```
A = A' s²
B = B' s²
C = C' s²
D = -2A' s² m_x - B' s² m_y + D' s
E = -B' s² m_x - 2C' s² m_y + E' s
F = A' s² m_x² + B' s² m_x m_y + C' s² m_y² - D' s m_x - E' s m_y + F'
```

These formulas are implemented directly in `denormalize_conic()`.

## Conversion to Geometric Ellipse Parameters

The conic coefficients `[A, B, C, D, E, F]` define the ellipse implicitly. For practical use, we convert to the geometric representation `(c_x, c_y, a, b, θ)` where `(c_x, c_y)` is the center, `a` and `b` are the semi-major and semi-minor axes, and `θ` is the rotation angle.

### Center

The center is the point where the gradient of the conic equation vanishes (apart from the constant terms). Setting the partial derivatives to zero:

```
∂f/∂x = 2Ax + By + D = 0
∂f/∂y = Bx + 2Cy + E = 0
```

This 2x2 linear system has the solution:

```
c_x = (BE - 2CD) / (4AC - B²)
c_y = (BD - 2AE) / (4AC - B²)
```

The denominator `4AC - B²` is positive for an ellipse (it equals the negative discriminant).

### Rotation Angle

The orientation of the ellipse axes is determined by the eigenvectors of the 2x2 quadratic-part matrix:

```
M₂ = | A    B/2 |
     | B/2  C   |
```

The rotation angle of the major axis from the positive x-axis is:

```
θ = (1/2) atan2(B, A - C)
```

with a special case when `A = C` (the ellipse axes are at 45 degrees).

### Semi-Axes

The eigenvalues of `M₂` are:

```
λ₁ = (A + C + √((A-C)² + B²)) / 2
λ₂ = (A + C - √((A-C)² + B²)) / 2
```

The value of the conic function at the center is:

```
F' = A c_x² + B c_x c_y + C c_y² + D c_x + E c_y + F
```

The squared semi-axes are:

```
a² = -F' / λ₁
b² = -F' / λ₂
```

For a valid ellipse, both must be positive, which requires `F'` and the eigenvalues to have opposite signs.

The ellipse is canonicalized so that `a >= b` (semi-major axis first), swapping axes and adjusting the angle by `π/2` if necessary. The angle is normalized to `(-π/2, π/2]`.

## Sampson Distance

The **Sampson distance** provides a first-order approximation to the geometric (Euclidean) distance from a point to the nearest point on a conic. It is much cheaper to compute than true geometric distance and is used as the error metric in RANSAC ellipse fitting.

For a conic `f(x, y) = Ax² + Bxy + Cy² + Dx + Ey + F`, the gradient at `(x, y)` is:

```
∇f = (∂f/∂x, ∂f/∂y) = (2Ax + By + D, Bx + 2Cy + E)
```

The Sampson distance is defined as:

```
d_S(x, y) = |f(x, y)| / ||∇f(x, y)||
```

where `||∇f||` is the Euclidean norm of the gradient:

```
||∇f|| = √((2Ax + By + D)² + (Bx + 2Cy + E)²)
```

Geometrically, this divides the algebraic distance by the "speed" at which the conic function changes in the direction normal to the curve. For points near the conic, this closely approximates the true orthogonal distance in pixels.

Ringgrid implements this in `Ellipse::sampson_distance()`:

```rust
pub fn sampson_distance(&self, x: f64, y: f64) -> f64 {
    let c = self.to_conic();
    let [ca, cb, cc, cd, ce, _cf] = c.0;
    let alg = c.algebraic_distance(x, y);
    let gx = 2.0 * ca * x + cb * y + cd;
    let gy = cb * x + 2.0 * cc * y + ce;
    let grad_mag_sq = gx * gx + gy * gy;
    if grad_mag_sq < 1e-30 {
        return alg.abs();
    }
    alg.abs() / grad_mag_sq.sqrt()
}
```

The Sampson distance has units of pixels (assuming coordinates are in pixels), making it directly interpretable as an inlier threshold in RANSAC.

## Minimum Point Count

The general conic has 6 coefficients but only 5 degrees of freedom (the overall scale is irrelevant). Thus, 5 points in general position determine a unique conic. However, the Fitzgibbon method imposes the ellipse constraint, which adds one equation, so the **minimum number of points is 6**. The implementation enforces this:

```rust
if n < 6 {
    return None;
}
```

With fewer than 6 points, the scatter matrix `S` does not have sufficient rank to reliably partition and invert `S₂₂`.

## Summary of the Algorithm

1. Normalize the input points (Hartley-style: center and scale)
2. Build the `n x 6` design matrix `D` in normalized coordinates
3. Compute the scatter matrix `S = Dᵀ D` and partition into 3x3 blocks
4. Compute the reduced matrix `M = S₁₁ - S₁₂ S₂₂⁻¹ S₂₁`
5. Solve the eigenvalue problem `C₁⁻¹ M a₁ = λ a₁`
6. Select the eigenvector with positive ellipse constraint: `a₁ᵀ C₁ a₁ > 0`
7. Recover linear coefficients: `a₂ = -S₂₂⁻¹ S₂₁ a₁`
8. Denormalize the conic coefficients to original coordinates
9. Validate the result (check it is a proper ellipse with finite positive semi-axes)
10. Optionally convert to geometric parameters `(c_x, c_y, a, b, θ)`

## Reference

Fitzgibbon, A., Pilu, M., and Fisher, R. B. "Direct Least Square Fitting of Ellipses." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 21(5):476--480, 1999.
