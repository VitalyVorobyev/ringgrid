//! Ellipse / conic fitting primitives.
//!
//! Implements:
//! - Direct least-squares conic fit (Fitzgibbon et al., "Direct Least Square Fitting of Ellipses", 1999).
//! - Conversion between general conic coefficients and geometric ellipse parameters.
//! - Algebraic and geometric residual computation.
//! - RANSAC wrapper for outlier-robust fitting.

use nalgebra::{DMatrix, Matrix3, Vector3, Vector6};
use serde::{Deserialize, Serialize};

// ── Error type ─────────────────────────────────────────────────────────────

/// Errors that can occur during conic/ellipse fitting.
#[derive(Debug, Clone, PartialEq)]
pub enum ConicError {
    /// Too few points for the requested operation.
    TooFewPoints { needed: usize, got: usize },
    /// The fitted conic is degenerate (e.g., a line pair or point).
    DegenerateConic,
    /// The fitted conic is not an ellipse (hyperbola or parabola).
    NotAnEllipse,
    /// Numerical failure (singular matrix, etc.).
    NumericalFailure(String),
    /// RANSAC could not find enough inliers.
    InsufficientInliers { needed: usize, found: usize },
}

impl std::fmt::Display for ConicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { needed, got } => {
                write!(f, "too few points: need {}, got {}", needed, got)
            }
            Self::DegenerateConic => write!(f, "degenerate conic"),
            Self::NotAnEllipse => write!(f, "conic is not an ellipse"),
            Self::NumericalFailure(msg) => write!(f, "numerical failure: {}", msg),
            Self::InsufficientInliers { needed, found } => {
                write!(f, "insufficient inliers: need {}, found {}", needed, found)
            }
        }
    }
}

impl std::error::Error for ConicError {}

// ── Types ──────────────────────────────────────────────────────────────────

/// General conic: A x² + B xy + C y² + D x + E y + F = 0
/// Stored as [A, B, C, D, E, F].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicCoeffs(pub [f64; 6]);

/// Geometric ellipse parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ellipse {
    /// Center x.
    pub cx: f64,
    /// Center y.
    pub cy: f64,
    /// Semi-major axis length.
    pub a: f64,
    /// Semi-minor axis length.
    pub b: f64,
    /// Rotation angle of the major axis from +x, in radians (−π/2, π/2].
    pub angle: f64,
}

/// 2D conic in homogeneous image coordinates: `x^T Q x = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conic2D {
    pub mat: Matrix3<f64>,
}

impl Conic2D {
    /// Build from general quadratic coefficients:
    /// `A x^2 + B xy + C y^2 + D x + E y + F = 0`.
    pub fn from_quadratic_coeffs(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self {
            mat: Matrix3::new(
                a,
                b * 0.5,
                d * 0.5,
                b * 0.5,
                c,
                e * 0.5,
                d * 0.5,
                e * 0.5,
                f,
            ),
        }
    }

    /// Build from conic coefficients.
    pub fn from_coeffs(c: &ConicCoeffs) -> Self {
        let [a, b, cc, d, e, f] = c.0;
        Self::from_quadratic_coeffs(a, b, cc, d, e, f)
    }

    /// Build from fitted geometric ellipse representation.
    pub fn from_ellipse(e: &Ellipse) -> Self {
        Self::from_coeffs(&ellipse_to_conic(e))
    }

    /// Build from serialized ellipse parameters.
    pub fn from_ellipse_params(e: &crate::EllipseParams) -> Self {
        let ellipse = Ellipse {
            cx: e.center_xy[0],
            cy: e.center_xy[1],
            a: e.semi_axes[0].abs(),
            b: e.semi_axes[1].abs(),
            angle: e.angle,
        };
        Self::from_ellipse(&ellipse)
    }

    /// Normalize conic scale to unit Frobenius norm.
    pub fn normalize_frobenius(&self) -> Option<Self> {
        let n = self.mat.norm();
        if !n.is_finite() || n <= 1e-15 {
            return None;
        }
        Some(Self { mat: self.mat / n })
    }

    /// Invert the conic matrix.
    pub fn invert(&self) -> Option<Matrix3<f64>> {
        self.mat.try_inverse()
    }

    /// Evaluate `x^T Q x` for homogeneous `x`.
    pub fn eval_h(&self, x: Vector3<f64>) -> f64 {
        x.dot(&(self.mat * x))
    }
}

/// Configuration for RANSAC ellipse fitting.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum number of RANSAC iterations.
    pub max_iters: usize,
    /// Inlier threshold (algebraic distance, after normalization).
    pub inlier_threshold: f64,
    /// Minimum number of inliers for a valid model.
    pub min_inliers: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iters: 500,
            inlier_threshold: 1.0, // Sampson distance in pixels
            min_inliers: 10,
            seed: 42,
        }
    }
}

/// Result of a RANSAC fit.
#[derive(Debug, Clone)]
pub struct RansacResult {
    pub ellipse: Ellipse,
    pub conic: ConicCoeffs,
    pub inlier_mask: Vec<bool>,
    pub num_inliers: usize,
}

// ── Conic normalization ────────────────────────────────────────────────────

impl ConicCoeffs {
    /// Normalize conic coefficients so that A + C = 1 (trace of quadratic part).
    /// This makes the algebraic distance comparable across different conics.
    /// Returns `None` if A + C ≈ 0 (degenerate / hyperbola-like).
    pub fn normalized(&self) -> Option<Self> {
        let [a, b, c, d, e, f] = self.0;
        let trace = a + c;
        if trace.abs() < 1e-15 {
            return None;
        }
        let s = 1.0 / trace;
        Some(Self([a * s, b * s, c * s, d * s, e * s, f * s]))
    }

    /// Algebraic distance of a point (x, y) to this conic.
    pub fn algebraic_distance(&self, x: f64, y: f64) -> f64 {
        let [a, b, c, d, e, f] = self.0;
        a * x * x + b * x * y + c * y * y + d * x + e * y + f
    }

    /// Check whether the conic represents an ellipse (discriminant B²−4AC < 0).
    pub fn is_ellipse(&self) -> bool {
        let [a, b, c, ..] = self.0;
        b * b - 4.0 * a * c < 0.0
    }

    /// Convert to geometric ellipse parameters.
    /// Returns `None` if the conic is not an ellipse.
    pub fn to_ellipse(&self) -> Option<Ellipse> {
        conic_to_ellipse(self)
    }
}

// ── Ellipse utilities ──────────────────────────────────────────────────────

impl Ellipse {
    /// Check basic validity: positive semi-axes, finite values.
    pub fn is_valid(&self) -> bool {
        self.a > 0.0
            && self.b > 0.0
            && self.a.is_finite()
            && self.b.is_finite()
            && self.cx.is_finite()
            && self.cy.is_finite()
            && self.angle.is_finite()
    }

    /// Aspect ratio a/b (always >= 1 when canonicalized).
    pub fn aspect_ratio(&self) -> f64 {
        if self.a >= self.b {
            self.a / self.b
        } else {
            self.b / self.a
        }
    }

    /// Convert back to conic coefficients.
    pub fn to_conic(&self) -> ConicCoeffs {
        ellipse_to_conic(self)
    }

    /// Sample `n` points on the ellipse boundary.
    pub fn sample_points(&self, n: usize) -> Vec<[f64; 2]> {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                let px = self.a * t.cos();
                let py = self.b * t.sin();
                let x = self.cx + cos_a * px - sin_a * py;
                let y = self.cy + sin_a * px + cos_a * py;
                [x, y]
            })
            .collect()
    }

    /// Approximate geometric distance from a point to the ellipse boundary.
    /// Uses the algebraic distance divided by the gradient magnitude as a
    /// first-order approximation (Sampson distance).
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
}

// ── Conversion: conic ↔ ellipse ────────────────────────────────────────────

/// Convert general conic coefficients to geometric ellipse parameters.
///
/// The general conic is A x² + B xy + C y² + D x + E y + F = 0.
/// Returns `None` if the conic does not represent a proper ellipse.
pub fn conic_to_ellipse(c: &ConicCoeffs) -> Option<Ellipse> {
    let [a, b, c_coeff, d, e, f] = c.0;

    // Discriminant check: must be an ellipse
    let disc = b * b - 4.0 * a * c_coeff;
    if disc >= 0.0 {
        return None;
    }

    // Matrix form: M = [[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]]
    let m = Matrix3::new(
        a,
        b / 2.0,
        d / 2.0,
        b / 2.0,
        c_coeff,
        e / 2.0,
        d / 2.0,
        e / 2.0,
        f,
    );
    let det_m = m.determinant();

    // For a proper ellipse, det(M) must be nonzero and have opposite sign to A+C
    // (so that the ellipse encloses a finite area).
    if det_m.abs() < 1e-15 {
        return None;
    }

    // Center by solving the 2x2 system:
    //   2A·cx + B·cy + D = 0
    //   B·cx + 2C·cy + E = 0
    let denom = 4.0 * a * c_coeff - b * b; // = -disc > 0
    let cx = (b * e - 2.0 * c_coeff * d) / denom;
    let cy = (b * d - 2.0 * a * e) / denom;

    // Rotation angle
    let angle = if (a - c_coeff).abs() < 1e-15 {
        if b > 0.0 {
            std::f64::consts::FRAC_PI_4
        } else if b < 0.0 {
            -std::f64::consts::FRAC_PI_4
        } else {
            0.0
        }
    } else {
        0.5 * (b).atan2(a - c_coeff)
    };

    // Semi-axes from eigenvalues of the 2x2 quadratic part
    let sum = a + c_coeff;
    let diff = ((a - c_coeff).powi(2) + b * b).sqrt();
    let lambda1 = (sum + diff) / 2.0;
    let lambda2 = (sum - diff) / 2.0;

    // F' = value of the conic at the center
    let f_prime = a * cx * cx + b * cx * cy + c_coeff * cy * cy + d * cx + e * cy + f;

    if f_prime.abs() < 1e-15 {
        return None;
    }

    let a_sq = -f_prime / lambda1;
    let b_sq = -f_prime / lambda2;

    if a_sq <= 0.0 || b_sq <= 0.0 {
        return None;
    }

    let semi_a = a_sq.sqrt();
    let semi_b = b_sq.sqrt();

    // Canonicalize so that a >= b, adjusting angle accordingly
    let (semi_a, semi_b, angle) = if semi_a >= semi_b {
        (semi_a, semi_b, angle)
    } else {
        (semi_b, semi_a, angle + std::f64::consts::FRAC_PI_2)
    };

    // Normalize angle to (−π/2, π/2]
    let angle = normalize_angle(angle);

    Some(Ellipse {
        cx,
        cy,
        a: semi_a,
        b: semi_b,
        angle,
    })
}

/// Convert geometric ellipse parameters to general conic coefficients.
pub fn ellipse_to_conic(e: &Ellipse) -> ConicCoeffs {
    let cos_a = e.angle.cos();
    let sin_a = e.angle.sin();
    let a2 = e.a * e.a;
    let b2 = e.b * e.b;

    let ca = cos_a * cos_a / a2 + sin_a * sin_a / b2;
    let cb = 2.0 * cos_a * sin_a * (1.0 / a2 - 1.0 / b2);
    let cc = sin_a * sin_a / a2 + cos_a * cos_a / b2;
    let cd = -2.0 * ca * e.cx - cb * e.cy;
    let ce = -cb * e.cx - 2.0 * cc * e.cy;
    let cf = ca * e.cx * e.cx + cb * e.cx * e.cy + cc * e.cy * e.cy - 1.0;

    ConicCoeffs([ca, cb, cc, cd, ce, cf])
}

/// Normalize angle to (−π/2, π/2].
fn normalize_angle(mut angle: f64) -> f64 {
    let pi = std::f64::consts::PI;
    while angle > pi / 2.0 {
        angle -= pi;
    }
    while angle <= -pi / 2.0 {
        angle += pi;
    }
    angle
}

// ── Direct Least-Squares Conic Fit ─────────────────────────────────────────

/// Fit an ellipse to a set of 2D points using the direct least-squares method
/// of Fitzgibbon et al. (1999).
///
/// The method solves a constrained eigenvalue problem enforcing the ellipse
/// constraint (B² − 4AC < 0) via the constraint matrix C₁.
///
/// Requires at least 6 points. Returns `None` if the fit fails or produces
/// a non-ellipse result.
pub fn fit_ellipse_direct(points: &[[f64; 2]]) -> Option<(ConicCoeffs, Ellipse)> {
    let n = points.len();
    if n < 6 {
        return None;
    }

    // Normalize points for numerical stability: shift to centroid, scale so
    // that mean distance from centroid ≈ √2.
    let (mean_x, mean_y, scale, _) = normalization_params(points);

    // Build the design matrix D = [x², xy, y², x, y, 1] for normalized coords
    let mut d = DMatrix::<f64>::zeros(n, 6);
    for (i, &[px, py]) in points.iter().enumerate() {
        let x = (px - mean_x) * scale;
        let y = (py - mean_y) * scale;
        d[(i, 0)] = x * x;
        d[(i, 1)] = x * y;
        d[(i, 2)] = y * y;
        d[(i, 3)] = x;
        d[(i, 4)] = y;
        d[(i, 5)] = 1.0;
    }

    // Scatter matrix S = Dᵀ D
    let s = d.transpose() * &d;

    // Partition S into 3x3 blocks:
    //   S = [S11  S12]
    //       [S21  S22]
    let s11 = s.fixed_view::<3, 3>(0, 0).into_owned();
    let s12 = s.fixed_view::<3, 3>(0, 3).into_owned();
    let s22 = s.fixed_view::<3, 3>(3, 3).into_owned();

    // Constraint matrix for the ellipse condition: 4AC − B² > 0
    //   C1 = [[0, 0, 2], [0, -1, 0], [2, 0, 0]]
    let c1 = Matrix3::new(0.0, 0.0, 2.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0);

    // Solve the reduced eigensystem:
    //   (S11 − S12 S22⁻¹ S21) a1 = λ C1 a1
    // which becomes: C1⁻¹ M a1 = λ a1
    let s22_inv = s22.try_inverse()?;
    let m = s11 - s12 * s22_inv * s12.transpose();

    // Solve the generalized eigenvalue problem M a1 = λ C1 a1.
    //
    // C1_inv * M is generally NOT symmetric, so we cannot use SymmetricEigen.
    // Instead, we solve it via real Schur decomposition of C1_inv * M,
    // then extract real eigenvalues and their eigenvectors.
    let c1_inv = c1.try_inverse()?;
    let system = c1_inv * m;

    let a1 = solve_gep_3x3(&system, &c1)?;
    let a2 = -s22_inv * s12.transpose() * a1;

    // Denormalize the conic coefficients
    let coeffs_norm = Vector6::new(a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]);
    let coeffs = denormalize_conic(&coeffs_norm, mean_x, mean_y, scale);

    let conic = ConicCoeffs(coeffs);

    if !conic.is_ellipse() {
        return None;
    }

    let ellipse = conic.to_ellipse()?;
    if !ellipse.is_valid() {
        return None;
    }

    Some((conic, ellipse))
}

/// Fit an ellipse, returning a detailed error on failure.
pub fn try_fit_ellipse_direct(points: &[[f64; 2]]) -> Result<(ConicCoeffs, Ellipse), ConicError> {
    let n = points.len();
    if n < 6 {
        return Err(ConicError::TooFewPoints { needed: 6, got: n });
    }
    fit_ellipse_direct(points).ok_or_else(|| {
        // Try to distinguish the failure mode
        if n < 6 {
            ConicError::TooFewPoints { needed: 6, got: n }
        } else {
            ConicError::NumericalFailure("direct fit returned None".into())
        }
    })
}

/// Fit an ellipse robustly via RANSAC, returning detailed errors.
pub fn try_fit_ellipse_ransac(
    points: &[[f64; 2]],
    config: &RansacConfig,
) -> Result<RansacResult, ConicError> {
    let n = points.len();
    if n < 6 {
        return Err(ConicError::TooFewPoints { needed: 6, got: n });
    }
    fit_ellipse_ransac(points, config).ok_or(ConicError::InsufficientInliers {
        needed: config.min_inliers,
        found: 0,
    })
}

/// Solve the generalized eigenvalue problem M a = λ C1 a for the 3×3 case.
///
/// Finds the eigenvector of C1⁻¹ M corresponding to the unique eigenvalue
/// that satisfies the ellipse constraint aᵀ C1 a > 0.
///
/// Uses explicit eigenvalue computation via the characteristic polynomial
/// (cubic formula) and inverse iteration for eigenvectors, avoiding the
/// SymmetricEigen pitfall (C1⁻¹ M is not symmetric in general).
fn solve_gep_3x3(system: &Matrix3<f64>, _c1: &Matrix3<f64>) -> Option<nalgebra::Vector3<f64>> {
    // Compute eigenvalues of `system` = C1⁻¹ M via characteristic polynomial.
    // For a 3x3 matrix A, the characteristic polynomial is:
    //   λ³ - tr(A) λ² + (sum of 2x2 minors) λ - det(A) = 0
    let a = system;
    let tr = a[(0, 0)] + a[(1, 1)] + a[(2, 2)];

    // Sum of 2×2 principal minors (cofactors of diagonal)
    let minor_sum = a[(0, 0)] * a[(1, 1)] - a[(0, 1)] * a[(1, 0)] + a[(0, 0)] * a[(2, 2)]
        - a[(0, 2)] * a[(2, 0)]
        + a[(1, 1)] * a[(2, 2)]
        - a[(1, 2)] * a[(2, 1)];

    let det = a.determinant();

    // Solve: λ³ - tr λ² + minor_sum λ - det = 0
    let eigenvalues = solve_cubic_real(1.0, -tr, minor_sum, -det);

    // For each real eigenvalue, compute the eigenvector via null space of (A - λI)
    // and check the ellipse constraint aᵀ C1 a > 0.
    let mut best_vec = None;
    let mut best_ev = f64::MAX;

    for &ev in &eigenvalues {
        let shifted = system - Matrix3::identity() * ev;

        // Find null vector via SVD-like approach: pick the column of the
        // adjugate (cofactor matrix) with the largest norm.
        let v = null_vector_3x3(&shifted)?;

        // Check ellipse constraint: 4 v[0] v[2] - v[1]² > 0
        let constraint = 4.0 * v[0] * v[2] - v[1] * v[1];
        if constraint > 0.0 {
            // Verify this is a reasonable eigenvalue (Fitzgibbon: we want
            // the one satisfying the constraint; there should be exactly one)
            if ev.abs() < best_ev {
                best_ev = ev.abs();
                best_vec = Some(v);
            }
        }
    }

    best_vec
}

/// Find a null vector of a (near-)singular 3×3 matrix.
///
/// Computes the row of the adjugate (cofactor) matrix with the largest norm.
/// For a rank-2 matrix, each row of the adjugate is proportional to the null
/// vector.
fn null_vector_3x3(m: &Matrix3<f64>) -> Option<nalgebra::Vector3<f64>> {
    // Cofactors for each row of the adjugate
    let cofactors = [
        nalgebra::Vector3::new(
            m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)],
            -(m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]),
            m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)],
        ),
        nalgebra::Vector3::new(
            -(m[(0, 1)] * m[(2, 2)] - m[(0, 2)] * m[(2, 1)]),
            m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)],
            -(m[(0, 0)] * m[(2, 1)] - m[(0, 1)] * m[(2, 0)]),
        ),
        nalgebra::Vector3::new(
            m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)],
            -(m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)]),
            m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
        ),
    ];

    // Pick the one with largest norm
    let mut best = &cofactors[0];
    let mut best_norm = best.norm_squared();
    for c in &cofactors[1..] {
        let n = c.norm_squared();
        if n > best_norm {
            best = c;
            best_norm = n;
        }
    }

    if best_norm < 1e-30 {
        return None;
    }

    Some(best / best_norm.sqrt())
}

/// Solve a real cubic equation a x³ + b x² + c x + d = 0.
/// Returns all real roots (1 or 3).
fn solve_cubic_real(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    // Reduce to depressed cubic: t³ + pt + q = 0 with x = t - b/(3a)
    let a_inv = 1.0 / a;
    let b_ = b * a_inv;
    let c_ = c * a_inv;
    let d_ = d * a_inv;

    let p = c_ - b_ * b_ / 3.0;
    let q = 2.0 * b_ * b_ * b_ / 27.0 - b_ * c_ / 3.0 + d_;

    let disc = -4.0 * p * p * p - 27.0 * q * q;
    let shift = -b_ / 3.0;

    if disc >= 0.0 {
        // Three real roots (or repeated roots)
        let r = (-p / 3.0).sqrt();
        let cos_arg = if r.abs() < 1e-15 {
            0.0
        } else {
            (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0)
        };
        let theta = cos_arg.acos();
        let two_r = 2.0 * r;

        vec![
            two_r * (theta / 3.0).cos() + shift,
            two_r * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() + shift,
            two_r * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() + shift,
        ]
    } else {
        // One real root (Cardano's formula)
        let sqrt_disc = (q * q / 4.0 + p * p * p / 27.0).sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        vec![u + v + shift]
    }
}

/// Compute normalization parameters for a point set.
/// Returns (mean_x, mean_y, scale, inv_scale).
fn normalization_params(points: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let n = points.len() as f64;
    let mean_x: f64 = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let mean_y: f64 = points.iter().map(|p| p[1]).sum::<f64>() / n;

    let mean_dist: f64 = points
        .iter()
        .map(|p| ((p[0] - mean_x).powi(2) + (p[1] - mean_y).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let scale = if mean_dist > 1e-15 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    (mean_x, mean_y, scale, 1.0 / scale)
}

/// Denormalize conic coefficients from normalized coordinates back to original.
///
/// If normalized coords are x' = s(x − mx), y' = s(y − my), then the conic
/// A'x'² + B'x'y' + C'y'² + D'x' + E'y' + F' = 0 transforms back via
/// substitution.
fn denormalize_conic(c: &Vector6<f64>, mx: f64, my: f64, s: f64) -> [f64; 6] {
    let [a_, b_, c_, d_, e_, f_] = [c[0], c[1], c[2], c[3], c[4], c[5]];
    let s2 = s * s;

    // x' = s(x - mx), y' = s(y - my)
    // Substitute into A'x'² + B'x'y' + C'y'² + D'x' + E'y' + F':
    let a = a_ * s2;
    let b = b_ * s2;
    let c = c_ * s2;
    let d = -2.0 * a_ * s2 * mx - b_ * s2 * my + d_ * s;
    let e = -b_ * s2 * mx - 2.0 * c_ * s2 * my + e_ * s;
    let f =
        a_ * s2 * mx * mx + b_ * s2 * mx * my + c_ * s2 * my * my - d_ * s * mx - e_ * s * my + f_;

    [a, b, c, d, e, f]
}

// ── RANSAC ─────────────────────────────────────────────────────────────────

/// Fit an ellipse robustly using RANSAC.
///
/// Samples 6-point minimal subsets, fits via direct least squares, and selects
/// the model with the most inliers. Final model is re-fit to all inliers.
pub fn fit_ellipse_ransac(points: &[[f64; 2]], config: &RansacConfig) -> Option<RansacResult> {
    use rand::prelude::*;

    let n = points.len();
    if n < 6 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut best_inlier_count = 0usize;
    let mut best_conic: Option<ConicCoeffs> = None;
    let mut best_ellipse: Option<Ellipse> = None;
    let mut best_mask: Vec<bool> = vec![false; n];

    for _ in 0..config.max_iters {
        // Sample 6 random points
        let sample = sample_indices(&mut rng, n, 6);
        let sample_pts: Vec<[f64; 2]> = sample.iter().map(|&i| points[i]).collect();

        // Fit ellipse to sample
        let Some((conic, ellipse)) = fit_ellipse_direct(&sample_pts) else {
            continue;
        };

        // Count inliers using Sampson distance (approximate geometric
        // distance in pixels), which is threshold-interpretable.
        let mut inlier_count = 0usize;
        let mut mask = vec![false; n];
        for (i, &[x, y]) in points.iter().enumerate() {
            let dist = ellipse.sampson_distance(x, y);
            if dist < config.inlier_threshold {
                mask[i] = true;
                inlier_count += 1;
            }
        }

        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_conic = Some(conic);
            best_ellipse = Some(ellipse);
            best_mask = mask;

            // Early exit: if >90% of points are inliers, stop searching
            if best_inlier_count * 10 > n * 9 {
                break;
            }
        }
    }

    // Check minimum inlier count
    if best_inlier_count < config.min_inliers {
        return None;
    }

    // Re-fit to all inliers
    let inlier_pts: Vec<[f64; 2]> = best_mask
        .iter()
        .zip(points.iter())
        .filter(|(&m, _)| m)
        .map(|(_, &p)| p)
        .collect();

    let (final_conic, final_ellipse) =
        fit_ellipse_direct(&inlier_pts).or_else(|| best_conic.zip(best_ellipse))?;

    // Recompute inlier mask with final model using Sampson distance
    let mut final_mask = vec![false; n];
    let mut final_count = 0;
    for (i, &[x, y]) in points.iter().enumerate() {
        let dist = final_ellipse.sampson_distance(x, y);
        if dist < config.inlier_threshold {
            final_mask[i] = true;
            final_count += 1;
        }
    }

    Some(RansacResult {
        ellipse: final_ellipse,
        conic: final_conic,
        inlier_mask: final_mask,
        num_inliers: final_count,
    })
}

/// Sample `k` distinct indices from `0..n` using Fisher–Yates partial shuffle.
fn sample_indices(rng: &mut impl rand::Rng, n: usize, k: usize) -> Vec<usize> {
    debug_assert!(k <= n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

// ── Residual computation ───────────────────────────────────────────────────

/// Compute RMS algebraic distance of points to a (normalized) conic.
pub fn rms_algebraic_distance(conic: &ConicCoeffs, points: &[[f64; 2]]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let conic_n = conic.normalized().unwrap_or(*conic);
    let sum_sq: f64 = points
        .iter()
        .map(|&[x, y]| {
            let d = conic_n.algebraic_distance(x, y);
            d * d
        })
        .sum();
    (sum_sq / points.len() as f64).sqrt()
}

/// Compute RMS Sampson distance of points to an ellipse.
pub fn rms_sampson_distance(ellipse: &Ellipse, points: &[[f64; 2]]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = points
        .iter()
        .map(|&[x, y]| {
            let d = ellipse.sampson_distance(x, y);
            d * d
        })
        .sum();
    (sum_sq / points.len() as f64).sqrt()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;

    /// Helper: create an ellipse and sample points on it.
    fn make_test_ellipse() -> Ellipse {
        Ellipse {
            cx: 100.0,
            cy: 80.0,
            a: 30.0,
            b: 15.0,
            angle: 0.3, // ~17 degrees
        }
    }

    #[test]
    fn test_ellipse_to_conic_roundtrip() {
        let e = make_test_ellipse();
        let c = e.to_conic();
        assert!(c.is_ellipse(), "conic should be an ellipse");
        let e2 = c.to_ellipse().expect("should convert back to ellipse");

        assert_relative_eq!(e.cx, e2.cx, epsilon = 1e-10);
        assert_relative_eq!(e.cy, e2.cy, epsilon = 1e-10);
        assert_relative_eq!(e.a, e2.a, epsilon = 1e-10);
        assert_relative_eq!(e.b, e2.b, epsilon = 1e-10);
        assert_relative_eq!(e.angle, e2.angle, epsilon = 1e-10);
    }

    #[test]
    fn test_algebraic_distance_on_ellipse() {
        let e = make_test_ellipse();
        let c = e.to_conic();
        let pts = e.sample_points(100);
        for &[x, y] in &pts {
            let d = c.algebraic_distance(x, y);
            assert!(
                d.abs() < 1e-10,
                "point on ellipse should have ~zero algebraic distance, got {}",
                d
            );
        }
    }

    #[test]
    fn test_fit_exact_points() {
        let e = make_test_ellipse();
        let pts = e.sample_points(50);

        let (conic, fitted) = fit_ellipse_direct(&pts).expect("fit should succeed");

        assert_relative_eq!(fitted.cx, e.cx, epsilon = 1e-6);
        assert_relative_eq!(fitted.cy, e.cy, epsilon = 1e-6);
        assert_relative_eq!(fitted.a, e.a, epsilon = 1e-6);
        assert_relative_eq!(fitted.b, e.b, epsilon = 1e-6);
        assert_relative_eq!(fitted.angle, e.angle, epsilon = 1e-6);

        let rms = rms_algebraic_distance(&conic, &pts);
        assert!(
            rms < 1e-10,
            "RMS algebraic distance should be ~0, got {}",
            rms
        );
    }

    #[test]
    fn test_fit_noisy_points() {
        let e = make_test_ellipse();
        let mut pts = e.sample_points(200);
        let mut rng = StdRng::seed_from_u64(123);
        let noise_sigma = 0.5; // pixels

        for p in &mut pts {
            p[0] += rng.gen::<f64>() * noise_sigma * 2.0 - noise_sigma;
            p[1] += rng.gen::<f64>() * noise_sigma * 2.0 - noise_sigma;
        }

        let (_conic, fitted) = fit_ellipse_direct(&pts).expect("fit should succeed with noise");

        // With noise_sigma=0.5 on ~30px semi-axis, center should be within ~1px
        assert_relative_eq!(fitted.cx, e.cx, epsilon = 1.0);
        assert_relative_eq!(fitted.cy, e.cy, epsilon = 1.0);
        assert_relative_eq!(fitted.a, e.a, epsilon = 2.0);
        assert_relative_eq!(fitted.b, e.b, epsilon = 2.0);
    }

    #[test]
    fn test_fit_circle() {
        // Special case: circle (a == b, angle irrelevant)
        let e = Ellipse {
            cx: 50.0,
            cy: 50.0,
            a: 20.0,
            b: 20.0,
            angle: 0.0,
        };
        let pts = e.sample_points(100);
        let (_conic, fitted) = fit_ellipse_direct(&pts).expect("circle fit should succeed");

        assert_relative_eq!(fitted.cx, 50.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.cy, 50.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.a, 20.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.b, 20.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ransac_no_outliers() {
        let e = make_test_ellipse();
        let pts = e.sample_points(100);

        let config = RansacConfig {
            max_iters: 100,
            inlier_threshold: 0.1, // Sampson distance in pixels
            min_inliers: 6,
            seed: 42,
        };

        let result = fit_ellipse_ransac(&pts, &config).expect("RANSAC should succeed");
        assert_eq!(result.num_inliers, 100);
        assert_relative_eq!(result.ellipse.cx, e.cx, epsilon = 1e-4);
        assert_relative_eq!(result.ellipse.cy, e.cy, epsilon = 1e-4);
    }

    #[test]
    fn test_ransac_with_outliers() {
        let e = make_test_ellipse();
        let mut pts = e.sample_points(80);
        let mut rng = StdRng::seed_from_u64(999);

        // Add 20 random outliers
        for _ in 0..20 {
            pts.push([rng.gen_range(0.0..200.0), rng.gen_range(0.0..200.0)]);
        }

        let config = RansacConfig {
            max_iters: 500,
            inlier_threshold: 0.1, // Sampson distance in pixels
            min_inliers: 20,
            seed: 42,
        };

        let result =
            fit_ellipse_ransac(&pts, &config).expect("RANSAC should succeed with outliers");

        // Should recover the original ellipse despite 20% outliers
        assert_relative_eq!(result.ellipse.cx, e.cx, epsilon = 0.5);
        assert_relative_eq!(result.ellipse.cy, e.cy, epsilon = 0.5);
        assert_relative_eq!(result.ellipse.a, e.a, epsilon = 0.5);
        assert_relative_eq!(result.ellipse.b, e.b, epsilon = 0.5);

        // Most true points should be inliers
        assert!(
            result.num_inliers >= 60,
            "expected >= 60 inliers, got {}",
            result.num_inliers
        );
    }

    #[test]
    fn test_ransac_with_noise_and_outliers() {
        let e = make_test_ellipse();
        let mut pts = e.sample_points(150);
        let mut rng = StdRng::seed_from_u64(777);
        let noise_sigma = 0.3;

        // Add Gaussian-ish noise to inliers
        for p in pts.iter_mut() {
            p[0] += (rng.gen::<f64>() - 0.5) * 2.0 * noise_sigma;
            p[1] += (rng.gen::<f64>() - 0.5) * 2.0 * noise_sigma;
        }

        // Add 50 outliers
        for _ in 0..50 {
            pts.push([rng.gen_range(20.0..180.0), rng.gen_range(20.0..160.0)]);
        }

        let config = RansacConfig {
            max_iters: 2000,
            inlier_threshold: 1.0, // Sampson distance in pixels; generous for σ=0.3
            min_inliers: 20,
            seed: 42,
        };

        let result =
            fit_ellipse_ransac(&pts, &config).expect("RANSAC should succeed with noise + outliers");

        assert_relative_eq!(result.ellipse.cx, e.cx, epsilon = 2.0);
        assert_relative_eq!(result.ellipse.cy, e.cy, epsilon = 2.0);
        assert_relative_eq!(result.ellipse.a, e.a, epsilon = 3.0);
        assert_relative_eq!(result.ellipse.b, e.b, epsilon = 3.0);
    }

    #[test]
    fn test_too_few_points() {
        let pts = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(fit_ellipse_direct(&pts).is_none());
    }

    #[test]
    fn test_collinear_points_rejected() {
        // 6 collinear points — should not form an ellipse
        let pts: Vec<[f64; 2]> = (0..6).map(|i| [i as f64, i as f64 * 2.0]).collect();
        assert!(fit_ellipse_direct(&pts).is_none());
    }

    #[test]
    fn test_conic_normalization() {
        let e = make_test_ellipse();
        let c = e.to_conic();
        let cn = c.normalized().expect("normalization should succeed");
        let [a, _, c_coeff, ..] = cn.0;
        assert_relative_eq!(a + c_coeff, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sampson_distance() {
        let e = make_test_ellipse();
        let pts = e.sample_points(50);

        // Points on the ellipse should have ~0 Sampson distance
        for &[x, y] in &pts {
            let d = e.sampson_distance(x, y);
            assert!(
                d < 1e-8,
                "Sampson distance on ellipse should be ~0, got {}",
                d
            );
        }

        // Center should have nonzero distance
        let d_center = e.sampson_distance(e.cx, e.cy);
        assert!(d_center > 1.0, "Center should be far from boundary");
    }

    #[test]
    fn test_rms_sampson_distance() {
        let e = make_test_ellipse();
        let pts = e.sample_points(100);
        let rms = rms_sampson_distance(&e, &pts);
        assert!(
            rms < 1e-8,
            "RMS Sampson distance for exact points should be ~0, got {}",
            rms
        );
    }

    #[test]
    fn test_various_ellipses() {
        // Test fitting with different aspect ratios and orientations
        let test_cases = [
            Ellipse {
                cx: 50.0,
                cy: 50.0,
                a: 40.0,
                b: 10.0,
                angle: 0.0,
            }, // very elongated, axis-aligned
            Ellipse {
                cx: 200.0,
                cy: 150.0,
                a: 25.0,
                b: 24.0,
                angle: 1.0,
            }, // nearly circular
            Ellipse {
                cx: 300.0,
                cy: 100.0,
                a: 50.0,
                b: 20.0,
                angle: -0.7,
            }, // tilted
            Ellipse {
                cx: 10.0,
                cy: 10.0,
                a: 8.0,
                b: 5.0,
                angle: std::f64::consts::FRAC_PI_4,
            }, // small, 45°
        ];

        for (i, e) in test_cases.iter().enumerate() {
            let pts = e.sample_points(100);
            let (_conic, fitted) = fit_ellipse_direct(&pts)
                .unwrap_or_else(|| panic!("fit should succeed for test case {}", i));

            assert_relative_eq!(fitted.cx, e.cx, epsilon = 1e-4);
            assert_relative_eq!(fitted.cy, e.cy, epsilon = 1e-4);
            assert_relative_eq!(fitted.a, e.a, epsilon = 1e-4);
            assert_relative_eq!(fitted.b, e.b, epsilon = 1e-4);
            // Angle comparison needs care near ±π/2 boundary
            let angle_diff = (fitted.angle - e.angle).abs();
            let angle_diff = angle_diff.min((angle_diff - std::f64::consts::PI).abs());
            assert!(
                angle_diff < 1e-4,
                "angle mismatch for case {}: expected {}, got {}",
                i,
                e.angle,
                fitted.angle
            );
        }
    }

    #[test]
    fn test_partial_arc_fit() {
        // Fit from only a quarter of the ellipse (partial arc)
        let e = make_test_ellipse();
        let all_pts = e.sample_points(400);
        // Keep only points in the first quadrant relative to center
        let arc_pts: Vec<[f64; 2]> = all_pts
            .into_iter()
            .filter(|&[x, y]| x > e.cx && y > e.cy)
            .collect();
        assert!(arc_pts.len() >= 20, "need enough arc points");

        let (_conic, fitted) =
            fit_ellipse_direct(&arc_pts).expect("partial arc fit should succeed");

        // Partial arc fits are less accurate, allow larger tolerance
        assert_relative_eq!(fitted.cx, e.cx, epsilon = 5.0);
        assert_relative_eq!(fitted.cy, e.cy, epsilon = 5.0);
        assert_relative_eq!(fitted.a, e.a, epsilon = 5.0);
        assert_relative_eq!(fitted.b, e.b, epsilon = 5.0);
    }

    #[test]
    fn test_strong_noise_fit() {
        let e = make_test_ellipse();
        let mut pts = e.sample_points(500);
        let mut rng = StdRng::seed_from_u64(2024);
        let noise_sigma = 2.0; // Strong: ~6% of semi-major axis

        for p in &mut pts {
            p[0] += (rng.gen::<f64>() - 0.5) * 2.0 * noise_sigma;
            p[1] += (rng.gen::<f64>() - 0.5) * 2.0 * noise_sigma;
        }

        let result = fit_ellipse_direct(&pts);
        assert!(
            result.is_some(),
            "fit should succeed even with strong noise"
        );
        let (_conic, fitted) = result.unwrap();
        // Allow generous tolerances
        assert_relative_eq!(fitted.cx, e.cx, epsilon = 5.0);
        assert_relative_eq!(fitted.cy, e.cy, epsilon = 5.0);
    }

    #[test]
    fn test_degenerate_inputs_dont_panic() {
        // Duplicate points
        let pts: Vec<[f64; 2]> = vec![[1.0, 1.0]; 10];
        assert!(fit_ellipse_direct(&pts).is_none());

        // Two clusters
        let mut pts2: Vec<[f64; 2]> = vec![[0.0, 0.0]; 5];
        pts2.extend(vec![[100.0, 100.0]; 5]);
        assert!(fit_ellipse_direct(&pts2).is_none());

        // Empty
        let empty: Vec<[f64; 2]> = vec![];
        assert!(fit_ellipse_direct(&empty).is_none());

        // Exactly 6 collinear
        let line: Vec<[f64; 2]> = (0..6).map(|i| [i as f64 * 10.0, 0.0]).collect();
        assert!(fit_ellipse_direct(&line).is_none());
    }

    #[test]
    fn test_try_fit_error_types() {
        // Too few points
        let pts = vec![[1.0, 2.0], [3.0, 4.0]];
        let err = try_fit_ellipse_direct(&pts).unwrap_err();
        assert!(matches!(
            err,
            ConicError::TooFewPoints { needed: 6, got: 2 }
        ));

        // Valid fit should succeed
        let e = make_test_ellipse();
        let pts = e.sample_points(50);
        let result = try_fit_ellipse_direct(&pts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ransac_early_exit() {
        // With all clean points, RANSAC should exit early
        let e = make_test_ellipse();
        let pts = e.sample_points(200);
        let config = RansacConfig {
            max_iters: 10000, // many iterations, but should exit early
            inlier_threshold: 0.5,
            min_inliers: 6,
            seed: 42,
        };
        // Should succeed quickly (early exit at 90% inliers)
        let result = fit_ellipse_ransac(&pts, &config).expect("should succeed");
        assert_eq!(result.num_inliers, 200);
    }

    #[test]
    fn test_ransac_partial_arc_with_outliers() {
        let e = make_test_ellipse();
        let all_pts = e.sample_points(400);
        // Keep only a half-arc
        let mut arc_pts: Vec<[f64; 2]> = all_pts.into_iter().filter(|&[_, y]| y > e.cy).collect();

        // Add outliers
        let mut rng = StdRng::seed_from_u64(333);
        for _ in 0..20 {
            arc_pts.push([rng.gen_range(0.0..200.0), rng.gen_range(0.0..200.0)]);
        }

        let config = RansacConfig {
            max_iters: 1000,
            inlier_threshold: 1.0,
            min_inliers: 10,
            seed: 42,
        };

        let result =
            fit_ellipse_ransac(&arc_pts, &config).expect("RANSAC should succeed on partial arc");

        assert_relative_eq!(result.ellipse.cx, e.cx, epsilon = 5.0);
        assert_relative_eq!(result.ellipse.cy, e.cy, epsilon = 5.0);
    }
}
