//! Generalized eigenvalue solver and cubic root finder for 3×3 systems.

use nalgebra::{Matrix3, Vector3};

/// Solve the generalized eigenvalue problem M a = λ C1 a for the 3×3 case.
///
/// Finds the eigenvector of C1⁻¹ M corresponding to the unique eigenvalue
/// that satisfies the ellipse constraint aᵀ C1 a > 0.
///
/// Uses explicit eigenvalue computation via the characteristic polynomial
/// (cubic formula) and inverse iteration for eigenvectors, avoiding the
/// SymmetricEigen pitfall (C1⁻¹ M is not symmetric in general).
pub(crate) fn solve_gep_3x3(system: &Matrix3<f64>, _c1: &Matrix3<f64>) -> Option<Vector3<f64>> {
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
fn null_vector_3x3(m: &Matrix3<f64>) -> Option<Vector3<f64>> {
    // Cofactors for each row of the adjugate
    let cofactors = [
        Vector3::new(
            m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)],
            -(m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]),
            m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)],
        ),
        Vector3::new(
            -(m[(0, 1)] * m[(2, 2)] - m[(0, 2)] * m[(2, 1)]),
            m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)],
            -(m[(0, 0)] * m[(2, 1)] - m[(0, 1)] * m[(2, 0)]),
        ),
        Vector3::new(
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
