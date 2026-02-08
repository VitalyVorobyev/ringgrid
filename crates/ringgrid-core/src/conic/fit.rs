//! Direct least-squares ellipse fitting (Fitzgibbon et al., 1999).

use nalgebra::{DMatrix, Matrix3, Vector6};

use super::eigen::solve_gep_3x3;
use super::types::{ConicCoeffs, ConicError, Ellipse};

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
    fit_ellipse_direct(points)
        .ok_or_else(|| ConicError::NumericalFailure("direct fit returned None".into()))
}

/// Compute normalization parameters for a point set.
/// Returns (mean_x, mean_y, scale, inv_scale).
pub(crate) fn normalization_params(points: &[[f64; 2]]) -> (f64, f64, f64, f64) {
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
