//! Direct least-squares ellipse fitting (Fitzgibbon et al., 1999).

use nalgebra::{DMatrix, Matrix3, Vector6};

use super::eigen::solve_gep_3x3;
use super::types::Ellipse;
use super::ConicCoeffs;

/// Fit an ellipse to a set of 2D points using the direct least-squares method
/// of Fitzgibbon et al. (1999).
///
/// The method solves a constrained eigenvalue problem enforcing the ellipse
/// constraint (B² − 4AC < 0) via the constraint matrix C₁.
///
/// Requires at least 6 points. Returns conic coefficients on success.
///
/// The fitted conic is validated to represent a proper ellipse.
pub fn fit_conic_direct(points: &[[f64; 2]]) -> Option<ConicCoeffs> {
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

    Some(conic)
}

/// Fit an ellipse and return geometric ellipse parameters.
///
/// Convenience wrapper for call sites that only need the geometric form.
pub fn fit_ellipse_direct(points: &[[f64; 2]]) -> Option<Ellipse> {
    fit_conic_direct(points)?.to_ellipse()
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

#[cfg(test)]
mod test {
    use super::super::types::Ellipse;
    use super::*;

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
    fn test_try_fit_error_types() {
        // Too few points
        let pts = vec![[1.0, 2.0], [3.0, 4.0]];
        let result = fit_conic_direct(&pts);
        assert!(result.is_none());

        // Valid fit should succeed
        let e = make_test_ellipse();
        let pts = e.sample_points(50);
        let result = fit_conic_direct(&pts);
        assert!(result.is_some());
    }
}

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

        let conic = fit_conic_direct(&pts).expect("fit should succeed");
        let fitted = conic.to_ellipse().expect("fit should return an ellipse");

        assert_relative_eq!(fitted.cx, e.cx, epsilon = 1e-6);
        assert_relative_eq!(fitted.cy, e.cy, epsilon = 1e-6);
        assert_relative_eq!(fitted.a, e.a, epsilon = 1e-6);
        assert_relative_eq!(fitted.b, e.b, epsilon = 1e-6);
        assert_relative_eq!(fitted.angle, e.angle, epsilon = 1e-6);

        let rms = rms_sampson_distance(&fitted, &pts);
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

        let conic = fit_conic_direct(&pts).expect("fit should succeed with noise");
        let fitted = conic.to_ellipse().expect("fit should return an ellipse");

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
        let conic = fit_conic_direct(&pts).expect("circle fit should succeed");
        let fitted = conic.to_ellipse().expect("fit should return an ellipse");

        assert_relative_eq!(fitted.cx, 50.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.cy, 50.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.a, 20.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.b, 20.0, epsilon = 1e-6);
    }

    #[test]
    fn test_too_few_points() {
        let pts = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(fit_conic_direct(&pts).is_none());
    }

    #[test]
    fn test_collinear_points_rejected() {
        // 6 collinear points — should not form an ellipse
        let pts: Vec<[f64; 2]> = (0..6).map(|i| [i as f64, i as f64 * 2.0]).collect();
        assert!(fit_conic_direct(&pts).is_none());
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
            let conic = fit_conic_direct(&pts)
                .unwrap_or_else(|| panic!("fit should succeed for test case {}", i));
            let fitted = conic.to_ellipse().expect("fit should return an ellipse");

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

        let conic = fit_conic_direct(&arc_pts).expect("partial arc fit should succeed");
        let fitted = conic.to_ellipse().expect("fit should return an ellipse");

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

        let result = fit_conic_direct(&pts);
        assert!(
            result.is_some(),
            "fit should succeed even with strong noise"
        );
        let conic = result.unwrap();
        let fitted = conic.to_ellipse().expect("fit should return an ellipse");
        // Allow generous tolerances
        assert_relative_eq!(fitted.cx, e.cx, epsilon = 5.0);
        assert_relative_eq!(fitted.cy, e.cy, epsilon = 5.0);
    }

    #[test]
    fn test_degenerate_inputs_dont_panic() {
        // Duplicate points
        let pts: Vec<[f64; 2]> = vec![[1.0, 1.0]; 10];
        assert!(fit_conic_direct(&pts).is_none());

        // Two clusters
        let mut pts2: Vec<[f64; 2]> = vec![[0.0, 0.0]; 5];
        pts2.extend(vec![[100.0, 100.0]; 5]);
        assert!(fit_conic_direct(&pts2).is_none());

        // Empty
        let empty: Vec<[f64; 2]> = vec![];
        assert!(fit_conic_direct(&empty).is_none());

        // Exactly 6 collinear
        let line: Vec<[f64; 2]> = (0..6).map(|i| [i as f64 * 10.0, 0.0]).collect();
        assert!(fit_conic_direct(&line).is_none());
    }
}
