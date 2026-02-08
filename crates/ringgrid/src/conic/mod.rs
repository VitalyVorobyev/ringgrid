//! Ellipse / conic fitting primitives.
//!
//! Implements:
//! - Direct least-squares conic fit (Fitzgibbon et al., "Direct Least Square Fitting of Ellipses", 1999).
//! - Conversion between general conic coefficients and geometric ellipse parameters.
//! - Algebraic and geometric residual computation.
//! - RANSAC wrapper for outlier-robust fitting.

mod eigen;
mod fit;
mod ransac;
mod types;

pub use fit::{fit_conic_direct, fit_ellipse_direct};
#[cfg(test)]
pub(crate) use ransac::{fit_ellipse_ransac, rms_algebraic_distance};
pub use ransac::{rms_sampson_distance, try_fit_ellipse_ransac};
pub use types::{Conic2D, ConicCoeffs, Ellipse, RansacConfig, RansacResult};

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
