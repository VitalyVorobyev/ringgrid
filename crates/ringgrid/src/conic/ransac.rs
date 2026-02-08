//! RANSAC wrapper for outlier-robust ellipse fitting.

use super::fit_conic_direct;
use super::types::{ConicError, Ellipse, RansacConfig, RansacResult};

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
    let mut best_ellipse: Option<Ellipse> = None;
    let mut best_mask: Vec<bool> = vec![false; n];

    for _ in 0..config.max_iters {
        // Sample 6 random points
        let sample = sample_indices(&mut rng, n, 6);
        let sample_pts: Vec<[f64; 2]> = sample.iter().map(|&i| points[i]).collect();

        // Fit ellipse to sample
        let Some(conic) = fit_conic_direct(&sample_pts) else {
            continue;
        };
        let Some(ellipse) = conic.to_ellipse() else {
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

    let final_ellipse = fit_conic_direct(&inlier_pts)
        .and_then(|conic| conic.to_ellipse())
        .or(best_ellipse)?;

    // Recompute inlier count with final model using Sampson distance
    let mut final_count = 0;
    for &[x, y] in points {
        let dist = final_ellipse.sampson_distance(x, y);
        if dist < config.inlier_threshold {
            final_count += 1;
        }
    }

    Some(RansacResult {
        ellipse: final_ellipse,
        num_inliers: final_count,
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
