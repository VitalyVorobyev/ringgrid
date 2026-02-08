//! RANSAC wrapper for outlier-robust ellipse fitting.

use super::fit_conic_direct;
use super::types::{ConicError, Ellipse, RansacConfig, RansacResult};
use super::ConicCoeffs;

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

    let (final_conic, final_ellipse) = fit_conic_direct(&inlier_pts)
        .and_then(|conic| conic.to_ellipse().map(|ellipse| (conic, ellipse)))
        .or_else(|| best_conic.zip(best_ellipse))?;

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
