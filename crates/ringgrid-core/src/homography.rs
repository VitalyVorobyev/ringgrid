//! Plane-to-image homography estimation via DLT with Hartley normalization.
//!
//! Provides:
//! - Direct Linear Transform (DLT) from ≥4 point correspondences.
//! - RANSAC wrapper for outlier-robust fitting.
//! - Reprojection error computation.

use nalgebra::{DMatrix, Matrix3, Vector3};

// ── Error type ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum HomographyError {
    TooFewPoints { needed: usize, got: usize },
    NumericalFailure(String),
    InsufficientInliers { needed: usize, found: usize },
}

impl std::fmt::Display for HomographyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { needed, got } => {
                write!(f, "too few points: need {}, got {}", needed, got)
            }
            Self::NumericalFailure(msg) => write!(f, "numerical failure: {}", msg),
            Self::InsufficientInliers { needed, found } => {
                write!(f, "insufficient inliers: need {}, found {}", needed, found)
            }
        }
    }
}

impl std::error::Error for HomographyError {}

// ── Projection ───────────────────────────────────────────────────────────

/// Project a 2D point through a 3×3 homography: H * [x, y, 1]^T → [u, v].
pub fn project(h: &Matrix3<f64>, x: f64, y: f64) -> [f64; 2] {
    let p = h * Vector3::new(x, y, 1.0);
    if p[2].abs() < 1e-15 {
        return [f64::NAN, f64::NAN];
    }
    [p[0] / p[2], p[1] / p[2]]
}

/// Reprojection error: ||project(H, src) - dst||.
pub fn reprojection_error(h: &Matrix3<f64>, src: &[f64; 2], dst: &[f64; 2]) -> f64 {
    let p = project(h, src[0], src[1]);
    let dx = p[0] - dst[0];
    let dy = p[1] - dst[1];
    (dx * dx + dy * dy).sqrt()
}

// ── Hartley normalization ────────────────────────────────────────────────

/// Compute a normalizing transform: translate centroid to origin, scale so
/// mean distance from origin is sqrt(2).
fn normalize_points(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
    let n = pts.len() as f64;
    let cx: f64 = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p[1]).sum::<f64>() / n;

    let mean_dist: f64 = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let s = if mean_dist > 1e-15 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);

    let normalized: Vec<[f64; 2]> = pts.iter().map(|p| [s * (p[0] - cx), s * (p[1] - cy)]).collect();

    (t, normalized)
}

// ── DLT ──────────────────────────────────────────────────────────────────

/// Estimate homography from ≥4 point correspondences using DLT.
///
/// `src`: source points (e.g., board coordinates in mm).
/// `dst`: destination points (e.g., image coordinates in pixels).
///
/// Returns the 3×3 homography H such that dst ≈ project(H, src).
pub fn estimate_homography_dlt(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
) -> Result<Matrix3<f64>, HomographyError> {
    let n = src.len();
    if n < 4 || dst.len() < 4 {
        return Err(HomographyError::TooFewPoints {
            needed: 4,
            got: n.min(dst.len()),
        });
    }
    if src.len() != dst.len() {
        return Err(HomographyError::NumericalFailure(
            "src and dst must have the same length".into(),
        ));
    }

    // Hartley normalization
    let (t_src, src_n) = normalize_points(src);
    let (t_dst, dst_n) = normalize_points(dst);

    // Build 2n × 9 matrix A
    let mut a = DMatrix::zeros(2 * n, 9);
    for i in 0..n {
        let (sx, sy) = (src_n[i][0], src_n[i][1]);
        let (dx, dy) = (dst_n[i][0], dst_n[i][1]);

        // Row 2i:   [  0  0  0 | -sx -sy -1 | dy*sx  dy*sy  dy ]
        a[(2 * i, 3)] = -sx;
        a[(2 * i, 4)] = -sy;
        a[(2 * i, 5)] = -1.0;
        a[(2 * i, 6)] = dy * sx;
        a[(2 * i, 7)] = dy * sy;
        a[(2 * i, 8)] = dy;

        // Row 2i+1: [ sx  sy  1 |  0  0  0 | -dx*sx -dx*sy -dx ]
        a[(2 * i + 1, 0)] = sx;
        a[(2 * i + 1, 1)] = sy;
        a[(2 * i + 1, 2)] = 1.0;
        a[(2 * i + 1, 6)] = -dx * sx;
        a[(2 * i + 1, 7)] = -dx * sy;
        a[(2 * i + 1, 8)] = -dx;
    }

    // Solve via A^T A: the solution h is the eigenvector of the smallest
    // eigenvalue of the 9×9 matrix A^T A. This avoids thin-SVD dimension issues.
    let ata = a.transpose() * &a;
    let eig = nalgebra::SymmetricEigen::new(ata);

    // Find eigenvector with smallest eigenvalue
    let mut min_idx = 0;
    let mut min_val = eig.eigenvalues[0].abs();
    for i in 1..9 {
        let v = eig.eigenvalues[i].abs();
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }
    let h_vec: Vec<f64> = (0..9).map(|j| eig.eigenvectors[(j, min_idx)]).collect();
    let h_norm = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2],
        h_vec[3], h_vec[4], h_vec[5],
        h_vec[6], h_vec[7], h_vec[8],
    );

    // Denormalize: H = T_dst^-1 * H_norm * T_src
    let t_dst_inv = t_dst
        .try_inverse()
        .ok_or_else(|| HomographyError::NumericalFailure("T_dst not invertible".into()))?;
    let h = t_dst_inv * h_norm * t_src;

    // Normalize so h[2][2] = 1 (if possible)
    let scale = h[(2, 2)];
    if scale.abs() < 1e-15 {
        Ok(h)
    } else {
        Ok(h / scale)
    }
}

// ── RANSAC ───────────────────────────────────────────────────────────────

/// RANSAC configuration for homography fitting.
#[derive(Debug, Clone)]
pub struct RansacHomographyConfig {
    /// Maximum number of RANSAC iterations.
    pub max_iters: usize,
    /// Inlier threshold (reprojection error in pixels).
    pub inlier_threshold: f64,
    /// Minimum number of inliers for a valid model.
    pub min_inliers: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for RansacHomographyConfig {
    fn default() -> Self {
        Self {
            max_iters: 2000,
            inlier_threshold: 5.0,
            min_inliers: 6,
            seed: 0,
        }
    }
}

/// Result of RANSAC homography fitting.
#[derive(Debug, Clone)]
pub struct RansacHomographyResult {
    /// The fitted homography.
    pub h: Matrix3<f64>,
    /// Boolean mask: true for inliers.
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub n_inliers: usize,
    /// Per-inlier reprojection errors (only for inliers; others are 0).
    pub errors: Vec<f64>,
}

/// Fit homography with RANSAC.
///
/// `src`: source points (board coords).
/// `dst`: destination points (image coords).
pub fn fit_homography_ransac(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    config: &RansacHomographyConfig,
) -> Result<RansacHomographyResult, HomographyError> {
    let n = src.len();
    if n < 4 {
        return Err(HomographyError::TooFewPoints { needed: 4, got: n });
    }

    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    let mut best_inliers = 0usize;
    let mut best_mask: Vec<bool> = vec![false; n];
    let mut best_h = Matrix3::identity();

    for _ in 0..config.max_iters {
        // Sample 4 distinct indices
        let mut indices = [0usize; 4];
        let mut attempts = 0;
        loop {
            for idx in &mut indices {
                *idx = rng.gen_range(0..n);
            }
            // Check distinct
            let mut ok = true;
            for i in 0..4 {
                for j in (i + 1)..4 {
                    if indices[i] == indices[j] {
                        ok = false;
                    }
                }
            }
            if ok {
                break;
            }
            attempts += 1;
            if attempts > 100 {
                break;
            }
        }

        let s4: Vec<[f64; 2]> = indices.iter().map(|&i| src[i]).collect();
        let d4: Vec<[f64; 2]> = indices.iter().map(|&i| dst[i]).collect();

        let h = match estimate_homography_dlt(&s4, &d4) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Count inliers
        let mut count = 0usize;
        let mut mask = vec![false; n];
        for i in 0..n {
            let err = reprojection_error(&h, &src[i], &dst[i]);
            if err < config.inlier_threshold {
                mask[i] = true;
                count += 1;
            }
        }

        if count > best_inliers {
            best_inliers = count;
            best_mask = mask;
            best_h = h;

            // Early exit if >90% inliers
            if count * 10 > n * 9 {
                break;
            }
        }
    }

    if best_inliers < config.min_inliers {
        return Err(HomographyError::InsufficientInliers {
            needed: config.min_inliers,
            found: best_inliers,
        });
    }

    // Refit using all inliers
    let inlier_src: Vec<[f64; 2]> = (0..n).filter(|&i| best_mask[i]).map(|i| src[i]).collect();
    let inlier_dst: Vec<[f64; 2]> = (0..n).filter(|&i| best_mask[i]).map(|i| dst[i]).collect();

    let h_refit = estimate_homography_dlt(&inlier_src, &inlier_dst).unwrap_or(best_h);

    // Recompute errors and mask with refined H
    let mut final_mask = vec![false; n];
    let mut errors = vec![0.0f64; n];
    let mut final_inliers = 0usize;
    for i in 0..n {
        let err = reprojection_error(&h_refit, &src[i], &dst[i]);
        errors[i] = err;
        if err < config.inlier_threshold {
            final_mask[i] = true;
            final_inliers += 1;
        }
    }

    Ok(RansacHomographyResult {
        h: h_refit,
        inlier_mask: final_mask,
        n_inliers: final_inliers,
        errors,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    fn make_test_homography() -> Matrix3<f64> {
        // Scale + translate + mild perspective
        Matrix3::new(
            3.5, 0.1, 640.0,
            -0.05, 3.3, 480.0,
            0.0001, -0.00005, 1.0,
        )
    }

    #[test]
    fn test_dlt_exact_4points() {
        let h_true = make_test_homography();
        let src = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let dst: Vec<[f64; 2]> = src.iter().map(|s| project(&h_true, s[0], s[1])).collect();

        let h_est = estimate_homography_dlt(&src, &dst).unwrap();

        // Check reprojection of all 4 points
        for (s, d) in src.iter().zip(&dst) {
            let err = reprojection_error(&h_est, s, d);
            assert!(err < 1e-6, "reprojection error too large: {}", err);
        }
    }

    #[test]
    fn test_dlt_overdetermined() {
        let h_true = make_test_homography();
        // Grid of 5x5 points
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let s = [i as f64 * 20.0, j as f64 * 20.0];
                let d = project(&h_true, s[0], s[1]);
                src.push(s);
                dst.push(d);
            }
        }

        let h_est = estimate_homography_dlt(&src, &dst).unwrap();

        for (s, d) in src.iter().zip(&dst) {
            let err = reprojection_error(&h_est, s, d);
            assert!(err < 1e-6, "reprojection error: {}", err);
        }
    }

    #[test]
    fn test_ransac_with_outliers() {
        let h_true = make_test_homography();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // 20 inlier points
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..20 {
            let s = [(i % 5) as f64 * 30.0, (i / 5) as f64 * 30.0];
            let d = project(&h_true, s[0], s[1]);
            // Add small noise
            let d = [
                d[0] + rng.gen_range(-0.5..0.5),
                d[1] + rng.gen_range(-0.5..0.5),
            ];
            src.push(s);
            dst.push(d);
        }

        // 8 outliers
        for _ in 0..8 {
            let s = [rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)];
            let d = [rng.gen_range(0.0..1280.0), rng.gen_range(0.0..960.0)];
            src.push(s);
            dst.push(d);
        }

        let config = RansacHomographyConfig {
            max_iters: 2000,
            inlier_threshold: 3.0,
            min_inliers: 6,
            seed: 99,
        };

        let result = fit_homography_ransac(&src, &dst, &config).unwrap();

        // Should find at least 18 of the 20 inliers
        assert!(result.n_inliers >= 18, "only {} inliers", result.n_inliers);

        // Check that reprojection errors for true inliers are small
        for i in 0..20 {
            let err = reprojection_error(&result.h, &src[i], &dst[i]);
            assert!(err < 5.0, "inlier {} has error {}", i, err);
        }
    }

    #[test]
    fn test_project_roundtrip() {
        let h = make_test_homography();
        let h_inv = h.try_inverse().unwrap();

        let p = [50.0, 75.0];
        let q = project(&h, p[0], p[1]);
        let p_back = project(&h_inv, q[0], q[1]);

        assert_relative_eq!(p[0], p_back[0], epsilon = 1e-8);
        assert_relative_eq!(p[1], p_back[1], epsilon = 1e-8);
    }

    #[test]
    fn test_too_few_points() {
        let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let dst = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let result = estimate_homography_dlt(&src, &dst);
        assert!(result.is_err());
    }
}
