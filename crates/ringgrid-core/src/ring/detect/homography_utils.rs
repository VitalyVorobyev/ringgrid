use crate::board_spec;
use crate::homography::{self, RansacHomographyConfig};
use crate::{DetectedMarker, RansacStats};

pub(super) fn refit_homography(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> (Option<[[f64; 3]; 3]>, Option<RansacStats>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for m in markers {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src.push([xy[0] as f64, xy[1] as f64]);
                dst.push(m.center);
            }
        }
    }

    if src.len() < 4 {
        return (None, None);
    }

    // Use a light RANSAC (most outliers already removed)
    let light_config = RansacHomographyConfig {
        max_iters: 500,
        inlier_threshold: config.inlier_threshold,
        min_inliers: config.min_inliers,
        seed: config.seed + 1,
    };

    match homography::fit_homography_ransac(&src, &dst, &light_config) {
        Ok(result) => {
            let mut errors: Vec<f64> = result
                .inlier_mask
                .iter()
                .zip(&result.errors)
                .filter(|(&m, _)| m)
                .map(|(_, &e)| e)
                .collect();
            errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean_err = if errors.is_empty() {
                0.0
            } else {
                errors.iter().sum::<f64>() / errors.len() as f64
            };
            let p95_err = if errors.is_empty() {
                0.0
            } else {
                let idx = ((errors.len() as f64 * 0.95) as usize).min(errors.len() - 1);
                errors[idx]
            };

            let stats = RansacStats {
                n_candidates: src.len(),
                n_inliers: result.n_inliers,
                threshold_px: light_config.inlier_threshold,
                mean_err_px: mean_err,
                p95_err_px: p95_err,
            };

            (Some(matrix3_to_array(&result.h)), Some(stats))
        }
        Err(_) => (None, None),
    }
}

pub(super) fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

pub(super) fn array_to_matrix3(m: &[[f64; 3]; 3]) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    )
}

pub(super) fn mean_reproj_error_px(h: &nalgebra::Matrix3<f64>, markers: &[DetectedMarker]) -> f64 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for m in markers {
        let Some(id) = m.id else {
            continue;
        };
        let Some(xy) = board_spec::xy_mm(id) else {
            continue;
        };
        let err = homography::reprojection_error(
            h,
            &[xy[0] as f64, xy[1] as f64],
            &[m.center[0], m.center[1]],
        );
        if err.is_finite() {
            sum += err;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

pub(super) fn compute_h_stats(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    thresh_px: f64,
) -> Option<RansacStats> {
    let mut errors: Vec<f64> = Vec::new();
    for m in markers {
        let Some(id) = m.id else {
            continue;
        };
        let Some(xy) = board_spec::xy_mm(id) else {
            continue;
        };
        let err = homography::reprojection_error(
            h,
            &[xy[0] as f64, xy[1] as f64],
            &[m.center[0], m.center[1]],
        );
        if err.is_finite() {
            errors.push(err);
        }
    }
    if errors.len() < 4 {
        return None;
    }

    let mut inlier_errors: Vec<f64> = errors.iter().copied().filter(|&e| e <= thresh_px).collect();
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    Some(RansacStats {
        n_candidates: errors.len(),
        n_inliers: inlier_errors.len(),
        threshold_px: thresh_px,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    })
}

pub(super) fn refit_homography_matrix(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> Option<(nalgebra::Matrix3<f64>, RansacStats)> {
    let (h_arr, stats) = refit_homography(markers, config);
    match (h_arr, stats) {
        (Some(h_arr), Some(stats)) => Some((array_to_matrix3(&h_arr), stats)),
        _ => None,
    }
}
