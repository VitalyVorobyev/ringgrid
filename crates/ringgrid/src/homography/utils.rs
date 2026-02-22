//! Higher-level homography utilities: refitting, statistics, format conversion.

use crate::board_layout::BoardLayout;
use crate::detector::DetectedMarker;

use super::core::{fit_homography_ransac, RansacHomographyConfig, RansacStats};
use super::{
    collect_marker_correspondences, collect_masked_inlier_errors, mean_and_p95,
    reprojection_errors, CorrespondenceDestinationFrame, DuplicateIdPolicy,
};

pub(crate) fn refit_homography(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> Option<(nalgebra::Matrix3<f64>, RansacStats)> {
    let correspondences = collect_marker_correspondences(
        markers,
        board,
        CorrespondenceDestinationFrame::Image,
        DuplicateIdPolicy::KeepAll,
        |m| Some(m.center),
    );
    debug_assert_eq!(
        correspondences.dst_frame,
        CorrespondenceDestinationFrame::Image
    );
    if correspondences.len() < 4 {
        return None;
    }

    // Use a light RANSAC (most outliers already removed)
    let light_config = RansacHomographyConfig {
        max_iters: 500,
        inlier_threshold: config.inlier_threshold,
        min_inliers: config.min_inliers,
        seed: config.seed + 1,
    };

    match fit_homography_ransac(
        &correspondences.src_board_mm,
        &correspondences.dst_points,
        &light_config,
    ) {
        Ok(result) => {
            let mut inlier_errors =
                collect_masked_inlier_errors(&result.errors, &result.inlier_mask);
            let (mean_err, p95_err) = mean_and_p95(&mut inlier_errors);

            let stats = RansacStats {
                n_candidates: correspondences.len(),
                n_inliers: result.n_inliers,
                threshold_px: light_config.inlier_threshold,
                mean_err_px: mean_err,
                p95_err_px: p95_err,
            };

            Some((result.h, stats))
        }
        Err(_) => None,
    }
}

pub(crate) fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

pub(crate) fn mean_reproj_error_px(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    board: &BoardLayout,
) -> f64 {
    let correspondences = collect_marker_correspondences(
        markers,
        board,
        CorrespondenceDestinationFrame::Image,
        DuplicateIdPolicy::KeepAll,
        |m| Some(m.center),
    );
    debug_assert_eq!(
        correspondences.dst_frame,
        CorrespondenceDestinationFrame::Image
    );
    if correspondences.len() == 0 {
        return f64::NAN;
    }

    let mut sum = 0.0f64;
    let mut n = 0usize;
    for err in reprojection_errors(h, &correspondences) {
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

pub(crate) fn compute_h_stats(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    thresh_px: f64,
    board: &BoardLayout,
) -> Option<RansacStats> {
    let correspondences = collect_marker_correspondences(
        markers,
        board,
        CorrespondenceDestinationFrame::Image,
        DuplicateIdPolicy::KeepAll,
        |m| Some(m.center),
    );
    debug_assert_eq!(
        correspondences.dst_frame,
        CorrespondenceDestinationFrame::Image
    );
    let errors: Vec<f64> = reprojection_errors(h, &correspondences)
        .into_iter()
        .filter(|e| e.is_finite())
        .collect();
    if errors.len() < 4 {
        return None;
    }

    let mut inlier_errors: Vec<f64> = errors.iter().copied().filter(|&e| e <= thresh_px).collect();
    let (mean_err, p95_err) = mean_and_p95(&mut inlier_errors);

    Some(RansacStats {
        n_candidates: errors.len(),
        n_inliers: inlier_errors.len(),
        threshold_px: thresh_px,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    })
}
