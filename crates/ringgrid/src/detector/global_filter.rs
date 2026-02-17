use crate::board_layout::BoardLayout;
use crate::homography::{self, RansacHomographyConfig};
use crate::{DetectedMarker, RansacStats};

/// Apply global homography RANSAC filter.
///
/// Returns `(filtered markers, RANSAC result, stats)`.
pub fn global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    let correspondences = homography::collect_marker_correspondences(
        markers,
        board,
        homography::CorrespondenceDestinationFrame::Image,
        homography::DuplicateIdPolicy::KeepAll,
        |m| Some(m.center),
    );
    debug_assert_eq!(
        correspondences.dst_frame,
        homography::CorrespondenceDestinationFrame::Image
    );

    tracing::info!(
        "Global filter: {} decoded candidates out of {} total detections",
        correspondences.len(),
        markers.len()
    );

    if correspondences.len() < 4 {
        tracing::warn!(
            "Too few decoded candidates for homography ({} < 4)",
            correspondences.len()
        );
        return (markers.to_vec(), None, None);
    }

    let result = match homography::fit_homography_ransac(
        &correspondences.src_board_mm,
        &correspondences.dst_points,
        config,
    ) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Homography RANSAC failed: {}", e);
            return (markers.to_vec(), None, None);
        }
    };

    // Collect inliers/outliers and per-id errors.
    let mut filtered: Vec<DetectedMarker> = Vec::new();
    for (j, marker_index) in correspondences.marker_indices.iter().enumerate() {
        if result.inlier_mask.get(j).copied().unwrap_or(false) {
            filtered.push(markers[*marker_index].clone());
        }
    }

    // Compute stats.
    let mut inlier_errors =
        homography::collect_masked_inlier_errors(&result.errors, &result.inlier_mask);
    let (mean_err, p95_err) = homography::mean_and_p95(&mut inlier_errors);

    let stats = RansacStats {
        n_candidates: correspondences.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    tracing::info!(
        "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
        result.n_inliers,
        correspondences.len(),
        mean_err,
        p95_err,
    );

    (filtered, Some(result), Some(stats))
}
