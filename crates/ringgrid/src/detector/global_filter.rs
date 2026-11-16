use crate::conic::RansacConfig;
use crate::detector::MarkerRecord;
use crate::homography;
use crate::homography::RansacStats;
use crate::target::TargetLayout;

/// Apply global homography RANSAC filter.
///
/// Returns `(filtered markers, RANSAC result, stats)`.
pub(crate) fn global_filter(
    markers: &[MarkerRecord],
    config: &RansacConfig,
    target: &TargetLayout,
) -> (
    Vec<MarkerRecord>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    let correspondences = homography::collect_marker_correspondences(
        markers,
        target,
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
    let mut filtered: Vec<MarkerRecord> = Vec::new();
    for (j, marker_index) in correspondences.marker_indices.iter().enumerate() {
        if result.inlier_mask.get(j).copied().unwrap_or(false) {
            filtered.push(markers[*marker_index].clone());
        }
    }

    // Compute stats.
    let stats = RansacStats::from_ransac(&result, correspondences.len(), config.inlier_threshold);

    tracing::info!(
        "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
        result.n_inliers,
        correspondences.len(),
        stats.mean_err_px,
        stats.p95_err_px,
    );

    (filtered, Some(result), Some(stats))
}
