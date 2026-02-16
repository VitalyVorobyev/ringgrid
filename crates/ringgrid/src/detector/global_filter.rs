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
    // Build correspondences from decoded markers.
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut decoded_ids: Vec<usize> = Vec::new();
    let mut candidate_indices: Vec<usize> = Vec::new();

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            if let Some(xy) = board.xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                decoded_ids.push(id);
                candidate_indices.push(i);
            }
        }
    }

    tracing::info!(
        "Global filter: {} decoded candidates out of {} total detections",
        candidate_indices.len(),
        markers.len()
    );

    if src_pts.len() < 4 {
        tracing::warn!(
            "Too few decoded candidates for homography ({} < 4)",
            src_pts.len()
        );
        return (markers.to_vec(), None, None);
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Homography RANSAC failed: {}", e);
            return (markers.to_vec(), None, None);
        }
    };

    // Collect inliers/outliers and per-id errors.
    let mut filtered: Vec<DetectedMarker> = Vec::new();
    let mut inlier_errors: Vec<f64> = Vec::new();
    for (j, _id) in decoded_ids.iter().enumerate() {
        let err = result.errors[j];
        if result.inlier_mask[j] {
            inlier_errors.push(err);
            filtered.push(markers[candidate_indices[j]].clone());
        }
    }

    // Compute stats.
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

    let stats = RansacStats {
        n_candidates: src_pts.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    tracing::info!(
        "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
        result.n_inliers,
        src_pts.len(),
        mean_err,
        p95_err,
    );

    (filtered, Some(result), Some(stats))
}
