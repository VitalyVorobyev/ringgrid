//! Helpers shared by the grid-first path and its fallbacks.

use crate::detector::MarkerRecord;
use crate::homography::RansacStats;
use crate::pipeline::geometric_verify::{
    GeometricVerifyStats, annotate_h_reproj_err_px, geometric_verify_filter,
};
use crate::pipeline::{DetectConfig, compute_h_stats, mean_reproj_error_px, refit_homography};
use crate::pixelmap::PixelMapper;
use crate::target::TargetLayout;

/// Run the geometric verification gate, or — when it is disabled — only annotate
/// the reprojection diagnostic so opt-out callers can filter on it themselves.
/// Returns the gate stats (empty when disabled).
pub(super) fn phase_geometric_verify(
    markers: &mut Vec<MarkerRecord>,
    final_h: Option<&nalgebra::Matrix3<f64>>,
    config: &DetectConfig,
) -> GeometricVerifyStats {
    if config.advanced.geometric_verify {
        geometric_verify_filter(
            markers,
            final_h,
            &config.target,
            config.advanced.ransac_homography.inlier_threshold,
        )
    } else {
        annotate_h_reproj_err_px(markers, final_h);
        GeometricVerifyStats::default()
    }
}

/// Final homography: with at least ten labeled markers, refit over all of
/// them and keep whichever of the refit and `h_current` reprojects better;
/// stats are recomputed for the kept homography.
pub(super) fn phase_final_h(
    final_markers: &[MarkerRecord],
    h_current: Option<nalgebra::Matrix3<f64>>,
    mut ransac_stats: Option<RansacStats>,
    config: &DetectConfig,
) -> (Option<nalgebra::Matrix3<f64>>, Option<RansacStats>) {
    let final_h_matrix = if final_markers.len() >= 10 {
        let h_refit = refit_homography(
            final_markers,
            &config.advanced.ransac_homography,
            &config.target,
        )
        .map(|(h, _)| h);
        match (h_current, h_refit) {
            (Some(h_cur), Some(h_new)) => {
                let cur_err = mean_reproj_error_px(&h_cur, final_markers, &config.target);
                let new_err = mean_reproj_error_px(&h_new, final_markers, &config.target);
                if new_err.is_finite() && (new_err < cur_err || !cur_err.is_finite()) {
                    Some(h_new)
                } else {
                    Some(h_cur)
                }
            }
            (None, Some(h_new)) => Some(h_new),
            (Some(h_cur), None) => Some(h_cur),
            (None, None) => None,
        }
    } else {
        h_current
    };

    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| {
            compute_h_stats(
                h,
                final_markers,
                config.advanced.ransac_homography.inlier_threshold,
                &config.target,
            )
        })
        .or_else(|| ransac_stats.take());

    (final_h_matrix, final_ransac)
}

fn drop_unmappable_markers(markers: &mut Vec<MarkerRecord>, mapper: &dyn PixelMapper) -> usize {
    let before = markers.len();
    markers.retain(|m| {
        mapper
            .working_to_image_pixel(m.center)
            .map(|p| p[0].is_finite() && p[1].is_finite())
            .unwrap_or(false)
    });
    before.saturating_sub(markers.len())
}

pub(super) fn drop_unmappable_markers_with_warning(
    markers: &mut Vec<MarkerRecord>,
    mapper: Option<&dyn PixelMapper>,
) {
    let Some(mapper) = mapper else {
        return;
    };
    let dropped = drop_unmappable_markers(markers, mapper);
    if dropped > 0 {
        tracing::warn!(
            dropped,
            "dropping markers whose mapped centers cannot be converted to image frame"
        );
    }
}

pub(super) fn map_centers_to_image(markers: &mut [MarkerRecord], mapper: &dyn PixelMapper) {
    for marker in markers.iter_mut() {
        let center_mapped = marker.center;
        marker.center_mapped = Some(center_mapped);
        if let Some(center_image) = mapper.working_to_image_pixel(center_mapped) {
            marker.center = center_image;
        }
    }
}

pub(super) fn sync_marker_board_correspondence(
    markers: &mut [MarkerRecord],
    target: &TargetLayout,
) -> usize {
    // Plain targets are labeled by grid coordinate, not ID: their board/frame
    // positions are owned by the assignment/anchor/completion stages and must
    // not be cleared here.
    if !target.is_coded() {
        return 0;
    }
    let mut cleared_invalid_ids = 0usize;
    for marker in markers.iter_mut() {
        match marker.id {
            Some(id) => {
                if let Some(board_xy) = target.xy_mm_of_id(id) {
                    marker.board_xy_mm = Some([board_xy[0] as f64, board_xy[1] as f64]);
                    marker.grid_coord = target.coord_of_id(id).map(|c| [c.u, c.v]);
                } else {
                    marker.id = None;
                    marker.board_xy_mm = None;
                    marker.grid_coord = None;
                    cleared_invalid_ids += 1;
                }
            }
            None => {
                marker.board_xy_mm = None;
                marker.grid_coord = None;
            }
        }
    }
    cleared_invalid_ids
}

pub(super) fn sync_marker_board_correspondence_with_logging(
    markers: &mut [MarkerRecord],
    target: &TargetLayout,
) {
    let cleared_invalid_ids = sync_marker_board_correspondence(markers, target);
    tracing::debug!(
        cleared_invalid_ids,
        n_markers = markers.len(),
        "synchronized marker id/board correspondence"
    );
}
