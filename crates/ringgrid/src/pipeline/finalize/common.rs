//! Helpers shared by the coded and plain finalize paths.

use crate::detector::MarkerRecord;
use crate::pipeline::DetectConfig;
use crate::pipeline::geometric_verify::{
    GeometricVerifyStats, annotate_h_reproj_err_px, geometric_verify_filter,
};
use crate::pixelmap::PixelMapper;
use crate::target::TargetLayout;

#[inline]
pub(super) fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

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
