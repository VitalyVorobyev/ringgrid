//! Neighbour-supported inner-ellipse recovery for labeled markers.
//!
//! A ring whose inner-edge estimate failed (small or blurred rings: the
//! per-ray peaks of a thin inner band rarely agree within the estimate's
//! tolerance) has no inner ellipse and so no perspective-corrected centre. Its
//! lattice neighbours, though, measure the inner/outer ratio of this view: the
//! inner edge is searched again at that ratio, ray by ray, and the marker gets
//! an inner ellipse — and then a two-conic centre — only if the edge is found
//! in the image and the fit passes the ordinary gates.

use image::GrayImage;

use super::inner_fit::{InnerFitStatus, fit_inner_ellipse_with_ratio_prior};
use super::{CenterSource, DetectConfig, MarkerRecord, apply_projective_centers};
use crate::pixelmap::PixelMapper;

/// Lattice neighbourhood (board distance, in centre spacings) whose
/// well-fitted rings supply the ratio: the first two hex rings, or the rect
/// 8-neighbourhood plus the next row/column.
const NEIGHBOUR_RADIUS_SPACINGS: f64 = 2.05;
/// Neighbours with a two-conic centre needed for a ratio.
const MIN_NEIGHBOURS: usize = 3;

/// Inner/outer mean-axis ratio of a marker with both ellipses.
fn inner_outer_ratio(m: &MarkerRecord) -> Option<f64> {
    let (inner, outer) = (m.ellipse_inner.as_ref()?, m.ellipse_outer.as_ref()?);
    Some(inner.mean_axis() / outer.mean_axis())
}

/// Median inner/outer ratio of the two-conic-centred markers within
/// [`NEIGHBOUR_RADIUS_SPACINGS`] of board position `xy`.
fn neighbour_ratio(anchors: &[([f64; 2], f64)], xy: [f64; 2], radius_mm: f64) -> Option<f64> {
    let mut ratios: Vec<f64> = anchors
        .iter()
        .filter(|(p, _)| (p[0] - xy[0]).hypot(p[1] - xy[1]) <= radius_mm)
        .map(|&(_, r)| r)
        .collect();
    if ratios.len() < MIN_NEIGHBOURS {
        return None;
    }
    ratios.sort_by(f64::total_cmp);
    Some(ratios[ratios.len() / 2])
}

/// Retry the inner fit of every labeled marker without an inner ellipse at
/// its neighbours' inner/outer ratio, then give the recovered markers their
/// two-conic centre (with the ordinary gates). Markers in the working frame;
/// runs before the final homography refit and geometric verify, which judge
/// the recovered markers like any other. Returns the number recovered.
pub(crate) fn recover_inner_from_neighbours(
    gray: &GrayImage,
    markers: &mut [MarkerRecord],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> usize {
    if !config.circle_refinement.uses_projective_center()
        || std::env::var("RINGGRID_EXP_OFF").is_ok()
    {
        return 0;
    }
    let anchors: Vec<([f64; 2], f64)> = markers
        .iter()
        .filter(|m| m.center_source == CenterSource::ProjectiveCenter)
        .filter_map(|m| Some((m.board_xy_mm?, inner_outer_ratio(m)?)))
        .collect();
    let radius_mm = NEIGHBOUR_RADIUS_SPACINGS * f64::from(config.target.min_center_spacing_mm());
    let mut n_recovered = 0usize;
    for m in markers.iter_mut() {
        if m.ellipse_inner.is_some() {
            continue;
        }
        let (Some(outer), Some(xy)) = (m.ellipse_outer, m.board_xy_mm) else {
            continue;
        };
        let Some(ratio) = neighbour_ratio(&anchors, xy, radius_mm) else {
            continue;
        };
        let Some(fit) = fit_inner_ellipse_with_ratio_prior(
            gray,
            &outer,
            ratio as f32,
            &config.advanced.marker_spec,
            mapper,
            &config.advanced.inner_fit,
        ) else {
            continue;
        };
        if fit.status != InnerFitStatus::Ok {
            continue;
        }
        m.ellipse_inner = fit.ellipse_inner;
        m.fit.n_points_inner = fit.points_inner.len();
        m.fit.ransac_inlier_ratio_inner = fit.ransac_inlier_ratio_inner;
        m.fit.rms_residual_inner = fit.rms_residual_inner;
        m.fit.max_angular_gap_inner = fit.max_angular_gap;
        m.fit.inner_fit_status = Some(InnerFitStatus::Ok);
        m.fit.inner_fit_reason = None;
        m.edge_points_inner = Some(fit.points_inner);
        apply_projective_centers(std::slice::from_mut(m), config);
        n_recovered += 1;
    }
    tracing::debug!(n_recovered, "neighbour-supported inner-ellipse recovery");
    n_recovered
}
