use crate::conic::{fit_ellipse_direct, try_fit_ellipse_ransac, Ellipse};
use crate::detector::DetectConfig;
use crate::ring::edge_sample::EdgeSampleResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OuterEllipseFitReject {
    TooFewPoints,
    DirectFitFailed,
    InvalidEllipse,
}

fn point_thresholds(config: &DetectConfig) -> (usize, usize) {
    let min_direct_fit_points = config.outer_fit.min_direct_fit_points.max(6);
    let min_ransac_points = config
        .outer_fit
        .min_ransac_points
        .max(min_direct_fit_points);
    (min_direct_fit_points, min_ransac_points)
}

pub(super) fn fit_outer_ellipse_with_reason(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Result<(Ellipse, Option<crate::conic::RansacResult>), OuterEllipseFitReject> {
    let (min_direct_fit_points, min_ransac_points) = point_thresholds(config);

    let (outer, outer_ransac) = if edge.outer_points.len() >= min_ransac_points {
        match try_fit_ellipse_ransac(&edge.outer_points, &config.outer_fit.ransac) {
            Ok(r) => (r.ellipse, Some(r)),
            Err(_) => {
                // Fall back to direct fit when robust fit fails.
                match fit_ellipse_direct(&edge.outer_points) {
                    Some(e) => (e, None),
                    None => return Err(OuterEllipseFitReject::DirectFitFailed),
                }
            }
        }
    } else if edge.outer_points.len() >= min_direct_fit_points {
        match fit_ellipse_direct(&edge.outer_points) {
            Some(e) => (e, None),
            None => return Err(OuterEllipseFitReject::DirectFitFailed),
        }
    } else {
        return Err(OuterEllipseFitReject::TooFewPoints);
    };

    if outer.a < config.min_semi_axis
        || outer.a > config.max_semi_axis
        || outer.b < config.min_semi_axis
        || outer.b > config.max_semi_axis
        || outer.aspect_ratio() > config.max_aspect_ratio
        || !outer.is_valid()
    {
        return Err(OuterEllipseFitReject::InvalidEllipse);
    }

    Ok((outer, outer_ransac))
}
