use std::collections::BTreeMap;

use crate::board_layout::BoardLayout;
use crate::conic::{fit_ellipse_direct, rms_sampson_distance};
use crate::homography;
use crate::DetectedMarker;

use super::super::{DivisionModel, PixelMapper};

/// Minimum number of edge points per ring required for conic refit.
const MIN_EDGE_POINTS: usize = 6;

/// Edge point data for a single marker: (outer_points, inner_points).
pub(super) type MarkerEdgeData = (Vec<[f64; 2]>, Vec<[f64; 2]>);

#[derive(Debug, Clone, Copy)]
pub(super) struct HomographySelfError {
    pub mean_error_px: f64,
    pub n_inliers: usize,
}

pub(super) fn collect_marker_edge_data(markers: &[DetectedMarker]) -> Vec<MarkerEdgeData> {
    markers
        .iter()
        .filter_map(|m| {
            let outer = m.edge_points_outer.as_ref()?;
            let inner = m.edge_points_inner.as_ref()?;
            if outer.len() >= MIN_EDGE_POINTS && inner.len() >= MIN_EDGE_POINTS {
                Some((outer.clone(), inner.clone()))
            } else {
                None
            }
        })
        .collect()
}

fn trimmed_mean(values: &mut [f64], trim_fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let trim = ((n as f64) * trim_fraction.clamp(0.0, 0.49)).floor() as usize;
    if 2 * trim >= n {
        return None;
    }
    let slice = &values[trim..(n - trim)];
    if slice.is_empty() {
        return None;
    }
    Some(slice.iter().sum::<f64>() / slice.len() as f64)
}

/// Compute a robust projective-conic objective across markers when edge points
/// are undistorted with the given lambda.
pub(super) fn conic_consistency_objective(
    lambda: f64,
    marker_edge_data: &[MarkerEdgeData],
    image_center: [f64; 2],
    trim_fraction: f64,
) -> f64 {
    let model = DivisionModel::new(lambda, image_center[0], image_center[1]);
    let mut marker_objective = Vec::with_capacity(marker_edge_data.len());

    for (outer_pts, inner_pts) in marker_edge_data {
        let outer_ud = model.undistort_points(outer_pts);
        let inner_ud = model.undistort_points(inner_pts);

        let Some(outer_ellipse) = fit_ellipse_direct(&outer_ud) else {
            continue;
        };
        let Some(inner_ellipse) = fit_ellipse_direct(&inner_ud) else {
            continue;
        };

        let rms_outer = rms_sampson_distance(&outer_ellipse, &outer_ud);
        let rms_inner = rms_sampson_distance(&inner_ellipse, &inner_ud);
        if !rms_outer.is_finite() || !rms_inner.is_finite() {
            continue;
        }

        let value = 0.5 * (rms_outer + rms_inner);
        if value.is_finite() {
            marker_objective.push(value);
        }
    }

    if marker_objective.is_empty() {
        return f64::MAX;
    }

    let Some(base) = trimmed_mean(&mut marker_objective, trim_fraction) else {
        return f64::MAX;
    };

    // Mild regularization to avoid unstable large-|lambda| solutions when
    // objective curvature is flat.
    let lambda_reg = 1e-6 * (lambda * 1.0e6).powi(2);
    base + lambda_reg
}

pub(super) fn homography_self_error_px(
    markers: &[DetectedMarker],
    board: &BoardLayout,
    mapper: &dyn PixelMapper,
) -> Option<HomographySelfError> {
    let mut by_id: BTreeMap<usize, (f32, [f64; 2])> = BTreeMap::new();

    for m in markers {
        let Some(id) = m.id else {
            continue;
        };
        let Some(center_w) = mapper.image_to_working_pixel(m.center) else {
            continue;
        };
        if !center_w[0].is_finite() || !center_w[1].is_finite() {
            continue;
        }

        let conf = m.confidence;
        match by_id.get_mut(&id) {
            Some((best_conf, best_center)) => {
                if conf > *best_conf {
                    *best_conf = conf;
                    *best_center = center_w;
                }
            }
            None => {
                by_id.insert(id, (conf, center_w));
            }
        }
    }

    let mut src = Vec::<[f64; 2]>::new();
    let mut dst = Vec::<[f64; 2]>::new();
    for (id, (_conf, center_w)) in by_id {
        let Some(xy) = board.xy_mm(id) else {
            continue;
        };
        src.push([xy[0] as f64, xy[1] as f64]);
        dst.push(center_w);
    }

    if src.len() < 4 {
        return None;
    }

    let ransac_cfg = homography::RansacHomographyConfig {
        max_iters: 1000,
        inlier_threshold: 5.0,
        min_inliers: 8,
        seed: 0,
    };

    let Ok(res) = homography::fit_homography_ransac(&src, &dst, &ransac_cfg) else {
        return None;
    };

    if res.n_inliers == 0 {
        return None;
    }

    let mut sum = 0.0;
    let mut n = 0usize;
    for (i, e) in res.errors.iter().enumerate() {
        if res.inlier_mask.get(i).copied().unwrap_or(false) && e.is_finite() {
            sum += *e;
            n += 1;
        }
    }

    if n == 0 {
        None
    } else {
        Some(HomographySelfError {
            mean_error_px: sum / n as f64,
            n_inliers: n,
        })
    }
}
