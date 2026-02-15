use image::GrayImage;

use crate::board_layout::BoardLayout;
use crate::homography::homography_project as project;
use crate::pixelmap::PixelMapper;
use crate::DetectedMarker;

use super::marker_build::{
    decode_metrics_from_result, fit_metrics_with_inner, marker_with_defaults,
};

pub(crate) fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &super::DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    refine_impl(gray, markers, h, config, board, mapper)
}

fn refine_impl(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &super::DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    let mut refined = Vec::with_capacity(markers.len());

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board.xy_mm(id) {
            Some(xy) => xy,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let prior = project(h, xy[0] as f64, xy[1] as f64);
        if prior[0].is_nan() || prior[1].is_nan() {
            refined.push(m.clone());
            continue;
        }

        let r_expected = super::mean_axis_px_from_marker(m)
            .unwrap_or(super::marker_outer_radius_expected_px(config));

        let fit_cand = match super::fit_outer_candidate_from_prior(
            gray,
            [prior[0] as f32, prior[1] as f32],
            r_expected,
            config,
            mapper,
        ) {
            Ok(v) => v,
            Err(_) => {
                refined.push(m.clone());
                continue;
            }
        };
        let edge = fit_cand.edge;
        let outer = fit_cand.outer;
        let outer_ransac = fit_cand.outer_ransac;
        let decode_result = fit_cand.decode_result.filter(|d| d.id == id);

        let mean_axis_new = ((outer.a + outer.b) * 0.5) as f32;
        let scale_ok = mean_axis_new.is_finite()
            && mean_axis_new >= (r_expected * 0.75)
            && mean_axis_new <= (r_expected * 1.33);
        if decode_result.is_none() || !scale_ok {
            refined.push(m.clone());
            continue;
        }

        let center = super::compute_center(&outer);

        let inner_fit = super::inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            mapper,
            &config.inner_fit,
            false,
        );
        let inner_params = inner_fit.ellipse_inner;
        let fit = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner_fit);

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_metrics_from_result(decode_result.as_ref());

        let updated = marker_with_defaults(
            Some(id),
            confidence,
            center,
            Some(outer),
            inner_params,
            Some(edge.outer_points.clone()),
            Some(inner_fit.points_inner.clone()),
            fit,
            decode_metrics,
        );

        refined.push(updated);
    }

    refined
}
