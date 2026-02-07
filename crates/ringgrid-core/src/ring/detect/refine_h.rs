use image::GrayImage;

use crate::board_spec;
use crate::debug_dump as dbg;
use crate::homography::project;
use crate::DetectedMarker;

use super::marker_build::{
    decode_metrics_from_result, fit_metrics_from_outer, marker_with_defaults,
};

pub(super) fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &super::DetectConfig,
) -> (Vec<DetectedMarker>, dbg::RefineDebugV1) {
    let mut refined = Vec::with_capacity(markers.len());
    let mut refined_dbg = Vec::with_capacity(markers.len());
    let inner_fit_cfg = super::inner_fit::InnerFitConfig::default();

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board_spec::xy_mm(id) {
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

        let fit_cand = match super::fit_outer_ellipse_robust_with_reason(
            gray,
            [prior[0] as f32, prior[1] as f32],
            r_expected,
            config,
            &config.edge_sample,
            false,
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
            // Refinement is best-effort and must not degrade decoded detections.
            refined.push(m.clone());
            refined_dbg.push(dbg::RefinedMarkerDebugV1 {
                id,
                prior_center_xy: [prior[0] as f32, prior[1] as f32],
                refined_center_xy: [m.center[0] as f32, m.center[1] as f32],
                ellipse_outer: m.ellipse_outer.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                    center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                    semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                    angle: p.angle as f32,
                }),
                ellipse_inner: m.ellipse_inner.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                    center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                    semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                    angle: p.angle as f32,
                }),
                fit: m.fit.clone(),
            });
            continue;
        }

        let center = super::compute_center(&outer);

        let inner_fit = super::inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            config.camera.as_ref(),
            &inner_fit_cfg,
            false,
        );
        let inner_params = inner_fit
            .ellipse_inner
            .as_ref()
            .map(super::ellipse_to_params);
        let fit = fit_metrics_from_outer(
            &edge,
            &outer,
            outer_ransac.as_ref(),
            inner_fit.points_inner.len(),
            inner_fit.ransac_inlier_ratio_inner,
            inner_fit.rms_residual_inner,
        );

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_metrics_from_result(decode_result.as_ref());

        let updated = marker_with_defaults(
            Some(id),
            confidence,
            center,
            Some(super::ellipse_to_params(&outer)),
            inner_params.clone(),
            fit.clone(),
            decode_metrics,
        );

        refined_dbg.push(dbg::RefinedMarkerDebugV1 {
            id,
            prior_center_xy: [prior[0] as f32, prior[1] as f32],
            refined_center_xy: [center[0] as f32, center[1] as f32],
            ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                center_xy: [outer.cx as f32, outer.cy as f32],
                semi_axes: [outer.a as f32, outer.b as f32],
                angle: outer.angle as f32,
            }),
            ellipse_inner: inner_params.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                angle: p.angle as f32,
            }),
            fit,
        });

        refined.push(updated);
    }

    (
        refined,
        dbg::RefineDebugV1 {
            h_prior: super::matrix3_to_array(h),
            refined_markers: refined_dbg,
            h_refit: None,
            notes: Vec::new(),
        },
    )
}
