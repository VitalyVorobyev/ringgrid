use image::GrayImage;

use crate::board_spec;
use crate::conic::rms_sampson_distance;
use crate::debug_dump as dbg;
use crate::homography::project;
use crate::ring::inner_estimate::{estimate_inner_scale_from_outer, InnerStatus};
use crate::{DecodeMetrics, DetectedMarker, EllipseParams, FitMetrics};

pub(super) fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &super::DetectConfig,
) -> (Vec<DetectedMarker>, dbg::RefineDebugV1) {
    let mut refined = Vec::with_capacity(markers.len());
    let mut refined_dbg = Vec::with_capacity(markers.len());

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

        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: 0,
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: None,
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: None,
        };

        let inner_est = estimate_inner_scale_from_outer(gray, &outer, &config.marker_spec, false);
        let inner_params = if inner_est.status == InnerStatus::Ok {
            let s = inner_est
                .r_inner_found
                .unwrap_or(config.marker_spec.r_inner_expected) as f64;
            Some(EllipseParams {
                center_xy: [outer.cx, outer.cy],
                semi_axes: [outer.a * s, outer.b * s],
                angle: outer.angle,
            })
        } else {
            None
        };

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let updated = DetectedMarker {
            id: Some(id),
            confidence,
            center,
            center_projective: None,
            vanishing_line: None,
            center_projective_residual: None,
            ellipse_outer: Some(super::ellipse_to_params(&outer)),
            ellipse_inner: inner_params.clone(),
            fit: fit.clone(),
            decode: decode_metrics,
        };

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
