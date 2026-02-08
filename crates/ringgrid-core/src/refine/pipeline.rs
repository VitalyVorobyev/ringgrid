use image::GrayImage;
use nalgebra as na;

use crate::board_spec;
use crate::camera::PixelMapper;
use crate::homography::project;

use super::*;

#[derive(Debug, Clone)]
struct MarkerProcessResult {
    record: MarkerRefineRecord,
    accepted_rms: Option<(f64, f64)>,
}

struct RunContext<'a> {
    gray: &'a GrayImage,
    h: &'a na::Matrix3<f64>,
    h_inv: &'a na::Matrix3<f64>,
    params: &'a RefineParams,
    radius_mm: f64,
    sample_cfg: OuterSampleConfig,
    mapper: Option<&'a dyn PixelMapper>,
    store_points: bool,
}

fn process_marker(ctx: &RunContext<'_>, m: &mut DetectedMarker) -> Option<MarkerProcessResult> {
    let id = m.id?;
    let center_img_before = m.center;

    let init_center_board_mm = match board_spec::xy_mm(id) {
        Some(v) => [v[0] as f64, v[1] as f64],
        None => {
            let rec = MarkerRefineRecord::new(
                id,
                [0.0, 0.0],
                center_img_before,
                MarkerRefineStatus::Skipped,
            )
            .with_reason("id_not_in_board_spec");
            return Some(MarkerProcessResult {
                record: rec,
                accepted_rms: None,
            });
        }
    };

    let ellipse_outer = match &m.ellipse_outer {
        Some(e) => e.clone(),
        None => {
            let rec = MarkerRefineRecord::new(
                id,
                init_center_board_mm,
                center_img_before,
                MarkerRefineStatus::Skipped,
            )
            .with_reason("missing_outer_ellipse");
            return Some(MarkerProcessResult {
                record: rec,
                accepted_rms: None,
            });
        }
    };

    let s_pos = sample_outer_points_around_ellipse(
        ctx.gray,
        &ellipse_outer,
        ctx.mapper,
        ctx.sample_cfg,
        Polarity::Pos,
    );
    let s_neg = sample_outer_points_around_ellipse(
        ctx.gray,
        &ellipse_outer,
        ctx.mapper,
        ctx.sample_cfg,
        Polarity::Neg,
    );

    let sampled_points = if s_pos.points.len() > s_neg.points.len() {
        s_pos.points
    } else if s_neg.points.len() > s_pos.points.len() {
        s_neg.points
    } else if s_pos.score_sum >= s_neg.score_sum {
        s_pos.points
    } else {
        s_neg.points
    };

    if sampled_points.len() < ctx.params.min_points {
        let rec = MarkerRefineRecord::new(
            id,
            init_center_board_mm,
            center_img_before,
            MarkerRefineStatus::Failed,
        )
        .with_reason(format!(
            "insufficient_edge_points({}<{})",
            sampled_points.len(),
            ctx.params.min_points
        ))
        .with_points(sampled_points.len(), Some(&sampled_points), None, ctx.store_points);
        return Some(MarkerProcessResult {
            record: rec,
            accepted_rms: None,
        });
    }

    let mut points_board: Vec<[f64; 2]> = Vec::with_capacity(sampled_points.len());
    for p in &sampled_points {
        if let Some(q) = try_unproject(ctx.h_inv, p[0], p[1]) {
            points_board.push(q);
        }
    }

    if points_board.len() < ctx.params.min_points {
        let rec = MarkerRefineRecord::new(
            id,
            init_center_board_mm,
            center_img_before,
            MarkerRefineStatus::Failed,
        )
        .with_reason(format!(
            "insufficient_unprojected_points({}<{})",
            points_board.len(),
            ctx.params.min_points
        ))
        .with_points(
            points_board.len(),
            Some(&sampled_points),
            Some(&points_board),
            ctx.store_points,
        );
        return Some(MarkerProcessResult {
            record: rec,
            accepted_rms: None,
        });
    }

    let before_rms = rms_circle_residual_mm(&points_board, init_center_board_mm, ctx.radius_mm);
    if !before_rms.is_finite() {
        let rec = MarkerRefineRecord::new(
            id,
            init_center_board_mm,
            center_img_before,
            MarkerRefineStatus::Failed,
        )
        .with_reason("before_rms_nan")
        .with_points(
            points_board.len(),
            Some(&sampled_points),
            Some(&points_board),
            ctx.store_points,
        );
        return Some(MarkerProcessResult {
            record: rec,
            accepted_rms: None,
        });
    }

    let refined_center = match solve_circle_center_mm(
        &points_board,
        init_center_board_mm,
        ctx.radius_mm,
        ctx.params.max_iters,
        ctx.params.huber_delta_mm.max(1e-6),
        ctx.params.solver,
    ) {
        Some(c) => c,
        None => {
            let mut rec = MarkerRefineRecord::new(
                id,
                init_center_board_mm,
                center_img_before,
                MarkerRefineStatus::Failed,
            )
            .with_reason("solver_failed")
            .with_points(
                points_board.len(),
                Some(&sampled_points),
                Some(&points_board),
                ctx.store_points,
            );
            rec.before_rms_mm = Some(before_rms);
            return Some(MarkerProcessResult {
                record: rec,
                accepted_rms: None,
            });
        }
    };

    let after_rms = rms_circle_residual_mm(&points_board, refined_center, ctx.radius_mm);
    let dx = refined_center[0] - init_center_board_mm[0];
    let dy = refined_center[1] - init_center_board_mm[1];
    let delta_center_mm = (dx * dx + dy * dy).sqrt();

    // Build common partial record for post-solve outcomes
    let mut rec = MarkerRefineRecord::new(
        id,
        init_center_board_mm,
        center_img_before,
        MarkerRefineStatus::Ok, // overridden below if rejected
    )
    .with_points(
        points_board.len(),
        Some(&sampled_points),
        Some(&points_board),
        ctx.store_points,
    );
    rec.before_rms_mm = Some(before_rms);
    rec.refined_center_board_mm = Some(refined_center);

    if !after_rms.is_finite() {
        rec.status = MarkerRefineStatus::Failed;
        rec.reason = Some("after_rms_nan".to_string());
        return Some(MarkerProcessResult {
            record: rec,
            accepted_rms: None,
        });
    }

    rec.after_rms_mm = Some(after_rms);
    rec.delta_center_mm = Some(delta_center_mm);

    if delta_center_mm > ctx.params.reject_thresh_mm.max(0.0) {
        rec.status = MarkerRefineStatus::Rejected;
        rec.reason = Some(format!(
            "delta_center_mm({:.3}>{:.3})",
            delta_center_mm, ctx.params.reject_thresh_mm
        ));
        return Some(MarkerProcessResult {
            record: rec,
            accepted_rms: None,
        });
    }

    if after_rms > before_rms {
        rec.status = MarkerRefineStatus::Rejected;
        rec.reason = Some("rms_not_improved".to_string());
        return Some(MarkerProcessResult {
            record: rec,
            accepted_rms: None,
        });
    }

    let refined_img = project(ctx.h, refined_center[0], refined_center[1]);
    m.center = refined_img;
    rec.center_img_after = Some(refined_img);

    Some(MarkerProcessResult {
        record: rec,
        accepted_rms: Some((before_rms, after_rms)),
    })
}

fn build_stats(
    n_inliers: usize,
    records: &[MarkerRefineRecord],
    mut rms_before_list: Vec<f64>,
    mut rms_after_list: Vec<f64>,
) -> RefineStats {
    rms_before_list.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rms_after_list.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_before = if rms_before_list.is_empty() {
        0.0
    } else {
        rms_before_list.iter().sum::<f64>() / rms_before_list.len() as f64
    };
    let mean_after = if rms_after_list.is_empty() {
        0.0
    } else {
        rms_after_list.iter().sum::<f64>() / rms_after_list.len() as f64
    };
    let p95_before = percentile(&rms_before_list, 0.95);
    let p95_after = percentile(&rms_after_list, 0.95);

    RefineStats {
        n_inliers,
        n_refined: records
            .iter()
            .filter(|r| r.status == MarkerRefineStatus::Ok)
            .count(),
        n_failed: records
            .iter()
            .filter(|r| matches!(r.status, MarkerRefineStatus::Failed))
            .count(),
        mean_before_mm: mean_before,
        mean_after_mm: mean_after,
        p95_before_mm: p95_before,
        p95_after_mm: p95_after,
    }
}

pub(super) fn run(
    gray: &GrayImage,
    h: &na::Matrix3<f64>,
    detections: &mut [DetectedMarker],
    params: &RefineParams,
    mapper: Option<&dyn PixelMapper>,
    store_points: bool,
) -> (RefineStats, Vec<MarkerRefineRecord>) {
    if !params.enabled {
        return (RefineStats::default(), Vec::new());
    }

    let radius_mm = board_spec::marker_outer_radius_mm() as f64;
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        let records: Vec<_> = detections
            .iter()
            .filter_map(|m| m.id)
            .map(|id| {
                MarkerRefineRecord::new(id, [0.0, 0.0], [0.0, 0.0], MarkerRefineStatus::Failed)
                    .with_reason("invalid_marker_outer_radius_mm")
            })
            .collect();
        return (RefineStats::default(), records);
    }

    let h_inv = match h.try_inverse() {
        Some(v) => v,
        None => {
            let records: Vec<_> = detections
                .iter()
                .filter_map(|m| {
                    let id = m.id?;
                    let board = board_spec::xy_mm(id).map(|v| [v[0] as f64, v[1] as f64]);
                    Some(
                        MarkerRefineRecord::new(
                            id,
                            board.unwrap_or([0.0, 0.0]),
                            m.center,
                            MarkerRefineStatus::Failed,
                        )
                        .with_reason("homography_not_invertible"),
                    )
                })
                .collect();
            return (RefineStats::default(), records);
        }
    };

    let sample_cfg = OuterSampleConfig {
        theta_samples: 96,
        search_halfwidth_px: 2.5,
        r_step_px: 0.5,
        min_ring_depth: 0.06,
    };

    let mut records: Vec<MarkerRefineRecord> = Vec::new();
    let mut rms_before_list: Vec<f64> = Vec::new();
    let mut rms_after_list: Vec<f64> = Vec::new();
    let ctx = RunContext {
        gray,
        h,
        h_inv: &h_inv,
        params,
        radius_mm,
        sample_cfg,
        mapper,
        store_points,
    };

    for m in detections.iter_mut() {
        let Some(outcome) = process_marker(&ctx, m) else {
            continue;
        };

        if let Some((before, after)) = outcome.accepted_rms {
            rms_before_list.push(before);
            rms_after_list.push(after);
        }
        records.push(outcome.record);
    }

    let stats = build_stats(detections.len(), &records, rms_before_list, rms_after_list);
    (stats, records)
}
