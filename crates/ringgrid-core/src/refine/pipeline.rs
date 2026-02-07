use image::GrayImage;
use nalgebra as na;

use crate::board_spec;
use crate::homography::project;

use super::*;

#[derive(Debug, Clone, Copy)]
struct SamplingConfig {
    theta_samples: usize,
    search_halfwidth_px: f32,
    r_step_px: f32,
    min_ring_depth: f32,
}

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
    sample_cfg: SamplingConfig,
    store_points: bool,
}

fn to_debug_points(points: &[[f64; 2]], store_points: bool) -> Option<Vec<[f32; 2]>> {
    if !store_points {
        return None;
    }
    Some(
        points
            .iter()
            .take(256)
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn make_record(
    id: usize,
    n_points: usize,
    init_center_board_mm: [f64; 2],
    center_img_before: [f64; 2],
    refined_center_board_mm: Option<[f64; 2]>,
    center_img_after: Option<[f64; 2]>,
    before_rms_mm: Option<f64>,
    after_rms_mm: Option<f64>,
    delta_center_mm: Option<f64>,
    sampled_points: Option<&[[f64; 2]]>,
    points_board: Option<&[[f64; 2]]>,
    store_points: bool,
    status: MarkerRefineStatus,
    reason: Option<String>,
) -> MarkerRefineRecord {
    MarkerRefineRecord {
        id,
        n_points,
        init_center_board_mm,
        refined_center_board_mm,
        center_img_before,
        center_img_after,
        before_rms_mm,
        after_rms_mm,
        delta_center_mm,
        edge_points_img: sampled_points.and_then(|p| to_debug_points(p, store_points)),
        edge_points_board_mm: points_board.and_then(|p| to_debug_points(p, store_points)),
        status,
        reason,
    }
}

fn invalid_radius_records(detections: &[DetectedMarker]) -> Vec<MarkerRefineRecord> {
    detections
        .iter()
        .filter_map(|m| m.id)
        .map(|id| {
            make_record(
                id,
                0,
                [0.0, 0.0],
                [0.0, 0.0],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
                MarkerRefineStatus::Failed,
                Some("invalid_marker_outer_radius_mm".to_string()),
            )
        })
        .collect()
}

fn homography_not_invertible_records(detections: &[DetectedMarker]) -> Vec<MarkerRefineRecord> {
    let mut records = Vec::new();
    for m in detections {
        if let Some(id) = m.id {
            let init_center_board_mm = board_spec::xy_mm(id).map(|v| [v[0] as f64, v[1] as f64]);
            records.push(make_record(
                id,
                0,
                init_center_board_mm.unwrap_or([0.0, 0.0]),
                m.center,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
                MarkerRefineStatus::Failed,
                Some("homography_not_invertible".to_string()),
            ));
        }
    }
    records
}

fn process_marker(ctx: &RunContext<'_>, m: &mut DetectedMarker) -> Option<MarkerProcessResult> {
    let id = m.id?;
    let center_img_before = m.center;

    let init_center_board_mm = match board_spec::xy_mm(id) {
        Some(v) => [v[0] as f64, v[1] as f64],
        None => {
            return Some(MarkerProcessResult {
                record: make_record(
                    id,
                    0,
                    [0.0, 0.0],
                    center_img_before,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                    MarkerRefineStatus::Skipped,
                    Some("id_not_in_board_spec".to_string()),
                ),
                accepted_rms: None,
            });
        }
    };

    let ellipse_outer = match &m.ellipse_outer {
        Some(e) => e.clone(),
        None => {
            return Some(MarkerProcessResult {
                record: make_record(
                    id,
                    0,
                    init_center_board_mm,
                    center_img_before,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                    MarkerRefineStatus::Skipped,
                    Some("missing_outer_ellipse".to_string()),
                ),
                accepted_rms: None,
            });
        }
    };

    let s_pos = sample_outer_points_around_ellipse(
        ctx.gray,
        &ellipse_outer,
        ctx.sample_cfg.theta_samples,
        ctx.sample_cfg.search_halfwidth_px,
        ctx.sample_cfg.r_step_px,
        ctx.sample_cfg.min_ring_depth,
        Polarity::Pos,
    );
    let s_neg = sample_outer_points_around_ellipse(
        ctx.gray,
        &ellipse_outer,
        ctx.sample_cfg.theta_samples,
        ctx.sample_cfg.search_halfwidth_px,
        ctx.sample_cfg.r_step_px,
        ctx.sample_cfg.min_ring_depth,
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
        return Some(MarkerProcessResult {
            record: make_record(
                id,
                sampled_points.len(),
                init_center_board_mm,
                center_img_before,
                None,
                None,
                None,
                None,
                None,
                Some(&sampled_points),
                None,
                ctx.store_points,
                MarkerRefineStatus::Failed,
                Some(format!(
                    "insufficient_edge_points({}<{})",
                    sampled_points.len(),
                    ctx.params.min_points
                )),
            ),
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
        return Some(MarkerProcessResult {
            record: make_record(
                id,
                points_board.len(),
                init_center_board_mm,
                center_img_before,
                None,
                None,
                None,
                None,
                None,
                Some(&sampled_points),
                Some(&points_board),
                ctx.store_points,
                MarkerRefineStatus::Failed,
                Some(format!(
                    "insufficient_unprojected_points({}<{})",
                    points_board.len(),
                    ctx.params.min_points
                )),
            ),
            accepted_rms: None,
        });
    }

    let before_rms = rms_circle_residual_mm(&points_board, init_center_board_mm, ctx.radius_mm);
    if !before_rms.is_finite() {
        return Some(MarkerProcessResult {
            record: make_record(
                id,
                points_board.len(),
                init_center_board_mm,
                center_img_before,
                None,
                None,
                None,
                None,
                None,
                Some(&sampled_points),
                Some(&points_board),
                ctx.store_points,
                MarkerRefineStatus::Failed,
                Some("before_rms_nan".to_string()),
            ),
            accepted_rms: None,
        });
    }

    let refined_center = match solve_circle_center_mm(
        &points_board,
        init_center_board_mm,
        ctx.radius_mm,
        ctx.params.max_iters,
        ctx.params.huber_delta_mm.max(1e-6),
    ) {
        Some(c) => c,
        None => {
            return Some(MarkerProcessResult {
                record: make_record(
                    id,
                    points_board.len(),
                    init_center_board_mm,
                    center_img_before,
                    None,
                    None,
                    Some(before_rms),
                    None,
                    None,
                    Some(&sampled_points),
                    Some(&points_board),
                    ctx.store_points,
                    MarkerRefineStatus::Failed,
                    Some("solver_failed".to_string()),
                ),
                accepted_rms: None,
            });
        }
    };

    let after_rms = rms_circle_residual_mm(&points_board, refined_center, ctx.radius_mm);
    let dx = refined_center[0] - init_center_board_mm[0];
    let dy = refined_center[1] - init_center_board_mm[1];
    let delta_center_mm = (dx * dx + dy * dy).sqrt();

    if !after_rms.is_finite() {
        return Some(MarkerProcessResult {
            record: make_record(
                id,
                points_board.len(),
                init_center_board_mm,
                center_img_before,
                None,
                None,
                Some(before_rms),
                None,
                None,
                Some(&sampled_points),
                Some(&points_board),
                ctx.store_points,
                MarkerRefineStatus::Failed,
                Some("after_rms_nan".to_string()),
            ),
            accepted_rms: None,
        });
    }

    if delta_center_mm > ctx.params.reject_thresh_mm.max(0.0) {
        return Some(MarkerProcessResult {
            record: make_record(
                id,
                points_board.len(),
                init_center_board_mm,
                center_img_before,
                Some(refined_center),
                None,
                Some(before_rms),
                Some(after_rms),
                Some(delta_center_mm),
                Some(&sampled_points),
                Some(&points_board),
                ctx.store_points,
                MarkerRefineStatus::Rejected,
                Some(format!(
                    "delta_center_mm({:.3}>{:.3})",
                    delta_center_mm, ctx.params.reject_thresh_mm
                )),
            ),
            accepted_rms: None,
        });
    }

    if after_rms > before_rms {
        return Some(MarkerProcessResult {
            record: make_record(
                id,
                points_board.len(),
                init_center_board_mm,
                center_img_before,
                Some(refined_center),
                None,
                Some(before_rms),
                Some(after_rms),
                Some(delta_center_mm),
                Some(&sampled_points),
                Some(&points_board),
                ctx.store_points,
                MarkerRefineStatus::Rejected,
                Some("rms_not_improved".to_string()),
            ),
            accepted_rms: None,
        });
    }

    let refined_img = project(ctx.h, refined_center[0], refined_center[1]);
    m.center = refined_img;

    Some(MarkerProcessResult {
        record: make_record(
            id,
            points_board.len(),
            init_center_board_mm,
            center_img_before,
            Some(refined_center),
            Some(refined_img),
            Some(before_rms),
            Some(after_rms),
            Some(delta_center_mm),
            Some(&sampled_points),
            Some(&points_board),
            ctx.store_points,
            MarkerRefineStatus::Ok,
            None,
        ),
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
    store_points: bool,
) -> (RefineStats, Vec<MarkerRefineRecord>) {
    if !params.enabled {
        return (RefineStats::default(), Vec::new());
    }

    let radius_mm = board_spec::marker_outer_radius_mm() as f64;
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return (RefineStats::default(), invalid_radius_records(detections));
    }

    let h_inv = match h.try_inverse() {
        Some(v) => v,
        None => {
            return (
                RefineStats::default(),
                homography_not_invertible_records(detections),
            )
        }
    };

    let sample_cfg = SamplingConfig {
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
