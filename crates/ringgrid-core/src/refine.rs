//! Non-linear refinement utilities.
//!
//! Milestone 6: refine per-marker centers in board coordinates (mm) using
//! measured outer-edge points and a known physical radius, then project the
//! refined center back to image space via the fitted homography.

use image::GrayImage;
use nalgebra as na;

use crate::board_spec;
use crate::homography::project;
use crate::ring::inner_estimate::Polarity;
use crate::{DetectedMarker, EllipseParams};
#[path = "refine/math.rs"]
mod math;
#[path = "refine/sampling.rs"]
mod sampling;
#[path = "refine/solver.rs"]
mod solver;
use sampling::SampleOutcome;

#[derive(Debug, Clone)]
pub struct RefineParams {
    pub enabled: bool,
    pub max_iters: usize,
    pub huber_delta_mm: f64,
    pub min_points: usize,
    pub reject_thresh_mm: f64,
    pub enable_h_refit: bool,
    pub h_refit_iters: usize,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            enabled: true,
            max_iters: 20,
            huber_delta_mm: 0.20,
            min_points: 20,
            reject_thresh_mm: 1.0,
            enable_h_refit: false,
            h_refit_iters: 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RefineStats {
    pub n_inliers: usize,
    pub n_refined: usize,
    pub n_failed: usize,
    pub mean_before_mm: f64,
    pub mean_after_mm: f64,
    pub p95_before_mm: f64,
    pub p95_after_mm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerRefineStatus {
    Ok,
    Rejected,
    Failed,
    Skipped,
}

#[derive(Debug, Clone)]
pub struct MarkerRefineRecord {
    pub id: usize,
    pub n_points: usize,
    pub init_center_board_mm: [f64; 2],
    pub refined_center_board_mm: Option<[f64; 2]>,
    pub center_img_before: [f64; 2],
    pub center_img_after: Option<[f64; 2]>,
    pub before_rms_mm: Option<f64>,
    pub after_rms_mm: Option<f64>,
    pub delta_center_mm: Option<f64>,
    pub edge_points_img: Option<Vec<[f32; 2]>>,
    pub edge_points_board_mm: Option<Vec<[f32; 2]>>,
    pub status: MarkerRefineStatus,
    pub reason: Option<String>,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    math::percentile(sorted, q)
}

fn rms_circle_residual_mm(points: &[[f64; 2]], center: [f64; 2], radius_mm: f64) -> f64 {
    math::rms_circle_residual_mm(points, center, radius_mm)
}

fn try_unproject(h_inv: &na::Matrix3<f64>, x: f64, y: f64) -> Option<[f64; 2]> {
    math::try_unproject(h_inv, x, y)
}

fn sample_outer_points_around_ellipse(
    gray: &GrayImage,
    ellipse: &EllipseParams,
    theta_samples: usize,
    search_halfwidth_px: f32,
    r_step_px: f32,
    min_ring_depth: f32,
    polarity: Polarity,
) -> SampleOutcome {
    sampling::sample_outer_points_around_ellipse(
        gray,
        ellipse,
        theta_samples,
        search_halfwidth_px,
        r_step_px,
        min_ring_depth,
        polarity,
    )
}

fn solve_circle_center_mm(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
) -> Option<[f64; 2]> {
    solver::solve_circle_center_mm(points, init_center_mm, radius_mm, max_iters, huber_delta_mm)
}

pub fn refine_markers_circle_board(
    gray: &GrayImage,
    h: &na::Matrix3<f64>,
    detections: &mut [DetectedMarker],
    params: &RefineParams,
    store_points: bool,
) -> (RefineStats, Vec<MarkerRefineRecord>) {
    let mut records: Vec<MarkerRefineRecord> = Vec::new();

    if !params.enabled {
        return (RefineStats::default(), records);
    }

    let radius_mm = board_spec::marker_outer_radius_mm() as f64;
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return (
            RefineStats::default(),
            detections
                .iter()
                .filter_map(|m| m.id)
                .map(|id| MarkerRefineRecord {
                    id,
                    n_points: 0,
                    init_center_board_mm: [0.0, 0.0],
                    refined_center_board_mm: None,
                    center_img_before: [0.0, 0.0],
                    center_img_after: None,
                    before_rms_mm: None,
                    after_rms_mm: None,
                    delta_center_mm: None,
                    edge_points_img: None,
                    edge_points_board_mm: None,
                    status: MarkerRefineStatus::Failed,
                    reason: Some("invalid_marker_outer_radius_mm".to_string()),
                })
                .collect(),
        );
    }

    let h_inv = match h.try_inverse() {
        Some(v) => v,
        None => {
            for m in detections.iter() {
                if let Some(id) = m.id {
                    let init_center_board_mm =
                        board_spec::xy_mm(id).map(|v| [v[0] as f64, v[1] as f64]);
                    records.push(MarkerRefineRecord {
                        id,
                        n_points: 0,
                        init_center_board_mm: init_center_board_mm.unwrap_or([0.0, 0.0]),
                        refined_center_board_mm: None,
                        center_img_before: m.center,
                        center_img_after: None,
                        before_rms_mm: None,
                        after_rms_mm: None,
                        delta_center_mm: None,
                        edge_points_img: None,
                        edge_points_board_mm: None,
                        status: MarkerRefineStatus::Failed,
                        reason: Some("homography_not_invertible".to_string()),
                    });
                }
            }
            return (RefineStats::default(), records);
        }
    };

    let theta_samples = 96usize;
    let search_halfwidth_px = 2.5f32;
    let r_step_px = 0.5f32;
    let min_ring_depth = 0.06f32;

    let mut rms_before_list: Vec<f64> = Vec::new();
    let mut rms_after_list: Vec<f64> = Vec::new();

    for m in detections.iter_mut() {
        let Some(id) = m.id else {
            continue;
        };

        let init_center_board_mm = match board_spec::xy_mm(id) {
            Some(v) => [v[0] as f64, v[1] as f64],
            None => {
                records.push(MarkerRefineRecord {
                    id,
                    n_points: 0,
                    init_center_board_mm: [0.0, 0.0],
                    refined_center_board_mm: None,
                    center_img_before: m.center,
                    center_img_after: None,
                    before_rms_mm: None,
                    after_rms_mm: None,
                    delta_center_mm: None,
                    edge_points_img: None,
                    edge_points_board_mm: None,
                    status: MarkerRefineStatus::Skipped,
                    reason: Some("id_not_in_board_spec".to_string()),
                });
                continue;
            }
        };

        let ellipse_outer = match &m.ellipse_outer {
            Some(e) => e.clone(),
            None => {
                records.push(MarkerRefineRecord {
                    id,
                    n_points: 0,
                    init_center_board_mm,
                    refined_center_board_mm: None,
                    center_img_before: m.center,
                    center_img_after: None,
                    before_rms_mm: None,
                    after_rms_mm: None,
                    delta_center_mm: None,
                    edge_points_img: None,
                    edge_points_board_mm: None,
                    status: MarkerRefineStatus::Skipped,
                    reason: Some("missing_outer_ellipse".to_string()),
                });
                continue;
            }
        };

        // Sample points for both polarities and choose the better-supported one.
        let s_pos = sample_outer_points_around_ellipse(
            gray,
            &ellipse_outer,
            theta_samples,
            search_halfwidth_px,
            r_step_px,
            min_ring_depth,
            Polarity::Pos,
        );
        let s_neg = sample_outer_points_around_ellipse(
            gray,
            &ellipse_outer,
            theta_samples,
            search_halfwidth_px,
            r_step_px,
            min_ring_depth,
            Polarity::Neg,
        );

        let chosen = if s_pos.points.len() > s_neg.points.len() {
            (Polarity::Pos, s_pos)
        } else if s_neg.points.len() > s_pos.points.len() {
            (Polarity::Neg, s_neg)
        } else if s_pos.score_sum >= s_neg.score_sum {
            (Polarity::Pos, s_pos)
        } else {
            (Polarity::Neg, s_neg)
        };

        let sampled_points = chosen.1.points;

        if sampled_points.len() < params.min_points {
            records.push(MarkerRefineRecord {
                id,
                n_points: sampled_points.len(),
                init_center_board_mm,
                refined_center_board_mm: None,
                center_img_before: m.center,
                center_img_after: None,
                before_rms_mm: None,
                after_rms_mm: None,
                delta_center_mm: None,
                edge_points_img: if store_points {
                    Some(
                        sampled_points
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                edge_points_board_mm: None,
                status: MarkerRefineStatus::Failed,
                reason: Some(format!(
                    "insufficient_edge_points({}<{})",
                    sampled_points.len(),
                    params.min_points
                )),
            });
            continue;
        }

        let mut points_board: Vec<[f64; 2]> = Vec::with_capacity(sampled_points.len());
        for p in &sampled_points {
            if let Some(q) = try_unproject(&h_inv, p[0], p[1]) {
                points_board.push(q);
            }
        }

        if points_board.len() < params.min_points {
            records.push(MarkerRefineRecord {
                id,
                n_points: points_board.len(),
                init_center_board_mm,
                refined_center_board_mm: None,
                center_img_before: m.center,
                center_img_after: None,
                before_rms_mm: None,
                after_rms_mm: None,
                delta_center_mm: None,
                edge_points_img: if store_points {
                    Some(
                        sampled_points
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                edge_points_board_mm: if store_points {
                    Some(
                        points_board
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                status: MarkerRefineStatus::Failed,
                reason: Some(format!(
                    "insufficient_unprojected_points({}<{})",
                    points_board.len(),
                    params.min_points
                )),
            });
            continue;
        }

        let before_rms = rms_circle_residual_mm(&points_board, init_center_board_mm, radius_mm);
        if !before_rms.is_finite() {
            records.push(MarkerRefineRecord {
                id,
                n_points: points_board.len(),
                init_center_board_mm,
                refined_center_board_mm: None,
                center_img_before: m.center,
                center_img_after: None,
                before_rms_mm: None,
                after_rms_mm: None,
                delta_center_mm: None,
                edge_points_img: if store_points {
                    Some(
                        sampled_points
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                edge_points_board_mm: if store_points {
                    Some(
                        points_board
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                status: MarkerRefineStatus::Failed,
                reason: Some("before_rms_nan".to_string()),
            });
            continue;
        }

        let refined_center = match solve_circle_center_mm(
            &points_board,
            init_center_board_mm,
            radius_mm,
            params.max_iters,
            params.huber_delta_mm.max(1e-6),
        ) {
            Some(c) => c,
            None => {
                records.push(MarkerRefineRecord {
                    id,
                    n_points: points_board.len(),
                    init_center_board_mm,
                    refined_center_board_mm: None,
                    center_img_before: m.center,
                    center_img_after: None,
                    before_rms_mm: Some(before_rms),
                    after_rms_mm: None,
                    delta_center_mm: None,
                    edge_points_img: if store_points {
                        Some(
                            sampled_points
                                .iter()
                                .take(256)
                                .map(|p| [p[0] as f32, p[1] as f32])
                                .collect(),
                        )
                    } else {
                        None
                    },
                    edge_points_board_mm: if store_points {
                        Some(
                            points_board
                                .iter()
                                .take(256)
                                .map(|p| [p[0] as f32, p[1] as f32])
                                .collect(),
                        )
                    } else {
                        None
                    },
                    status: MarkerRefineStatus::Failed,
                    reason: Some("solver_failed".to_string()),
                });
                continue;
            }
        };

        let after_rms = rms_circle_residual_mm(&points_board, refined_center, radius_mm);
        let dx = refined_center[0] - init_center_board_mm[0];
        let dy = refined_center[1] - init_center_board_mm[1];
        let delta_center_mm = (dx * dx + dy * dy).sqrt();

        if !after_rms.is_finite() {
            records.push(MarkerRefineRecord {
                id,
                n_points: points_board.len(),
                init_center_board_mm,
                refined_center_board_mm: None,
                center_img_before: m.center,
                center_img_after: None,
                before_rms_mm: Some(before_rms),
                after_rms_mm: None,
                delta_center_mm: None,
                edge_points_img: if store_points {
                    Some(
                        sampled_points
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                edge_points_board_mm: if store_points {
                    Some(
                        points_board
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                status: MarkerRefineStatus::Failed,
                reason: Some("after_rms_nan".to_string()),
            });
            continue;
        }

        if delta_center_mm > params.reject_thresh_mm.max(0.0) {
            records.push(MarkerRefineRecord {
                id,
                n_points: points_board.len(),
                init_center_board_mm,
                refined_center_board_mm: Some(refined_center),
                center_img_before: m.center,
                center_img_after: None,
                before_rms_mm: Some(before_rms),
                after_rms_mm: Some(after_rms),
                delta_center_mm: Some(delta_center_mm),
                edge_points_img: if store_points {
                    Some(
                        sampled_points
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                edge_points_board_mm: if store_points {
                    Some(
                        points_board
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                status: MarkerRefineStatus::Rejected,
                reason: Some(format!(
                    "delta_center_mm({:.3}>{:.3})",
                    delta_center_mm, params.reject_thresh_mm
                )),
            });
            continue;
        }

        if after_rms > before_rms {
            records.push(MarkerRefineRecord {
                id,
                n_points: points_board.len(),
                init_center_board_mm,
                refined_center_board_mm: Some(refined_center),
                center_img_before: m.center,
                center_img_after: None,
                before_rms_mm: Some(before_rms),
                after_rms_mm: Some(after_rms),
                delta_center_mm: Some(delta_center_mm),
                edge_points_img: if store_points {
                    Some(
                        sampled_points
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                edge_points_board_mm: if store_points {
                    Some(
                        points_board
                            .iter()
                            .take(256)
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                status: MarkerRefineStatus::Rejected,
                reason: Some("rms_not_improved".to_string()),
            });
            continue;
        }

        let center_img_before = m.center;
        let refined_img = project(h, refined_center[0], refined_center[1]);
        m.center = refined_img;

        records.push(MarkerRefineRecord {
            id,
            n_points: points_board.len(),
            init_center_board_mm,
            refined_center_board_mm: Some(refined_center),
            center_img_before,
            center_img_after: Some(refined_img),
            before_rms_mm: Some(before_rms),
            after_rms_mm: Some(after_rms),
            delta_center_mm: Some(delta_center_mm),
            edge_points_img: if store_points {
                Some(
                    sampled_points
                        .iter()
                        .take(256)
                        .map(|p| [p[0] as f32, p[1] as f32])
                        .collect(),
                )
            } else {
                None
            },
            edge_points_board_mm: if store_points {
                Some(
                    points_board
                        .iter()
                        .take(256)
                        .map(|p| [p[0] as f32, p[1] as f32])
                        .collect(),
                )
            } else {
                None
            },
            status: MarkerRefineStatus::Ok,
            reason: None,
        });

        rms_before_list.push(before_rms);
        rms_after_list.push(after_rms);
    }

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

    let stats = RefineStats {
        n_inliers: detections.len(),
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
    };

    (stats, records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_points(
        center: [f64; 2],
        radius: f64,
        n: usize,
        theta0: f64,
        theta1: f64,
    ) -> Vec<[f64; 2]> {
        let n = n.max(2);
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let t = theta0 + (theta1 - theta0) * (i as f64) / ((n - 1) as f64);
            pts.push([center[0] + radius * t.cos(), center[1] + radius * t.sin()]);
        }
        pts
    }

    #[test]
    fn solve_circle_center_mm_recovers_center_full_circle() {
        let true_center = [10.0, -5.0];
        let radius = 4.8;
        let pts = circle_points(true_center, radius, 96, 0.0, 2.0 * std::f64::consts::PI);
        let init = [true_center[0] + 0.8, true_center[1] - 0.6];

        let est = solve_circle_center_mm(&pts, init, radius, 50, 0.2).expect("solver result");
        assert!((est[0] - true_center[0]).abs() < 1e-3);
        assert!((est[1] - true_center[1]).abs() < 1e-3);
    }

    #[test]
    fn solve_circle_center_mm_handles_partial_arc() {
        let true_center = [-12.0, 7.5];
        let radius = 4.8;
        // ~80° arc
        let pts = circle_points(true_center, radius, 64, 0.4, 1.8);
        let init = [true_center[0] + 0.5, true_center[1] + 0.4];

        let est = solve_circle_center_mm(&pts, init, radius, 80, 0.2).expect("solver result");
        assert!((est[0] - true_center[0]).abs() < 5e-2);
        assert!((est[1] - true_center[1]).abs() < 5e-2);
    }

    #[test]
    fn solve_circle_center_mm_is_robust_to_outliers() {
        use rand::prelude::*;

        let true_center = [3.0, 2.0];
        let radius = 4.8;
        let mut pts = circle_points(true_center, radius, 80, 0.0, 2.0 * std::f64::consts::PI);

        // Add small isotropic noise.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for p in &mut pts {
            let nx: f64 = rng.gen_range(-0.02..0.02);
            let ny: f64 = rng.gen_range(-0.02..0.02);
            p[0] += nx;
            p[1] += ny;
        }

        // Inject a few large outliers.
        for _ in 0..12 {
            pts.push([
                true_center[0] + rng.gen_range(-30.0..30.0),
                true_center[1] + rng.gen_range(-30.0..30.0),
            ]);
        }

        let init = [true_center[0] - 0.9, true_center[1] + 0.7];
        let est = solve_circle_center_mm(&pts, init, radius, 80, 0.10).expect("solver result");
        assert!((est[0] - true_center[0]).abs() < 5e-2);
        assert!((est[1] - true_center[1]).abs() < 5e-2);
    }
}
