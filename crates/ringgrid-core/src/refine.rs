//! Non-linear refinement utilities.
//!
//! Milestone 6: refine per-marker centers in board coordinates (mm) using
//! measured outer-edge points and a known physical radius, then project the
//! refined center back to image space via the fitted homography.

use std::collections::HashMap;

use image::GrayImage;
use nalgebra as na;

use crate::board_spec;
use crate::homography::project;
use crate::ring::edge_sample::bilinear_sample_u8_checked;
use crate::ring::inner_estimate::Polarity;
use crate::{DetectedMarker, EllipseParams};

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
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn rms_circle_residual_mm(points: &[[f64; 2]], center: [f64; 2], radius_mm: f64) -> f64 {
    if points.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0f64;
    for p in points {
        let dx = p[0] - center[0];
        let dy = p[1] - center[1];
        let r = (dx * dx + dy * dy).sqrt();
        let e = r - radius_mm;
        sum += e * e;
    }
    (sum / points.len() as f64).sqrt()
}

fn try_unproject(h_inv: &na::Matrix3<f64>, x: f64, y: f64) -> Option<[f64; 2]> {
    let v = h_inv * na::Vector3::new(x, y, 1.0);
    let w = v[2];
    if !w.is_finite() || w.abs() < 1e-12 {
        return None;
    }
    let bx = v[0] / w;
    let by = v[1] / w;
    if !bx.is_finite() || !by.is_finite() {
        return None;
    }
    Some([bx, by])
}

fn ellipse_direction_radius_px(e: &EllipseParams, dir: [f32; 2]) -> Option<f32> {
    let a = e.semi_axes[0] as f32;
    let b = e.semi_axes[1] as f32;
    if !a.is_finite() || !b.is_finite() || a <= 1e-3 || b <= 1e-3 {
        return None;
    }
    let phi = e.angle as f32;
    let (c, s) = (phi.cos(), phi.sin());
    // Rotate direction into ellipse frame: d' = R(-phi) * d
    let dx = dir[0];
    let dy = dir[1];
    let dxp = c * dx + s * dy;
    let dyp = -s * dx + c * dy;
    let denom = (dxp * dxp) / (a * a) + (dyp * dyp) / (b * b);
    if !denom.is_finite() || denom <= 1e-9 {
        return None;
    }
    Some(1.0 / denom.sqrt())
}

#[derive(Debug, Clone)]
struct SampleOutcome {
    points: Vec<[f64; 2]>,
    score_sum: f32,
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
    let n_t = theta_samples.max(8);
    let cx = ellipse.center_xy[0] as f32;
    let cy = ellipse.center_xy[1] as f32;

    let hw = search_halfwidth_px.max(0.0);
    let step = r_step_px.clamp(0.25, 1.0);
    let n_ref = ((hw / step).ceil() as i32).max(1);

    let mut points = Vec::with_capacity(n_t);
    let mut score_sum = 0.0f32;

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let dir = [theta.cos(), theta.sin()];

        let r_pred = match ellipse_direction_radius_px(ellipse, dir) {
            Some(r) => r,
            None => continue,
        };

        let mut best_score = f32::NEG_INFINITY;
        let mut best_r = None::<f32>;

        for k in -n_ref..=n_ref {
            let r = r_pred + k as f32 * step;
            if r <= 0.5 {
                continue;
            }

            // dI/dr at r via small central difference
            let h = 0.25f32;
            if r <= h {
                continue;
            }

            let x1 = cx + dir[0] * (r + h);
            let y1 = cy + dir[1] * (r + h);
            let x0 = cx + dir[0] * (r - h);
            let y0 = cy + dir[1] * (r - h);
            let i1 = match bilinear_sample_u8_checked(gray, x1, y1) {
                Some(v) => v,
                None => continue,
            };
            let i0 = match bilinear_sample_u8_checked(gray, x0, y0) {
                Some(v) => v,
                None => continue,
            };
            let d = (i1 - i0) / (2.0 * h);

            let score = match polarity {
                Polarity::Pos => d,
                Polarity::Neg => -d,
            };

            if score > best_score {
                best_score = score;
                best_r = Some(r);
            }
        }

        let r = match best_r {
            Some(r) if best_score.is_finite() && best_score > 0.0 => r,
            _ => continue,
        };

        // Ring depth check across the chosen edge.
        let band = 2.0f32;
        let x_in = cx + dir[0] * (r - band);
        let y_in = cy + dir[1] * (r - band);
        let x_out = cx + dir[0] * (r + band);
        let y_out = cy + dir[1] * (r + band);
        let i_in = match bilinear_sample_u8_checked(gray, x_in, y_in) {
            Some(v) => v,
            None => continue,
        };
        let i_out = match bilinear_sample_u8_checked(gray, x_out, y_out) {
            Some(v) => v,
            None => continue,
        };
        let signed_depth = match polarity {
            Polarity::Pos => i_out - i_in,
            Polarity::Neg => i_in - i_out,
        };
        if signed_depth < min_ring_depth {
            continue;
        }

        let x = cx + dir[0] * r;
        let y = cy + dir[1] * r;
        points.push([x as f64, y as f64]);
        score_sum += best_score.max(0.0);
    }

    SampleOutcome { points, score_sum }
}

fn solve_circle_center_mm(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
) -> Option<[f64; 2]> {
    if points.is_empty() {
        return None;
    }

    use tiny_solver::factors::na as ts_na;
    use tiny_solver::Optimizer;

    #[derive(Debug, Clone)]
    struct CircleFactor {
        x: f64,
        y: f64,
        radius: f64,
    }
    impl<T: ts_na::RealField> tiny_solver::factors::Factor<T> for CircleFactor {
        fn residual_func(&self, params: &[ts_na::DVector<T>]) -> ts_na::DVector<T> {
            let c = &params[0];
            let cx = c[0].clone();
            let cy = c[1].clone();
            let dx = T::from_f64(self.x).unwrap() - cx;
            let dy = T::from_f64(self.y).unwrap() - cy;
            let dist = (dx.clone() * dx + dy.clone() * dy).sqrt();
            let r = dist - T::from_f64(self.radius).unwrap();
            ts_na::DVector::<T>::from_vec(vec![r])
        }
    }

    let mut problem = tiny_solver::Problem::new();
    for p in points {
        problem.add_residual_block(
            1,
            &["c"],
            Box::new(CircleFactor {
                x: p[0],
                y: p[1],
                radius: radius_mm,
            }),
            Some(Box::new(tiny_solver::loss_functions::HuberLoss::new(
                huber_delta_mm,
            ))),
        );
    }

    let mut initial_values = HashMap::<String, ts_na::DVector<f64>>::new();
    initial_values.insert(
        "c".to_string(),
        ts_na::DVector::<f64>::from_vec(vec![init_center_mm[0], init_center_mm[1]]),
    );

    let optimizer = tiny_solver::LevenbergMarquardtOptimizer::default();
    let options = tiny_solver::OptimizerOptions {
        max_iteration: max_iters.clamp(1, 200),
        verbosity_level: 0,
        ..Default::default()
    };

    let result = optimizer.optimize(&problem, &initial_values, Some(options))?;
    let c = result.get("c")?;
    if c.len() != 2 {
        return None;
    }
    let cx = c[0];
    let cy = c[1];
    if !cx.is_finite() || !cy.is_finite() {
        return None;
    }
    Some([cx, cy])
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
