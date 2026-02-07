//! Non-linear refinement utilities.
//!
//! Milestone 6: refine per-marker centers in board coordinates (mm) using
//! measured outer-edge points and a known physical radius, then project the
//! refined center back to image space via the fitted homography.

use image::GrayImage;
use nalgebra as na;

use crate::ring::inner_estimate::Polarity;
use crate::{DetectedMarker, EllipseParams};
#[path = "refine/math.rs"]
mod math;
#[path = "refine/pipeline.rs"]
mod pipeline;
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
    pipeline::run(gray, h, detections, params, store_points)
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
