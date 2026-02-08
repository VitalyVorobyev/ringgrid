//! Board-plane center refinement utilities.
//!
//! Milestone 6: refine per-marker centers in board coordinates (mm) using
//! measured outer-edge points and a known physical radius, then project the
//! refined center back to image space via the fitted homography.

use image::GrayImage;
use nalgebra as na;

use crate::board_layout::BoardLayout;
use crate::camera::PixelMapper;
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
use sampling::{OuterSampleConfig, SampleOutcome};

/// Solver backend used for fixed-radius circle-center optimization in board space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CircleCenterSolver {
    /// Robust Gauss-Newton / IRLS update on geometric residuals.
    Irls,
    /// Levenberg-Marquardt backend (`tiny-solver`) with Huber loss.
    #[default]
    Lm,
}

/// Configuration for board-plane center refinement.
#[derive(Debug, Clone)]
pub struct RefineParams {
    /// Master enable switch for board-plane center refinement.
    pub enabled: bool,
    /// Maximum solver iterations per marker.
    pub max_iters: usize,
    /// Huber delta (mm) used for robust residual weighting.
    pub huber_delta_mm: f64,
    /// Minimum number of sampled edge points required to refine a marker.
    pub min_points: usize,
    /// Reject refined centers that move more than this distance (mm) in board space.
    pub reject_thresh_mm: f64,
    /// Enable homography re-fit loop after refinement.
    pub enable_h_refit: bool,
    /// Number of H re-fit iterations when `enable_h_refit` is true.
    pub h_refit_iters: usize,
    /// Solver backend for per-marker board-plane fixed-radius circle center fitting.
    pub solver: CircleCenterSolver,
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
            solver: CircleCenterSolver::default(),
        }
    }
}

/// Aggregate statistics produced by one refinement pass.
#[derive(Debug, Clone, Default)]
pub struct RefineStats {
    /// Number of detections with a valid board correspondence.
    pub n_inliers: usize,
    /// Number of markers successfully refined and accepted.
    pub n_refined: usize,
    /// Number of markers that failed/refused refinement.
    pub n_failed: usize,
    /// Mean RMS residual before refinement (mm).
    pub mean_before_mm: f64,
    /// Mean RMS residual after refinement (mm).
    pub mean_after_mm: f64,
    /// 95th percentile RMS residual before refinement (mm).
    pub p95_before_mm: f64,
    /// 95th percentile RMS residual after refinement (mm).
    pub p95_after_mm: f64,
}

/// Final status of one per-marker refinement attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerRefineStatus {
    /// Refinement succeeded and update was accepted.
    Ok,
    /// Refinement solved but failed an acceptance gate.
    Rejected,
    /// Refinement could not be solved.
    Failed,
    /// Refinement was skipped (for example, missing prerequisites).
    Skipped,
}

/// Per-marker refinement record for debug tooling and analysis.
#[derive(Debug, Clone)]
pub struct MarkerRefineRecord {
    /// Marker id.
    pub id: usize,
    /// Number of sampled edge points used for this marker.
    pub n_points: usize,
    /// Initial center in board coordinates (mm).
    pub init_center_board_mm: [f64; 2],
    /// Refined center in board coordinates (mm), if solver succeeded.
    pub refined_center_board_mm: Option<[f64; 2]>,
    /// Marker center in image coordinates before refinement.
    pub center_img_before: [f64; 2],
    /// Marker center in image coordinates after refinement, if accepted.
    pub center_img_after: Option<[f64; 2]>,
    /// RMS circle residual before refinement (mm).
    pub before_rms_mm: Option<f64>,
    /// RMS circle residual after refinement (mm).
    pub after_rms_mm: Option<f64>,
    /// Board-space center shift magnitude (mm).
    pub delta_center_mm: Option<f64>,
    /// Optional sampled image-space edge points.
    pub edge_points_img: Option<Vec<[f32; 2]>>,
    /// Optional sampled board-space edge points (mm).
    pub edge_points_board_mm: Option<Vec<[f32; 2]>>,
    /// Final per-marker status.
    pub status: MarkerRefineStatus,
    /// Optional reject/failure reason.
    pub reason: Option<String>,
}

impl MarkerRefineRecord {
    /// Create a record with required fields and `None` defaults for optional fields.
    fn new(
        id: usize,
        init_center_board_mm: [f64; 2],
        center_img_before: [f64; 2],
        status: MarkerRefineStatus,
    ) -> Self {
        Self {
            id,
            n_points: 0,
            init_center_board_mm,
            refined_center_board_mm: None,
            center_img_before,
            center_img_after: None,
            before_rms_mm: None,
            after_rms_mm: None,
            delta_center_mm: None,
            edge_points_img: None,
            edge_points_board_mm: None,
            status,
            reason: None,
        }
    }

    fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    fn with_points(
        mut self,
        n_points: usize,
        img: Option<&[[f64; 2]]>,
        board: Option<&[[f64; 2]]>,
        store: bool,
    ) -> Self {
        self.n_points = n_points;
        if store {
            self.edge_points_img = img.map(truncate_f32);
            self.edge_points_board_mm = board.map(truncate_f32);
        }
        self
    }
}

fn truncate_f32(points: &[[f64; 2]]) -> Vec<[f32; 2]> {
    points
        .iter()
        .take(256)
        .map(|p| [p[0] as f32, p[1] as f32])
        .collect()
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
    mapper: Option<&dyn PixelMapper>,
    cfg: OuterSampleConfig,
    polarity: Polarity,
) -> SampleOutcome {
    sampling::sample_outer_points_around_ellipse(gray, ellipse, mapper, cfg, polarity)
}

fn solve_circle_center_mm(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
    solver: CircleCenterSolver,
) -> Option<[f64; 2]> {
    solver::solve_circle_center_mm(
        points,
        init_center_mm,
        radius_mm,
        max_iters,
        huber_delta_mm,
        solver,
    )
}

/// Refine marker centers in board coordinates using fixed-radius circle fitting.
///
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub fn refine_markers_circle_board_with_mapper(
    gray: &GrayImage,
    h: &na::Matrix3<f64>,
    detections: &mut [DetectedMarker],
    params: &RefineParams,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
    store_points: bool,
) -> (RefineStats, Vec<MarkerRefineRecord>) {
    pipeline::run(gray, h, detections, params, board, mapper, store_points)
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

        let est = solve_circle_center_mm(&pts, init, radius, 50, 0.2, CircleCenterSolver::Lm)
            .expect("solver result");
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

        let est = solve_circle_center_mm(&pts, init, radius, 80, 0.2, CircleCenterSolver::Lm)
            .expect("solver result");
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
        let est = solve_circle_center_mm(&pts, init, radius, 80, 0.10, CircleCenterSolver::Lm)
            .expect("solver result");
        assert!((est[0] - true_center[0]).abs() < 5e-2);
        assert!((est[1] - true_center[1]).abs() < 5e-2);
    }
}
