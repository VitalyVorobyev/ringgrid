//! Self-undistort: intrinsics-free distortion estimation from ring markers.
//!
//! Uses a 1-parameter division model to estimate lens distortion from the
//! projective-center residuals of detected inner/outer ring conic pairs.
//! The division model maps distorted → undistorted coordinates:
//!
//!   x_u = cx + (x_d - cx) / (1 + λ r²)
//!   y_u = cy + (y_d - cy) / (1 + λ r²)
//!
//! where r² = (x_d - cx)² + (y_d - cy)² and (cx, cy) is the distortion center
//! (typically image center).

use serde::{Deserialize, Serialize};

use crate::camera::PixelMapper;

/// Edge point data for a single marker: (outer_points, inner_points).
type MarkerEdgeData = (Vec<[f64; 2]>, Vec<[f64; 2]>);
use crate::conic::{fit_ellipse_direct, Conic2D};
use crate::projective_center::{ring_center_projective_with_debug, RingCenterProjectiveOptions};
use crate::DetectedMarker;

/// Single-parameter division distortion model.
///
/// Negative lambda corresponds to barrel distortion (most common),
/// positive to pincushion distortion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DivisionModel {
    /// Distortion parameter.
    pub lambda: f64,
    /// Distortion center x (pixels).
    pub cx: f64,
    /// Distortion center y (pixels).
    pub cy: f64,
}

impl DivisionModel {
    /// Create a division model with explicit parameters.
    pub fn new(lambda: f64, cx: f64, cy: f64) -> Self {
        Self { lambda, cx, cy }
    }

    /// Create a division model centered on the image.
    pub fn centered(lambda: f64, width: u32, height: u32) -> Self {
        Self {
            lambda,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
        }
    }

    /// Identity model (zero distortion) centered on the image.
    pub fn identity(width: u32, height: u32) -> Self {
        Self::centered(0.0, width, height)
    }

    /// Undistort a single point.
    pub fn undistort_point(&self, distorted_xy: [f64; 2]) -> [f64; 2] {
        let dx = distorted_xy[0] - self.cx;
        let dy = distorted_xy[1] - self.cy;
        let r2 = dx * dx + dy * dy;
        let denom = 1.0 + self.lambda * r2;
        if denom.abs() < 1e-12 || !denom.is_finite() {
            return distorted_xy;
        }
        let scale = 1.0 / denom;
        [self.cx + dx * scale, self.cy + dy * scale]
    }

    /// Distort a point (inverse mapping: undistorted → distorted).
    ///
    /// Uses iterative fixed-point method since the inverse is not closed-form.
    pub fn distort_point(&self, undistorted_xy: [f64; 2]) -> Option<[f64; 2]> {
        if self.lambda.abs() < 1e-18 {
            return Some(undistorted_xy);
        }
        let ux = undistorted_xy[0] - self.cx;
        let uy = undistorted_xy[1] - self.cy;
        let mut dx = ux;
        let mut dy = uy;
        for _ in 0..20 {
            let r2 = dx * dx + dy * dy;
            let factor = 1.0 + self.lambda * r2;
            if factor.abs() < 1e-12 || !factor.is_finite() {
                return None;
            }
            let dx_next = ux * factor;
            let dy_next = uy * factor;
            if !dx_next.is_finite() || !dy_next.is_finite() {
                return None;
            }
            let delta = (dx_next - dx).powi(2) + (dy_next - dy).powi(2);
            dx = dx_next;
            dy = dy_next;
            if delta.sqrt() < 1e-12 {
                break;
            }
        }
        let out = [self.cx + dx, self.cy + dy];
        if out[0].is_finite() && out[1].is_finite() {
            Some(out)
        } else {
            None
        }
    }

    /// Undistort a batch of points.
    pub fn undistort_points(&self, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        points.iter().map(|p| self.undistort_point(*p)).collect()
    }
}

impl PixelMapper for DivisionModel {
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
        Some(self.undistort_point(image_xy))
    }

    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
        self.distort_point(working_xy)
    }
}

/// Configuration for self-undistort estimation.
#[derive(Debug, Clone)]
pub struct SelfUndistortConfig {
    /// Enable self-undistort refinement.
    pub enable: bool,
    /// Search range for lambda: [lambda_min, lambda_max].
    pub lambda_range: [f64; 2],
    /// Maximum function evaluations for the 1D optimizer.
    pub max_evals: usize,
    /// Minimum number of markers with both inner+outer edge points required.
    pub min_markers: usize,
    /// Relative improvement threshold: accept only if
    /// `(baseline - optimum) / baseline > improvement_threshold`.
    pub improvement_threshold: f64,
}

impl Default for SelfUndistortConfig {
    fn default() -> Self {
        Self {
            enable: false,
            lambda_range: [-2e-7, 2e-7],
            max_evals: 40,
            min_markers: 6,
            improvement_threshold: 0.01,
        }
    }
}

/// Result of self-undistort estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfUndistortResult {
    /// Estimated division model.
    pub model: DivisionModel,
    /// Objective value at the estimated lambda.
    pub objective_at_lambda: f64,
    /// Objective value at lambda=0 (baseline).
    pub objective_at_zero: f64,
    /// Number of markers used in estimation.
    pub n_markers_used: usize,
    /// Whether the estimated model was applied (improvement exceeded threshold).
    pub applied: bool,
}

/// Minimum number of edge points per ring required for conic refit.
const MIN_EDGE_POINTS: usize = 6;

/// Compute the mean projective-center residual across markers when edge points
/// are undistorted with the given lambda.
fn self_undistort_objective(
    lambda: f64,
    marker_edge_data: &[MarkerEdgeData],
    image_center: [f64; 2],
    proj_opts: &RingCenterProjectiveOptions,
) -> f64 {
    let model = DivisionModel::new(lambda, image_center[0], image_center[1]);
    let mut total_residual = 0.0;
    let mut count = 0usize;

    for (outer_pts, inner_pts) in marker_edge_data {
        // Undistort edge points.
        let outer_ud = model.undistort_points(outer_pts);
        let inner_ud = model.undistort_points(inner_pts);

        // Refit conics.
        let Some((_outer_coeffs, _outer_ellipse)) = fit_ellipse_direct(&outer_ud) else {
            continue;
        };
        let Some((_inner_coeffs, _inner_ellipse)) = fit_ellipse_direct(&inner_ud) else {
            continue;
        };

        let q_outer = Conic2D::from_coeffs(&_outer_coeffs).mat;
        let q_inner = Conic2D::from_coeffs(&_inner_coeffs).mat;

        // Compute projective center residual.
        let Ok(res) = ring_center_projective_with_debug(&q_inner, &q_outer, *proj_opts) else {
            continue;
        };

        if res.debug.selected_residual.is_finite() {
            total_residual += res.debug.selected_residual;
            count += 1;
        }
    }

    if count == 0 {
        return f64::MAX;
    }
    total_residual / count as f64
}

/// Golden-section search for the minimum of `f` on `[a, b]`.
///
/// Returns `(x_min, f_min)`.
fn golden_section_minimize(
    f: impl Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    max_evals: usize,
) -> (f64, f64) {
    const PHI: f64 = 1.618_033_988_749_895;
    const RESP: f64 = 2.0 - PHI; // ~0.382

    let mut x1 = a + RESP * (b - a);
    let mut x2 = b - RESP * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    let mut evals = 2;

    while evals < max_evals && (b - a).abs() > 1e-18 {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + RESP * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - RESP * (b - a);
            f2 = f(x2);
        }
        evals += 1;
    }

    if f1 < f2 {
        (x1, f1)
    } else {
        (x2, f2)
    }
}

/// Estimate a division-model distortion parameter from detected markers.
///
/// Uses the projective-center residual of inner/outer ring conic pairs as the
/// objective: correct distortion makes conic pairs more consistent, yielding
/// lower residuals.
///
/// Returns `None` if fewer than `min_markers` have both inner and outer edge
/// points with sufficient count.
pub fn estimate_self_undistort(
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
) -> Option<SelfUndistortResult> {
    // Collect edge data from markers with sufficient points.
    let marker_edge_data: Vec<MarkerEdgeData> = markers
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
        .collect();

    if marker_edge_data.len() < config.min_markers {
        return None;
    }

    let image_center = [image_size[0] as f64 / 2.0, image_size[1] as f64 / 2.0];
    let proj_opts = RingCenterProjectiveOptions::default();

    // Baseline: objective at lambda = 0.
    let objective_at_zero =
        self_undistort_objective(0.0, &marker_edge_data, image_center, &proj_opts);
    if !objective_at_zero.is_finite() {
        return None;
    }

    // Optimize lambda via golden-section search.
    let (lambda_opt, objective_at_lambda) = golden_section_minimize(
        |lambda| self_undistort_objective(lambda, &marker_edge_data, image_center, &proj_opts),
        config.lambda_range[0],
        config.lambda_range[1],
        config.max_evals,
    );

    let improvement = if objective_at_zero > 1e-18 {
        (objective_at_zero - objective_at_lambda) / objective_at_zero
    } else {
        0.0
    };

    let applied = improvement > config.improvement_threshold && objective_at_lambda.is_finite();

    Some(SelfUndistortResult {
        model: DivisionModel::centered(lambda_opt, image_size[0], image_size[1]),
        objective_at_lambda,
        objective_at_zero,
        n_markers_used: marker_edge_data.len(),
        applied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    fn circle_conic(radius: f64) -> Matrix3<f64> {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -(radius * radius))
    }

    fn project_conic(q_plane: &Matrix3<f64>, h: &Matrix3<f64>) -> Matrix3<f64> {
        let h_inv = h.try_inverse().expect("invertible homography");
        h_inv.transpose() * q_plane * h_inv
    }

    /// Sample N points on a conic boundary.
    fn sample_conic_boundary(q: &Matrix3<f64>, n: usize) -> Vec<[f64; 2]> {
        let q_sym = 0.5 * (q + q.transpose());
        let coeffs = crate::conic::ConicCoeffs([
            q_sym[(0, 0)],
            2.0 * q_sym[(0, 1)],
            q_sym[(1, 1)],
            2.0 * q_sym[(0, 2)],
            2.0 * q_sym[(1, 2)],
            q_sym[(2, 2)],
        ]);
        let e = coeffs.to_ellipse().expect("should be an ellipse");
        e.sample_points(n)
    }

    fn synthetic_h() -> Matrix3<f64> {
        Matrix3::new(1.12, 0.21, 321.0, -0.17, 0.94, 245.0, 8.0e-4, -6.0e-4, 1.0)
    }

    fn projected_center(h: &Matrix3<f64>) -> [f64; 2] {
        let p = h * Vector3::new(0.0, 0.0, 1.0);
        [p[0] / p[2], p[1] / p[2]]
    }

    /// Build a synthetic marker with edge points from projected circles.
    ///
    /// When `distort` is provided, points are distorted to simulate lens distortion.
    /// The edge points stored are the *distorted* image-space points (what a camera
    /// would observe). The ellipse params are fitted to these distorted points.
    fn make_synthetic_marker(
        h: &Matrix3<f64>,
        r_inner: f64,
        r_outer: f64,
        n_points: usize,
        distort: Option<&DivisionModel>,
    ) -> DetectedMarker {
        let q_inner = project_conic(&circle_conic(r_inner), h);
        let q_outer = project_conic(&circle_conic(r_outer), h);

        let mut outer_pts = sample_conic_boundary(&q_outer, n_points);
        let mut inner_pts = sample_conic_boundary(&q_inner, n_points);

        // Optionally apply distortion to the sampled points.
        if let Some(model) = distort {
            outer_pts = outer_pts
                .iter()
                .filter_map(|p| model.distort_point(*p))
                .collect();
            inner_pts = inner_pts
                .iter()
                .filter_map(|p| model.distort_point(*p))
                .collect();
        }

        // Fit ellipses to (possibly distorted) points.
        let outer_ellipse = fit_ellipse_direct(&outer_pts).map(|(_, e)| e);
        let inner_ellipse = fit_ellipse_direct(&inner_pts).map(|(_, e)| e);

        let center = projected_center(h);
        DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center,
            center_projective: None,
            vanishing_line: None,
            center_projective_residual: None,
            ellipse_outer: outer_ellipse.as_ref().map(crate::EllipseParams::from),
            ellipse_inner: inner_ellipse.as_ref().map(crate::EllipseParams::from),
            edge_points_outer: Some(outer_pts),
            edge_points_inner: Some(inner_pts),
            fit: crate::FitMetrics::default(),
            decode: None,
        }
    }

    #[test]
    fn division_model_identity_is_noop() {
        let model = DivisionModel::identity(640, 480);
        let p = [123.45, 67.89];
        let u = model.undistort_point(p);
        assert!(
            (u[0] - p[0]).abs() < 1e-10,
            "x: got {}, expected {}",
            u[0],
            p[0]
        );
        assert!(
            (u[1] - p[1]).abs() < 1e-10,
            "y: got {}, expected {}",
            u[1],
            p[1]
        );
    }

    #[test]
    fn division_model_roundtrip() {
        for &lambda in &[-1e-7, -5e-8, 1e-8, 1e-7] {
            let model = DivisionModel::centered(lambda, 1280, 960);
            let p = [400.0, 300.0];
            let u = model.undistort_point(p);
            let d = model.distort_point(u).expect("distort should succeed");
            assert!(
                (d[0] - p[0]).abs() < 1e-8,
                "roundtrip failed for lambda={}: dx={}",
                lambda,
                (d[0] - p[0]).abs()
            );
            assert!(
                (d[1] - p[1]).abs() < 1e-8,
                "roundtrip failed for lambda={}: dy={}",
                lambda,
                (d[1] - p[1]).abs()
            );
        }
    }

    #[test]
    fn objective_zero_for_perfect_circles() {
        let h = synthetic_h();
        let marker = make_synthetic_marker(&h, 4.0, 7.0, 64, None);
        let edge_data = vec![(
            marker.edge_points_outer.unwrap(),
            marker.edge_points_inner.unwrap(),
        )];
        let image_center = [320.0, 240.0];
        let proj_opts = RingCenterProjectiveOptions::default();
        let obj = self_undistort_objective(0.0, &edge_data, image_center, &proj_opts);
        assert!(
            obj < 1e-6,
            "objective for perfect circles should be near zero, got {}",
            obj
        );
    }

    #[test]
    fn objective_minimum_near_true_lambda() {
        // lambda = -1e-6 on 1000x1000 image: at r=500px (corner), displacement
        // is ~125 px -- clearly visible, enough for the conic pencil to break.
        let true_lambda = -1e-6;
        let distort_model = DivisionModel::centered(true_lambda, 1000, 1000);

        let homographies: Vec<Matrix3<f64>> = vec![
            Matrix3::new(1.12, 0.21, 100.0, -0.17, 0.94, 80.0, 8.0e-4, -6.0e-4, 1.0),
            Matrix3::new(0.95, -0.10, 850.0, 0.08, 1.05, 900.0, 5.0e-4, 3.0e-4, 1.0),
            Matrix3::new(1.05, 0.15, 900.0, -0.12, 0.90, 100.0, -3.0e-4, 7.0e-4, 1.0),
            Matrix3::new(0.88, 0.05, 120.0, 0.03, 1.10, 880.0, 6.0e-4, -2.0e-4, 1.0),
            Matrix3::new(1.20, -0.08, 750.0, 0.14, 0.85, 800.0, -5.0e-4, 4.0e-4, 1.0),
            Matrix3::new(0.92, 0.18, 180.0, -0.05, 0.98, 150.0, 4.0e-4, -5.0e-4, 1.0),
        ];

        let edge_data: Vec<MarkerEdgeData> = homographies
            .iter()
            .map(|hi| {
                let m = make_synthetic_marker(hi, 8.0, 14.0, 128, Some(&distort_model));
                (m.edge_points_outer.unwrap(), m.edge_points_inner.unwrap())
            })
            .collect();

        let image_center = [500.0, 500.0];
        let proj_opts = RingCenterProjectiveOptions::default();

        let obj_zero = self_undistort_objective(0.0, &edge_data, image_center, &proj_opts);
        let obj_true = self_undistort_objective(true_lambda, &edge_data, image_center, &proj_opts);
        let obj_wrong =
            self_undistort_objective(-true_lambda, &edge_data, image_center, &proj_opts);

        assert!(
            obj_true < obj_zero,
            "objective at true lambda ({:.3e}) should be less than at zero ({:.3e})",
            obj_true,
            obj_zero
        );
        assert!(
            obj_true < obj_wrong,
            "objective at true lambda ({:.3e}) should be less than at wrong sign ({:.3e})",
            obj_true,
            obj_wrong
        );
    }

    #[test]
    fn golden_section_finds_quadratic_min() {
        let f = |x: f64| (x - 0.3).powi(2);
        let (x_min, f_min) = golden_section_minimize(f, 0.0, 1.0, 50);
        assert!(
            (x_min - 0.3).abs() < 1e-10,
            "expected min near 0.3, got {}",
            x_min
        );
        assert!(f_min < 1e-18);
    }

    #[test]
    fn estimate_returns_none_for_few_markers() {
        let markers: Vec<DetectedMarker> = vec![];
        let config = SelfUndistortConfig {
            enable: true,
            min_markers: 6,
            ..Default::default()
        };
        let result = estimate_self_undistort(&markers, [640, 480], &config);
        assert!(result.is_none());
    }

    #[test]
    fn estimate_recovers_lambda_synthetic() {
        let true_lambda = -1e-6;
        let image_w = 1000u32;
        let image_h = 1000u32;
        let distort_model = DivisionModel::centered(true_lambda, image_w, image_h);

        let homographies: Vec<Matrix3<f64>> = vec![
            Matrix3::new(1.12, 0.21, 100.0, -0.17, 0.94, 80.0, 8.0e-4, -6.0e-4, 1.0),
            Matrix3::new(0.95, -0.10, 850.0, 0.08, 1.05, 900.0, 5.0e-4, 3.0e-4, 1.0),
            Matrix3::new(1.05, 0.15, 900.0, -0.12, 0.90, 100.0, -3.0e-4, 7.0e-4, 1.0),
            Matrix3::new(0.88, 0.05, 120.0, 0.03, 1.10, 880.0, 6.0e-4, -2.0e-4, 1.0),
            Matrix3::new(1.20, -0.08, 750.0, 0.14, 0.85, 800.0, -5.0e-4, 4.0e-4, 1.0),
            Matrix3::new(0.92, 0.18, 180.0, -0.05, 0.98, 150.0, 4.0e-4, -5.0e-4, 1.0),
            Matrix3::new(1.00, 0.12, 700.0, -0.09, 1.02, 600.0, 2.0e-4, -3.0e-4, 1.0),
            Matrix3::new(0.97, -0.15, 300.0, 0.11, 0.91, 50.0, -4.0e-4, 6.0e-4, 1.0),
        ];

        let markers: Vec<DetectedMarker> = homographies
            .iter()
            .enumerate()
            .map(|(i, hi)| {
                let mut m = make_synthetic_marker(hi, 8.0, 14.0, 128, Some(&distort_model));
                m.id = Some(i);
                m
            })
            .collect();

        let config = SelfUndistortConfig {
            enable: true,
            lambda_range: [-5e-6, 5e-6],
            max_evals: 60,
            min_markers: 6,
            improvement_threshold: 0.001,
        };

        let result = estimate_self_undistort(&markers, [image_w, image_h], &config)
            .expect("should find result");

        // Tolerance is generous: the 1D division model is an approximation
        // and the sparse synthetic marker set limits precision.
        let lambda_err = (result.model.lambda - true_lambda).abs();
        assert!(
            lambda_err < 2e-6,
            "expected lambda near {:.3e}, got {:.3e} (err={:.3e})",
            true_lambda,
            result.model.lambda,
            lambda_err
        );
        assert!(
            result.objective_at_lambda < result.objective_at_zero,
            "optimized objective ({:.3e}) should be less than baseline ({:.3e})",
            result.objective_at_lambda,
            result.objective_at_zero
        );
        assert!(result.applied, "should be marked as applied");
    }
}
