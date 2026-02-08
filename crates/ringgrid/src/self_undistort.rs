//! Self-undistort: intrinsics-free distortion estimation from ring markers.
//!
//! Uses a 1-parameter division model to estimate lens distortion from
//! conic-consistency of detected inner/outer ring edge points.
//! The division model maps distorted → undistorted coordinates:
//!
//!   x_u = cx + (x_d - cx) / (1 + λ r²)
//!   y_u = cy + (y_d - cy) / (1 + λ r²)
//!
//! where r² = (x_d - cx)² + (y_d - cy)² and (cx, cy) is the distortion center
//! (typically image center).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::board_layout::BoardLayout;
use crate::camera::PixelMapper;
use crate::homography;

/// Edge point data for a single marker: (outer_points, inner_points).
type MarkerEdgeData = (Vec<[f64; 2]>, Vec<[f64; 2]>);
use crate::conic::{fit_ellipse_direct, rms_sampson_distance};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Minimum absolute objective improvement required to apply the model.
    ///
    /// This prevents applying when the objective is near numerical noise floor.
    pub min_abs_improvement: f64,
    /// Trim fraction for robust aggregation of per-marker objective values.
    ///
    /// `0.1` means drop 10% low and 10% high scores before averaging.
    pub trim_fraction: f64,
    /// Minimum |lambda| required for applying the model.
    ///
    /// Very small lambda values are effectively identity and are treated as
    /// "no correction" even if relative improvement is non-zero.
    pub min_lambda_abs: f64,
    /// Reject solutions that land too close to lambda-range boundaries.
    pub reject_range_edge: bool,
    /// Relative margin of the lambda range treated as unstable boundary area.
    pub range_edge_margin_frac: f64,
    /// Minimum decoded-ID correspondences needed for homography validation.
    pub validation_min_markers: usize,
    /// Minimum absolute homography self-error improvement (pixels) required.
    pub validation_abs_improvement_px: f64,
    /// Minimum relative homography self-error improvement required.
    pub validation_rel_improvement: f64,
}

impl Default for SelfUndistortConfig {
    fn default() -> Self {
        Self {
            enable: false,
            lambda_range: [-8e-7, 8e-7],
            max_evals: 40,
            min_markers: 6,
            improvement_threshold: 0.01,
            min_abs_improvement: 1e-4,
            trim_fraction: 0.1,
            min_lambda_abs: 5e-9,
            reject_range_edge: true,
            range_edge_margin_frac: 0.02,
            validation_min_markers: 24,
            validation_abs_improvement_px: 0.05,
            validation_rel_improvement: 0.03,
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
fn trimmed_mean(values: &mut [f64], trim_fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let trim = ((n as f64) * trim_fraction.clamp(0.0, 0.49)).floor() as usize;
    if 2 * trim >= n {
        return None;
    }
    let slice = &values[trim..(n - trim)];
    if slice.is_empty() {
        return None;
    }
    Some(slice.iter().sum::<f64>() / slice.len() as f64)
}

/// Compute a robust projective-conic objective across markers when edge points
/// are undistorted with the given lambda.
fn self_undistort_objective(
    lambda: f64,
    marker_edge_data: &[MarkerEdgeData],
    image_center: [f64; 2],
    trim_fraction: f64,
) -> f64 {
    let model = DivisionModel::new(lambda, image_center[0], image_center[1]);
    let mut marker_objective = Vec::with_capacity(marker_edge_data.len());

    for (outer_pts, inner_pts) in marker_edge_data {
        // Undistort edge points.
        let outer_ud = model.undistort_points(outer_pts);
        let inner_ud = model.undistort_points(inner_pts);

        // Refit conics.
        let Some(outer_ellipse) = fit_ellipse_direct(&outer_ud) else {
            continue;
        };
        let Some(inner_ellipse) = fit_ellipse_direct(&inner_ud) else {
            continue;
        };

        let rms_outer = rms_sampson_distance(&outer_ellipse, &outer_ud);
        let rms_inner = rms_sampson_distance(&inner_ellipse, &inner_ud);
        if !rms_outer.is_finite() || !rms_inner.is_finite() {
            continue;
        }
        let value = 0.5 * (rms_outer + rms_inner);
        if value.is_finite() {
            marker_objective.push(value);
        }
    }

    if marker_objective.is_empty() {
        return f64::MAX;
    }
    let Some(base) = trimmed_mean(&mut marker_objective, trim_fraction) else {
        return f64::MAX;
    };
    // Mild regularization to avoid unstable large-|lambda| solutions when
    // objective curvature is flat.
    let lambda_reg = 1e-6 * (lambda * 1.0e6).powi(2);
    base + lambda_reg
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

fn homography_self_error_px(
    markers: &[DetectedMarker],
    board: &BoardLayout,
    mapper: &dyn PixelMapper,
) -> Option<(f64, usize)> {
    let mut by_id: HashMap<usize, (f32, [f64; 2])> = HashMap::new();
    for m in markers {
        let Some(id) = m.id else {
            continue;
        };
        let Some(center_w) = mapper.image_to_working_pixel(m.center) else {
            continue;
        };
        if !center_w[0].is_finite() || !center_w[1].is_finite() {
            continue;
        }
        let conf = m.confidence;
        match by_id.get_mut(&id) {
            Some((best_conf, best_center)) => {
                if conf > *best_conf {
                    *best_conf = conf;
                    *best_center = center_w;
                }
            }
            None => {
                by_id.insert(id, (conf, center_w));
            }
        }
    }

    let mut src = Vec::<[f64; 2]>::new();
    let mut dst = Vec::<[f64; 2]>::new();
    for (id, (_conf, center_w)) in by_id {
        let Some(xy) = board.xy_mm(id) else {
            continue;
        };
        src.push([xy[0] as f64, xy[1] as f64]);
        dst.push(center_w);
    }
    if src.len() < 4 {
        return None;
    }
    let ransac_cfg = homography::RansacHomographyConfig {
        max_iters: 1000,
        inlier_threshold: 5.0,
        min_inliers: 8,
        seed: 0,
    };
    let Ok(res) = homography::fit_homography_ransac(&src, &dst, &ransac_cfg) else {
        return None;
    };
    if res.n_inliers == 0 {
        return None;
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    for (i, e) in res.errors.iter().enumerate() {
        if res.inlier_mask.get(i).copied().unwrap_or(false) && e.is_finite() {
            sum += *e;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some((sum / n as f64, n))
    }
}

/// Estimate a division-model distortion parameter from detected markers.
///
/// Uses a robust mean of Sampson residuals of fitted inner/outer ellipses:
/// correct distortion makes ring boundaries more conic-like.
///
/// Returns `None` if fewer than `min_markers` have both inner and outer edge
/// points with sufficient count.
pub fn estimate_self_undistort(
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
    board: Option<&BoardLayout>,
) -> Option<SelfUndistortResult> {
    let image_center = [image_size[0] as f64 / 2.0, image_size[1] as f64 / 2.0];

    // Prefer homography-based objective when enough decoded markers are present.
    let board_zero = board.and_then(|b| {
        let zero_model = DivisionModel::centered(0.0, image_size[0], image_size[1]);
        homography_self_error_px(markers, b, &zero_model).and_then(|(err, n)| {
            if n >= config.validation_min_markers {
                Some((b, err, n))
            } else {
                None
            }
        })
    });

    let (objective_at_zero, lambda_opt, objective_at_lambda, n_markers_used) =
        if let Some((board, err0, n0)) = board_zero {
            let (lambda_opt, objective_at_lambda) = golden_section_minimize(
                |lambda| {
                    let model = DivisionModel::centered(lambda, image_size[0], image_size[1]);
                    homography_self_error_px(markers, board, &model)
                        .map(|(e, _)| e)
                        .unwrap_or(f64::MAX)
                },
                config.lambda_range[0],
                config.lambda_range[1],
                config.max_evals,
            );
            (err0, lambda_opt, objective_at_lambda, n0)
        } else {
            // Fallback objective: conic-consistency from edge points.
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
            let objective_at_zero = self_undistort_objective(
                0.0,
                &marker_edge_data,
                image_center,
                config.trim_fraction,
            );
            if !objective_at_zero.is_finite() {
                return None;
            }
            let (lambda_opt, objective_at_lambda) = golden_section_minimize(
                |lambda| {
                    self_undistort_objective(
                        lambda,
                        &marker_edge_data,
                        image_center,
                        config.trim_fraction,
                    )
                },
                config.lambda_range[0],
                config.lambda_range[1],
                config.max_evals,
            );
            (
                objective_at_zero,
                lambda_opt,
                objective_at_lambda,
                marker_edge_data.len(),
            )
        };

    let abs_improvement = objective_at_zero - objective_at_lambda;
    let improvement = if objective_at_zero > 1e-18 {
        abs_improvement / objective_at_zero
    } else {
        0.0
    };

    let mut applied = objective_at_lambda.is_finite()
        && abs_improvement.is_finite()
        && abs_improvement > config.min_abs_improvement
        && improvement > config.improvement_threshold
        && lambda_opt.abs() >= config.min_lambda_abs;

    if applied && config.reject_range_edge {
        let lo = config.lambda_range[0].min(config.lambda_range[1]);
        let hi = config.lambda_range[0].max(config.lambda_range[1]);
        let span = (hi - lo).abs();
        let margin = span * config.range_edge_margin_frac.clamp(0.0, 0.49);
        if span > 0.0 && ((lambda_opt - lo).abs() <= margin || (hi - lambda_opt).abs() <= margin) {
            applied = false;
        }
    }

    if applied {
        if let Some(board) = board {
            let zero_model = DivisionModel::centered(0.0, image_size[0], image_size[1]);
            let opt_model = DivisionModel::centered(lambda_opt, image_size[0], image_size[1]);
            let err0 = homography_self_error_px(markers, board, &zero_model);
            let err1 = homography_self_error_px(markers, board, &opt_model);
            if let (Some((err0, n0)), Some((err1, n1))) = (err0, err1) {
                let abs_gain = err0 - err1;
                let rel_gain = if err0 > 1e-12 { abs_gain / err0 } else { 0.0 };
                let by_abs = abs_gain >= config.validation_abs_improvement_px;
                let by_rel = rel_gain >= config.validation_rel_improvement;
                let enough_ids =
                    n0 >= config.validation_min_markers && n1 >= config.validation_min_markers;
                if enough_ids && !(by_abs && by_rel) {
                    applied = false;
                }
            }
        }
    }

    Some(SelfUndistortResult {
        model: DivisionModel::centered(lambda_opt, image_size[0], image_size[1]),
        objective_at_lambda,
        objective_at_zero,
        n_markers_used,
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
        let outer_ellipse = fit_ellipse_direct(&outer_pts);
        let inner_ellipse = fit_ellipse_direct(&inner_pts);

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
        let obj = self_undistort_objective(0.0, &edge_data, image_center, 0.0);
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
        let obj_zero = self_undistort_objective(0.0, &edge_data, image_center, 0.0);
        let obj_true = self_undistort_objective(true_lambda, &edge_data, image_center, 0.0);
        let obj_wrong = self_undistort_objective(-true_lambda, &edge_data, image_center, 0.0);

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
        let result = estimate_self_undistort(&markers, [640, 480], &config, None);
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
            min_abs_improvement: 0.0,
            ..SelfUndistortConfig::default()
        };

        let result = estimate_self_undistort(&markers, [image_w, image_h], &config, None)
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
