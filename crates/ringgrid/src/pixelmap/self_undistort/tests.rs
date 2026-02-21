use nalgebra::{Matrix3, Vector3};

use super::super::DivisionModel;
use super::*;
use crate::conic::fit_ellipse_direct;
use crate::DetectedMarker;

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

    let outer_ellipse = fit_ellipse_direct(&outer_pts);
    let inner_ellipse = fit_ellipse_direct(&inner_pts);

    let center = projected_center(h);
    DetectedMarker {
        id: Some(0),
        confidence: 1.0,
        center,
        center_mapped: None,
        board_xy_mm: None,
        ellipse_outer: outer_ellipse,
        ellipse_inner: inner_ellipse,
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
    let obj = super::objective::conic_consistency_objective(0.0, &edge_data, image_center, 0.0);
    assert!(
        obj < 1e-6,
        "objective for perfect circles should be near zero, got {}",
        obj
    );
}

#[test]
fn objective_minimum_near_true_lambda() {
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

    let edge_data: Vec<super::objective::MarkerEdgeData> = homographies
        .iter()
        .map(|hi| {
            let m = make_synthetic_marker(hi, 8.0, 14.0, 128, Some(&distort_model));
            (m.edge_points_outer.unwrap(), m.edge_points_inner.unwrap())
        })
        .collect();

    let image_center = [500.0, 500.0];
    let obj_zero =
        super::objective::conic_consistency_objective(0.0, &edge_data, image_center, 0.0);
    let obj_true =
        super::objective::conic_consistency_objective(true_lambda, &edge_data, image_center, 0.0);
    let obj_wrong =
        super::objective::conic_consistency_objective(-true_lambda, &edge_data, image_center, 0.0);

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
    let (x_min, f_min) = super::optimizer::golden_section_minimize(f, 0.0, 1.0, 50);
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
