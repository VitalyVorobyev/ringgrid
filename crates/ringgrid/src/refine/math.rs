use nalgebra as na;

pub(super) fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub(super) fn rms_circle_residual_mm(points: &[[f64; 2]], center: [f64; 2], radius_mm: f64) -> f64 {
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

pub(super) fn try_unproject(h_inv: &na::Matrix3<f64>, x: f64, y: f64) -> Option<[f64; 2]> {
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
