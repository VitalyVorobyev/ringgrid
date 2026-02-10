//! Projective unbiased ring-center recovery from two conics.
//!
//! The center of an observed ellipse is generally biased under perspective.
//! For a ring marker, two conics (inner/outer) from the same concentric-circle
//! family allow recovery of the true projected center without intrinsics using
//! a conic-pencil eigen approach (Wang et al., 2019).

use nalgebra::{Matrix3, Point2, Vector3};

use crate::conic::Conic2D;

type C64 = nalgebra::Complex<f64>;

/// Selection and numerical options for projective center recovery.
#[derive(Debug, Clone, Copy)]
pub struct RingCenterProjectiveOptions {
    /// Optional expected radius ratio `Rin/Rout`.
    pub expected_ratio: Option<f64>,
    /// Penalty weight for `|lambda - k^2|` if `expected_ratio` is set.
    pub ratio_penalty_weight: f64,
    /// Soft preference weight for small imaginary part of eigenvalue.
    pub imag_lambda_weight: f64,
    /// Soft preference weight for small imaginary norm of eigenvector.
    pub imag_vec_weight: f64,
    /// Numerical epsilon.
    pub eps: f64,
}

impl Default for RingCenterProjectiveOptions {
    fn default() -> Self {
        Self {
            expected_ratio: None,
            ratio_penalty_weight: 1.0,
            imag_lambda_weight: 1e-3,
            imag_vec_weight: 1e-3,
            eps: 1e-12,
        }
    }
}

/// Debug information for selected eigenpair/candidate.
#[derive(Debug, Clone, Copy)]
pub struct RingCenterProjectiveDebug {
    /// Geometric residual for the selected candidate.
    pub selected_residual: f64,
    /// Separation from nearest competing eigenvalue.
    pub selected_eig_separation: f64,
}

/// Full result including debug score.
#[derive(Debug, Clone, Copy)]
pub struct RingCenterProjectiveResult {
    /// Recovered projective-unbiased center in image coordinates.
    pub center: Point2<f64>,
    /// Selector diagnostics for the chosen eigenpair.
    pub debug: RingCenterProjectiveDebug,
}

/// Errors returned by projective-center recovery.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectiveCenterError {
    /// Input conics contain non-finite values.
    NonFiniteInput,
    /// Inner conic is singular and cannot be inverted.
    SingularInnerConic,
    /// Conic pair is degenerate after normalization.
    DegenerateConics,
    /// No viable eigenpair candidate passed numerical checks.
    NoViableEigenpair,
    /// Recovered center is invalid/non-finite.
    InvalidCenter,
}

impl std::fmt::Display for ProjectiveCenterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "non-finite conic input"),
            Self::SingularInnerConic => write!(f, "inner conic is singular"),
            Self::DegenerateConics => write!(f, "degenerate conic pair"),
            Self::NoViableEigenpair => write!(f, "no viable eigenpair candidate"),
            Self::InvalidCenter => write!(f, "invalid recovered center"),
        }
    }
}

impl std::error::Error for ProjectiveCenterError {}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    center: Point2<f64>,
    residual: f64,
    score: f64,
    eig_separation: f64,
}

fn as_complex_matrix(m: &Matrix3<f64>) -> Matrix3<C64> {
    m.map(|v| C64::new(v, 0.0))
}

fn complex_vec_norm(v: &Vector3<C64>) -> f64 {
    (v[0].norm_sqr() + v[1].norm_sqr() + v[2].norm_sqr()).sqrt()
}

fn complex_imag_vec_norm(v: &Vector3<C64>) -> f64 {
    (v[0].im * v[0].im + v[1].im * v[1].im + v[2].im * v[2].im).sqrt()
}

fn real_null_vector_3x3(a: &Matrix3<f64>) -> Option<Vector3<f64>> {
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let s = svd.singular_values;
    let mut min_i = 0usize;
    if s[1] < s[min_i] {
        min_i = 1;
    }
    if s[2] < s[min_i] {
        min_i = 2;
    }
    let row = v_t.row(min_i);
    let v = Vector3::new(row[0], row[1], row[2]);
    let n = v.norm();
    if !n.is_finite() || n <= 1e-18 {
        return None;
    }
    Some(v / n)
}

fn complex_null_vector_3x3(a: &Matrix3<C64>) -> Option<Vector3<C64>> {
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let s = svd.singular_values;
    let mut min_i = 0usize;
    if s[1] < s[min_i] {
        min_i = 1;
    }
    if s[2] < s[min_i] {
        min_i = 2;
    }
    let row = v_t.row(min_i);
    // For complex SVD, `v_t` is V^H, so convert row to the corresponding
    // right-singular vector by conjugation.
    let v = Vector3::new(row[0].conj(), row[1].conj(), row[2].conj());
    let n = complex_vec_norm(&v);
    if !n.is_finite() || n <= 1e-18 {
        return None;
    }
    Some(v / C64::new(n, 0.0))
}

fn normalize_line(line: Vector3<f64>, eps: f64) -> Option<Vector3<f64>> {
    let n_xy = (line[0] * line[0] + line[1] * line[1]).sqrt();
    if n_xy > eps {
        return Some(line / n_xy);
    }
    let n = line.norm();
    if n > eps {
        return Some(line / n);
    }
    None
}

/// Compute projective unbiased ring center and expose selection residual/debug.
pub fn ring_center_projective_with_debug(
    q_inner: &Matrix3<f64>,
    q_outer: &Matrix3<f64>,
    opts: RingCenterProjectiveOptions,
) -> Result<RingCenterProjectiveResult, ProjectiveCenterError> {
    if !q_inner.iter().all(|v| v.is_finite()) || !q_outer.iter().all(|v| v.is_finite()) {
        return Err(ProjectiveCenterError::NonFiniteInput);
    }

    let q1 = Conic2D { mat: *q_inner }
        .normalize_frobenius()
        .ok_or(ProjectiveCenterError::DegenerateConics)?
        .mat;
    let q2 = Conic2D { mat: *q_outer }
        .normalize_frobenius()
        .ok_or(ProjectiveCenterError::DegenerateConics)?
        .mat;

    let q1_inv = q1
        .try_inverse()
        .ok_or(ProjectiveCenterError::SingularInnerConic)?;
    let q1_inv_c = as_complex_matrix(&q1_inv);
    let a = q2 * q1_inv;
    let ac = as_complex_matrix(&a);

    let eigvals = a.complex_eigenvalues();
    let mut eig_sep = [0.0f64; 3];
    for i in 0..3 {
        let mut min_d = f64::INFINITY;
        for j in 0..3 {
            if i == j {
                continue;
            }
            let d = (eigvals[i] - eigvals[j]).norm();
            if d < min_d {
                min_d = d;
            }
        }
        eig_sep[i] = min_d;
    }
    let mut best: Option<Candidate> = None;
    let eps = opts.eps.max(1e-15);
    let ratio_target = opts.expected_ratio.map(|k| k * k);

    for i in 0..3 {
        let lambda = eigvals[i];
        if !lambda.re.is_finite() || !lambda.im.is_finite() {
            continue;
        }
        let systems = [
            ac - Matrix3::<C64>::identity() * lambda,
            ac.transpose() - Matrix3::<C64>::identity() * lambda,
        ];
        for sys in &systems {
            let Some(u) = complex_null_vector_3x3(sys) else {
                continue;
            };
            let lambda_re = lambda.re;
            let imag_u_norm = complex_imag_vec_norm(&u);
            let mut p_candidates: Vec<Vector3<f64>> = Vec::new();

            // Method A (Wang): p~ = inv(Q1) * u.
            let p_h_c = q1_inv_c * u;
            if complex_vec_norm(&p_h_c) > eps && p_h_c[2].norm() > eps {
                let cx_c = p_h_c[0] / p_h_c[2];
                let cy_c = p_h_c[1] / p_h_c[2];
                if cx_c.re.is_finite()
                    && cx_c.im.is_finite()
                    && cy_c.re.is_finite()
                    && cy_c.im.is_finite()
                {
                    p_candidates.push(Vector3::new(cx_c.re, cy_c.re, 1.0));
                }
            }

            // Method B (equivalent in exact arithmetic): (Q2 - lambda Q1) p = 0.
            let m = q2 - q1 * lambda_re;
            if let Some(p_h) = real_null_vector_3x3(&m) {
                if p_h[2].abs() > eps {
                    p_candidates.push(Vector3::new(p_h[0] / p_h[2], p_h[1] / p_h[2], 1.0));
                }
            }

            for p in p_candidates {
                if !p.iter().all(|v| v.is_finite()) {
                    continue;
                }

                let q1p = q1 * p;
                let q2p = q2 * p;
                let denom = q1p.norm() * q2p.norm() + eps;
                if !denom.is_finite() || denom <= eps {
                    continue;
                }
                let residual = q1p.cross(&q2p).norm() / denom;

                let imag_lambda = lambda.im.abs();
                let ratio_penalty = ratio_target
                    .map(|t| {
                        let inv_t = if t.abs() > eps { 1.0 / t } else { t };
                        let e = (lambda.re - t).abs().min((lambda.re - inv_t).abs());
                        e * opts.ratio_penalty_weight.max(0.0)
                    })
                    .unwrap_or(0.0);
                let score = residual
                    + opts.imag_lambda_weight.max(0.0) * imag_lambda
                    + opts.imag_vec_weight.max(0.0) * imag_u_norm
                    + ratio_penalty;

                if normalize_line(q1 * p, eps).is_none() {
                    continue;
                }
                let cand = Candidate {
                    center: Point2::new(p[0], p[1]),
                    residual,
                    score,
                    eig_separation: eig_sep[i],
                };

                match best {
                    Some(b) => {
                        let sep_tol = 1e-12;
                        if cand.eig_separation > b.eig_separation + sep_tol
                            || ((cand.eig_separation - b.eig_separation).abs() <= sep_tol
                                && cand.score < b.score)
                        {
                            best = Some(cand);
                        }
                    }
                    None => best = Some(cand),
                }
            }
        }
    }

    let b = best.ok_or(ProjectiveCenterError::NoViableEigenpair)?;
    if !b.center.x.is_finite() || !b.center.y.is_finite() {
        return Err(ProjectiveCenterError::InvalidCenter);
    }

    Ok(RingCenterProjectiveResult {
        center: b.center,
        debug: RingCenterProjectiveDebug {
            selected_residual: b.residual,
            selected_eig_separation: b.eig_separation,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn circle_conic(radius: f64) -> Matrix3<f64> {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -(radius * radius))
    }

    fn project_conic(q_plane: &Matrix3<f64>, h: &Matrix3<f64>) -> Matrix3<f64> {
        let h_inv = h.try_inverse().expect("invertible homography");
        h_inv.transpose() * q_plane * h_inv
    }

    fn synthetic_h() -> Matrix3<f64> {
        Matrix3::new(1.12, 0.21, 321.0, -0.17, 0.94, 245.0, 8.0e-4, -6.0e-4, 1.0)
    }

    fn gt_center(h: &Matrix3<f64>) -> Point2<f64> {
        let p = h * Vector3::new(0.0, 0.0, 1.0);
        Point2::new(p[0] / p[2], p[1] / p[2])
    }

    fn symmetric_noise(rng: &mut StdRng, scale: f64) -> Matrix3<f64> {
        let mut n = Matrix3::<f64>::zeros();
        for i in 0..3 {
            for j in i..3 {
                let v = rng.gen_range(-1.0..1.0) * scale;
                n[(i, j)] = v;
                n[(j, i)] = v;
            }
        }
        n
    }

    #[test]
    fn ring_center_projective_exact_synthetic_homography() {
        let h = synthetic_h();
        let r_in = 4.0;
        let r_out = 7.0;
        let q1 = project_conic(&circle_conic(r_in), &h);
        let q2 = project_conic(&circle_conic(r_out), &h);

        let c = ring_center_projective_with_debug(
            &q1,
            &q2,
            RingCenterProjectiveOptions {
                expected_ratio: Some(r_in / r_out),
                ..Default::default()
            },
        )
        .expect("center recovery")
        .center;
        let gt = gt_center(&h);
        let err = ((c.x - gt.x).powi(2) + (c.y - gt.y).powi(2)).sqrt();
        assert!(
            err < 1e-8,
            "expected near-exact center, got err={:.3e} px",
            err
        );
    }

    #[test]
    fn ring_center_projective_is_scale_invariant() {
        let h = synthetic_h();
        let r_in = 3.2;
        let r_out = 8.5;
        let q1 = project_conic(&circle_conic(r_in), &h);
        let q2 = project_conic(&circle_conic(r_out), &h);

        let c0 = ring_center_projective_with_debug(&q1, &q2, Default::default())
            .expect("base")
            .center;
        let c1 = ring_center_projective_with_debug(&(q1 * 3.7), &(q2 * -2.1), Default::default())
            .expect("scaled conics")
            .center;

        let err = ((c0.x - c1.x).powi(2) + (c0.y - c1.y).powi(2)).sqrt();
        assert!(err < 1e-10, "scale invariance violated, err={:.3e}", err);
    }

    #[test]
    fn ring_center_projective_mild_noise_is_stable() {
        let h = synthetic_h();
        let r_in = 5.0;
        let r_out = 9.0;
        let q1 = project_conic(&circle_conic(r_in), &h);
        let q2 = project_conic(&circle_conic(r_out), &h);
        let gt = gt_center(&h);
        let q1_base = Conic2D { mat: q1 }
            .normalize_frobenius()
            .expect("normalized q1")
            .mat;
        let q2_base = Conic2D { mat: q2 }
            .normalize_frobenius()
            .expect("normalized q2")
            .mat;

        let mut rng = StdRng::seed_from_u64(7);
        let eps1 = 1e-10;
        let eps2 = 1e-10;
        let q1_pert = q1_base + symmetric_noise(&mut rng, eps1);
        let q2_pert = q2_base + symmetric_noise(&mut rng, eps2);
        let q1n = (q1_pert + q1_pert.transpose()) * 0.5;
        let q2n = (q2_pert + q2_pert.transpose()) * 0.5;

        let res = ring_center_projective_with_debug(
            &q1n,
            &q2n,
            RingCenterProjectiveOptions {
                expected_ratio: Some(r_in / r_out),
                ..Default::default()
            },
        )
        .expect("noisy center recovery");

        let err = ((res.center.x - gt.x).powi(2) + (res.center.y - gt.y).powi(2)).sqrt();
        assert!(err < 1e-2, "noise robustness degraded, err={:.3e} px", err);
        assert!(res.debug.selected_residual.is_finite());
    }
}
