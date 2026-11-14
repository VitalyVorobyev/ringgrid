//! Photometric (intensity-model) refinement of one coded marker.
//!
//! The marker patch is explained pixel by pixel as
//!
//! ```text
//! I(x) = B(x) - A(x) * P(u(x); sigma)
//! ```
//!
//! - `u(x)` is the board-local position (mm, relative to the marker centre)
//!   that pixel `x` sees, through a local projective map seeded from the board
//!   homography and corrected per marker by a free translation `p` and a free
//!   2×2 matrix `C`: `y = h0 + C (x - p)`, `u = H_loc^-1(y)`. The marker centre
//!   (`u = 0`) is imaged exactly at `p`, so the reported centre is `p` — no
//!   ellipse-centre or two-conic construction is involved.
//! - `P` is the dark-ink coverage of the printed marker (both rings, the
//!   decoded code sectors and the outer rings of the neighbouring markers)
//!   blurred by an isotropic Gaussian PSF of standard deviation `sigma` px. Each
//!   edge contributes a normal-CDF step of its signed image-space distance (the
//!   board-space distance divided by the local gradient norm); code sectors are
//!   the product of a radial band and an angular indicator whose sector
//!   transitions are blurred the same way.
//! - `B` (paper) and `A` (paper minus ink) are affine in the image, which absorbs
//!   vignetting and illumination gradients.
//!
//! The parameters are fitted by Levenberg–Marquardt on the pixel residuals with
//! Huber weights; saturated pixels (and their immediate neighbours) are
//! masked. The centre covariance is the `p` block of the inverse Fisher
//! information `(J^T W J / s^2 + prior)^-1`, with `s^2` the weighted residual
//! variance.

use image::GrayImage;
use nalgebra::{DMatrix, DVector};

/// Number of sectors in the coded band.
const CODE_SECTORS: usize = 16;
/// Nonlinear geometric parameters: p (2), C (4), log sigma (1).
const N_GEOM: usize = 7;
/// Linear photometric coefficients: B up to cubic (10) and A affine (3).
const N_LIN: usize = 13;
/// Active linear coefficients with an affine background.
const LIN_AFFINE: [usize; 6] = [0, 1, 2, 10, 11, 12];
/// Active linear coefficients with a cubic background.
const LIN_CUBIC: [usize; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
/// Index of the constant term of `A` (the contrast) in the linear block.
const CONTRAST: usize = 10;
/// Beyond this many PSF sigmas an edge's contribution is exactly 0 or 1.
const EDGE_REACH_SIGMAS: f64 = 6.0;
/// Prior standard deviation of `C - I` (dimensionless). Keeps the rotation of
/// a marker whose code band is blurred away well-posed; far wider than the
/// error of a board-homography seed.
const C_PRIOR_SIGMA: f64 = 0.15;
/// Prior standard deviation of free print radii (mm).
const RADIUS_PRIOR_SIGMA_MM: f64 = 0.5;
/// Huber threshold in units of the robust residual scale.
const HUBER_K: f64 = 2.0;
/// Blur search grid for the initial sigma (px).
const SIGMA_GRID: [f64; 4] = [0.7, 1.4, 2.6, 4.5];
/// Iterations during which the robust residual scale is re-estimated.
const SCALE_UPDATE_ITERATIONS: usize = 3;

/// Board → image projective map, row-major.
pub(crate) type Mat3 = [[f64; 3]; 3];

/// Radial print geometry of a coded marker, board mm, from the centre out:
/// `[hole edge, inner ring outer edge, outer ring inner edge, outer edge]`.
/// Ink covers `[r0, r1]` and `[r2, r3]`; the code band is `[r1, r2]`.
pub(crate) type PrintRadii = [f64; 4];

/// Everything the fitter needs for one marker.
pub(crate) struct PhotometricProblem<'a> {
    pub gray: &'a GrayImage,
    /// Board-local mm (relative to this marker's centre) → image px.
    pub local_map: Mat3,
    /// Initial centre estimate, image px.
    pub center_seed: [f64; 2],
    /// Decoded codeword; bit `s` set ⇔ sector `s` is paper (white).
    pub codeword: u16,
    pub radii: PrintRadii,
    /// Free the three inner radii (`r3` stays fixed: a common scale of all
    /// radii is indistinguishable from the scale in `C`).
    pub fit_radii: bool,
    /// Model the paper level `B` as a cubic instead of an affine function of
    /// the image position, for patches where a smooth but strongly curved additive term
    /// (a glare halo) is present. Costs precision elsewhere.
    pub cubic_background: bool,
    /// Centres of neighbouring markers, board mm relative to this marker.
    pub neighbors_mm: &'a [[f64; 2]],
    /// Pixels up to `r3 + patch_margin_mm` (board mm) enter the fit.
    pub patch_margin_mm: f64,
    /// Grey level at or above which a pixel counts as saturated.
    pub saturation_level: u8,
    pub max_iterations: usize,
}

/// Fitted parameters and their uncertainty.
#[derive(Debug, Clone)]
pub(crate) struct PhotometricFit {
    pub center: [f64; 2],
    /// Covariance of `center`, px².
    pub center_cov: [[f64; 2]; 2],
    pub blur_sigma_px: f64,
    /// Paper minus ink at the marker centre, grey levels.
    pub contrast: f64,
    /// Robust residual standard deviation, grey levels.
    pub residual_sigma: f64,
    pub n_pixels: usize,
    /// Fraction of the patch masked as saturated.
    pub masked_fraction: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Fitted radii (equal to the input unless `fit_radii`).
    pub radii: PrintRadii,
    /// Standard deviations of `r0..r2` when fitted.
    pub radii_std: Option<[f64; 3]>,
}

/// Why a fit was not produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhotometricFailure {
    /// Too few unmasked pixels inside the image.
    InsufficientPixels,
    /// The local map is singular at the marker.
    DegenerateMap,
    /// The normal equations could not be solved.
    SingularSystem,
}

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Standard normal CDF and density at `z` with a single exponential:
/// Abramowitz & Stegun 7.1.26 for erfc (absolute error below 1.5e-7).
#[inline]
fn norm_cdf_pdf(z: f64) -> (f64, f64) {
    let e = (-0.5 * z * z).exp();
    let x = z.abs() * std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let tail = 0.5 * poly * e; // Phi(-|z|)
    let cdf = if z >= 0.0 { 1.0 - tail } else { tail };
    (cdf, INV_SQRT_2PI * e)
}

/// Value and first derivatives of a blurred step `Phi(d / sigma)`.
#[derive(Clone, Copy)]
struct Step {
    value: f64,
    /// d value / d d
    slope: f64,
    /// d value / d log(sigma)
    dlog_sigma: f64,
}

#[inline]
fn step(d: f64, sigma: f64) -> Step {
    step_curved(d, sigma, f64::INFINITY)
}

/// Blurred step across a circular edge of image-space radius `radius_px`,
/// `d` positive outside the circle. To first order in `sigma / radius` the
/// blur of a disc moves its half-level contour inward by
/// `sigma^2 / (2 radius)`, the same for a dark disc and a light hole.
#[inline]
fn step_curved(d: f64, sigma: f64, radius_px: f64) -> Step {
    let shift = 0.5 * sigma / radius_px;
    let z = d / sigma + shift;
    if z > EDGE_REACH_SIGMAS {
        return Step {
            value: 1.0,
            slope: 0.0,
            dlog_sigma: 0.0,
        };
    }
    if z < -EDGE_REACH_SIGMAS {
        return Step {
            value: 0.0,
            slope: 0.0,
            dlog_sigma: 0.0,
        };
    }
    let (cdf, pdf) = norm_cdf_pdf(z);
    Step {
        value: cdf,
        slope: pdf / sigma,
        dlog_sigma: pdf * (shift - d / sigma),
    }
}

fn mat3_inverse(m: &Mat3) -> Option<Mat3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let c =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [
            c(1, 2, 1, 2) / det,
            -c(0, 2, 1, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 2, 0, 2) / det,
            c(0, 2, 0, 2) / det,
            -c(0, 1, 0, 2) / det,
        ],
        [
            c(1, 2, 0, 1) / det,
            -c(0, 2, 0, 1) / det,
            c(0, 1, 0, 1) / det,
        ],
    ])
}

#[inline]
fn project(h: &Mat3, p: [f64; 2]) -> Option<[f64; 2]> {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    if w.abs() < 1e-12 {
        return None;
    }
    Some([
        (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
        (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
    ])
}

/// Current parameter vector, unpacked.
#[derive(Clone, Copy, Debug)]
struct Params {
    p: [f64; 2],
    c: [[f64; 2]; 2],
    log_sigma: f64,
    /// `B` then `A` coefficients, see [`lin_row`]; inactive ones stay zero.
    lin: [f64; N_LIN],
    radii: PrintRadii,
}

/// One patch pixel.
#[derive(Clone, Copy)]
struct PatchPixel {
    x: [f64; 2],
    /// Normalized offset from the seed centre used by the affine B and A.
    xi: [f64; 2],
    value: f64,
}

/// Linear-parameter regressors at a pixel with ink coverage `p`:
/// `B` on the monomials of `(xi, eta)` up to degree 3, `A` on `[1, xi, eta]`,
/// with the model `B - A p`.
#[inline]
fn lin_row(xi: [f64; 2], p: f64) -> [f64; N_LIN] {
    let [u, v] = xi;
    [
        1.0,
        u,
        v,
        u * u,
        u * v,
        v * v,
        u * u * u,
        u * u * v,
        u * v * v,
        v * v * v,
        -p,
        -p * u,
        -p * v,
    ]
}

/// Photometric model value and the local amplitude `A` at a pixel.
#[inline]
fn photometry(l: &[f64; N_LIN], xi: [f64; 2], p: f64) -> (f64, f64) {
    let row = lin_row(xi, p);
    let value = row.iter().zip(l).map(|(r, c)| r * c).sum();
    let a = l[CONTRAST] + l[CONTRAST + 1] * xi[0] + l[CONTRAST + 2] * xi[1];
    (value, a)
}

/// Value and gradient of the ink-coverage pattern at one pixel.
struct PatternEval {
    p: f64,
    /// dP/dy (the pre-image coordinate `y = h0 + C (x - p)`).
    grad_y: [f64; 2],
    dlog_sigma: f64,
    /// dP/dr0, dP/dr1, dP/dr2.
    d_radii: [f64; 3],
}

/// Transition of the angular code indicator at a sector boundary.
#[derive(Clone, Copy)]
struct Transition {
    /// Unit direction of the boundary ray (board frame).
    dir: [f64; 2],
    /// White(after) - white(before), counter-clockwise.
    delta: f64,
}

struct Model<'a> {
    h_inv: Mat3,
    h0: [f64; 2],
    bits: [f64; CODE_SECTORS],
    transitions: Vec<Transition>,
    neighbors: &'a [[f64; 2]],
}

impl Model<'_> {
    /// `u = H^-1(y)` and its Jacobian `du/dy`.
    #[inline]
    fn unproject(&self, y: [f64; 2]) -> Option<([f64; 2], [[f64; 2]; 2])> {
        let h = &self.h_inv;
        let uu = h[0][0] * y[0] + h[0][1] * y[1] + h[0][2];
        let vv = h[1][0] * y[0] + h[1][1] * y[1] + h[1][2];
        let ww = h[2][0] * y[0] + h[2][1] * y[1] + h[2][2];
        if ww.abs() < 1e-12 {
            return None;
        }
        let u = [uu / ww, vv / ww];
        let j = [
            [
                (h[0][0] - u[0] * h[2][0]) / ww,
                (h[0][1] - u[0] * h[2][1]) / ww,
            ],
            [
                (h[1][0] - u[1] * h[2][0]) / ww,
                (h[1][1] - u[1] * h[2][1]) / ww,
            ],
        ];
        Some((u, j))
    }

    /// Ink coverage and derivatives at pixel `x`.
    fn pattern(&self, x: [f64; 2], prm: &Params) -> Option<PatternEval> {
        let sigma = prm.log_sigma.exp();
        let dx = [x[0] - prm.p[0], x[1] - prm.p[1]];
        let c = &prm.c;
        let y = [
            self.h0[0] + c[0][0] * dx[0] + c[0][1] * dx[1],
            self.h0[1] + c[1][0] * dx[0] + c[1][1] * dx[1],
        ];
        let (u, j) = self.unproject(y)?;

        // Covector `dq/dy` for a board-space linear functional with gradient
        // `n` (board frame), and its image-space norm `|C^T J^T n|`.
        let covector = |n: [f64; 2]| -> ([f64; 2], f64) {
            let v = [
                j[0][0] * n[0] + j[1][0] * n[1],
                j[0][1] * n[0] + j[1][1] * n[1],
            ];
            let g = [
                c[0][0] * v[0] + c[1][0] * v[1],
                c[0][1] * v[0] + c[1][1] * v[1],
            ];
            (v, g[0].hypot(g[1]))
        };

        let rho = u[0].hypot(u[1]);
        let mut out = PatternEval {
            p: 0.0,
            grad_y: [0.0; 2],
            dlog_sigma: 0.0,
            d_radii: [0.0; 3],
        };
        if rho < 1e-9 {
            // At the exact centre the pattern is flat paper.
            return Some(out);
        }
        let n_r = [u[0] / rho, u[1] / rho];
        let (v_r, gnorm) = covector(n_r);
        if gnorm < 1e-12 {
            return None;
        }
        // Image px per board mm along the radial direction is 1 / (dρ/dx).
        let inv_g = 1.0 / gnorm;
        let rq = [v_r[0] * inv_g, v_r[1] * inv_g]; // d(dist)/dy for radial edges
        let r = &prm.radii;
        let s = |k: usize| step_curved((rho - r[k]) * inv_g, sigma, r[k] * inv_g);

        // Annulus [r0, r3]: Phi(d0) - Phi(d3).
        let s0 = s(0);
        let s3 = s(3);
        out.p = s0.value - s3.value;
        let mut radial_slope = s0.slope - s3.slope;
        out.dlog_sigma = s0.dlog_sigma - s3.dlog_sigma;
        out.d_radii[0] = -s0.slope * inv_g;

        // Board-mm reach of a blurred edge (gnorm is mm per image px), with
        // slack for the anisotropy of the local map.
        let reach_mm = EDGE_REACH_SIGMAS * sigma * gnorm * 1.5;

        // Code band [r1, r2] times the angular paper indicator.
        let s1 = s(1);
        let s2 = s(2);
        let band = s1.value - s2.value;
        if band > 1e-9 || s1.slope != 0.0 || s2.slope != 0.0 {
            let (theta, dtheta_y, dtheta_ls) = self.angular(u, sigma, reach_mm, &covector);
            out.p -= band * theta;
            radial_slope -= (s1.slope - s2.slope) * theta;
            out.dlog_sigma -= (s1.dlog_sigma - s2.dlog_sigma) * theta + band * dtheta_ls;
            out.grad_y[0] -= band * dtheta_y[0];
            out.grad_y[1] -= band * dtheta_y[1];
            out.d_radii[1] = s1.slope * inv_g * theta;
            out.d_radii[2] = -s2.slope * inv_g * theta;
        }
        out.grad_y[0] += radial_slope * rq[0];
        out.grad_y[1] += radial_slope * rq[1];

        // Outer rings of the neighbours.
        let reach_sq = (r[3] + reach_mm) * (r[3] + reach_mm);
        for nb in self.neighbors {
            let un = [u[0] - nb[0], u[1] - nb[1]];
            let rho_sq = un[0] * un[0] + un[1] * un[1];
            if rho_sq > reach_sq || rho_sq < 1e-18 {
                continue;
            }
            let rho_n = rho_sq.sqrt();
            let (v_n, g_n) = covector([un[0] / rho_n, un[1] / rho_n]);
            if g_n < 1e-12 {
                continue;
            }
            let sn = step_curved((rho_n - r[3]) / g_n, sigma, r[3] / g_n);
            out.p += 1.0 - sn.value;
            out.grad_y[0] -= sn.slope * v_n[0] / g_n;
            out.grad_y[1] -= sn.slope * v_n[1] / g_n;
            out.dlog_sigma -= sn.dlog_sigma;
        }
        Some(out)
    }

    /// Blurred paper indicator of the code sectors at board point `u`, with its
    /// gradient in `y` and derivative in log sigma.
    fn angular(
        &self,
        u: [f64; 2],
        sigma: f64,
        reach_mm: f64,
        covector: &dyn Fn([f64; 2]) -> ([f64; 2], f64),
    ) -> (f64, [f64; 2], f64) {
        let angle = u[1].atan2(u[0]);
        let pos =
            (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * CODE_SECTORS as f64;
        let sector = (pos.floor() as isize).rem_euclid(CODE_SECTORS as isize) as usize;
        let mut theta = self.bits[sector];
        let mut grad = [0.0; 2];
        let mut dls = 0.0;
        for tr in &self.transitions {
            // Only the boundary ray on this side of the centre.
            if tr.dir[0] * u[0] + tr.dir[1] * u[1] <= 0.0 {
                continue;
            }
            // Signed board distance, positive counter-clockwise of the ray.
            let n = [-tr.dir[1], tr.dir[0]];
            let t_mm = n[0] * u[0] + n[1] * u[1];
            if t_mm.abs() > reach_mm {
                continue;
            }
            let (v, g) = covector(n);
            if g < 1e-12 {
                continue;
            }
            let t = t_mm / g;
            if t.abs() > EDGE_REACH_SIGMAS * sigma {
                continue;
            }
            let st = step(t, sigma);
            let heaviside = if t >= 0.0 { 1.0 } else { 0.0 };
            theta += tr.delta * (st.value - heaviside);
            grad[0] += tr.delta * st.slope * v[0] / g;
            grad[1] += tr.delta * st.slope * v[1] / g;
            dls += tr.delta * st.dlog_sigma;
        }
        (theta, grad, dls)
    }
}

fn build_model<'a>(problem: &'a PhotometricProblem<'_>) -> Option<Model<'a>> {
    let h_inv = mat3_inverse(&problem.local_map)?;
    let h0 = project(&problem.local_map, [0.0, 0.0])?;
    let mut bits = [0.0; CODE_SECTORS];
    for (s, b) in bits.iter_mut().enumerate() {
        *b = f64::from((problem.codeword >> s) & 1);
    }
    let dtheta = 2.0 * std::f64::consts::PI / CODE_SECTORS as f64;
    let transitions = (0..CODE_SECTORS)
        .filter_map(|s| {
            let before = bits[(s + CODE_SECTORS - 1) % CODE_SECTORS];
            let delta = bits[s] - before;
            (delta != 0.0).then(|| {
                let a = -std::f64::consts::PI + s as f64 * dtheta;
                Transition {
                    dir: [a.cos(), a.sin()],
                    delta,
                }
            })
        })
        .collect();
    Some(Model {
        h_inv,
        h0,
        bits,
        transitions,
        neighbors: problem.neighbors_mm,
    })
}

/// Gather the patch: pixels whose seed board radius is within the marker plus
/// margin, minus saturated pixels and their 8-neighbours.
fn collect_patch(problem: &PhotometricProblem<'_>, model: &Model<'_>) -> (Vec<PatchPixel>, f64) {
    let gray = problem.gray;
    let (w, h) = gray.dimensions();
    let r_max = problem.radii[3] + problem.patch_margin_mm;
    // Image extent of the patch from the seed map.
    let mut reach = 0.0f64;
    for k in 0..16 {
        let a = k as f64 * std::f64::consts::PI / 8.0;
        if let Some(q) = project(&problem.local_map, [r_max * a.cos(), r_max * a.sin()]) {
            reach = reach.max((q[0] - model.h0[0]).hypot(q[1] - model.h0[1]));
        }
    }
    let reach = reach + 1.0;
    let c = problem.center_seed;
    let x0 = (c[0] - reach).floor().max(0.0) as i64;
    let x1 = (c[0] + reach).ceil().min(f64::from(w) - 1.0) as i64;
    let y0 = (c[1] - reach).floor().max(0.0) as i64;
    let y1 = (c[1] + reach).ceil().min(f64::from(h) - 1.0) as i64;
    let sat = problem.saturation_level;
    let saturated_near = |x: i64, y: i64| {
        for yy in (y - 1).max(0)..=(y + 1).min(i64::from(h) - 1) {
            for xx in (x - 1).max(0)..=(x + 1).min(i64::from(w) - 1) {
                if gray.get_pixel(xx as u32, yy as u32)[0] >= sat {
                    return true;
                }
            }
        }
        false
    };
    let mut pixels = Vec::new();
    let mut n_masked = 0usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let xy = [x as f64, y as f64];
            // Seed pre-image: y = h0 + (x - c).
            let pre = [model.h0[0] + xy[0] - c[0], model.h0[1] + xy[1] - c[1]];
            let Some((u, _)) = model.unproject(pre) else {
                continue;
            };
            if u[0].hypot(u[1]) > r_max {
                continue;
            }
            if saturated_near(x, y) {
                n_masked += 1;
                continue;
            }
            pixels.push(PatchPixel {
                x: xy,
                xi: [(xy[0] - c[0]) / reach, (xy[1] - c[1]) / reach],
                value: f64::from(gray.get_pixel(x as u32, y as u32)[0]),
            });
        }
    }
    let masked_fraction = n_masked as f64 / (n_masked + pixels.len()).max(1) as f64;
    (pixels, masked_fraction)
}

/// Layout of the full parameter vector: geometry, the active linear
/// coefficients, then the free radii.
struct Layout {
    fit_radii: bool,
    lin: &'static [usize],
}

impl Layout {
    fn new(fit_radii: bool, cubic: bool) -> Self {
        Self {
            fit_radii,
            lin: if cubic { &LIN_CUBIC } else { &LIN_AFFINE },
        }
    }
    fn radii_base(&self) -> usize {
        N_GEOM + self.lin.len()
    }
    fn n(&self) -> usize {
        self.radii_base() + if self.fit_radii { 3 } else { 0 }
    }
    fn pack(&self, prm: &Params) -> DVector<f64> {
        let mut v = DVector::zeros(self.n());
        v[0] = prm.p[0];
        v[1] = prm.p[1];
        v[2] = prm.c[0][0];
        v[3] = prm.c[0][1];
        v[4] = prm.c[1][0];
        v[5] = prm.c[1][1];
        v[6] = prm.log_sigma;
        for (k, &li) in self.lin.iter().enumerate() {
            v[N_GEOM + k] = prm.lin[li];
        }
        if self.fit_radii {
            for k in 0..3 {
                v[self.radii_base() + k] = prm.radii[k];
            }
        }
        v
    }
    fn unpack(&self, v: &DVector<f64>, template: &Params) -> Params {
        let mut prm = *template;
        prm.p = [v[0], v[1]];
        prm.c = [[v[2], v[3]], [v[4], v[5]]];
        prm.log_sigma = v[6];
        for (k, &li) in self.lin.iter().enumerate() {
            prm.lin[li] = v[N_GEOM + k];
        }
        if self.fit_radii {
            for k in 0..3 {
                prm.radii[k] = v[self.radii_base() + k];
            }
        }
        prm
    }
}

/// Maximum parameter count (geometry, photometry, three free radii).
const N_MAX: usize = N_GEOM + N_LIN + 3;

/// Robust normal equations over the patch at one parameter point.
struct Accumulated {
    residuals: Vec<f64>,
    /// `J^T W J` (no noise scaling).
    jtwj: DMatrix<f64>,
    /// `J^T W r`.
    jtwr: DVector<f64>,
    /// `sum huber(r)` at the scale used for the weights.
    robust_cost: f64,
    /// `sum w r^2` and `sum w`, for the residual variance.
    wrss: f64,
    wsum: f64,
}

/// Evaluate the model over the patch and accumulate the Huber-weighted normal
/// equations in one pass (no per-pixel Jacobian storage).
fn accumulate(
    model: &Model<'_>,
    pixels: &[PatchPixel],
    prm: &Params,
    layout: &Layout,
    scale: f64,
) -> Option<Accumulated> {
    let n = layout.n();
    let k_huber = HUBER_K * scale;
    let mut residuals = Vec::with_capacity(pixels.len());
    let mut h = [[0.0f64; N_MAX]; N_MAX];
    let mut g = [0.0f64; N_MAX];
    let mut robust_cost = 0.0;
    let mut wrss = 0.0;
    let mut wsum = 0.0;
    let c = &prm.c;
    let l = &prm.lin;
    let mut row = [0.0f64; N_MAX];
    for px in pixels {
        let pe = model.pattern(px.x, prm)?;
        let (value, a) = photometry(l, px.xi, pe.p);
        let r = px.value - value;
        residuals.push(r);
        let ar = r.abs();
        let w = if ar <= k_huber { 1.0 } else { k_huber / ar };
        robust_cost += if ar <= k_huber {
            0.5 * r * r
        } else {
            k_huber * (ar - 0.5 * k_huber)
        };
        wrss += w * r * r;
        wsum += w;

        let gy = pe.grad_y;
        let dx = [px.x[0] - prm.p[0], px.x[1] - prm.p[1]];
        // dy/dp = -C  =>  dI/dp = A * C^T grad_y.
        row[0] = a * (c[0][0] * gy[0] + c[1][0] * gy[1]);
        row[1] = a * (c[0][1] * gy[0] + c[1][1] * gy[1]);
        // dy_i/dC_ij = dx_j.
        row[2] = -a * gy[0] * dx[0];
        row[3] = -a * gy[0] * dx[1];
        row[4] = -a * gy[1] * dx[0];
        row[5] = -a * gy[1] * dx[1];
        row[6] = -a * pe.dlog_sigma;
        let full = lin_row(px.xi, pe.p);
        for (k, &li) in layout.lin.iter().enumerate() {
            row[N_GEOM + k] = full[li];
        }
        if layout.fit_radii {
            let base = layout.radii_base();
            for k in 0..3 {
                row[base + k] = -a * pe.d_radii[k];
            }
        }
        for i in 0..n {
            let wi = w * row[i];
            g[i] += wi * r;
            for j in 0..=i {
                h[i][j] += wi * row[j];
            }
        }
    }
    let jtwj = DMatrix::from_fn(n, n, |i, j| if j <= i { h[i][j] } else { h[j][i] });
    let jtwr = DVector::from_fn(n, |i, _| g[i]);
    Some(Accumulated {
        residuals,
        jtwj,
        jtwr,
        robust_cost,
        wrss,
        wsum,
    })
}

fn median_abs(v: &[f64]) -> f64 {
    let mut a: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    if a.is_empty() {
        return 0.0;
    }
    let mid = a.len() / 2;
    a.select_nth_unstable_by(mid, |x, y| x.total_cmp(y));
    a[mid]
}

/// Prior information (diagonal) and mean for the regularized parameters.
fn prior_terms(layout: &Layout, prior: &Params) -> (DVector<f64>, DVector<f64>) {
    let n = layout.n();
    let mut info = DVector::zeros(n);
    let mean = layout.pack(prior);
    let ic = 1.0 / (C_PRIOR_SIGMA * C_PRIOR_SIGMA);
    for k in 2..6 {
        info[k] = ic;
    }
    if layout.fit_radii {
        let ir = 1.0 / (RADIUS_PRIOR_SIGMA_MM * RADIUS_PRIOR_SIGMA_MM);
        for k in 0..3 {
            info[layout.radii_base() + k] = ir;
        }
    }
    (info, mean)
}

/// Solve the linear photometric parameters for a fixed geometry by least
/// squares; returns the parameters with the solution and the residual sum of
/// squares.
fn solve_linear(
    model: &Model<'_>,
    pixels: &[PatchPixel],
    prm: &Params,
    layout: &Layout,
) -> Option<(Params, f64)> {
    let mut ata = nalgebra::SMatrix::<f64, N_LIN, N_LIN>::zeros();
    let mut atb = nalgebra::SVector::<f64, N_LIN>::zeros();
    let mut btb = 0.0;
    for px in pixels {
        let pe = model.pattern(px.x, prm)?;
        let row = nalgebra::SVector::<f64, N_LIN>::from(lin_row(px.xi, pe.p));
        ata += row * row.transpose();
        atb += row * px.value;
        btb += px.value * px.value;
    }
    // Inactive coefficients are pinned to zero.
    for k in 0..N_LIN {
        if !layout.lin.contains(&k) {
            ata.row_mut(k).fill(0.0);
            ata.column_mut(k).fill(0.0);
            ata[(k, k)] = 1.0;
            atb[k] = 0.0;
        }
    }
    let sol = ata.cholesky()?.solve(&atb);
    let mut out = *prm;
    for k in 0..N_LIN {
        out.lin[k] = sol[k];
    }
    Some((out, btb - sol.dot(&atb)))
}

/// The fixed parts of one marker's fit.
struct Fitting<'a> {
    model: Model<'a>,
    pixels: Vec<PatchPixel>,
    layout: Layout,
    prior_info: DVector<f64>,
    prior_mean: DVector<f64>,
}

/// Optimizer state at an accepted point.
struct Solution {
    prm: Params,
    acc: Accumulated,
    iterations: usize,
    converged: bool,
}

impl Fitting<'_> {
    fn accumulate(&self, prm: &Params, scale: f64) -> Result<Accumulated, PhotometricFailure> {
        accumulate(&self.model, &self.pixels, prm, &self.layout, scale)
            .ok_or(PhotometricFailure::DegenerateMap)
    }

    fn prior_cost(&self, v: &DVector<f64>) -> f64 {
        0.5 * (v - &self.prior_mean)
            .iter()
            .zip(self.prior_info.iter())
            .map(|(d, w)| d * d * w)
            .sum::<f64>()
    }

    /// Blur and photometry from a coarse search over sigma with the geometry
    /// at the seed (the linear parameters are solved exactly per sigma).
    fn initial_params(&self, seed: &Params) -> Result<Params, PhotometricFailure> {
        let mut best: Option<(Params, f64)> = None;
        for s in SIGMA_GRID {
            let trial = Params {
                log_sigma: s.ln(),
                ..*seed
            };
            if let Some((prm, cost)) = solve_linear(&self.model, &self.pixels, &trial, &self.layout)
                && best.as_ref().is_none_or(|b| cost < b.1)
            {
                best = Some((prm, cost));
            }
        }
        best.map(|b| b.0).ok_or(PhotometricFailure::SingularSystem)
    }

    /// Levenberg–Marquardt on the Huber-weighted, prior-regularized cost. The
    /// robust scale is re-estimated during the first iterations only, so the
    /// cost stays comparable between accepted steps afterwards.
    fn solve(
        &self,
        mut prm: Params,
        max_iterations: usize,
    ) -> Result<Solution, PhotometricFailure> {
        let n = self.layout.n();
        let initial = self.accumulate(&prm, 1.0)?;
        let mut scale = (1.4826 * median_abs(&initial.residuals)).max(0.5);
        let mut acc = self.accumulate(&prm, scale)?;
        let mut x = self.layout.pack(&prm);
        let mut cost = acc.robust_cost / (scale * scale) + self.prior_cost(&x);
        let mut lambda = 1e-3;
        let mut iterations = 0;
        let mut converged = false;
        while iterations < max_iterations {
            iterations += 1;
            // (J^T W J / s^2 + P) dx = J^T W r / s^2 - P (x - m).
            let inv_s2 = 1.0 / (scale * scale);
            let mut hmat = &acc.jtwj * inv_s2;
            let mut grad = &acc.jtwr * inv_s2;
            for k in 0..n {
                hmat[(k, k)] += self.prior_info[k];
                grad[k] -= self.prior_info[k] * (x[k] - self.prior_mean[k]);
            }
            let mut accepted = false;
            for _ in 0..8 {
                let mut damped = hmat.clone();
                for k in 0..n {
                    damped[(k, k)] += lambda * hmat[(k, k)].max(1e-12);
                }
                let Some(chol) = damped.cholesky() else {
                    lambda *= 10.0;
                    continue;
                };
                let delta = chol.solve(&grad);
                let x_new = &x + &delta;
                let prm_new = self.layout.unpack(&x_new, &prm);
                let Ok(acc_new) = self.accumulate(&prm_new, scale) else {
                    lambda *= 10.0;
                    continue;
                };
                let cost_new = acc_new.robust_cost * inv_s2 + self.prior_cost(&x_new);
                if cost_new.is_finite() && cost_new <= cost {
                    converged = delta[0].abs().max(delta[1].abs()) < 1e-4
                        && delta[6].abs() < 1e-4
                        && (cost - cost_new) <= 1e-9 * cost.max(1.0);
                    x = x_new;
                    prm = prm_new;
                    acc = acc_new;
                    cost = cost_new;
                    lambda = (lambda * 0.3).max(1e-7);
                    accepted = true;
                    break;
                }
                lambda *= 10.0;
            }
            if !accepted {
                // No downhill step: at a minimum to numerical precision.
                converged = true;
            }
            let new_scale = (1.4826 * median_abs(&acc.residuals)).max(0.5);
            if iterations <= SCALE_UPDATE_ITERATIONS && (new_scale - scale).abs() > 0.02 * scale {
                scale = new_scale;
                acc = self.accumulate(&prm, scale)?;
                cost = acc.robust_cost / (scale * scale) + self.prior_cost(&x);
                converged = false;
            }
            if converged {
                break;
            }
        }
        Ok(Solution {
            prm,
            acc,
            iterations,
            converged,
        })
    }

    /// Parameter covariance: inverse Fisher information with the noise
    /// variance estimated from the weighted residuals; also returns that
    /// variance.
    fn covariance(&self, acc: &Accumulated) -> Result<(DMatrix<f64>, f64), PhotometricFailure> {
        let n = self.layout.n();
        let dof = (acc.wsum - n as f64).max(1.0);
        let s2 = (acc.wrss / dof).max(1e-6);
        let mut info = &acc.jtwj / s2;
        for k in 0..n {
            info[(k, k)] += self.prior_info[k];
        }
        let cov = info
            .cholesky()
            .ok_or(PhotometricFailure::SingularSystem)?
            .inverse();
        Ok((cov, s2))
    }
}

/// Fit one marker.
pub(crate) fn fit_marker(
    problem: &PhotometricProblem<'_>,
) -> Result<PhotometricFit, PhotometricFailure> {
    let model = build_model(problem).ok_or(PhotometricFailure::DegenerateMap)?;
    let (pixels, masked_fraction) = collect_patch(problem, &model);
    let layout = Layout::new(problem.fit_radii, problem.cubic_background);
    if pixels.len() < 4 * layout.n() + 32 {
        return Err(PhotometricFailure::InsufficientPixels);
    }
    let seed = Params {
        p: problem.center_seed,
        c: [[1.0, 0.0], [0.0, 1.0]],
        log_sigma: 0.0,
        lin: [0.0; N_LIN],
        radii: problem.radii,
    };
    let (prior_info, prior_mean) = prior_terms(&layout, &seed);
    let fitting = Fitting {
        model,
        pixels,
        layout,
        prior_info,
        prior_mean,
    };
    let start = fitting.initial_params(&seed)?;
    let sol = fitting.solve(start, problem.max_iterations)?;
    let (cov, s2) = fitting.covariance(&sol.acc)?;
    let radii_std = fitting.layout.fit_radii.then(|| {
        let b = fitting.layout.radii_base();
        [
            cov[(b, b)].sqrt(),
            cov[(b + 1, b + 1)].sqrt(),
            cov[(b + 2, b + 2)].sqrt(),
        ]
    });
    Ok(PhotometricFit {
        center: sol.prm.p,
        center_cov: [[cov[(0, 0)], cov[(0, 1)]], [cov[(1, 0)], cov[(1, 1)]]],
        blur_sigma_px: sol.prm.log_sigma.exp(),
        contrast: sol.prm.lin[CONTRAST],
        residual_sigma: s2.sqrt(),
        n_pixels: fitting.pixels.len(),
        masked_fraction,
        iterations: sol.iterations,
        converged: sol.converged,
        radii: sol.prm.radii,
        radii_std,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use image::Luma;
    use rand::{RngExt, SeedableRng};

    /// Nominal default-board print: centerline radii 3.2 / 4.8 mm, 1.152 mm strokes.
    pub(super) const RADII: PrintRadii = [2.624, 3.776, 4.224, 5.376];
    pub(super) const CODEWORD: u16 = 0b1011_0010_1110_0100;
    pub(super) const PITCH_SPACING_MM: f64 = 13.856_406;

    pub(super) fn neighbors() -> Vec<[f64; 2]> {
        (0..6)
            .map(|k| {
                let a = k as f64 * std::f64::consts::PI / 3.0;
                [PITCH_SPACING_MM * a.cos(), PITCH_SPACING_MM * a.sin()]
            })
            .collect()
    }

    /// Board-local mm → image px for a board tilted by `tilt_deg`, ≈3.45 px/mm,
    /// marker imaged near (40, 40) of the patch.
    pub(super) fn true_map(tilt_deg: f64, center: [f64; 2]) -> Mat3 {
        let f = 1400.0;
        let z = f / 3.45;
        let (st, ct) = tilt_deg.to_radians().sin_cos();
        // Rotation about the axis (cos 30°, sin 30°, 0), then a 10° roll.
        let (ax, ay) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let t = [
            [ct + ax * ax * (1.0 - ct), ax * ay * (1.0 - ct), ay * st],
            [ax * ay * (1.0 - ct), ct + ay * ay * (1.0 - ct), -ax * st],
            [-ay * st, ax * st, ct],
        ];
        let (sr, cr) = 10f64.to_radians().sin_cos();
        let r = [
            [cr * t[0][0] - sr * t[1][0], cr * t[0][1] - sr * t[1][1]],
            [sr * t[0][0] + cr * t[1][0], sr * t[0][1] + cr * t[1][1]],
            [t[2][0], t[2][1]],
        ];
        // The marker sits 60 mm off-axis so the projective terms are not zero.
        let off = [60.0, -40.0];
        let tx = r[0][0] * off[0] + r[0][1] * off[1];
        let ty = r[1][0] * off[0] + r[1][1] * off[1];
        let tz = z + r[2][0] * off[0] + r[2][1] * off[1];
        // Image of the marker centre: (f tx / tz, f ty / tz); shift it to `center`.
        let sx = center[0] - f * tx / tz;
        let sy = center[1] - f * ty / tz;
        [
            [
                f * r[0][0] + sx * r[2][0],
                f * r[0][1] + sx * r[2][1],
                f * tx + sx * tz,
            ],
            [
                f * r[1][0] + sy * r[2][0],
                f * r[1][1] + sy * r[2][1],
                f * ty + sy * tz,
            ],
            [r[2][0], r[2][1], tz],
        ]
    }

    fn ink(u: [f64; 2], code: u16) -> f64 {
        let rho = u[0].hypot(u[1]);
        for nb in neighbors() {
            if (u[0] - nb[0]).hypot(u[1] - nb[1]) <= RADII[3] {
                return 1.0;
            }
        }
        if rho < RADII[0] || rho > RADII[3] {
            return 0.0;
        }
        if rho > RADII[1] && rho < RADII[2] {
            let a = u[1].atan2(u[0]);
            let s =
                (((a + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 16.0).floor() as i64;
            let s = s.rem_euclid(16) as u16;
            return if (code >> s) & 1 == 1 { 0.0 } else { 1.0 };
        }
        1.0
    }

    /// Exact-geometry render: 8×8 supersampled coverage, Gaussian blur,
    /// affine illumination, noise, 8-bit quantization.
    fn render(map: &Mat3, size: u32, blur: f64, noise: f64, seed: u64, code: u16) -> GrayImage {
        quantize(&radiance(map, size, blur, code), size, noise, seed)
    }

    /// Noise-free radiance of the patch, row-major.
    pub(super) fn radiance(map: &Mat3, size: u32, blur: f64, code: u16) -> Vec<f64> {
        let inv = mat3_inverse(map).unwrap();
        let pad = (5.0 * blur).ceil() as i64 + 2;
        let n = size as i64 + 2 * pad;
        let mut cov = vec![0.0f64; (n * n) as usize];
        let ss = 8;
        for y in 0..n {
            for x in 0..n {
                let mut acc = 0.0;
                for sy in 0..ss {
                    for sx in 0..ss {
                        let px = (x - pad) as f64 + (sx as f64 + 0.5) / ss as f64 - 0.5;
                        let py = (y - pad) as f64 + (sy as f64 + 0.5) / ss as f64 - 0.5;
                        acc += ink(project(&inv, [px, py]).unwrap(), code);
                    }
                }
                cov[(y * n + x) as usize] = acc / (ss * ss) as f64;
            }
        }
        if blur > 0.0 {
            let r = (4.0 * blur).ceil() as i64;
            let k: Vec<f64> = (-r..=r)
                .map(|i| (-((i * i) as f64) / (2.0 * blur * blur)).exp())
                .collect();
            let ks: f64 = k.iter().sum();
            let mut tmp = vec![0.0; cov.len()];
            for y in 0..n {
                for x in 0..n {
                    let mut a = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let xx = (x + j as i64 - r).clamp(0, n - 1);
                        a += kv * cov[(y * n + xx) as usize];
                    }
                    tmp[(y * n + x) as usize] = a / ks;
                }
            }
            for y in 0..n {
                for x in 0..n {
                    let mut a = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let yy = (y + j as i64 - r).clamp(0, n - 1);
                        a += kv * tmp[(yy * n + x) as usize];
                    }
                    cov[(y * n + x) as usize] = a / ks;
                }
            }
        }
        let mut out = Vec::with_capacity((size * size) as usize);
        for y in 0..size as i64 {
            for x in 0..size as i64 {
                let c = cov[((y + pad) * n + x + pad) as usize];
                let illum = 1.0 + 0.004 * (x as f64 - 40.0);
                out.push(105.0 * illum - 40.0 * illum * c);
            }
        }
        out
    }

    /// Add Gaussian noise and quantize to 8 bits.
    pub(super) fn quantize(radiance: &[f64], size: u32, noise: f64, seed: u64) -> GrayImage {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut img = GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let v0 = radiance[(y * size + x) as usize];
                let u1: f64 = rng.random_range(f64::EPSILON..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let v = (v0 + noise * g).round().clamp(0.0, 255.0);
                img.put_pixel(x, y, Luma([v as u8]));
            }
        }
        img
    }

    /// A seed map with a 2 % scale and 2° rotation error, as a board
    /// homography fitted over the whole image would carry locally.
    pub(super) fn perturbed(map: &Mat3) -> Mat3 {
        let (s, c) = (
            2f64.to_radians().sin() * 1.02,
            2f64.to_radians().cos() * 1.02,
        );
        let a = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = (0..3).map(|k| map[i][k] * a[k][j]).sum();
            }
        }
        out
    }

    pub(super) fn problem<'a>(
        gray: &'a GrayImage,
        map: Mat3,
        seed: [f64; 2],
        nbs: &'a [[f64; 2]],
    ) -> PhotometricProblem<'a> {
        PhotometricProblem {
            gray,
            local_map: map,
            center_seed: seed,
            codeword: CODEWORD,
            radii: RADII,
            fit_radii: false,
            cubic_background: false,
            neighbors_mm: nbs,
            patch_margin_mm: 1.5,
            saturation_level: 250,
            max_iterations: 30,
        }
    }

    #[test]
    fn norm_cdf_matches_reference_values() {
        for (z, want) in [
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_543),
            (-2.0, 0.022_750_131_948_179),
            (3.0, 0.998_650_101_968_37),
        ] {
            let (cdf, pdf) = norm_cdf_pdf(z);
            assert!((cdf - want).abs() < 2e-7, "Phi({z})");
            let want_pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            assert!((pdf - want_pdf).abs() < 1e-15, "phi({z})");
        }
    }

    #[test]
    fn noiseless_fit_recovers_the_true_center_across_blur() {
        let nbs = neighbors();
        for (blur, tilt) in [
            (0.8, 0.0),
            (1.5, 20.0),
            (3.0, 0.0),
            (3.0, 30.0),
            (4.0, 0.0),
            (4.0, 30.0),
        ] {
            let truth = [40.3, 39.6];
            let map = true_map(tilt, truth);
            for code in [CODEWORD, 0u16, 0xFFFF] {
                let gray = render(&map, 80, blur, 0.0, 1, code);
                let seed = [truth[0] + 0.6, truth[1] - 0.4];
                let mut pb = problem(&gray, perturbed(&map), seed, &nbs);
                pb.codeword = code;
                let fit = fit_marker(&pb).unwrap();
                let err = (fit.center[0] - truth[0]).hypot(fit.center[1] - truth[1]);
                eprintln!(
                    "blur {blur} tilt {tilt} code {code:04x}: err {err:.4} px, sigma {:.3}, iters {}, conv {}",
                    fit.blur_sigma_px, fit.iterations, fit.converged
                );
                assert!(err < 0.5, "blur {blur}: centre error {err} px");
                assert!((fit.blur_sigma_px - blur.max(0.3)).abs() < 0.25 * blur + 0.15);
            }
        }
    }
}

#[cfg(test)]
mod monte_carlo {
    use super::tests::*;
    use super::*;

    /// The reported centre covariance matches the scatter of repeated fits
    /// under independent noise (σ = 2 grey levels, 40 levels of contrast).
    #[test]
    fn reported_covariance_matches_monte_carlo_scatter() {
        let nbs = neighbors();
        let truth = [40.3, 39.6];
        for (blur, tilt) in [(1.0, 10.0), (3.0, 30.0)] {
            let map = true_map(tilt, truth);
            let rad = radiance(&map, 80, blur, CODEWORD);
            let n = 120;
            let mut centers = Vec::with_capacity(n);
            let mut trace_sum = 0.0;
            for seed in 0..n as u64 {
                let gray = quantize(&rad, 80, 2.0, 1000 + seed);
                let fit = fit_marker(&problem(&gray, perturbed(&map), [40.8, 39.2], &nbs)).unwrap();
                centers.push(fit.center);
                trace_sum += fit.center_cov[0][0] + fit.center_cov[1][1];
            }
            let m = [
                centers.iter().map(|c| c[0]).sum::<f64>() / n as f64,
                centers.iter().map(|c| c[1]).sum::<f64>() / n as f64,
            ];
            let mut emp = [[0.0; 2]; 2];
            for c in &centers {
                let d = [c[0] - m[0], c[1] - m[1]];
                for i in 0..2 {
                    for j in 0..2 {
                        emp[i][j] += d[i] * d[j] / (n - 1) as f64;
                    }
                }
            }
            let rep_tr = trace_sum / n as f64;
            let emp_tr = emp[0][0] + emp[1][1];
            let bias = (m[0] - truth[0]).hypot(m[1] - truth[1]);
            eprintln!(
                "blur {blur}: bias {bias:.4} px, empirical rms {:.4} px, reported rms {:.4} px, ratio {:.3}",
                emp_tr.sqrt(),
                rep_tr.sqrt(),
                (rep_tr / emp_tr).sqrt()
            );
            let ratio = (rep_tr / emp_tr).sqrt();
            assert!(
                (0.77..=1.3).contains(&ratio),
                "blur {blur}: sigma ratio {ratio}"
            );
        }
    }
}

#[cfg(test)]
mod timing {
    use super::tests::*;
    use super::*;

    #[test]
    #[ignore = "timing probe"]
    fn fit_time_per_marker() {
        let nbs = neighbors();
        let truth = [40.3, 39.6];
        for blur in [1.0, 3.0] {
            let map = true_map(20.0, truth);
            let gray = quantize(&radiance(&map, 80, blur, CODEWORD), 80, 2.0, 7);
            for fit_radii in [false, true] {
                let mut pb = problem(&gray, perturbed(&map), [40.8, 39.2], &nbs);
                pb.fit_radii = fit_radii;
                let n = 200;
                let t0 = std::time::Instant::now();
                let mut it = 0;
                for _ in 0..n {
                    it += fit_marker(&pb).unwrap().iterations;
                }
                eprintln!(
                    "blur {blur} fit_radii {fit_radii}: {:.3} ms per fit, {:.1} iterations",
                    t0.elapsed().as_secs_f64() * 1e3 / n as f64,
                    it as f64 / n as f64
                );
            }
        }
    }
}
