//! Photometric (intensity-model) refinement of one ring marker.
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
//!   ellipse-centre or two-conic construction is involved. A plain marker,
//!   symmetric under rotation, keeps the board homography's rotation: its `C`
//!   has three free directions instead of four (see [`CBasis`]).
//! - `P` is the dark-ink coverage of the printed pattern around the marker,
//!   blurred by a Gaussian PSF of standard deviation `sigma` px (optionally
//!   anisotropic, two more parameters, see [`Blur`]):
//!   the marker's ink annulus `[r0, r3]`, minus the paper sectors of its code
//!   band `[r1, r2]` when the target is coded, plus the ink discs around it
//!   (the outer edges of neighbouring markers, origin fiducial dots). Each
//!   edge contributes a normal-CDF step of its signed image-space distance
//!   (the board-space distance divided by the local gradient norm); code
//!   sectors are the product of a radial band and an angular indicator whose
//!   sector transitions are blurred the same way. A plain marker is the same
//!   model without a code band.
//! - `B` (paper) and `A` (paper minus ink) are affine in the image, which absorbs
//!   vignetting and illumination gradients.
//!
//! The parameters are fitted by Levenberg–Marquardt on the pixel residuals with
//! Huber weights; saturated pixels (and their immediate neighbours) are
//! masked. The covariance of the centre and of the PSF (`sigma` and its
//! anisotropy) is a sandwich estimate that stays honest when the residuals
//! are spatially correlated (see [`sandwich_covariance`]); it reduces to
//! the corresponding block of the inverse Fisher information
//! `(J^T W J / s^2 + prior)^-1` when they are white.

use image::GrayImage;
use nalgebra::{DMatrix, DVector};

use super::MAD_TO_SIGMA;
use crate::homography::homography_project_rows;

/// Number of sectors in the coded band.
const CODE_SECTORS: usize = 16;
/// Nonlinear geometric parameters of a coded marker: p (2), C (4), log sigma
/// (1). A plain marker fits one fewer (see [`CBasis`]).
const N_GEOM: usize = 7;
/// PSF anisotropy parameters, when fitted.
const N_ANISO: usize = 2;
/// Largest PSF anisotropy `|[a, b]|` (axis ratio `sqrt((1 + r) / (1 - r))`).
const MAX_ANISO: f64 = 0.9;
/// Prior standard deviation of the PSF anisotropy: keeps it at zero where the
/// pattern cannot tell (blur far below a pixel) without limiting real
/// astigmatism.
const ANISO_PRIOR_SIGMA: f64 = 0.5;
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
/// a coded marker whose code band is blurred away well-posed (a plain marker
/// does not fit its rotation, see [`CBasis`]); far wider than the error of a
/// board-homography seed.
const C_PRIOR_SIGMA: f64 = 0.15;
/// Prior standard deviation of free print radii, as a fraction of the outer
/// radius `r3` (scale-free, so any marker size gets the same prior).
const RADIUS_PRIOR_SIGMA_REL: f64 = 0.1;
/// Huber threshold in units of the robust residual scale.
const HUBER_K: f64 = 2.0;
/// Blur search grid for the initial sigma (px).
const SIGMA_GRID: [f64; 4] = [0.7, 1.4, 2.6, 4.5];
/// Iterations during which the robust residual scale is re-estimated.
const SCALE_UPDATE_ITERATIONS: usize = 3;
/// Lags (px) of the reported residual autocorrelation diagnostic.
pub(crate) const RESIDUAL_ACF_LAGS: usize = 6;
/// Largest lag (px, per axis) at which the residual autocovariance model is
/// fitted.
const ACF_FIT_LAGS: i64 = 4;
/// Candidate correlation lengths (px) of the correlated residual component.
const ACF_LENGTHS: [f64; 8] = [0.4, 0.6, 0.85, 1.2, 1.7, 2.4, 3.4, 4.8];
/// A correlated residual component is used only when its estimate exceeds
/// this many standard errors.
const ACF_SIGNIFICANCE: f64 = 3.0;
/// The covariance window extends this many correlation lengths.
const ACF_WINDOW_LENGTHS: f64 = 3.0;
/// Largest covariance window half-width (px).
const MAX_ACF_WINDOW: i64 = 16;

/// Board → image projective map, row-major.
pub(crate) type Mat3 = [[f64; 3]; 3];

/// Radial print geometry of a marker, board mm, from the centre out:
/// `[hole edge, inner ring outer edge, outer ring inner edge, outer edge]`.
/// Ink covers `[r0, r3]` except the paper sectors of the code band `[r1, r2]`.
/// A plain marker (a filled annulus, no code band) has `r1 = r2 = r3`.
pub(crate) type PrintRadii = [f64; 4];

/// A filled ink disc near the marker, board mm relative to the marker centre:
/// the outer edge of a neighbouring marker (whose interior never reaches the
/// patch) or an origin fiducial dot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InkDisc {
    pub center: [f64; 2],
    pub radius: f64,
}

/// Everything the fitter needs for one marker.
pub(crate) struct PhotometricProblem<'a> {
    pub gray: &'a GrayImage,
    /// Board-local mm (relative to this marker's centre) → image px.
    pub local_map: Mat3,
    /// Initial centre estimate, image px.
    pub center_seed: [f64; 2],
    /// Codeword of a coded marker (bit `s` set ⇔ sector `s` is paper);
    /// `None` for a plain marker, which has no code band.
    pub codeword: Option<u16>,
    pub radii: PrintRadii,
    /// Free the inner radii: `r0`, and `r1`, `r2` when there is a code band.
    /// `r3` stays fixed: a common scale of all radii is indistinguishable
    /// from the scale in `C`.
    pub fit_radii: bool,
    /// Model the paper level `B` as a cubic instead of an affine function of
    /// the image position, for patches where a smooth but strongly curved additive term
    /// (a glare halo) is present. Costs precision elsewhere.
    pub cubic_background: bool,
    /// Other ink near the marker (neighbours' outer edges, fiducial dots).
    pub ink_discs: &'a [InkDisc],
    /// Fit an anisotropic Gaussian PSF (two more parameters) instead of an
    /// isotropic one.
    pub anisotropic_blur: bool,
    /// Pixels up to `r3 + patch_margin_mm` (board mm) enter the fit.
    pub patch_margin_mm: f64,
    /// Grey level at or above which a pixel counts as saturated.
    pub saturation_level: u8,
    pub max_iterations: usize,
}

/// Spatial autocorrelation of the (Huber-weighted) fit residuals at lags
/// `1..=RESIDUAL_ACF_LAGS` px along the image x and y axes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct ResidualAcf {
    pub x: [f64; RESIDUAL_ACF_LAGS],
    pub y: [f64; RESIDUAL_ACF_LAGS],
}

/// Fitted parameters and their uncertainty.
#[derive(Debug, Clone)]
pub(crate) struct PhotometricFit {
    pub center: [f64; 2],
    /// Covariance of `center`, px², honest under spatially correlated
    /// residuals.
    pub center_cov: [[f64; 2]; 2],
    /// The same covariance assuming white residuals (the inverse Fisher
    /// information); `center_cov` reduces to it when the residuals are white.
    pub center_cov_white: [[f64; 2]; 2],
    /// Residual autocorrelation (diagnostic).
    pub residual_acf: ResidualAcf,
    /// The fitted PSF and its uncertainty.
    pub blur: BlurEstimate,
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
    /// Standard deviations of `r0..r2` when fitted; entries of radii that are
    /// not free (`r1`, `r2` of a plain marker) are 0.
    pub radii_std: Option<[f64; 3]>,
    /// The fitted parameter vector, kept so the residual probe can re-evaluate
    /// the model ([`residual_samples`]).
    #[cfg(feature = "study")]
    pub params: Params,
}

/// The fitted Gaussian PSF, covariance `sigma^2 [[1 + a, b], [b, 1 - a]]`
/// (see [`Blur`]), with the covariance of `(sigma, a, b)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BlurEstimate {
    /// PSF sigma, px.
    pub sigma_px: f64,
    /// PSF anisotropy `[a, b]`; zero when the fit is isotropic.
    pub aniso: [f64; 2],
    /// Covariance of `(sigma px, a, b)`: the same sandwich as the centre's
    /// (honest under spatially correlated residuals), with `sigma` from log
    /// sigma by the delta method. The `a`, `b` rows are zero when the fit is
    /// isotropic. Where the sandwich fails (or its blur block is not positive
    /// definite) this is the prior-regularized inverse Fisher information,
    /// as for the centre.
    pub cov: [[f64; 3]; 3],
    /// The same covariance assuming white residuals.
    pub cov_white: [[f64; 3]; 3],
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

/// Current parameter vector, unpacked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    p: [f64; 2],
    c: [[f64; 2]; 2],
    log_sigma: f64,
    /// PSF anisotropy `[a, b]`, see [`Blur`]; zero unless fitted.
    aniso: [f64; 2],
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
    /// dP/d(anisotropy) of the PSF.
    d_aniso: [f64; 2],
    /// dP/dr0, dP/dr1, dP/dr2.
    d_radii: [f64; 3],
}

impl PatternEval {
    /// Add `dls` (a derivative in log sigma of an edge's step) to the blur
    /// derivatives, splitting it onto the anisotropy through the edge's
    /// direction.
    #[inline]
    fn add_dlog_sigma(&mut self, dls: f64, e: &Edge) {
        self.dlog_sigma += dls;
        self.d_aniso[0] += dls * e.dlog_aniso[0];
        self.d_aniso[1] += dls * e.dlog_aniso[1];
    }
}

/// The PSF: a Gaussian of covariance `sigma^2 [[1 + a, b], [b, 1 - a]]`,
/// `aniso = [a, b]` (isotropic when zero). Across a straight edge with image
/// normal `g` it blurs like a 1-D Gaussian of `sigma sqrt(g^T S g / |g|^2)`.
#[derive(Clone, Copy)]
struct Blur {
    sigma: f64,
    aniso: [f64; 2],
}

impl Blur {
    /// Effective sigma across an edge with image gradient `g` (norm `gnorm`)
    /// and the derivatives of its log in `aniso`.
    #[inline]
    fn along(&self, g: [f64; 2], gnorm: f64) -> (f64, [f64; 2]) {
        if gnorm <= 0.0 {
            return (self.sigma, [0.0; 2]);
        }
        let (nx, ny) = (g[0] / gnorm, g[1] / gnorm);
        let c2 = nx * nx - ny * ny;
        let s2 = 2.0 * nx * ny;
        let q = 1.0 + self.aniso[0] * c2 + self.aniso[1] * s2;
        (self.sigma * q.sqrt(), [0.5 * c2 / q, 0.5 * s2 / q])
    }

    /// The largest effective sigma over directions.
    fn max_sigma(&self) -> f64 {
        self.sigma * (1.0 + self.aniso[0].hypot(self.aniso[1])).sqrt()
    }
}

/// A board-space edge at one pixel: its covector `dq/dy`, image gradient
/// norm (mm per px) and blur across it.
#[derive(Clone, Copy)]
struct Edge {
    v: [f64; 2],
    gnorm: f64,
    sigma: f64,
    /// d log(sigma) / d aniso.
    dlog_aniso: [f64; 2],
}

/// Transition of the angular code indicator at a sector boundary.
#[derive(Clone, Copy)]
struct Transition {
    /// Unit direction of the boundary ray (board frame).
    dir: [f64; 2],
    /// White(after) - white(before), counter-clockwise.
    delta: f64,
}

/// The paper sectors of a coded marker's code band.
struct CodeBand {
    bits: [f64; CODE_SECTORS],
    transitions: Vec<Transition>,
}

impl CodeBand {
    fn new(codeword: u16) -> Self {
        let mut bits = [0.0; CODE_SECTORS];
        for (s, b) in bits.iter_mut().enumerate() {
            *b = f64::from((codeword >> s) & 1);
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
        Self { bits, transitions }
    }
}

struct Model<'a> {
    h_inv: Mat3,
    h0: [f64; 2],
    code: Option<CodeBand>,
    ink_discs: &'a [InkDisc],
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
        let blur = Blur {
            sigma: prm.log_sigma.exp(),
            aniso: prm.aniso,
        };
        let dx = [x[0] - prm.p[0], x[1] - prm.p[1]];
        let c = &prm.c;
        let y = [
            self.h0[0] + c[0][0] * dx[0] + c[0][1] * dx[1],
            self.h0[1] + c[1][0] * dx[0] + c[1][1] * dx[1],
        ];
        let (u, j) = self.unproject(y)?;

        // A board-space edge with unit normal `n` (board frame): its covector
        // `dq/dy`, its image-space gradient norm `|C^T J^T n|` and its blur.
        let edge = |n: [f64; 2]| -> Edge {
            let v = [
                j[0][0] * n[0] + j[1][0] * n[1],
                j[0][1] * n[0] + j[1][1] * n[1],
            ];
            let g = [
                c[0][0] * v[0] + c[1][0] * v[1],
                c[0][1] * v[0] + c[1][1] * v[1],
            ];
            let gnorm = g[0].hypot(g[1]);
            let (sigma, dlog_aniso) = blur.along(g, gnorm);
            Edge {
                v,
                gnorm,
                sigma,
                dlog_aniso,
            }
        };

        let rho = u[0].hypot(u[1]);
        let mut out = PatternEval {
            p: 0.0,
            grad_y: [0.0; 2],
            dlog_sigma: 0.0,
            d_aniso: [0.0; 2],
            d_radii: [0.0; 3],
        };
        if rho < 1e-9 {
            // At the exact centre the pattern is flat paper.
            return Some(out);
        }
        let er = edge([u[0] / rho, u[1] / rho]);
        if er.gnorm < 1e-12 {
            return None;
        }
        // Image px per board mm along the radial direction is 1 / (dρ/dx).
        let inv_g = 1.0 / er.gnorm;
        let rq = [er.v[0] * inv_g, er.v[1] * inv_g]; // d(dist)/dy for radial edges
        let r = &prm.radii;
        let s = |k: usize| step_curved((rho - r[k]) * inv_g, er.sigma, r[k] * inv_g);

        // Ink annulus [r0, r3]: Phi(d0) - Phi(d3).
        let s0 = s(0);
        let s3 = s(3);
        out.p = s0.value - s3.value;
        let mut radial_slope = s0.slope - s3.slope;
        let mut radial_dls = s0.dlog_sigma - s3.dlog_sigma;
        out.d_radii[0] = -s0.slope * inv_g;

        // Board-mm reach of a blurred edge (gnorm is mm per image px), with
        // slack for the anisotropy of the local map.
        let reach_mm = EDGE_REACH_SIGMAS * blur.max_sigma() * er.gnorm * 1.5;

        // Code band [r1, r2] times the angular paper indicator.
        if let Some(code) = &self.code {
            let s1 = s(1);
            let s2 = s(2);
            let band = s1.value - s2.value;
            if band > 1e-9 || s1.slope != 0.0 || s2.slope != 0.0 {
                let ang = code.angular(u, reach_mm, &edge);
                out.p -= band * ang.theta;
                radial_slope -= (s1.slope - s2.slope) * ang.theta;
                radial_dls -= (s1.dlog_sigma - s2.dlog_sigma) * ang.theta;
                out.dlog_sigma -= band * ang.dlog_sigma;
                for k in 0..2 {
                    out.grad_y[k] -= band * ang.grad_y[k];
                    out.d_aniso[k] -= band * ang.d_aniso[k];
                }
                out.d_radii[1] = s1.slope * inv_g * ang.theta;
                out.d_radii[2] = -s2.slope * inv_g * ang.theta;
            }
        }
        out.grad_y[0] += radial_slope * rq[0];
        out.grad_y[1] += radial_slope * rq[1];
        out.add_dlog_sigma(radial_dls, &er);

        self.add_ink_discs(u, reach_mm, &edge, &mut out);
        Some(out)
    }

    /// Add the blurred coverage of the surrounding ink discs.
    fn add_ink_discs(
        &self,
        u: [f64; 2],
        reach_mm: f64,
        edge: &dyn Fn([f64; 2]) -> Edge,
        out: &mut PatternEval,
    ) {
        for disc in self.ink_discs {
            let un = [u[0] - disc.center[0], u[1] - disc.center[1]];
            let rho_sq = un[0] * un[0] + un[1] * un[1];
            let outer = disc.radius + reach_mm;
            if rho_sq > outer * outer {
                continue;
            }
            let inner = disc.radius - reach_mm;
            if inner > 0.0 && rho_sq < inner * inner {
                // Deep inside the disc: solid ink.
                out.p += 1.0;
                continue;
            }
            if rho_sq < 1e-18 {
                continue;
            }
            let rho_n = rho_sq.sqrt();
            let e = edge([un[0] / rho_n, un[1] / rho_n]);
            if e.gnorm < 1e-12 {
                continue;
            }
            let sn = step_curved(
                (rho_n - disc.radius) / e.gnorm,
                e.sigma,
                disc.radius / e.gnorm,
            );
            out.p += 1.0 - sn.value;
            out.grad_y[0] -= sn.slope * e.v[0] / e.gnorm;
            out.grad_y[1] -= sn.slope * e.v[1] / e.gnorm;
            out.add_dlog_sigma(-sn.dlog_sigma, &e);
        }
    }
}

/// Index of the code sector containing board point `u` (sectors start at
/// angle `-pi`, counter-clockwise).
fn sector_index(u: [f64; 2]) -> usize {
    let pos = (u[1].atan2(u[0]) + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
        * CODE_SECTORS as f64;
    (pos.floor() as isize).rem_euclid(CODE_SECTORS as isize) as usize
}

/// The blurred paper indicator of a code band at one board point.
struct Angular {
    theta: f64,
    grad_y: [f64; 2],
    dlog_sigma: f64,
    d_aniso: [f64; 2],
}

impl CodeBand {
    /// Blurred paper indicator of the code sectors at board point `u`, with its
    /// gradient in `y` and derivatives in the blur parameters.
    fn angular(&self, u: [f64; 2], reach_mm: f64, edge: &dyn Fn([f64; 2]) -> Edge) -> Angular {
        let mut out = Angular {
            theta: self.bits[sector_index(u)],
            grad_y: [0.0; 2],
            dlog_sigma: 0.0,
            d_aniso: [0.0; 2],
        };
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
            let e = edge(n);
            if e.gnorm < 1e-12 {
                continue;
            }
            let t = t_mm / e.gnorm;
            if t.abs() > EDGE_REACH_SIGMAS * e.sigma {
                continue;
            }
            let st = step(t, e.sigma);
            let heaviside = if t >= 0.0 { 1.0 } else { 0.0 };
            out.theta += tr.delta * (st.value - heaviside);
            out.grad_y[0] += tr.delta * st.slope * e.v[0] / e.gnorm;
            out.grad_y[1] += tr.delta * st.slope * e.v[1] / e.gnorm;
            let dls = tr.delta * st.dlog_sigma;
            out.dlog_sigma += dls;
            out.d_aniso[0] += dls * e.dlog_aniso[0];
            out.d_aniso[1] += dls * e.dlog_aniso[1];
        }
        out
    }
}

/// Jacobian of `map` at the board origin (px per board mm).
fn local_jacobian(map: &Mat3) -> [[f64; 2]; 2] {
    let w = map[2][2];
    let x = [map[0][2] / w, map[1][2] / w];
    let mut j = [[0.0; 2]; 2];
    for (i, row) in j.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = (map[i][k] - x[i] * map[2][k]) / w;
        }
    }
    j
}

fn build_model<'a>(problem: &'a PhotometricProblem<'_>) -> Option<Model<'a>> {
    let h_inv = mat3_inverse(&problem.local_map)?;
    let h0 = homography_project_rows(&problem.local_map, [0.0, 0.0])?;
    Some(Model {
        h_inv,
        h0,
        code: problem.codeword.map(CodeBand::new),
        ink_discs: problem.ink_discs,
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
        if let Some(q) =
            homography_project_rows(&problem.local_map, [r_max * a.cos(), r_max * a.sin()])
        {
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

/// The free directions of the local correction `C`: an orthonormal basis
/// (Frobenius inner product) of the 2×2 matrices the fit may move `C` along,
/// with `C = sum_k v_k E_k`.
///
/// A coded marker frees all four entries (the standard basis, so `v` is `C`
/// itself). A plain marker is a rotationally symmetric annulus: rotating the
/// board about its centre, `C -> M R(t) M^-1 C` with `M` the local map's
/// Jacobian at the centre, leaves the pattern's image unchanged up to the
/// projective terms and the far-away neighbour ink. Free, that direction is
/// held by the weak `C` prior alone; it forms a long curved valley in which
/// Levenberg–Marquardt creeps by thousandths per iteration (slower the larger
/// the marker), dragging the centre along. The plain basis spans the
/// complement of its generator `M Omega M^-1` at `C = I`, so the marker keeps
/// the board homography's rotation, which the whole board determines far
/// better. The identity lies in that complement (the generator is
/// traceless), so the seed `C = I` stays representable.
#[derive(Clone, Copy)]
struct CBasis {
    e: [[[f64; 2]; 2]; 4],
    n: usize,
}

impl CBasis {
    fn full() -> Self {
        Self {
            e: [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ],
            n: 4,
        }
    }

    /// The complement of the in-plane rotation of a symmetric pattern imaged
    /// through `m` (board mm -> px at the centre); `None` when `m` is
    /// singular.
    fn without_rotation(m: [[f64; 2]; 2]) -> Option<Self> {
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if !det.is_finite() || det.abs() < 1e-300 {
            return None;
        }
        // M Omega M^-1, Omega = [[0, -1], [1, 0]].
        let mo = [[m[0][1], -m[0][0]], [m[1][1], -m[1][0]]];
        let inv = [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ];
        let mut g = [[0.0; 2]; 2];
        for (i, row) in g.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = mo[i][0] * inv[0][j] + mo[i][1] * inv[1][j];
            }
        }
        let dot = |a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]| {
            a[0][0] * b[0][0] + a[0][1] * b[0][1] + a[1][0] * b[1][0] + a[1][1] * b[1][1]
        };
        let norm = dot(&g, &g).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        let g = g.map(|r| r.map(|v| v / norm));
        // Gram-Schmidt of the standard basis against g, dropping the unit
        // matrix most aligned with g (deterministic).
        let units = Self::full().e;
        let drop = (0..4)
            .max_by(|&a, &b| {
                dot(&units[a], &g)
                    .abs()
                    .total_cmp(&dot(&units[b], &g).abs())
            })
            .unwrap_or(0);
        let mut out = Self {
            e: [[[0.0; 2]; 2]; 4],
            n: 0,
        };
        for (k, u) in units.iter().enumerate() {
            if k == drop {
                continue;
            }
            let mut v = *u;
            for b in std::iter::once(&g).chain(out.e[..out.n].iter()) {
                let c = dot(&v, b);
                for i in 0..2 {
                    for j in 0..2 {
                        v[i][j] -= c * b[i][j];
                    }
                }
            }
            let nv = dot(&v, &v).sqrt();
            out.e[out.n] = v.map(|r| r.map(|x| x / nv));
            out.n += 1;
        }
        Some(out)
    }

    /// Coordinates of `c` in the basis.
    fn coords(&self, c: &[[f64; 2]; 2], k: usize) -> f64 {
        let e = &self.e[k];
        e[0][0] * c[0][0] + e[0][1] * c[0][1] + e[1][0] * c[1][0] + e[1][1] * c[1][1]
    }

    /// The matrix with coordinates `v`.
    fn matrix(&self, v: &[f64]) -> [[f64; 2]; 2] {
        let mut c = [[0.0; 2]; 2];
        for (e, vk) in self.e[..self.n].iter().zip(v) {
            for i in 0..2 {
                for j in 0..2 {
                    c[i][j] += vk * e[i][j];
                }
            }
        }
        c
    }
}

/// Layout of the full parameter vector: centre, the free directions of `C`,
/// log sigma, the PSF anisotropy when fitted, the active linear coefficients,
/// then the free radii.
struct Layout {
    c_basis: CBasis,
    /// Fit the PSF anisotropy (two more geometric parameters).
    aniso: bool,
    /// Number of free radii: 0, 1 (`r0` of a plain marker) or 3 (`r0..r2`).
    n_radii: usize,
    lin: &'static [usize],
}

impl Layout {
    fn new(c_basis: CBasis, fit_radii: bool, coded: bool, cubic: bool, aniso: bool) -> Self {
        Self {
            c_basis,
            aniso,
            n_radii: match (fit_radii, coded) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 3,
            },
            lin: if cubic { &LIN_CUBIC } else { &LIN_AFFINE },
        }
    }
    /// Index of log sigma (after the centre and `C`).
    fn sigma_index(&self) -> usize {
        2 + self.c_basis.n
    }
    /// Index of the first anisotropy parameter, when fitted.
    fn aniso_base(&self) -> usize {
        self.sigma_index() + 1
    }
    /// Index of the first linear coefficient (after the geometry).
    fn lin_base(&self) -> usize {
        self.aniso_base() + if self.aniso { N_ANISO } else { 0 }
    }
    fn radii_base(&self) -> usize {
        self.lin_base() + self.lin.len()
    }
    /// The parameters whose sampling covariance is reported (see
    /// [`sandwich_covariance`]): the centre, log sigma, and the anisotropy
    /// when fitted. Returns the indices and their count.
    fn reported(&self) -> ([usize; N_REPORTED], usize) {
        let s = self.sigma_index();
        if self.aniso {
            ([0, 1, s, s + 1, s + 2], N_REPORTED)
        } else {
            ([0, 1, s, 0, 0], 3)
        }
    }
    fn n(&self) -> usize {
        self.radii_base() + self.n_radii
    }
    fn pack(&self, prm: &Params) -> DVector<f64> {
        let mut v = DVector::zeros(self.n());
        v[0] = prm.p[0];
        v[1] = prm.p[1];
        for k in 0..self.c_basis.n {
            v[2 + k] = self.c_basis.coords(&prm.c, k);
        }
        v[self.sigma_index()] = prm.log_sigma;
        if self.aniso {
            v[self.aniso_base()] = prm.aniso[0];
            v[self.aniso_base() + 1] = prm.aniso[1];
        }
        for (k, &li) in self.lin.iter().enumerate() {
            v[self.lin_base() + k] = prm.lin[li];
        }
        for k in 0..self.n_radii {
            v[self.radii_base() + k] = prm.radii[k];
        }
        v
    }
    fn unpack(&self, v: &DVector<f64>, template: &Params) -> Params {
        let mut prm = *template;
        prm.p = [v[0], v[1]];
        prm.c = self.c_basis.matrix(&v.as_slice()[2..2 + self.c_basis.n]);
        prm.log_sigma = v[self.sigma_index()];
        if self.aniso {
            // Keep the PSF covariance positive definite (axis ratio ≤ 4.4).
            let a = [v[self.aniso_base()], v[self.aniso_base() + 1]];
            let r = a[0].hypot(a[1]);
            let k = if r > MAX_ANISO { MAX_ANISO / r } else { 1.0 };
            prm.aniso = [k * a[0], k * a[1]];
        }
        for (k, &li) in self.lin.iter().enumerate() {
            prm.lin[li] = v[self.lin_base() + k];
        }
        for k in 0..self.n_radii {
            prm.radii[k] = v[self.radii_base() + k];
        }
        prm
    }
}

/// Maximum parameter count (geometry with anisotropy, photometry, three free
/// radii).
const N_MAX: usize = N_GEOM + N_ANISO + N_LIN + 3;

/// Largest number of reported parameters: centre (2), log sigma, anisotropy
/// (2).
const N_REPORTED: usize = 5;

/// Residual and Jacobian row (`d model / d params`, active entries first) of
/// one pixel.
fn pixel_row(
    model: &Model<'_>,
    px: &PatchPixel,
    prm: &Params,
    layout: &Layout,
    row: &mut [f64; N_MAX],
) -> Option<f64> {
    let pe = model.pattern(px.x, prm)?;
    let (value, a) = photometry(&prm.lin, px.xi, pe.p);
    let c = &prm.c;
    let gy = pe.grad_y;
    let dx = [px.x[0] - prm.p[0], px.x[1] - prm.p[1]];
    // dy/dp = -C  =>  dI/dp = A * C^T grad_y.
    row[0] = a * (c[0][0] * gy[0] + c[1][0] * gy[1]);
    row[1] = a * (c[0][1] * gy[0] + c[1][1] * gy[1]);
    // dy_i/dC_ij = dx_j, projected on the free directions of C.
    let d_c = [
        [-a * gy[0] * dx[0], -a * gy[0] * dx[1]],
        [-a * gy[1] * dx[0], -a * gy[1] * dx[1]],
    ];
    for k in 0..layout.c_basis.n {
        row[2 + k] = layout.c_basis.coords(&d_c, k);
    }
    row[layout.sigma_index()] = -a * pe.dlog_sigma;
    if layout.aniso {
        row[layout.aniso_base()] = -a * pe.d_aniso[0];
        row[layout.aniso_base() + 1] = -a * pe.d_aniso[1];
    }
    let full = lin_row(px.xi, pe.p);
    let lin_base = layout.lin_base();
    for (k, &li) in layout.lin.iter().enumerate() {
        row[lin_base + k] = full[li];
    }
    let base = layout.radii_base();
    for k in 0..layout.n_radii {
        row[base + k] = -a * pe.d_radii[k];
    }
    Some(px.value - value)
}

/// Huber IRLS weight of a residual.
#[inline]
fn huber_weight(r: f64, k_huber: f64) -> f64 {
    let ar = r.abs();
    if ar <= k_huber { 1.0 } else { k_huber / ar }
}

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
    let mut row = [0.0f64; N_MAX];
    for px in pixels {
        let r = pixel_row(model, px, prm, layout, &mut row)?;
        residuals.push(r);
        let ar = r.abs();
        let w = huber_weight(r, k_huber);
        robust_cost += if ar <= k_huber {
            0.5 * r * r
        } else {
            k_huber * (ar - 0.5 * k_huber)
        };
        wrss += w * r * r;
        wsum += w;
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
    // Isotropic in the entries of C, so the same in any orthonormal basis.
    for k in 0..layout.c_basis.n {
        info[2 + k] = ic;
    }
    if layout.aniso {
        let ia = 1.0 / (ANISO_PRIOR_SIGMA * ANISO_PRIOR_SIGMA);
        info[layout.aniso_base()] = ia;
        info[layout.aniso_base() + 1] = ia;
    }
    let sigma_r = RADIUS_PRIOR_SIGMA_REL * prior.radii[3];
    let ir = 1.0 / (sigma_r * sigma_r);
    for k in 0..layout.n_radii {
        info[layout.radii_base() + k] = ir;
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
    /// Robust residual scale the final weights use.
    scale: f64,
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
        let mut scale = (MAD_TO_SIGMA * median_abs(&initial.residuals)).max(0.5);
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
                        && delta[self.layout.sigma_index()].abs() < 1e-4
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
            let new_scale = (MAD_TO_SIGMA * median_abs(&acc.residuals)).max(0.5);
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
            scale,
            iterations,
            converged,
        })
    }

    /// Parameter covariance assuming white residuals: inverse Fisher
    /// information with the noise variance estimated from the weighted
    /// residuals; also returns that variance.
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

/// Weighted residuals and influence vectors of the patch on a dense grid over
/// its bounding box (zero outside the patch).
struct ResidualGrid {
    width: usize,
    height: usize,
    /// `sqrt(w) r`, the standardized-weight residual.
    e: Vec<f64>,
    /// `sqrt(w) B J`, the pixel's influence on the reported parameters
    /// ([`Layout::reported`]), `n_reported` per pixel.
    z: Vec<f64>,
    n_reported: usize,
    /// `L^T sqrt(w) J` with `L L^T = A^-1`, `n` per pixel: the hat matrix of
    /// the fit is `H_ij = q_i . q_j`.
    q: Vec<f64>,
    n: usize,
    inside: Vec<bool>,
    /// The point-antisymmetric parts of `e` and `q` about the fitted centre,
    /// `(f(x) - f(x'))/sqrt(2)` with `x'` the reflection of `x`, and where
    /// both `x` and `x'` are in the patch.
    e_odd: Vec<f64>,
    q_odd: Vec<f64>,
    inside_odd: Vec<bool>,
}

impl ResidualGrid {
    /// Build the grid for the solution; `a_inv` is `(J^T W J + s^2 P)^-1`.
    fn new(fitting: &Fitting<'_>, sol: &Solution, a_inv: &DMatrix<f64>) -> Option<Self> {
        let n = fitting.layout.n();
        let (reported, n_reported) = fitting.layout.reported();
        let reported = &reported[..n_reported];
        let l = a_inv.clone().cholesky()?.l();
        let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for px in &fitting.pixels {
            let (x, y) = (px.x[0] as i64, px.x[1] as i64);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        if x1 < x0 {
            return None;
        }
        let width = (x1 - x0 + 1) as usize;
        let height = (y1 - y0 + 1) as usize;
        let mut grid = Self {
            width,
            height,
            e: vec![0.0; width * height],
            z: vec![0.0; width * height * n_reported],
            n_reported,
            q: vec![0.0; width * height * n],
            n,
            inside: vec![false; width * height],
            e_odd: vec![0.0; width * height],
            q_odd: vec![0.0; width * height * n],
            inside_odd: vec![false; width * height],
        };
        let k_huber = HUBER_K * sol.scale;
        let mut row = [0.0f64; N_MAX];
        for px in &fitting.pixels {
            let r = pixel_row(&fitting.model, px, &sol.prm, &fitting.layout, &mut row)?;
            let sw = huber_weight(r, k_huber).sqrt();
            let idx = (px.x[1] as i64 - y0) as usize * width + (px.x[0] as i64 - x0) as usize;
            let z = &mut grid.z[idx * n_reported..(idx + 1) * n_reported];
            for (zk, &k) in z.iter_mut().zip(reported) {
                *zk = sw * (0..n).map(|j| a_inv[(k, j)] * row[j]).sum::<f64>();
            }
            for k in 0..n {
                grid.q[idx * n + k] = sw * (k..n).map(|j| l[(j, k)] * row[j]).sum::<f64>();
            }
            grid.e[idx] = sw * r;
            grid.inside[idx] = true;
        }
        // Reflection through the grid point nearest the fitted centre.
        let cx2 = (2.0 * (sol.prm.p[0] - x0 as f64)).round() as i64;
        let cy2 = (2.0 * (sol.prm.p[1] - y0 as f64)).round() as i64;
        let h = std::f64::consts::FRAC_1_SQRT_2;
        for gy in 0..height as i64 {
            for gx in 0..width as i64 {
                let (rx, ry) = (cx2 - gx, cy2 - gy);
                if !(0..width as i64).contains(&rx) || !(0..height as i64).contains(&ry) {
                    continue;
                }
                let i = gy as usize * width + gx as usize;
                let j = ry as usize * width + rx as usize;
                if !(grid.inside[i] && grid.inside[j]) {
                    continue;
                }
                grid.inside_odd[i] = true;
                grid.e_odd[i] = h * (grid.e[i] - grid.e[j]);
                for k in 0..n {
                    grid.q_odd[i * n + k] = h * (grid.q[i * n + k] - grid.q[j * n + k]);
                }
            }
        }
        Some(grid)
    }

    /// `sum_x f(x, x + d)` over pixel pairs both inside the patch.
    fn lag_sum<T: Default + std::ops::AddAssign>(
        &self,
        d: [i64; 2],
        f: impl Fn(usize, usize) -> T,
    ) -> T {
        self.masked_lag_sum(&self.inside, d, f)
    }

    /// `sum_x f(x, x + d)` over pixel pairs both inside `mask`.
    fn masked_lag_sum<T: Default + std::ops::AddAssign>(
        &self,
        mask: &[bool],
        d: [i64; 2],
        f: impl Fn(usize, usize) -> T,
    ) -> T {
        let mut acc = T::default();
        let (w, h) = (self.width as i64, self.height as i64);
        for y in 0.max(-d[1])..h.min(h - d[1]) {
            for x in 0.max(-d[0])..w.min(w - d[0]) {
                let i = (y * w + x) as usize;
                let j = ((y + d[1]) * w + x + d[0]) as usize;
                if mask[i] && mask[j] {
                    acc += f(i, j);
                }
            }
        }
        acc
    }

    /// Residual autocorrelation coefficient at lag `d` over the pairs that
    /// exist at that lag (diagnostic).
    fn correlation(&self, d: [i64; 2]) -> f64 {
        let (num, (ei, ej)) = (
            self.lag_sum(d, |i, j| self.e[i] * self.e[j]),
            (
                self.lag_sum(d, |i, _| self.e[i] * self.e[i]),
                self.lag_sum(d, |_, j| self.e[j] * self.e[j]),
            ),
        );
        let den = (ei * ej).sqrt();
        if den > 0.0 { num / den } else { 0.0 }
    }
}

/// Lag statistics of the residual grid at one lag `d`.
#[derive(Default, Clone, Copy)]
struct LagStats {
    /// `sum e_x e_{x+d}` over the pixel pairs at this lag.
    resid: f64,
    /// Number of pixel pairs at this lag.
    pairs: f64,
    /// `sum H_{x,x+d}`: the hat matrix's share of this lag.
    hat: f64,
}

impl std::ops::AddAssign for LagStats {
    fn add_assign(&mut self, o: Self) {
        self.resid += o.resid;
        self.pairs += o.pairs;
        self.hat += o.hat;
    }
}

impl ResidualGrid {
    /// Residual-noise statistics at lag `d` (antisymmetric part, see
    /// [`sandwich_covariance`]).
    fn noise_lag_stats(&self, d: [i64; 2]) -> LagStats {
        let n = self.n;
        let (e, q) = (&self.e_odd, &self.q_odd);
        self.masked_lag_sum(&self.inside_odd, d, |i, j| LagStats {
            resid: e[i] * e[j],
            pairs: 1.0,
            hat: (0..n).map(|m| q[i * n + m] * q[j * n + m]).sum(),
        })
    }

    /// `sum_x z_x z_{x+d}^T`.
    fn influence(&self, d: [i64; 2]) -> Block {
        let k = self.n_reported;
        self.lag_sum(d, |i, j| {
            let (a, b) = (&self.z[i * k..(i + 1) * k], &self.z[j * k..(j + 1) * k]);
            let mut out = Block::default();
            for (r, ar) in a.iter().enumerate() {
                for (c, bc) in b.iter().enumerate() {
                    out.0[r][c] = ar * bc;
                }
            }
            out
        })
    }
}

/// Accumulated products over the reported parameters (entries beyond their
/// count stay zero).
#[derive(Default, Clone, Copy)]
struct Block([[f64; N_REPORTED]; N_REPORTED]);

impl std::ops::AddAssign for Block {
    fn add_assign(&mut self, o: Self) {
        for (a, b) in self.0.as_flattened_mut().iter_mut().zip(o.0.as_flattened()) {
            *a += b;
        }
    }
}

/// Stationary autocovariance of the residual noise: a white part plus a
/// Gaussian-correlated part, `C(d) = white δ(d) + corr exp(-|d|^2 / (2 len^2))`.
/// Positive semidefinite for any non-negative `white`, `corr`.
#[derive(Debug, Clone, Copy)]
struct NoiseAcf {
    white: f64,
    corr: f64,
    len: f64,
}

impl NoiseAcf {
    fn kernel(len: f64, d: [i64; 2]) -> f64 {
        (-((d[0] * d[0] + d[1] * d[1]) as f64) / (2.0 * len * len)).exp()
    }

    fn at(&self, d: [i64; 2]) -> f64 {
        let white = if d == [0, 0] { self.white } else { 0.0 };
        white + self.corr * Self::kernel(self.len, d)
    }

    /// Covariance window half-width.
    fn window(&self) -> i64 {
        if self.corr <= 0.0 {
            return 0;
        }
        ((ACF_WINDOW_LENGTHS * self.len).ceil() as i64).min(MAX_ACF_WINDOW)
    }
}

/// Fit [`NoiseAcf`] to the residual lag statistics.
///
/// The fit absorbs part of any noise (`e = (I - H) eps`). For the white part
/// this is exact, `E[sum e_x e_{x+d}] = white (pairs δ(d) - h(d))` with
/// `h(d) = sum_x H_{x,x+d}`; for the correlated part, with `H` varying slowly
/// over the correlation length, `E ≈ corr (g(d) pairs(d) - G h(d))`,
/// `G = sum_d g(d)`. Both are linear in `(white, corr)`, which are solved by
/// non-negative least squares for each candidate length.
fn fit_noise_acf(stats: &[([i64; 2], LagStats)]) -> Option<NoiseAcf> {
    // Residual variance and the sampling variance of one lag statistic
    // `c/p` (white residuals, the typical pair count).
    let (d0, st0) = stats.first()?;
    debug_assert_eq!(*d0, [0, 0]);
    let var = st0.resid / st0.pairs.max(1.0);
    let noise_var = var * var / st0.pairs.max(1.0);
    let mut best: Option<(NoiseAcf, f64)> = None;
    for len in ACF_LENGTHS {
        let reach = (6.0 * len).ceil() as i64;
        let mut g_sum = 0.0;
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                g_sum += NoiseAcf::kernel(len, [dx, dy]);
            }
        }
        // Normal equations of c/p = white a + corr b.
        let (mut aa, mut ab, mut bb, mut ac, mut bc, mut cc) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (d, st) in stats {
            let p = st.pairs;
            if p < 1.0 {
                continue;
            }
            let delta = if *d == [0, 0] { 1.0 } else { 0.0 };
            let a = delta - st.hat / p;
            let b = NoiseAcf::kernel(len, *d) - g_sum * st.hat / p;
            let c = st.resid / p;
            aa += a * a;
            ab += a * b;
            bb += b * b;
            ac += a * c;
            bc += b * c;
            cc += c * c;
        }
        let det = aa * bb - ab * ab;
        let (white, corr) = if det > 1e-12 * aa * bb {
            ((bb * ac - ab * bc) / det, (aa * bc - ab * ac) / det)
        } else {
            (-1.0, -1.0)
        };
        // A correlated part must stand out of the sampling noise of the lag
        // statistics (`c/p` of white residuals scatters by `s^2 / sqrt(p)`):
        // the non-negativity constraint would otherwise turn noise into a
        // positive correlation that a long length integrates into a large
        // inflation.
        let corr_se = if det > 0.0 {
            (aa / det * noise_var).sqrt()
        } else {
            f64::INFINITY
        };
        let significant = corr > ACF_SIGNIFICANCE * corr_se;
        // Non-negative solution: interior, else the better single term.
        let candidates = [
            (white, corr),
            ((ac / aa).max(0.0), 0.0),
            (0.0, if bb > 0.0 { (bc / bb).max(0.0) } else { 0.0 }),
        ];
        for (w, c) in candidates {
            if !(w >= 0.0 && c >= 0.0) || (c > 0.0 && !significant) {
                continue;
            }
            let sse = cc - 2.0 * (w * ac + c * bc) + w * w * aa + 2.0 * w * c * ab + c * c * bb;
            if best.as_ref().is_none_or(|b| sse < b.1) {
                best = Some((
                    NoiseAcf {
                        white: w,
                        corr: c,
                        len,
                    },
                    sse,
                ));
            }
        }
    }
    best.map(|b| b.0)
}

/// Covariance of the reported parameters (the centre, log sigma and the PSF
/// anisotropy, [`Layout::reported`]) under spatially correlated residuals,
/// and the residual autocorrelation diagnostic. `None` when the centre block
/// is not positive definite.
///
/// The estimate is the sampling covariance of the regularized estimator, the
/// sandwich `A^-1 J^T W^½ C W^½ J A^-1` restricted to those parameters, with
/// `A = J^T W J + s^2 P` and `C` the autocovariance of the weighted residual
/// noise over pixel lags: a white part plus a Gaussian-correlated part
/// ([`NoiseAcf`], positive semidefinite by construction, so the result is a
/// valid covariance), fitted to the residual lag statistics with the fit's own
/// absorption of the noise accounted for ([`fit_noise_acf`]). With white
/// residuals `C = s^2 I` and the estimate is `s^2 A^-1 J^T W J A^-1`, which
/// equals the inverse Fisher information `s^2 A^-1` wherever the data, not the
/// prior, determine the centre. Where a direction the data barely constrain
/// (the in-plane rotation of a plain ring under perspective) couples into the
/// centre, the posterior `s^2 A^-1` would report the prior's width there,
/// while the fitted centre does not scatter by it.
///
/// The noise statistics come from the point-antisymmetric part of the
/// residual field about the fitted centre: a misfit of the pattern model that
/// is symmetric about the centre (a PSF shape or response curve the model
/// does not have, applied to a centrally symmetric marker) cannot move the
/// centre and is not mistaken for noise, while stationary noise splits evenly
/// between the symmetric and antisymmetric parts.
fn sandwich_covariance(
    fitting: &Fitting<'_>,
    sol: &Solution,
    cov: &DMatrix<f64>,
    s2: f64,
) -> Option<SandwichCovariance> {
    let a_inv = cov / s2;
    let grid = ResidualGrid::new(fitting, sol, &a_inv)?;
    // Half plane of lags (d and -d are symmetric), lag 0 first.
    let half_plane = |reach: i64| {
        let mut lags = vec![[0i64, 0i64]];
        for dy in 0..=reach {
            for dx in -reach..=reach {
                if dy > 0 || dx > 0 {
                    lags.push([dx, dy]);
                }
            }
        }
        lags
    };
    let stats: Vec<([i64; 2], LagStats)> = half_plane(ACF_FIT_LAGS)
        .into_iter()
        .map(|d| (d, grid.noise_lag_stats(d)))
        .collect();
    let acf = fit_noise_acf(&stats)?;
    let window = acf.window().min((grid.width.min(grid.height) / 3) as i64);
    let n_reported = grid.n_reported;
    // Symmetrized: lag -d contributes the transpose.
    let add_symmetric = |m: &mut Block, k: f64, zz: &Block| {
        for r in 0..n_reported {
            m.0[r][r] += k * zz.0[r][r];
            for c in r + 1..n_reported {
                let off = 0.5 * k * (zz.0[r][c] + zz.0[c][r]);
                m.0[r][c] += off;
                m.0[c][r] += off;
            }
        }
    };
    let mut m = Block::default();
    for d in half_plane(window) {
        let mult = if d == [0, 0] { 1.0 } else { 2.0 };
        let k = mult * acf.at(d);
        if k == 0.0 {
            continue;
        }
        add_symmetric(&mut m, k, &grid.influence(d));
    }
    let c = &m.0;
    let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    if !(c[0][0] > 0.0 && det > 0.0 && det.is_finite()) {
        return None;
    }
    let mut white = Block::default();
    add_symmetric(&mut white, s2, &grid.influence([0, 0]));
    let mut diag = ResidualAcf::default();
    for k in 0..RESIDUAL_ACF_LAGS {
        let lag = k as i64 + 1;
        diag.x[k] = grid.correlation([lag, 0]);
        diag.y[k] = grid.correlation([0, lag]);
    }
    Some(SandwichCovariance {
        cov: m,
        white,
        acf: diag,
    })
}

/// The covariance of the reported parameters and its diagnostics.
struct SandwichCovariance {
    /// Under the fitted residual noise correlation.
    cov: Block,
    /// The same sandwich with white residuals of variance `s^2`.
    white: Block,
    acf: ResidualAcf,
}

impl Block {
    /// The reported parameters' block of a full parameter covariance.
    fn of(cov: &DMatrix<f64>, layout: &Layout) -> Self {
        let (rows, n) = layout.reported();
        let mut out = Self::default();
        for r in 0..n {
            for c in 0..n {
                out.0[r][c] = cov[(rows[r], rows[c])];
            }
        }
        out
    }

    fn center(&self) -> [[f64; 2]; 2] {
        [[self.0[0][0], self.0[0][1]], [self.0[1][0], self.0[1][1]]]
    }

    /// Covariance of `(sigma, a, b)` from the `(log sigma, a, b)` block at
    /// `sigma` (delta method: `d sigma = sigma d log sigma`); the `a`, `b`
    /// entries are zero when the anisotropy is not reported.
    fn blur(&self, sigma: f64) -> [[f64; 3]; 3] {
        let scale = [sigma, 1.0, 1.0];
        let mut out = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = scale[r] * scale[c] * self.0[2 + r][2 + c];
            }
        }
        out
    }
}

/// Whether a symmetric 3×3 matrix is positive definite on its leading
/// `n` rows (the rest are zero).
fn positive_definite(m: &[[f64; 3]; 3], n: usize) -> bool {
    m.as_flattened().iter().all(|v| v.is_finite())
        && nalgebra::DMatrix::from_fn(n, n, |r, c| m[r][c])
            .cholesky()
            .is_some()
}

impl PhotometricFit {
    /// Reported centre standard deviation over the one white residuals would
    /// give (`sqrt` of the covariance trace ratio).
    pub(crate) fn covariance_inflation(&self) -> Option<f64> {
        let trace = |c: &[[f64; 2]; 2]| c[0][0] + c[1][1];
        let inflation = (trace(&self.center_cov) / trace(&self.center_cov_white)).sqrt();
        inflation.is_finite().then_some(inflation)
    }
}

/// Fit one marker.
pub(crate) fn fit_marker(
    problem: &PhotometricProblem<'_>,
) -> Result<PhotometricFit, PhotometricFailure> {
    let model = build_model(problem).ok_or(PhotometricFailure::DegenerateMap)?;
    let (pixels, masked_fraction) = collect_patch(problem, &model);
    let c_basis = match problem.codeword {
        Some(_) => CBasis::full(),
        None => CBasis::without_rotation(local_jacobian(&problem.local_map))
            .ok_or(PhotometricFailure::DegenerateMap)?,
    };
    let layout = Layout::new(
        c_basis,
        problem.fit_radii,
        problem.codeword.is_some(),
        problem.cubic_background,
        problem.anisotropic_blur,
    );
    if pixels.len() < 4 * layout.n() + 32 {
        return Err(PhotometricFailure::InsufficientPixels);
    }
    let seed = Params {
        p: problem.center_seed,
        c: [[1.0, 0.0], [0.0, 1.0]],
        log_sigma: 0.0,
        aniso: [0.0; 2],
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
    let sigma = sol.prm.log_sigma.exp();
    // Fallback: the prior-regularized inverse Fisher information.
    let posterior = Block::of(&cov, &fitting.layout);
    let sandwich = sandwich_covariance(&fitting, &sol, &cov, s2);
    let (center_cov, center_cov_white, residual_acf) = match &sandwich {
        Some(c) => (c.cov.center(), c.white.center(), c.acf),
        None => (
            posterior.center(),
            posterior.center(),
            ResidualAcf::default(),
        ),
    };
    let n_blur = fitting.layout.reported().1 - 2;
    let (blur_cov, blur_cov_white) = match &sandwich {
        Some(c) if positive_definite(&c.cov.blur(sigma), n_blur) => {
            (c.cov.blur(sigma), c.white.blur(sigma))
        }
        _ => (posterior.blur(sigma), posterior.blur(sigma)),
    };
    let radii_std = (fitting.layout.n_radii > 0).then(|| {
        let b = fitting.layout.radii_base();
        let mut std = [0.0; 3];
        for (k, s) in std.iter_mut().enumerate().take(fitting.layout.n_radii) {
            *s = cov[(b + k, b + k)].sqrt();
        }
        std
    });
    Ok(PhotometricFit {
        center: sol.prm.p,
        center_cov,
        center_cov_white,
        residual_acf,
        blur: BlurEstimate {
            sigma_px: sigma,
            aniso: sol.prm.aniso,
            cov: blur_cov,
            cov_white: blur_cov_white,
        },
        contrast: sol.prm.lin[CONTRAST],
        residual_sigma: s2.sqrt(),
        n_pixels: fitting.pixels.len(),
        masked_fraction,
        iterations: sol.iterations,
        converged: sol.converged,
        radii: sol.prm.radii,
        radii_std,
        #[cfg(feature = "study")]
        params: sol.prm,
    })
}

#[cfg(feature = "study")]
mod residuals;
#[cfg(feature = "study")]
pub use residuals::PhotometricResidualSample;
#[cfg(feature = "study")]
pub(crate) use residuals::residual_samples;

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use image::Luma;
    use rand::{RngExt, SeedableRng};

    /// Nominal default-board print: centerline radii 3.2 / 4.8 mm, 1.152 mm strokes.
    pub(super) const RADII: PrintRadii = [2.624, 3.776, 4.224, 5.376];
    pub(super) const CODEWORD: u16 = 0b1011_0010_1110_0100;
    pub(super) const PITCH_SPACING_MM: f64 = 13.856_406;

    pub(super) fn neighbors() -> Vec<InkDisc> {
        (0..6)
            .map(|k| {
                let a = k as f64 * std::f64::consts::PI / 3.0;
                InkDisc {
                    center: [PITCH_SPACING_MM * a.cos(), PITCH_SPACING_MM * a.sin()],
                    radius: RADII[3],
                }
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

    /// Printed ink at board point `u`; `code = None` is a plain annulus.
    fn ink(u: [f64; 2], code: Option<u16>) -> f64 {
        let rho = u[0].hypot(u[1]);
        for nb in neighbors() {
            if (u[0] - nb.center[0]).hypot(u[1] - nb.center[1]) <= nb.radius {
                return 1.0;
            }
        }
        if rho < RADII[0] || rho > RADII[3] {
            return 0.0;
        }
        if let Some(code) = code
            && rho > RADII[1]
            && rho < RADII[2]
        {
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
    fn render(
        map: &Mat3,
        size: u32,
        blur: f64,
        noise: f64,
        seed: u64,
        code: Option<u16>,
    ) -> GrayImage {
        quantize(&radiance(map, size, blur, code), size, noise, seed)
    }

    /// Noise-free radiance of the patch, row-major.
    pub(super) fn radiance(map: &Mat3, size: u32, blur: f64, code: Option<u16>) -> Vec<f64> {
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
                        acc += ink(homography_project_rows(&inv, [px, py]).unwrap(), code);
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
        nbs: &'a [InkDisc],
    ) -> PhotometricProblem<'a> {
        PhotometricProblem {
            gray,
            local_map: map,
            center_seed: seed,
            codeword: Some(CODEWORD),
            radii: RADII,
            fit_radii: false,
            cubic_background: false,
            ink_discs: nbs,
            anisotropic_blur: true,
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
            for code in [Some(CODEWORD), Some(0u16), Some(0xFFFF), None] {
                let gray = render(&map, 80, blur, 0.0, 1, code);
                let seed = [truth[0] + 0.6, truth[1] - 0.4];
                let mut pb = problem(&gray, perturbed(&map), seed, &nbs);
                pb.codeword = code;
                if code.is_none() {
                    pb.radii = [RADII[0], RADII[3], RADII[3], RADII[3]];
                }
                let fit = fit_marker(&pb).unwrap();
                let err = (fit.center[0] - truth[0]).hypot(fit.center[1] - truth[1]);
                eprintln!(
                    "blur {blur} tilt {tilt} code {code:?}: err {err:.4} px, sigma {:.3}, iters {}, conv {}",
                    fit.blur.sigma_px, fit.iterations, fit.converged
                );
                assert!(err < 0.5, "blur {blur}: centre error {err} px");
                assert!((fit.blur.sigma_px - blur.max(0.3)).abs() < 0.25 * blur + 0.15);
            }
        }
    }

    /// `map` magnified `k` times about the image point `c`: the same marker
    /// imaged `k` times larger.
    fn magnified(map: &Mat3, k: f64, c: [f64; 2]) -> Mat3 {
        let s = [
            [k, 0.0, c[0] * (1.0 - k)],
            [0.0, k, c[1] * (1.0 - k)],
            [0.0, 0.0, 1.0],
        ];
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|m| s[i][m] * map[m][j]).sum();
            }
        }
        out
    }

    /// A tilted plain annulus at every size from 20 to 111 px across: the
    /// fit converges within a few iterations and lands within the noise of
    /// the truth. Its rotation about the centre is fixed by the seed map
    /// (here 2° off); fitted freely, it is held only by the `C` prior, and
    /// the solver crept along it on large markers (10–17 iterations, 1 of 4
    /// fits not converging at 111 px and blur 5, centre errors 2–3× the
    /// noise).
    #[test]
    fn plain_marker_converges_at_every_size() {
        let nbs = neighbors();
        let plain = [RADII[0], RADII[3], RADII[3], RADII[3]];
        for (k, blur) in [
            (0.54f64, 0.8),
            (0.54, 2.0),
            (2.0, 3.0),
            (3.0, 1.5),
            (3.0, 4.0),
        ] {
            let size = (80.0 * k).ceil() as u32;
            let truth = [f64::from(size) / 2.0 + 0.3, f64::from(size) / 2.0 - 0.4];
            let map = magnified(&true_map(30.0, truth), k, truth);
            let rad = radiance(&map, size, blur, None);
            for seed in 0..3 {
                let gray = quantize(&rad, size, 2.0, 100 + seed);
                let seed_xy = [truth[0] + 0.6, truth[1] - 0.4];
                let mut pb = problem(&gray, perturbed(&map), seed_xy, &nbs);
                pb.codeword = None;
                pb.radii = plain;
                let fit = fit_marker(&pb).unwrap();
                let err = (fit.center[0] - truth[0]).hypot(fit.center[1] - truth[1]);
                let sd = (fit.center_cov[0][0] + fit.center_cov[1][1]).sqrt();
                let diameter = 2.0 * RADII[3] * 3.45 * k;
                eprintln!(
                    "diameter {diameter:.0} px blur {blur}: {} iterations, error {err:.4} px, sd {sd:.4} px",
                    fit.iterations
                );
                assert!(
                    fit.converged,
                    "{diameter:.0} px, blur {blur}: not converged"
                );
                assert!(
                    fit.iterations <= 10,
                    "{diameter:.0} px, blur {blur}: {} iterations",
                    fit.iterations
                );
                assert!(
                    err <= 4.0 * sd,
                    "{diameter:.0} px, blur {blur}: error {err} px, sd {sd} px"
                );
            }
        }
    }
}

#[cfg(test)]
mod monte_carlo {
    use super::tests::*;
    use super::*;
    use rand::{RngExt, SeedableRng};

    /// Radiance plus Gaussian noise of standard deviation `noise` whose
    /// spatial correlation is a Gaussian of `corr_px` (0 = white), quantized.
    pub(super) fn noisy(
        radiance: &[f64],
        size: u32,
        noise: f64,
        corr_px: f64,
        seed: u64,
    ) -> GrayImage {
        if corr_px <= 0.0 {
            return quantize(radiance, size, noise, seed);
        }
        let n = size as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut field: Vec<f64> = (0..n * n)
            .map(|_| {
                let u1: f64 = rng.random_range(f64::EPSILON..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let r = (4.0 * corr_px).ceil() as i64;
        let k: Vec<f64> = (-r..=r)
            .map(|i| (-((i * i) as f64) / (2.0 * corr_px * corr_px)).exp())
            .collect();
        // A unit-variance field after the separable blur.
        let norm = k.iter().map(|v| v * v).sum::<f64>();
        for pass in 0..2 {
            let src = field.clone();
            for y in 0..n as i64 {
                for x in 0..n as i64 {
                    let mut acc = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let o = j as i64 - r;
                        let (xx, yy) = if pass == 0 { (x + o, y) } else { (x, y + o) };
                        if (0..n as i64).contains(&xx) && (0..n as i64).contains(&yy) {
                            acc += kv * src[(yy * n as i64 + xx) as usize];
                        }
                    }
                    field[(y * n as i64 + x) as usize] = acc / norm.sqrt();
                }
            }
        }
        let mut img = GrayImage::new(size, size);
        for (i, v) in radiance.iter().enumerate() {
            let q = (v + noise * field[i]).round().clamp(0.0, 255.0);
            img.put_pixel((i % n) as u32, (i / n) as u32, image::Luma([q as u8]));
        }
        img
    }

    /// Reported over empirical standard deviation of repeated fits, for the
    /// white-residual and the correlated-residual covariance.
    #[derive(Debug, Clone, Copy)]
    struct Ratios {
        white: f64,
        corr: f64,
    }

    /// Monte-Carlo honesty of one fit configuration.
    struct MonteCarlo {
        /// Centre (pooled over both axes).
        center: Ratios,
        /// `sigma`, `a`, `b` of the PSF.
        blur: [Ratios; 3],
        /// Mean fitted sigma and anisotropy over the seeds, and their
        /// empirical standard deviations.
        blur_mean: [f64; 3],
        blur_sd: [f64; 3],
    }

    /// Repeated fits of one noise-free radiance under independent noise.
    fn monte_carlo(blur: f64, tilt: f64, code: Option<u16>, corr_px: f64) -> MonteCarlo {
        let nbs = neighbors();
        let truth = [40.3, 39.6];
        let map = true_map(tilt, truth);
        let rad = radiance(&map, 80, blur, code);
        let n = 120;
        let mut centers = Vec::with_capacity(n);
        let mut blurs: Vec<[f64; 3]> = Vec::with_capacity(n);
        let (mut tr_white, mut tr_corr) = (0.0, 0.0);
        let (mut var_white, mut var_corr) = ([0.0; 3], [0.0; 3]);
        for seed in 0..n as u64 {
            let gray = noisy(&rad, 80, 2.0, corr_px, 1000 + seed);
            let mut pb = problem(&gray, perturbed(&map), [40.8, 39.2], &nbs);
            pb.codeword = code;
            if code.is_none() {
                pb.radii = [RADII[0], RADII[3], RADII[3], RADII[3]];
            }
            let fit = fit_marker(&pb).unwrap();
            centers.push(fit.center);
            tr_white += fit.center_cov_white[0][0] + fit.center_cov_white[1][1];
            tr_corr += fit.center_cov[0][0] + fit.center_cov[1][1];
            let b = &fit.blur;
            blurs.push([b.sigma_px, b.aniso[0], b.aniso[1]]);
            for k in 0..3 {
                var_white[k] += b.cov_white[k][k];
                var_corr[k] += b.cov[k][k];
            }
        }
        let mean = |f: &dyn Fn(usize) -> f64| (0..n).map(f).sum::<f64>() / n as f64;
        let var = |f: &dyn Fn(usize) -> f64| {
            let m = mean(f);
            (0..n).map(|i| (f(i) - m).powi(2)).sum::<f64>() / (n - 1) as f64
        };
        let m = [mean(&|i| centers[i][0]), mean(&|i| centers[i][1])];
        let emp_tr = var(&|i| centers[i][0]) + var(&|i| centers[i][1]);
        let bias = (m[0] - truth[0]).hypot(m[1] - truth[1]);
        let ratio = |reported: f64, empirical: f64| (reported / n as f64 / empirical).sqrt();
        let center = Ratios {
            white: ratio(tr_white, emp_tr),
            corr: ratio(tr_corr, emp_tr),
        };
        let blur_mean: [f64; 3] = std::array::from_fn(|k| mean(&|i| blurs[i][k]));
        let blur_var: [f64; 3] = std::array::from_fn(|k| var(&|i| blurs[i][k]));
        let blur_ratios: [Ratios; 3] = std::array::from_fn(|k| Ratios {
            white: ratio(var_white[k], blur_var[k]),
            corr: ratio(var_corr[k], blur_var[k]),
        });
        // The renderer integrates over the pixel (a box, variance 1/12 per
        // axis) before the Gaussian blur.
        let effective = (blur * blur + 1.0 / 12.0).sqrt();
        eprintln!(
            "blur {blur} tilt {tilt} code {:?} noise corr {corr_px}: centre bias {bias:.4} px, empirical rms {:.4} px, reported/empirical white {:.3} correlated {:.3}",
            code.is_some(),
            emp_tr.sqrt(),
            center.white,
            center.corr
        );
        eprintln!(
            "    sigma mean {:.4} (effective {effective:.4}, bias {:+.4}) sd {:.4} ratio white {:.3} corr {:.3} | a mean {:+.4} sd {:.4} ratio {:.3} {:.3} | b mean {:+.4} sd {:.4} ratio {:.3} {:.3}",
            blur_mean[0],
            blur_mean[0] - effective,
            blur_var[0].sqrt(),
            blur_ratios[0].white,
            blur_ratios[0].corr,
            blur_mean[1],
            blur_var[1].sqrt(),
            blur_ratios[1].white,
            blur_ratios[1].corr,
            blur_mean[2],
            blur_var[2].sqrt(),
            blur_ratios[2].white,
            blur_ratios[2].corr,
        );
        MonteCarlo {
            center,
            blur: blur_ratios,
            blur_mean,
            blur_sd: blur_var.map(f64::sqrt),
        }
    }

    /// With white noise (σ = 2 grey levels, 40 levels of contrast) both
    /// covariances match the scatter of repeated fits, for the centre and for
    /// the PSF `(sigma, a, b)`.
    #[test]
    fn reported_covariance_matches_monte_carlo_scatter() {
        for (blur, tilt, code) in [
            (1.0, 10.0, Some(CODEWORD)),
            (3.0, 30.0, Some(CODEWORD)),
            (1.5, 20.0, None),
        ] {
            let mc = monte_carlo(blur, tilt, code, 0.0);
            for (name, r) in [("centre", mc.center)]
                .into_iter()
                .chain(["sigma", "a", "b"].into_iter().zip(mc.blur))
            {
                for ratio in [r.white, r.corr] {
                    assert!(
                        (0.8..=1.25).contains(&ratio),
                        "blur {blur}: {name} sd ratio {ratio}"
                    );
                }
            }
        }
    }

    /// Spatially correlated noise: the white-residual covariance is
    /// overconfident; the correlated-residual one is not (centre and PSF).
    #[test]
    fn correlated_noise_covariance_matches_monte_carlo_scatter() {
        for (blur, tilt, code) in [(1.5, 20.0, Some(CODEWORD)), (1.5, 20.0, None)] {
            let mc = monte_carlo(blur, tilt, code, 1.0);
            for (name, r) in [("centre", mc.center)]
                .into_iter()
                .chain(["sigma", "a", "b"].into_iter().zip(mc.blur))
            {
                assert!(
                    r.white < 0.8,
                    "{name}: white covariance should be overconfident: {}",
                    r.white
                );
                assert!(
                    (0.8..=1.25).contains(&r.corr),
                    "blur {blur}: {name} sd ratio {}",
                    r.corr
                );
            }
        }
    }

    /// Honesty of the blur covariance `(sigma, a, b)` across blur, marker
    /// kind and noise correlation (study; prints the table).
    #[test]
    #[ignore = "study: blur covariance honesty sweep"]
    fn blur_covariance_sweep() {
        let cases: Vec<(f64, f64, Option<u16>, f64)> = [0.0, 1.0]
            .into_iter()
            .flat_map(|corr| {
                [Some(CODEWORD), None].into_iter().flat_map(move |code| {
                    [(0.4, 10.0), (0.8, 10.0), (1.5, 20.0), (3.0, 30.0)]
                        .into_iter()
                        .map(move |(blur, tilt)| (blur, tilt, code, corr))
                })
            })
            .collect();
        let rows: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = cases
                .iter()
                .map(|&(blur, tilt, code, corr)| {
                    scope.spawn(move || {
                        let mc = monte_carlo(blur, tilt, code, corr);
                        format!(
                            "| {} | {blur} | {tilt} | {corr} | {:.3}/{:.3} | {:.4} ± {:.4} | {:.3}/{:.3} | {:+.4} ± {:.4} | {:.3}/{:.3} | {:+.4} ± {:.4} | {:.3}/{:.3} |",
                            if code.is_some() { "coded" } else { "plain" },
                            mc.center.white,
                            mc.center.corr,
                            mc.blur_mean[0],
                            mc.blur_sd[0],
                            mc.blur[0].white,
                            mc.blur[0].corr,
                            mc.blur_mean[1],
                            mc.blur_sd[1],
                            mc.blur[1].white,
                            mc.blur[1].corr,
                            mc.blur_mean[2],
                            mc.blur_sd[2],
                            mc.blur[2].white,
                            mc.blur[2].corr,
                        )
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        println!(
            "| kind | blur | tilt | noise corr | centre w/c | sigma | sigma w/c | a | a w/c | b | b w/c |"
        );
        for r in rows {
            println!("{r}");
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
        for (blur, corr_px) in [(1.0, 0.0), (3.0, 0.0), (1.5, 1.0)] {
            let map = true_map(20.0, truth);
            let rad = radiance(&map, 80, blur, Some(CODEWORD));
            let gray = super::monte_carlo::noisy(&rad, 80, 2.0, corr_px, 7);
            for (fit_radii, aniso) in [(false, false), (false, true), (true, true)] {
                let mut pb = problem(&gray, perturbed(&map), [40.8, 39.2], &nbs);
                pb.fit_radii = fit_radii;
                pb.anisotropic_blur = aniso;
                let n = 200;
                let t0 = std::time::Instant::now();
                let mut it = 0;
                for _ in 0..n {
                    it += fit_marker(&pb).unwrap().iterations;
                }
                eprintln!(
                    "blur {blur} noise corr {corr_px} fit_radii {fit_radii} anisotropic {aniso}: {:.3} ms per fit, {:.1} iterations",
                    t0.elapsed().as_secs_f64() * 1e3 / n as f64,
                    it as f64 / n as f64
                );
            }
        }
    }
}
