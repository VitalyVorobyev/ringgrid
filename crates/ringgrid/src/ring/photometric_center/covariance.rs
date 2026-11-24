//! Fit uncertainty from the final weighted residuals and Jacobian.

use super::{
    Accumulated, Fitting, HUBER_K, Layout, N_MAX, PhotometricFailure, Solution, huber_weight,
    pixel_row,
};
use nalgebra::DMatrix;

/// Lags (px) of the reported residual autocorrelation diagnostic.
const RESIDUAL_ACF_LAGS: usize = 6;
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

/// Largest number of reported parameters: centre (2), log sigma, anisotropy
/// (2).
pub(super) const N_REPORTED: usize = 5;

/// Spatial autocorrelation of the (Huber-weighted) fit residuals at lags
/// `1..=RESIDUAL_ACF_LAGS` px along the image x and y axes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct ResidualAcf {
    pub x: [f64; RESIDUAL_ACF_LAGS],
    pub y: [f64; RESIDUAL_ACF_LAGS],
}

/// Sampling covariance blocks and diagnostics returned at the fit boundary.
pub(super) struct FitUncertainty {
    pub center: [[f64; 2]; 2],
    pub center_white: [[f64; 2]; 2],
    pub residual_acf: ResidualAcf,
    pub blur: [[f64; 3]; 3],
    #[cfg(any(test, feature = "study"))]
    pub blur_white: [[f64; 3]; 3],
    pub residual_sigma: f64,
}

impl FitUncertainty {
    pub(super) fn estimate(
        fitting: &Fitting<'_>,
        sol: &Solution,
        sigma: f64,
    ) -> Result<Self, PhotometricFailure> {
        let (cov, s2) = fitting.inverse_fisher(&sol.acc)?;
        // Fallback: the prior-regularized inverse Fisher information.
        let posterior = Block::of(&cov, &fitting.layout);
        let sandwich = sandwich_covariance(fitting, sol, &cov, s2);
        let (center_cov, center_cov_white, residual_acf) = match &sandwich {
            Some(c) => (c.cov.center(), c.white.center(), c.acf),
            None => (
                posterior.center(),
                posterior.center(),
                ResidualAcf::default(),
            ),
        };
        let n_blur = fitting.layout.reported().1 - 2;
        // The blur block (correlated, white): the sandwich's, else the posterior.
        let blur_blocks = match &sandwich {
            Some(c) if positive_definite(&c.cov.blur(sigma), n_blur) => (&c.cov, &c.white),
            _ => (&posterior, &posterior),
        };
        Ok(Self {
            center: center_cov,
            center_white: center_cov_white,
            residual_acf,
            blur: blur_blocks.0.blur(sigma),
            #[cfg(any(test, feature = "study"))]
            blur_white: blur_blocks.1.blur(sigma),
            residual_sigma: s2.sqrt(),
        })
    }
}

impl Fitting<'_> {
    /// Parameter covariance assuming white residuals: inverse Fisher
    /// information with the noise variance estimated from the weighted
    /// residuals; also returns that variance.
    fn inverse_fisher(&self, acc: &Accumulated) -> Result<(DMatrix<f64>, f64), PhotometricFailure> {
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
