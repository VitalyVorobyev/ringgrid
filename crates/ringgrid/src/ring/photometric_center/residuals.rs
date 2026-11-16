//! Per-pixel residuals of a finished fit, for studying where the intensity
//! model fails (`study` feature).

use super::{
    Params, PhotometricFit, PhotometricProblem, build_model, collect_patch, photometry,
    sector_index,
};

impl Params {
    /// The PSF anisotropy `[a, b]`.
    pub(crate) fn aniso(&self) -> [f64; 2] {
        self.aniso
    }
}

/// One patch pixel of a fit, with its residual and its position in marker
/// coordinates. Diagnostic data for studying where the intensity model fails
/// (see `Detector::detect_with_photometric_residuals`, `study` feature).
///
/// Edge distances are in image pixels, positive on the ink side of the edge
/// (so a positive `d_outer` is inside the outer ring), and measured to the
/// nominal sharp edge: the small curvature shift of a blurred edge is not
/// included. They are `NaN` when the marker has no edge of that class near the
/// pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotometricResidualSample {
    /// Pixel column.
    pub x: i32,
    /// Pixel row.
    pub y: i32,
    /// Observed grey level.
    pub observed: f32,
    /// Model grey level at the fitted parameters.
    pub model: f32,
    /// Blurred ink coverage `P` of the model, 0 (paper) to 1 (ink).
    pub ink: f32,
    /// Board-frame radius in units of the marker's outer print radius.
    pub radius: f32,
    /// Board-frame polar angle of the pixel, radians in `(-pi, pi]`.
    pub angle: f32,
    /// Signed distance to the nearest ring edge of the inner ring (the hole
    /// edge, and for coded markers the code band's inner edge where it is paper).
    pub d_inner: f32,
    /// Signed distance to the nearest edge of the outer ring (for coded
    /// markers the code band's outer edge where it is paper, and the outer
    /// edge).
    pub d_outer: f32,
    /// Signed distance to the nearest code-sector boundary (coded markers,
    /// radially inside the code band only).
    pub d_code: f32,
    /// Distance (unsigned) to the nearest edge of any kind, including
    /// neighbouring markers and fiducial dots; infinite when none is modelled.
    pub d_any: f32,
}

impl PhotometricResidualSample {
    /// Observed minus model grey level.
    pub fn residual(&self) -> f32 {
        self.observed - self.model
    }
}

/// Nearest signed distance by magnitude, `NaN` when empty.
fn nearest(cands: impl IntoIterator<Item = f64>) -> f64 {
    cands
        .into_iter()
        .filter(|d| d.is_finite())
        .min_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(f64::NAN)
}

/// Re-evaluate a finished fit over its patch and return every pixel's
/// residual with its marker coordinates. `problem` must be the one the fit
/// was made from (its `center_seed` fixes the patch); nothing is refitted.
/// Used only on request, so the detection path pays nothing for it.
pub(crate) fn residual_samples(
    problem: &PhotometricProblem<'_>,
    fit: &PhotometricFit,
) -> Option<Vec<PhotometricResidualSample>> {
    let model = build_model(problem)?;
    let (pixels, _) = collect_patch(problem, &model);
    let prm = &fit.params;
    let r = fit.radii;
    let r_out = r[3];
    let c = &prm.c;
    let mut out = Vec::with_capacity(pixels.len());
    for px in &pixels {
        let Some(pe) = model.pattern(px.x, prm) else {
            continue;
        };
        let (value, _) = photometry(&prm.lin, px.xi, pe.p);
        let dx = [px.x[0] - prm.p[0], px.x[1] - prm.p[1]];
        let y = [
            model.h0[0] + c[0][0] * dx[0] + c[0][1] * dx[1],
            model.h0[1] + c[1][0] * dx[0] + c[1][1] * dx[1],
        ];
        let Some((u, j)) = model.unproject(y) else {
            continue;
        };
        // Image px per board mm across an edge with board normal `n`.
        let gnorm = |n: [f64; 2]| {
            let v = [
                j[0][0] * n[0] + j[1][0] * n[1],
                j[0][1] * n[0] + j[1][1] * n[1],
            ];
            (c[0][0] * v[0] + c[1][0] * v[1]).hypot(c[0][1] * v[0] + c[1][1] * v[1])
        };
        let rho = u[0].hypot(u[1]);
        let (mut d_inner, mut d_outer, mut d_code, mut d_any) =
            (f64::NAN, f64::NAN, f64::NAN, f64::INFINITY);
        if rho > 1e-9 {
            let g = gnorm([u[0] / rho, u[1] / rho]);
            if g > 1e-12 {
                let d = |k: usize| (rho - r[k]) / g;
                // Ink on the outside of r0 and r2, on the inside of r1 and r3.
                let mut inner = vec![d(0)];
                let mut outer = vec![-d(3)];
                let mut all = vec![d(0), d(3)];
                if let Some(code) = &model.code {
                    // r1 and r2 are edges only where the code sector is paper.
                    if code.bits[sector_index(u)] > 0.5 {
                        inner.push(-d(1));
                        outer.push(d(2));
                        all.push(d(1));
                        all.push(d(2));
                    }
                    if rho >= r[1] && rho <= r[2] {
                        let mut sectors = Vec::new();
                        for tr in &code.transitions {
                            if tr.dir[0] * u[0] + tr.dir[1] * u[1] <= 0.0 {
                                continue;
                            }
                            let n = [-tr.dir[1], tr.dir[0]];
                            let gn = gnorm(n);
                            if gn <= 1e-12 {
                                continue;
                            }
                            let t = (n[0] * u[0] + n[1] * u[1]) / gn;
                            // After the boundary is paper when delta > 0: ink
                            // lies on the negative side.
                            sectors.push(if tr.delta > 0.0 { -t } else { t });
                        }
                        all.extend(sectors.iter().copied());
                        d_code = nearest(sectors);
                    }
                }
                d_inner = nearest(inner);
                d_outer = nearest(outer);
                for disc in model.ink_discs {
                    let un = [u[0] - disc.center[0], u[1] - disc.center[1]];
                    let rn = un[0].hypot(un[1]);
                    if rn < 1e-9 {
                        continue;
                    }
                    let gd = gnorm([un[0] / rn, un[1] / rn]);
                    if gd > 1e-12 {
                        all.push((rn - disc.radius) / gd);
                    }
                }
                d_any = all
                    .into_iter()
                    .filter(|d| d.is_finite())
                    .map(f64::abs)
                    .fold(f64::INFINITY, f64::min);
            }
        }
        out.push(PhotometricResidualSample {
            x: px.x[0] as i32,
            y: px.x[1] as i32,
            observed: px.value as f32,
            model: value as f32,
            ink: pe.p as f32,
            radius: (rho / r_out) as f32,
            angle: u[1].atan2(u[0]) as f32,
            d_inner: d_inner as f32,
            d_outer: d_outer as f32,
            d_code: d_code as f32,
            d_any: d_any as f32,
        });
    }
    Some(out)
}
