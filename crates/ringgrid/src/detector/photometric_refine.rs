//! Photometric centre refinement stage.
//!
//! Runs once on the final marker set of a coded target that has a board
//! homography: every decoded marker is re-fitted with the blurred-pattern
//! model of [`crate::ring::photometric_center`], seeded from its current
//! (projective) centre and the board homography. Accepted fits replace the
//! centre and attach a covariance; rejected ones leave the marker untouched
//! and record why.

use image::GrayImage;

use crate::detector::{
    CenterSource, DetectConfig, MarkerRecord, PhotometricMetrics, PhotometricStatus,
};
use crate::marker::codec::Codebook;
use crate::pixelmap::PixelMapper;
use crate::ring::photometric_center::{
    Mat3, PhotometricFailure, PhotometricFit, PhotometricProblem, PrintRadii, fit_marker,
};
use crate::target::{MarkerCoding, TargetLayout};

/// Neighbours are the cells within this multiple of the lattice spacing.
const NEIGHBOR_SPACING_FACTOR: f64 = 1.15;
/// Minimum number of well-determined markers for a print-radii consensus.
const MIN_RADII_SAMPLES: usize = 6;
/// Fits whose residual exceeds this multiple of the image median are retried
/// with a cubic paper level.
const CUBIC_RETRY_RATIO: f64 = 1.5;
/// A cubic refit replaces the first fit only when it lowers the residual
/// standard deviation below this fraction: four more background terms always
/// lower it a little, at a cost in centre precision.
const CUBIC_ADOPT_RATIO: f64 = 0.85;
/// Markers fitted with free radii for the print-radii consensus.
const MAX_RADII_MARKERS: usize = 16;
/// Per-marker radius estimates less precise than this (mm) are not used.
const MAX_RADIUS_STD_MM: f64 = 0.1;
/// Plausible fitted PSF sigma range (pixels).
const SIGMA_RANGE_PX: (f64, f64) = (0.2, 15.0);
/// Minimum fitted contrast in units of the residual noise.
const MIN_CONTRAST_SNR: f64 = 3.0;
/// A fit whose residual standard deviation exceeds this multiple of the
/// image-wide median is a model misfit (a glare halo, an occluder, a print
/// defect); its centre is not trusted.
const MAX_RESIDUAL_RATIO: f64 = 3.0;

fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn project(h: &Mat3, p: [f64; 2]) -> Option<[f64; 2]> {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    (w.abs() > 1e-12).then(|| {
        [
            (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
            (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
        ]
    })
}

/// Board-local (mm relative to `cell_mm`) → image-pixel projective map.
///
/// Without a mapper the board homography is already board → image and the
/// map is exact. With a mapper the homography is board → working frame; the
/// image-space map is then the best projective fit to the mapped marker
/// neighbourhood (the per-marker affine correction of the fit absorbs the rest).
fn local_map(
    h: &Mat3,
    cell_mm: [f64; 2],
    r_outer_mm: f64,
    mapper: Option<&dyn PixelMapper>,
) -> Option<Mat3> {
    let shift = [
        [1.0, 0.0, cell_mm[0]],
        [0.0, 1.0, cell_mm[1]],
        [0.0, 0.0, 1.0],
    ];
    let to_working = mat3_mul(h, &shift);
    let Some(mapper) = mapper else {
        return Some(to_working);
    };
    let mut src = Vec::with_capacity(25);
    let mut dst = Vec::with_capacity(25);
    for i in -2..=2 {
        for j in -2..=2 {
            let u = [
                0.65 * r_outer_mm * f64::from(i),
                0.65 * r_outer_mm * f64::from(j),
            ];
            let w = project(&to_working, u)?;
            let x = mapper.working_to_image_pixel(w)?;
            src.push(u);
            dst.push(x);
        }
    }
    let m = crate::homography::estimate_homography_dlt(&src, &dst).ok()?;
    Some(crate::homography::matrix3_to_array(&m))
}

/// Nominal printed radii of a coded marker (board mm).
fn nominal_radii(target: &TargetLayout) -> Option<PrintRadii> {
    let MarkerCoding::Coded16(spec) = target.coding() else {
        return None;
    };
    let ring = target.ring();
    let h = 0.5 * f64::from(spec.ring_width_mm);
    let inner = f64::from(ring.inner_radius_mm);
    let outer = f64::from(ring.outer_radius_mm);
    Some([inner - h, inner + h, outer - h, outer + h])
}

/// One marker's fit inputs.
struct Task {
    index: usize,
    local_map: Mat3,
    codeword: u16,
    neighbors_mm: Vec<[f64; 2]>,
    /// Image px per board mm at the marker (geometric mean).
    px_per_mm: f64,
}

fn build_tasks(
    markers: &[MarkerRecord],
    h: &Mat3,
    mapper: Option<&dyn PixelMapper>,
    target: &TargetLayout,
    codebook: &Codebook,
    r_outer_mm: f64,
) -> Vec<Task> {
    let spacing = f64::from(target.min_center_spacing_mm());
    let reach = NEIGHBOR_SPACING_FACTOR * spacing;
    let cells: Vec<[f64; 2]> = target
        .cells()
        .iter()
        .map(|c| [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])])
        .collect();
    let mut tasks = Vec::new();
    for (index, m) in markers.iter().enumerate() {
        let Some(id) = m.id else { continue };
        let Some(xy) = target.xy_mm_of_id(id) else {
            continue;
        };
        let Some(codeword) = codebook.word(id) else {
            continue;
        };
        if !(m.center[0].is_finite() && m.center[1].is_finite()) {
            continue;
        }
        let cell = [f64::from(xy[0]), f64::from(xy[1])];
        let Some(map) = local_map(h, cell, r_outer_mm, mapper) else {
            continue;
        };
        // Local scale from the map's Jacobian at the marker centre.
        let Some(c0) = project(&map, [0.0, 0.0]) else {
            continue;
        };
        let (Some(cx), Some(cy)) = (project(&map, [1.0, 0.0]), project(&map, [0.0, 1.0])) else {
            continue;
        };
        let det = ((cx[0] - c0[0]) * (cy[1] - c0[1]) - (cx[1] - c0[1]) * (cy[0] - c0[0])).abs();
        if !(det.is_finite() && det > 0.0) {
            continue;
        }
        let neighbors_mm = cells
            .iter()
            .filter_map(|c| {
                let d = [c[0] - cell[0], c[1] - cell[1]];
                let r = d[0].hypot(d[1]);
                (r > 1e-6 && r <= reach).then_some(d)
            })
            .collect();
        tasks.push(Task {
            index,
            local_map: map,
            codeword,
            neighbors_mm,
            px_per_mm: det.sqrt(),
        });
    }
    tasks
}

/// How a marker is fitted.
#[derive(Clone, Copy)]
struct FitMode {
    fit_radii: bool,
    cubic_background: bool,
}

/// Inputs shared by every fit of one detection.
struct Stage<'a> {
    gray: &'a GrayImage,
    config: &'a DetectConfig,
    tasks: Vec<Task>,
    patch_margin_mm: f64,
}

impl Stage<'_> {
    fn fit(
        &self,
        task: &Task,
        seed: [f64; 2],
        radii: PrintRadii,
        mode: FitMode,
    ) -> Result<PhotometricFit, PhotometricFailure> {
        let photometric = &self.config.advanced.photometric;
        fit_marker(&PhotometricProblem {
            gray: self.gray,
            local_map: task.local_map,
            center_seed: seed,
            codeword: task.codeword,
            radii,
            fit_radii: mode.fit_radii,
            cubic_background: mode.cubic_background,
            neighbors_mm: &task.neighbors_mm,
            patch_margin_mm: self.patch_margin_mm,
            saturation_level: photometric.saturation_level,
            max_iterations: photometric.max_iterations.max(1),
        })
    }

    /// Printed radii from free-radii fits of an evenly spread subset of the
    /// markers (a handful of shared radii needs no more).
    fn estimate_print_radii(
        &self,
        markers: &[MarkerRecord],
        nominal: PrintRadii,
    ) -> Option<PrintRadii> {
        let stride = self.tasks.len().div_ceil(MAX_RADII_MARKERS).max(1);
        let mode = FitMode {
            fit_radii: true,
            cubic_background: false,
        };
        let fits: Vec<PhotometricFit> = self
            .tasks
            .iter()
            .step_by(stride)
            .filter_map(|t| self.fit(t, markers[t.index].center, nominal, mode).ok())
            .collect();
        let estimated = radii_consensus(&fits, nominal);
        tracing::debug!(?estimated, n_fits = fits.len(), "photometric print radii");
        estimated
    }

    /// Centre fits with fixed radii, plus the image's typical fit residual.
    ///
    /// An affine paper level cannot follow a steep glare halo, so markers
    /// whose residual is well above the typical one are refitted with a cubic
    /// paper level and keep whichever fit explains the pixels better.
    fn fit_centres(
        &self,
        markers: &[MarkerRecord],
        radii: PrintRadii,
    ) -> (Vec<Result<PhotometricFit, PhotometricFailure>>, f64) {
        let mode = |cubic_background| FitMode {
            fit_radii: false,
            cubic_background,
        };
        let mut fits: Vec<_> = self
            .tasks
            .iter()
            .map(|t| self.fit(t, markers[t.index].center, radii, mode(false)))
            .collect();
        let mut residuals: Vec<f64> = fits
            .iter()
            .filter_map(|f| f.as_ref().ok())
            .filter(|f| f.converged && f.residual_sigma.is_finite())
            .map(|f| f.residual_sigma)
            .collect();
        residuals.sort_by(|a, b| a.total_cmp(b));
        let typical = residuals
            .get(residuals.len() / 2)
            .copied()
            .unwrap_or(f64::INFINITY);
        for (t, fit) in self.tasks.iter().zip(fits.iter_mut()) {
            let Ok(first) = fit.as_ref() else {
                continue;
            };
            if first.residual_sigma <= CUBIC_RETRY_RATIO * typical {
                continue;
            }
            if let Ok(cubic) = self.fit(t, markers[t.index].center, radii, mode(true))
                && cubic.converged
                && cubic.residual_sigma < CUBIC_ADOPT_RATIO * first.residual_sigma
            {
                *fit = Ok(cubic);
            }
        }
        (fits, typical)
    }
}

/// Precision-weighted consensus of per-marker radius estimates with a
/// 4-sigma outlier cut; `None` when too few markers determine them.
fn radii_consensus(fits: &[PhotometricFit], nominal: PrintRadii) -> Option<PrintRadii> {
    let samples: Vec<([f64; 3], [f64; 3])> = fits
        .iter()
        .filter(|f| f.converged)
        .filter_map(|f| {
            let std = f.radii_std?;
            std.iter()
                .all(|s| s.is_finite() && *s > 0.0 && *s < MAX_RADIUS_STD_MM)
                .then_some(([f.radii[0], f.radii[1], f.radii[2]], std))
        })
        .collect();
    if samples.len() < MIN_RADII_SAMPLES {
        return None;
    }
    let mut out = nominal;
    for k in 0..3 {
        let mean_of = |keep: &dyn Fn(f64, f64) -> bool| {
            let (mut sw, mut swx, mut n) = (0.0, 0.0, 0usize);
            for (r, s) in &samples {
                if keep(r[k], s[k]) {
                    let w = 1.0 / (s[k] * s[k]);
                    sw += w;
                    swx += w * r[k];
                    n += 1;
                }
            }
            (n >= MIN_RADII_SAMPLES).then(|| swx / sw)
        };
        let first = mean_of(&|_, _| true)?;
        // Spread of the samples around the first mean, robustly.
        let mut dev: Vec<f64> = samples.iter().map(|(r, _)| (r[k] - first).abs()).collect();
        dev.sort_by(|a, b| a.total_cmp(b));
        let spread = 1.4826 * dev[dev.len() / 2];
        out[k] = mean_of(&|r, s| (r - first).abs() <= 4.0 * s.hypot(spread))?;
    }
    let ordered = out[0] > 0.0 && out[0] < out[1] && out[1] < out[2] && out[2] < out[3];
    ordered.then_some(out)
}

fn failure_status(f: PhotometricFailure) -> PhotometricStatus {
    match f {
        PhotometricFailure::InsufficientPixels => PhotometricStatus::InsufficientPixels,
        PhotometricFailure::DegenerateMap | PhotometricFailure::SingularSystem => {
            PhotometricStatus::Degenerate
        }
    }
}

/// Accept or reject one fit and summarize it.
fn judge(
    fit: &PhotometricFit,
    seed: [f64; 2],
    max_shift_px: f64,
    typical_residual: f64,
) -> PhotometricMetrics {
    let shift = (fit.center[0] - seed[0]).hypot(fit.center[1] - seed[1]);
    let cov_ok = fit.center_cov[0][0] > 0.0
        && fit.center_cov[1][1] > 0.0
        && fit.center_cov.iter().flatten().all(|v| v.is_finite());
    let status = if !fit.converged {
        PhotometricStatus::NotConverged
    } else if !(shift.is_finite() && shift <= max_shift_px) {
        PhotometricStatus::ShiftTooLarge
    } else if !(SIGMA_RANGE_PX.0..=SIGMA_RANGE_PX.1).contains(&fit.blur_sigma_px)
        || fit.contrast < MIN_CONTRAST_SNR * fit.residual_sigma
        || !cov_ok
    {
        PhotometricStatus::Implausible
    } else if fit.residual_sigma > MAX_RESIDUAL_RATIO * typical_residual {
        PhotometricStatus::PoorFit
    } else {
        PhotometricStatus::Accepted
    };
    PhotometricMetrics {
        status,
        blur_sigma_px: Some(fit.blur_sigma_px),
        contrast: Some(fit.contrast),
        residual_sigma: Some(fit.residual_sigma),
        n_pixels: fit.n_pixels,
        masked_fraction: fit.masked_fraction,
        center_shift_px: Some(shift),
    }
}

/// Refine the centres of decoded markers in place.
///
/// `homography` maps board mm to the working frame; `mapper` maps that frame
/// to image pixels and must be `Some` exactly when the two differ. Marker
/// centres are image pixels on entry and on exit. Returns the print radii the
/// final fits used when they were estimated from the image.
pub(crate) fn apply_photometric_centers(
    gray: &GrayImage,
    markers: &mut [MarkerRecord],
    homography: Option<&Mat3>,
    mapper: Option<&dyn PixelMapper>,
    config: &DetectConfig,
) -> Option<PrintRadii> {
    if !config.circle_refinement.uses_photometric() {
        return None;
    }
    let target = &config.target;
    let h = homography?;
    let nominal = nominal_radii(target)?;
    let MarkerCoding::Coded16(spec) = target.coding() else {
        return None;
    };
    let codebook = Codebook::from_profile(spec.codebook_profile);
    let tasks = build_tasks(markers, h, mapper, target, &codebook, nominal[3]);
    if tasks.is_empty() {
        return None;
    }
    let spacing = f64::from(target.min_center_spacing_mm());
    let stage = Stage {
        gray,
        config,
        tasks,
        patch_margin_mm: (0.5 * (spacing - 2.0 * nominal[3])).clamp(0.2, 0.5 * nominal[3]),
    };

    let estimated = if config.advanced.photometric.estimate_print_radii {
        stage.estimate_print_radii(markers, nominal)
    } else {
        None
    };
    let (fits, typical_residual) = stage.fit_centres(markers, estimated.unwrap_or(nominal));

    let mut n_accepted = 0usize;
    let mut iterations = 0usize;
    for (t, fit) in stage.tasks.iter().zip(fits) {
        let m = &mut markers[t.index];
        let max_shift_px = config
            .advanced
            .photometric
            .max_center_shift_px
            .unwrap_or(0.5 * nominal[3] * t.px_per_mm);
        let metrics = match fit {
            Err(f) => PhotometricMetrics {
                status: failure_status(f),
                blur_sigma_px: None,
                contrast: None,
                residual_sigma: None,
                n_pixels: 0,
                masked_fraction: 0.0,
                center_shift_px: None,
            },
            Ok(fit) => {
                iterations += fit.iterations;
                let metrics = judge(&fit, m.center, max_shift_px, typical_residual);
                if metrics.status == PhotometricStatus::Accepted {
                    m.center = fit.center;
                    m.center_cov = Some(fit.center_cov);
                    m.center_source = CenterSource::Photometric;
                    if let Some(mapper) = mapper {
                        m.center_mapped = mapper.image_to_working_pixel(fit.center);
                    }
                    n_accepted += 1;
                }
                metrics
            }
        };
        m.photometric = Some(metrics);
    }
    tracing::debug!(
        n_markers = stage.tasks.len(),
        n_accepted,
        iterations,
        typical_residual,
        "photometric centre refinement summary"
    );
    estimated
}
