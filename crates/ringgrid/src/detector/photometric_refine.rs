//! Photometric centre refinement stage.
//!
//! Runs once on the final marker set of any target with a board homography:
//! every marker with a position on the board (a decoded id on a coded target,
//! a lattice label on a plain one) is re-fitted with the blurred-pattern model
//! of [`crate::ring::photometric_center`], seeded from its current
//! (projective) centre and the board homography. The printed pattern — ink
//! annulus, code band, neighbouring markers, origin dots — is derived from the
//! target definition, so coded and plain markers on hex and rect lattices go
//! through the same model. Accepted fits replace the centre and attach a
//! covariance; every other marker keeps its centre and records why.

use image::GrayImage;
use projective_grid::Coord;

use crate::detector::marker_build::marker_confidence;
use crate::detector::{
    CenterSource, DetectConfig, MarkerRecord, PhotometricMetrics, PhotometricStatus,
};
use crate::homography::homography_project_rows;
use crate::marker::codec::Codebook;
use crate::pipeline::BoardFrame;
use crate::pipeline::stats::median_f64;
use crate::pixelmap::PixelMapper;
use crate::ring::MAD_TO_SIGMA;
use crate::ring::photometric_center::{
    BlurEstimate, InkDisc, Mat3, PhotometricFailure, PhotometricFit, PhotometricProblem,
    PrintRadii, fit_marker,
};
use crate::target::{MarkerCoding, TargetLayout};

#[cfg(feature = "study")]
mod residuals;
#[cfg(feature = "study")]
use residuals::record_residuals;
#[cfg(feature = "study")]
pub use residuals::{PhotometricResidualMarker, PhotometricStudyOptions, ResidualAutocorrelation};
#[cfg(feature = "study")]
pub(crate) use residuals::{ResidualSink, StudySink};

/// Where the stage sends each fitted marker's residuals: nowhere without the
/// `study` feature (the sink can only be `None`).
#[cfg(not(feature = "study"))]
pub(crate) type ResidualSink<'a> = Option<&'a mut std::convert::Infallible>;

#[cfg(not(feature = "study"))]
fn record_residuals(
    sink: &mut std::convert::Infallible,
    _: &Stage<'_>,
    _: &Task,
    _: [f64; 2],
    _: PrintRadii,
    _: &PhotometricFit,
) {
    match *sink {}
}

/// Minimum number of converged free-radii fits for a print-radii consensus.
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
/// Iteration budget of a free-radii fit, in multiples of the configured
/// per-marker budget. Where the blur is comparable to the ink stroke, stroke
/// width and blur trade off along a shallow valley that the fit descends
/// slowly, and the fits still descending after the centre-fit budget are the
/// thin-stroke ones: leaving them out biased the consensus low (photometric
/// target matrix, thin ring at 17 px, blur 1.5: −0.3 … −0.8 %).
const RADII_FIT_ITERATION_FACTOR: usize = 4;
/// Largest standard error of a consensus radius (from the scatter of its
/// samples), as a fraction of the outer radius: the precision six samples
/// of 2 % each give.
const MAX_RADIUS_SE_REL: f64 = 0.008;
/// Radius samples further from their median than this many robust standard
/// deviations are outliers.
const RADII_OUTLIER_CUT: f64 = 4.0;
/// Plausible fitted PSF sigma range (pixels).
const SIGMA_RANGE_PX: (f64, f64) = (0.2, 15.0);
/// Largest fitted PSF sigma, as a fraction of the marker's outer radius in
/// pixels, at which the blur is reported (and print radii are estimated, see
/// [`radii_consensus`]). Above it the ring's own features blur together and
/// σ is overestimated: on the photometric target matrix
/// (24 noise seeds, Gaussian PSF) σ ≈ 0.19 R is within ±0.5 % of the true
/// width, σ ≈ 0.375 R (outer radius 8 px, blur 3) +18 … +41 %.
const MAX_BLUR_OUTER_RADIUS_RATIO: f64 = 0.25;
/// Minimum fitted contrast in units of the residual noise.
const MIN_CONTRAST_SNR: f64 = 3.0;
/// A fit whose residual standard deviation exceeds this multiple of the
/// image-wide median is a model misfit (a glare halo, an occluder, a print
/// defect); its centre is not trusted.
const MAX_RESIDUAL_RATIO: f64 = 3.0;
/// Board-mm reach of a blurred edge for selecting the ink discs of a patch,
/// in PSF sigmas at the largest plausible blur.
const DISC_REACH_SIGMAS: f64 = 6.0;
/// Ink discs are looked for no further than this many lattice spacings away.
const MAX_DISC_SPACINGS: f64 = 3.0;
/// Bias of a photometric centre per unit of blur × blur gradient: under a
/// blur sigma `σ` (px) changing by `g` px per px across the marker, the fit
/// (an isotropic PSF of one width) lands about `1.24 g σ` px towards the
/// sharper side (`tests/blur_gradient.rs`, `blur_gradient_bias_matrix`,
/// scatter form: the physical one of depth-dependent defocus). Not refitted:
/// with it the predicted error is calibrated at every blur level of the
/// synthetic confidence study (ADR-029).
const BLUR_GRADIENT_BIAS_PER_SIGMA: f64 = 1.24;
/// Accepted fits nearest a marker (besides its own) its local blur gradient
/// is fitted over.
const BLUR_GRADIENT_NEIGHBOURS: usize = 8;

fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
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
            let w = homography_project_rows(&to_working, u)?;
            let x = mapper.working_to_image_pixel(w)?;
            src.push(u);
            dst.push(x);
        }
    }
    let m = crate::homography::estimate_homography_dlt(&src, &dst).ok()?;
    Some(crate::homography::matrix3_to_array(&m))
}

/// Where markers sit in the source plane of the result homography (board mm
/// in the absolute frame, assignment-frame mm in a plain target's relative
/// frame) and what else is printed around them there.
struct PrintedBoard<'a> {
    target: &'a TargetLayout,
    /// The homography's source plane is the physical board frame.
    absolute: bool,
    codebook: Option<Codebook>,
    /// Centres of every printed marker in the frame.
    cells_mm: Vec<[f64; 2]>,
    /// Fiducial dots in the frame (absolute frame only: the relative frame
    /// does not know where they are).
    dots: Vec<InkDisc>,
}

impl<'a> PrintedBoard<'a> {
    /// `None` when the markers have no frame position (a plain target whose
    /// grid assignment did not run).
    fn new(
        target: &'a TargetLayout,
        board_frame: Option<BoardFrame>,
        markers: &[MarkerRecord],
    ) -> Option<Self> {
        let f32_pair = |p: [f32; 2]| [f64::from(p[0]), f64::from(p[1])];
        let (absolute, codebook) = match target.coding() {
            MarkerCoding::Coded16(spec) => {
                (true, Some(Codebook::from_profile(spec.codebook_profile)))
            }
            MarkerCoding::Plain => (board_frame?.origin_resolved(), None),
        };
        let mut board = Self {
            target,
            absolute,
            codebook,
            cells_mm: Vec::new(),
            dots: Vec::new(),
        };
        if absolute {
            board.cells_mm = target.cells().iter().map(|c| f32_pair(c.xy_mm)).collect();
            if let Some(f) = target.fiducials() {
                let radius = f64::from(f.dot_radius_mm);
                board.dots = target
                    .fiducial_dots_mm()
                    .iter()
                    .map(|&d| InkDisc {
                        center: f32_pair(d),
                        radius,
                    })
                    .collect();
            }
        } else {
            // Relative frame: only the labelled markers are known to exist.
            board.cells_mm = markers.iter().filter_map(|m| board.frame_xy(m)).collect();
        }
        Some(board)
    }

    /// Lattice cell of a plain marker's label.
    fn coord(m: &MarkerRecord) -> Option<Coord> {
        m.grid_coord.map(|c| Coord::new(c[0], c[1]))
    }

    /// The marker's centre in the frame.
    fn frame_xy(&self, m: &MarkerRecord) -> Option<[f64; 2]> {
        let xy = match (&self.codebook, self.absolute) {
            (Some(_), _) => self.target.xy_mm_of_id(m.id?)?,
            (None, true) => self.target.cell_xy_mm(Self::coord(m)?)?,
            (None, false) => {
                return Some(crate::pipeline::frame_xy_mm(self.target, Self::coord(m)?));
            }
        };
        Some([f64::from(xy[0]), f64::from(xy[1])])
    }

    /// The marker's codeword; `Ok(None)` for a plain marker, `Err` when a
    /// coded marker's id has no word.
    fn codeword(&self, m: &MarkerRecord) -> Result<Option<u16>, ()> {
        match &self.codebook {
            Some(cb) => m.id.and_then(|id| cb.word(id)).map(Some).ok_or(()),
            None => Ok(None),
        }
    }

    /// Ink discs within reach of a patch of radius `r_patch_mm` around
    /// `center_mm`, relative to it.
    fn ink_discs(
        &self,
        center_mm: [f64; 2],
        r_patch_mm: f64,
        r_outer_mm: f64,
        px_per_mm: f64,
    ) -> Vec<InkDisc> {
        let spacing = f64::from(self.target.min_center_spacing_mm());
        let edge_reach = DISC_REACH_SIGMAS * SIGMA_RANGE_PX.1 / px_per_mm;
        let neighbours = self.cells_mm.iter().map(|&c| InkDisc {
            center: c,
            radius: r_outer_mm,
        });
        neighbours
            .chain(self.dots.iter().copied())
            .filter_map(|d| {
                let rel = [d.center[0] - center_mm[0], d.center[1] - center_mm[1]];
                let dist = rel[0].hypot(rel[1]);
                let reach = (r_patch_mm + d.radius + edge_reach).min(MAX_DISC_SPACINGS * spacing);
                (dist > 1e-6 && dist <= reach).then_some(InkDisc {
                    center: rel,
                    radius: d.radius,
                })
            })
            .collect()
    }
}

/// One marker's fit inputs.
struct Task {
    index: usize,
    local_map: Mat3,
    codeword: Option<u16>,
    ink_discs: Vec<InkDisc>,
    /// Image px per board mm at the marker (geometric mean).
    px_per_mm: f64,
}

/// Image px per board mm of a local map at the marker centre.
fn local_scale(map: &Mat3) -> Option<f64> {
    let c0 = homography_project_rows(map, [0.0, 0.0])?;
    let cx = homography_project_rows(map, [1.0, 0.0])?;
    let cy = homography_project_rows(map, [0.0, 1.0])?;
    let det = ((cx[0] - c0[0]) * (cy[1] - c0[1]) - (cx[1] - c0[1]) * (cy[0] - c0[0])).abs();
    (det.is_finite() && det > 0.0).then(|| det.sqrt())
}

fn build_tasks(
    markers: &[MarkerRecord],
    h: &Mat3,
    mapper: Option<&dyn PixelMapper>,
    board: &PrintedBoard<'_>,
    r_outer_mm: f64,
    r_patch_mm: f64,
) -> Vec<Task> {
    let mut tasks = Vec::new();
    for (index, m) in markers.iter().enumerate() {
        if !(m.center[0].is_finite() && m.center[1].is_finite()) {
            continue;
        }
        let Some(cell) = board.frame_xy(m) else {
            continue;
        };
        let Ok(codeword) = board.codeword(m) else {
            continue;
        };
        let Some(map) = local_map(h, cell, r_outer_mm, mapper) else {
            continue;
        };
        let Some(px_per_mm) = local_scale(&map) else {
            continue;
        };
        tasks.push(Task {
            index,
            local_map: map,
            codeword,
            ink_discs: board.ink_discs(cell, r_patch_mm, r_outer_mm, px_per_mm),
            px_per_mm,
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
    /// Whether the markers carry a code band (radii `r1`, `r2` exist).
    coded: bool,
}

impl Stage<'_> {
    fn problem<'a>(
        &'a self,
        task: &'a Task,
        seed: [f64; 2],
        radii: PrintRadii,
        mode: FitMode,
    ) -> PhotometricProblem<'a> {
        let photometric = &self.config.advanced.photometric;
        PhotometricProblem {
            gray: self.gray,
            local_map: task.local_map,
            center_seed: seed,
            codeword: task.codeword,
            radii,
            fit_radii: mode.fit_radii,
            cubic_background: mode.cubic_background,
            ink_discs: &task.ink_discs,
            // Real lenses are often astigmatic: an isotropic PSF leaves a
            // structured residual there; the two extra parameters cost little
            // and are held at zero by their prior where the blur is too small
            // to tell.
            anisotropic_blur: true,
            patch_margin_mm: self.patch_margin_mm,
            saturation_level: photometric.saturation_level,
            max_iterations: photometric.max_iterations.max(1)
                * if mode.fit_radii {
                    RADII_FIT_ITERATION_FACTOR
                } else {
                    1
                },
        }
    }

    fn fit(
        &self,
        task: &Task,
        seed: [f64; 2],
        radii: PrintRadii,
        mode: FitMode,
    ) -> Result<PhotometricFit, PhotometricFailure> {
        fit_marker(&self.problem(task, seed, radii, mode))
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
        let samples: Vec<RadiiSample> = self
            .tasks
            .iter()
            .step_by(stride)
            .filter_map(|t| {
                let fit = self.fit(t, markers[t.index].center, nominal, mode).ok()?;
                fit.converged.then(|| RadiiSample {
                    radii: fit.radii,
                    blur_measured: blur_is_measured(fit.blur.sigma_px, nominal[3] * t.px_per_mm),
                })
            })
            .collect();
        let n_free = if self.coded { 3 } else { 1 };
        let estimated = radii_consensus(&samples, nominal, n_free);
        tracing::debug!(
            ?estimated,
            n_samples = samples.len(),
            "photometric print radii"
        );
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

/// One converged free-radii fit.
struct RadiiSample {
    radii: PrintRadii,
    /// Whether its blur is within the range the model measures it in
    /// ([`blur_is_measured`]).
    blur_measured: bool,
}

/// Consensus of the per-marker estimates of the first `n_free` radii: the
/// plain mean of the samples within [`RADII_OUTLIER_CUT`] robust standard
/// deviations of their median.
///
/// The samples are neither weighted nor gated by their own covariance: where
/// the blur is comparable to the stroke, a fit's radius uncertainty grows with
/// the radius it found (blur and stroke width trade off), so precision
/// weights and per-sample precision gates both favour thick-stroke fits and
/// biased the consensus low (photometric target matrix, thin ring at 17 px,
/// blur 1.5: −1 … −2.7 %). The consensus itself must be precise instead: its
/// standard error, from the scatter of its samples, within
/// [`MAX_RADIUS_SE_REL`] of the outer radius.
///
/// `None` (keep the nominal radii) with fewer than [`MIN_RADII_SAMPLES`]
/// samples, when most samples' blur is outside the range the model measures
/// it in (there the fitted radii absorb the model error: −15 … −18 % at an
/// outer radius of 8 px and blur 3), when a consensus is not precise enough,
/// or when the radii come out unordered.
fn radii_consensus(
    samples: &[RadiiSample],
    nominal: PrintRadii,
    n_free: usize,
) -> Option<PrintRadii> {
    let n_measured = samples.iter().filter(|s| s.blur_measured).count();
    if samples.len() < MIN_RADII_SAMPLES || 2 * n_measured <= samples.len() {
        return None;
    }
    let max_se = MAX_RADIUS_SE_REL * nominal[3];
    let mut out = nominal;
    for k in 0..n_free {
        let values: Vec<f64> = samples.iter().map(|s| s.radii[k]).collect();
        let centre = median_f64(values.clone())?;
        let spread =
            MAD_TO_SIGMA * median_f64(values.iter().map(|r| (r - centre).abs()).collect())?;
        let kept: Vec<f64> = values
            .into_iter()
            .filter(|r| (r - centre).abs() <= RADII_OUTLIER_CUT * spread)
            .collect();
        let n = kept.len();
        if n < MIN_RADII_SAMPLES {
            return None;
        }
        let mean = kept.iter().sum::<f64>() / n as f64;
        let variance = kept.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let standard_error = (variance / n as f64).sqrt();
        // Also rejects a non-finite mean or spread.
        if !standard_error.is_finite() || standard_error > max_se {
            return None;
        }
        out[k] = mean;
    }
    // Plain markers keep r1 = r2 = r3, hence the non-strict comparisons.
    let ordered = out[0] > 0.0 && out[0] < out[1] && out[1] <= out[2] && out[2] <= out[3];
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

/// Metrics of a marker that has no fit.
fn status_only(status: PhotometricStatus) -> PhotometricMetrics {
    PhotometricMetrics {
        status,
        contrast: None,
        residual_sigma: None,
        n_pixels: 0,
        masked_fraction: 0.0,
        center_shift_px: None,
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
    } else if !(SIGMA_RANGE_PX.0..=SIGMA_RANGE_PX.1).contains(&fit.blur.sigma_px)
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
        contrast: Some(fit.contrast),
        residual_sigma: Some(fit.residual_sigma),
        n_pixels: fit.n_pixels,
        masked_fraction: fit.masked_fraction,
        center_shift_px: Some(shift),
    }
}

/// The local blur gradient `∇σ` (px per px, image frame) at every accepted
/// photometric centre, aligned with `accepted`: the gradient of the
/// least-squares plane through the fitted blur of it and its
/// [`BLUR_GRADIENT_NEIGHBOURS`] nearest accepted neighbours (ties by index);
/// `None` when the plane is undetermined.
///
/// `accepted` holds `(marker index, fitted blur sigma)`; the markers' centres
/// are the accepted (image-pixel) ones.
fn blur_gradients(markers: &[MarkerRecord], accepted: &[(usize, f64)]) -> Vec<Option<[f64; 2]>> {
    let points: Vec<([f64; 2], f64)> = accepted
        .iter()
        .map(|&(i, sigma)| (markers[i].center, sigma))
        .collect();
    let dist2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
    let mut order: Vec<usize> = (0..points.len()).collect();
    points
        .iter()
        .map(|&(c, _)| {
            // The K + 1 nearest (the marker itself included), by distance then
            // index: a partial selection, O(n) per marker, then the few sorted.
            let nearer = |&a: &usize, &b: &usize| {
                dist2(points[a].0, c)
                    .total_cmp(&dist2(points[b].0, c))
                    .then(a.cmp(&b))
            };
            let k = BLUR_GRADIENT_NEIGHBOURS.min(order.len() - 1);
            order.select_nth_unstable_by(k, nearer);
            order[..=k].sort_unstable_by(nearer);
            plane_gradient(order[..=k].iter().map(|&j| points[j]))
        })
        .collect()
}

/// Attach to every accepted photometric centre the blur-gradient bias its
/// fit does not model ([`MarkerRecord::center_bias_px`]): `1.24 |∇σ| σ`,
/// `σ` its fitted blur and `∇σ` its local blur gradient ([`blur_gradients`],
/// aligned with `accepted`). `None` (no bias known) when the gradient is
/// undetermined.
fn attach_blur_gradient_bias(
    markers: &mut [MarkerRecord],
    accepted: &[(usize, f64)],
    gradients: &[Option<[f64; 2]>],
) {
    for (&(i, sigma), g) in accepted.iter().zip(gradients) {
        markers[i].center_bias_px =
            g.map(|g| BLUR_GRADIENT_BIAS_PER_SIGMA * g[0].hypot(g[1]) * sigma);
    }
}

/// Gradient `g` of the least-squares plane `v = a + g · p` through `(p, v)`
/// samples; `None` for fewer than four samples or positions too close to a
/// line to determine it.
fn plane_gradient(samples: impl Iterator<Item = ([f64; 2], f64)>) -> Option<[f64; 2]> {
    let samples: Vec<([f64; 2], f64)> = samples.collect();
    if samples.len() < 4 {
        return None;
    }
    let n = samples.len() as f64;
    let [mx, my, mv] = samples
        .iter()
        .fold([0.0; 3], |m, (p, v)| [m[0] + p[0], m[1] + p[1], m[2] + v])
        .map(|sum| sum / n);
    let (mut sxx, mut sxy, mut syy, mut sxv, mut syv) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (p, v) in &samples {
        let (x, y, v) = (p[0] - mx, p[1] - my, v - mv);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxv += x * v;
        syv += y * v;
    }
    let det = sxx * syy - sxy * sxy;
    // Both principal spreads must be a fair fraction of the total (also
    // rejects NaN positions).
    if det.is_nan() || det <= 1e-3 * (sxx + syy).powi(2) {
        return None;
    }
    let gx = (syy * sxv - sxy * syv) / det;
    let gy = (sxx * syv - sxy * sxv) / det;
    Some([gx, gy])
}

/// Record [`PhotometricStatus::NoBoardPosition`] on every marker the stage
/// did not reach, so a requested refinement never leaves a marker silent.
pub(crate) fn mark_unrefined(markers: &mut [MarkerRecord]) {
    for m in markers.iter_mut().filter(|m| m.photometric.is_none()) {
        m.photometric = Some(status_only(PhotometricStatus::NoBoardPosition));
    }
}

/// Refine the centres of every marker with a board position in place.
/// `residuals`, when given (`study` feature), receives every fitted marker's
/// per-pixel residuals; nothing is computed for it otherwise.
///
/// `homography` maps the result's board frame (`board_frame`; absolute board
/// mm for coded targets) to the working frame; `mapper` maps that frame to
/// image pixels and must be `Some` exactly when the two differ. Marker
/// centres are image pixels on entry and on exit. Returns the print radii the
/// final fits used when they were estimated from the image.
pub(crate) fn apply_photometric_centers(
    gray: &GrayImage,
    markers: &mut [MarkerRecord],
    homography: Option<&Mat3>,
    board_frame: Option<BoardFrame>,
    mapper: Option<&dyn PixelMapper>,
    config: &DetectConfig,
    residuals: ResidualSink<'_>,
) -> Option<PrintRadii> {
    if !config.circle_refinement.uses_photometric() {
        return None;
    }
    let estimated = refine(
        gray,
        markers,
        homography,
        board_frame,
        mapper,
        config,
        residuals,
    );
    mark_unrefined(markers);
    estimated
}

fn refine(
    gray: &GrayImage,
    markers: &mut [MarkerRecord],
    homography: Option<&Mat3>,
    board_frame: Option<BoardFrame>,
    mapper: Option<&dyn PixelMapper>,
    config: &DetectConfig,
    mut residuals: ResidualSink<'_>,
) -> Option<PrintRadii> {
    let target = &config.target;
    let h = homography?;
    let board = PrintedBoard::new(target, board_frame, markers)?;
    #[cfg(feature = "study")]
    let fixed = residuals
        .as_deref()
        .and_then(|sink| sink.options.fixed_radii_mm());
    #[cfg(not(feature = "study"))]
    let fixed: Option<PrintRadii> = None;
    let nominal = fixed.unwrap_or_else(|| target.coding().ink_radii_mm_f64(&target.ring()));
    let spacing = f64::from(target.min_center_spacing_mm());
    let patch_margin_mm =
        (0.5 * (spacing - 2.0 * nominal[3])).clamp(0.04 * nominal[3], 0.5 * nominal[3]);
    let tasks = build_tasks(
        markers,
        h,
        mapper,
        &board,
        nominal[3],
        nominal[3] + patch_margin_mm,
    );
    if tasks.is_empty() {
        return None;
    }
    let stage = Stage {
        gray,
        config,
        tasks,
        patch_margin_mm,
        coded: target.is_coded(),
    };

    let estimated = if fixed.is_none() && config.advanced.photometric.estimate_print_radii {
        stage.estimate_print_radii(markers, nominal)
    } else {
        None
    };
    let final_radii = estimated.unwrap_or(nominal);
    let (fits, typical_residual) = stage.fit_centres(markers, final_radii);

    // (marker index, fitted blur sigma) of every accepted fit.
    let mut accepted: Vec<(usize, f64)> = Vec::new();
    let mut iterations = 0usize;
    for (t, fit) in stage.tasks.iter().zip(fits) {
        let m = &mut markers[t.index];
        let max_shift_px = config
            .advanced
            .photometric
            .max_center_shift_px
            .unwrap_or(0.5 * nominal[3] * t.px_per_mm);
        let metrics = match fit {
            Err(f) => status_only(failure_status(f)),
            Ok(fit) => {
                iterations += fit.iterations;
                if let Some(sink) = residuals.as_deref_mut() {
                    // The seed is still the pre-fit centre.
                    record_residuals(sink, &stage, t, m.center, final_radii, &fit);
                }
                let metrics = judge(&fit, m.center, max_shift_px, typical_residual);
                tracing::trace!(
                    marker = t.index,
                    status = ?metrics.status,
                    covariance_inflation = fit.covariance_inflation(),
                    residual_acf = ?fit.residual_acf,
                    "photometric fit"
                );
                if metrics.status == PhotometricStatus::Accepted {
                    m.center = fit.center;
                    m.center_cov = Some(fit.center_cov);
                    m.blur = reported_blur(fit.blur, nominal[3] * t.px_per_mm);
                    m.center_source = CenterSource::Photometric;
                    if let Some(mapper) = mapper {
                        m.center_mapped = mapper.image_to_working_pixel(fit.center);
                    }
                    accepted.push((t.index, fit.blur.sigma_px));
                }
                metrics
            }
        };
        m.photometric = Some(metrics);
    }
    let gradients = blur_gradients(markers, &accepted);
    attach_blur_gradient_bias(markers, &accepted, &gradients);
    #[cfg(feature = "study")]
    if let Some(sink) = residuals {
        residuals::record_blur_gradient_bias(&mut sink.rows, &accepted, &gradients);
    }
    for &(i, _) in &accepted {
        markers[i].confidence = marker_confidence(&markers[i]);
    }
    tracing::debug!(
        n_markers = stage.tasks.len(),
        n_accepted = accepted.len(),
        iterations,
        typical_residual,
        "photometric centre refinement summary"
    );
    estimated
}

/// Whether a fitted blur is within the width the model measures it in
/// ([`MAX_BLUR_OUTER_RADIUS_RATIO`]).
fn blur_is_measured(sigma_px: f64, outer_radius_px: f64) -> bool {
    sigma_px <= MAX_BLUR_OUTER_RADIUS_RATIO * outer_radius_px
}

/// The blur of an accepted fit as reported, or `None` outside the width it
/// is measured in.
fn reported_blur(blur: BlurEstimate, outer_radius_px: f64) -> Option<BlurEstimate> {
    blur_is_measured(blur.sigma_px, outer_radius_px).then_some(blur)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_is_reported_up_to_a_quarter_of_the_outer_radius() {
        let blur = |sigma_px| BlurEstimate {
            sigma_px,
            anisotropy: [0.1, -0.05],
            cov: [[1e-4, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 1e-3]],
        };
        // Outer radius 17 px: up to 4.25 px.
        assert_eq!(reported_blur(blur(1.5), 17.0), Some(blur(1.5)));
        assert_eq!(reported_blur(blur(4.25), 17.0), Some(blur(4.25)));
        assert_eq!(reported_blur(blur(4.3), 17.0), None);
        // Outer radius 8 px at blur 3 (σ ≈ 0.375 R): not reported.
        assert_eq!(reported_blur(blur(3.0), 8.0), None);
    }

    /// A hex patch of fitted blur sigmas on the plane `1.5 + 0.01 x − 0.02 y`.
    fn blur_plane_markers() -> (Vec<MarkerRecord>, Vec<(usize, f64)>) {
        let mut markers = Vec::new();
        let mut accepted = Vec::new();
        for row in -3i32..=3 {
            for col in -3i32..=3 {
                let x = 40.0 * f64::from(col) + 20.0 * f64::from(row.rem_euclid(2));
                let y = 35.0 * f64::from(row);
                accepted.push((markers.len(), 1.5 + 0.01 * x - 0.02 * y));
                markers.push(MarkerRecord {
                    center: [300.0 + x, 200.0 + y],
                    ..MarkerRecord::default()
                });
            }
        }
        (markers, accepted)
    }

    #[test]
    fn blur_gradient_bias_follows_the_bias_law() {
        let (mut markers, accepted) = blur_plane_markers();
        let gradients = blur_gradients(&markers, &accepted);
        attach_blur_gradient_bias(&mut markers, &accepted, &gradients);
        let g = 0.01f64.hypot(0.02);
        for &(i, sigma) in &accepted {
            let b = markers[i].center_bias_px.expect("bias");
            assert!((b - 1.24 * g * sigma).abs() < 1e-9, "{b}");
            let v = gradients[i].expect("gradient");
            assert!(
                (v[0] - 0.01).abs() < 1e-12 && (v[1] + 0.02).abs() < 1e-12,
                "{v:?}"
            );
        }
    }

    #[test]
    fn blur_gradient_needs_a_two_dimensional_neighbourhood() {
        // Three fits, then four on one line: no plane.
        let line =
            |n: usize| (0..n).map(|k| ([10.0 * k as f64, 5.0 * k as f64], 1.0 + 0.1 * k as f64));
        assert_eq!(plane_gradient(line(3)), None);
        assert_eq!(plane_gradient(line(4)), None);
        let square = [
            ([0.0, 0.0], 1.0),
            ([1.0, 0.0], 1.5),
            ([0.0, 1.0], 1.0),
            ([1.0, 1.0], 1.5),
        ];
        let g = plane_gradient(square.into_iter()).expect("plane");
        assert!((g[0] - 0.5).abs() < 1e-12 && g[1].abs() < 1e-12, "{g:?}");
    }

    /// Accepted fits of a `side × side` plate (58 × 58: the IPGS plate),
    /// jittered off the lattice, with a curved blur field.
    fn plate_markers(side: usize) -> (Vec<MarkerRecord>, Vec<(usize, f64)>) {
        let mut markers = Vec::new();
        let mut accepted = Vec::new();
        for k in 0..side * side {
            let (r, c) = ((k / side) as f64, (k % side) as f64);
            // Deterministic sub-pixel jitter, so no two distances tie.
            let j = ((k as f64) * 0.618_034).fract() - 0.5;
            let p = [30.0 * c + 3.0 * j, 30.0 * r - 2.0 * j];
            accepted.push((k, 1.0 + 1e-3 * p[0] + 2e-6 * p[1] * p[1] + 0.05 * j));
            markers.push(MarkerRecord {
                center: p,
                ..MarkerRecord::default()
            });
        }
        (markers, accepted)
    }

    /// The neighbour selection is the plain `K + 1` nearest by a full sort.
    #[test]
    fn blur_gradient_bias_matches_a_full_sort() {
        let (mut markers, accepted) = plate_markers(12);
        let gradients = blur_gradients(&markers, &accepted);
        attach_blur_gradient_bias(&mut markers, &accepted, &gradients);
        for &(i, sigma) in &accepted {
            let c = markers[i].center;
            let mut all: Vec<(f64, usize)> = accepted
                .iter()
                .map(|&(j, _)| {
                    let p = markers[j].center;
                    ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2), j)
                })
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let near = all[..=BLUR_GRADIENT_NEIGHBOURS].iter().map(|&(_, j)| {
                let s = accepted.iter().find(|a| a.0 == j).unwrap().1;
                (markers[j].center, s)
            });
            let expected = plane_gradient(near)
                .map(|g| BLUR_GRADIENT_BIAS_PER_SIGMA * g[0].hypot(g[1]) * sigma);
            assert_eq!(markers[i].center_bias_px, expected, "marker {i}");
        }
    }

    const PLAIN_NOMINAL: PrintRadii = [2.4, 4.8, 4.8, 4.8];

    fn plain_samples(r0: &[f64]) -> Vec<RadiiSample> {
        r0.iter()
            .map(|&r| RadiiSample {
                radii: [r, 4.8, 4.8, 4.8],
                blur_measured: true,
            })
            .collect()
    }

    #[test]
    fn radii_consensus_is_the_plain_mean_after_the_outlier_cut() {
        let samples = plain_samples(&[2.30, 2.32, 2.34, 2.36, 2.38, 2.40, 2.42, 2.44, 3.4]);
        let r = radii_consensus(&samples, PLAIN_NOMINAL, 1).expect("consensus");
        assert!((r[0] - 2.37).abs() < 1e-12, "r0 {}", r[0]);
        assert_eq!(r[1..], PLAIN_NOMINAL[1..]);
    }

    #[test]
    fn radii_consensus_needs_enough_precise_samples_with_measurable_blur() {
        let five = plain_samples(&[2.38, 2.39, 2.40, 2.41, 2.42]);
        assert_eq!(radii_consensus(&five, PLAIN_NOMINAL, 1), None);
        // Standard error 0.06 mm against the 0.008 × 4.8 = 0.038 mm limit.
        let scattered = plain_samples(&[2.2, 2.6, 2.25, 2.55, 2.3, 2.5, 2.35, 2.45]);
        assert_eq!(radii_consensus(&scattered, PLAIN_NOMINAL, 1), None);
        let mut blurred = plain_samples(&[2.38, 2.39, 2.40, 2.41, 2.42, 2.40]);
        for s in &mut blurred[..3] {
            s.blur_measured = false;
        }
        assert_eq!(radii_consensus(&blurred, PLAIN_NOMINAL, 1), None);
        blurred[0].blur_measured = true;
        assert!(radii_consensus(&blurred, PLAIN_NOMINAL, 1).is_some());
    }

    #[test]
    #[ignore = "timing probe"]
    fn blur_gradient_bias_time_on_a_plate() {
        let (mut markers, accepted) = plate_markers(58);
        let t = std::time::Instant::now();
        let gradients = blur_gradients(&markers, &accepted);
        attach_blur_gradient_bias(&mut markers, &accepted, &gradients);
        println!(
            "{} accepted fits: {:.1} ms",
            accepted.len(),
            1e3 * t.elapsed().as_secs_f64()
        );
    }
}
