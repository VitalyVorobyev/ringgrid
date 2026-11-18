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
use crate::pixelmap::PixelMapper;
use crate::ring::MAD_TO_SIGMA;
use crate::ring::photometric_center::{
    InkDisc, Mat3, PhotometricFailure, PhotometricFit, PhotometricProblem, PrintRadii, fit_marker,
};
use crate::target::{MarkerCoding, TargetLayout};

#[cfg(feature = "study")]
mod residuals;
#[cfg(feature = "study")]
pub(crate) use residuals::ResidualSink;
#[cfg(feature = "study")]
use residuals::record_residuals;
#[cfg(feature = "study")]
pub use residuals::{PhotometricResidualMarker, ResidualAutocorrelation};

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
/// Per-marker radius estimates less precise than this fraction of the outer
/// radius are not used.
const MAX_RADIUS_STD_REL: f64 = 0.02;
/// Plausible fitted PSF sigma range (pixels).
const SIGMA_RANGE_PX: (f64, f64) = (0.2, 15.0);
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

/// Nominal printed radii of the target's markers (board mm), see
/// [`PrintRadii`]: stroked rings around a code band for coded markers, the
/// filled annulus between the two radii for plain ones.
fn nominal_radii(target: &TargetLayout) -> PrintRadii {
    let ring = target.ring();
    let inner = f64::from(ring.inner_radius_mm);
    let outer = f64::from(ring.outer_radius_mm);
    match target.coding() {
        MarkerCoding::Coded16(spec) => {
            let h = 0.5 * f64::from(spec.ring_width_mm);
            [inner - h, inner + h, outer - h, outer + h]
        }
        MarkerCoding::Plain => [inner, outer, outer, outer],
    }
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
            max_iterations: photometric.max_iterations.max(1),
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
        let fits: Vec<PhotometricFit> = self
            .tasks
            .iter()
            .step_by(stride)
            .filter_map(|t| self.fit(t, markers[t.index].center, nominal, mode).ok())
            .collect();
        let n_free = if self.coded { 3 } else { 1 };
        let estimated = radii_consensus(&fits, nominal, n_free);
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

/// Precision-weighted consensus of per-marker estimates of the first
/// `n_free` radii with a 4-sigma outlier cut; `None` when too few markers
/// determine them.
fn radii_consensus(
    fits: &[PhotometricFit],
    nominal: PrintRadii,
    n_free: usize,
) -> Option<PrintRadii> {
    let max_std = MAX_RADIUS_STD_REL * nominal[3];
    let samples: Vec<(PrintRadii, [f64; 3])> = fits
        .iter()
        .filter(|f| f.converged)
        .filter_map(|f| {
            let std = f.radii_std?;
            std[..n_free]
                .iter()
                .all(|s| s.is_finite() && *s > 0.0 && *s < max_std)
                .then_some((f.radii, std))
        })
        .collect();
    if samples.len() < MIN_RADII_SAMPLES {
        return None;
    }
    let mut out = nominal;
    for k in 0..n_free {
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
        let spread = MAD_TO_SIGMA * dev[dev.len() / 2];
        out[k] = mean_of(&|r, s| (r - first).abs() <= 4.0 * s.hypot(spread))?;
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
        blur_sigma_px: None,
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
    let nominal = nominal_radii(target);
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

    let estimated = if config.advanced.photometric.estimate_print_radii {
        stage.estimate_print_radii(markers, nominal)
    } else {
        None
    };
    let final_radii = estimated.unwrap_or(nominal);
    let (fits, typical_residual) = stage.fit_centres(markers, final_radii);

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
                    m.center_source = CenterSource::Photometric;
                    m.confidence = marker_confidence(m);
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
