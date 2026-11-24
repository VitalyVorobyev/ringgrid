//! Photometric centre accuracy and covariance honesty across every
//! first-class target kind, against exact synthetic ground truth.
//!
//! A cell is (target kind, marker scale, blur, contrast, tilt, scene effect).
//! Per cell, one noise-free radiance is rendered and exposed with several
//! noise seeds; each exposure is detected with the photometric method (and
//! the first with the projective-centre method as the baseline). Reported:
//!
//! - centre RMS against the truth for both methods (photometric over the
//!   markers whose centre it produced, with its coverage);
//! - Monte-Carlo honesty of the covariance: reported σ over the empirical
//!   scatter of each marker's centre across noise seeds, pooled, for the
//!   correlated-residual covariance (`corr`) and the white-residual one
//!   (`white`, recovered from the covariance inflation; `study` feature only);
//! - reported σ over the error against the truth (which also contains bias,
//!   which no covariance covers), and the bias itself;
//! - the result homography's RMS error at the true cell centres for both
//!   methods (first exposure; absolute board frame only);
//! - with the `study` feature, the fitted PSF `(sigma, a, b)`: Monte-Carlo
//!   honesty of its covariance (as for the centre) and the bias of the mean
//!   fitted `sigma`, `a`, `b` against the rendered PSF's second moments.
//!
//! The default run checks a representative subset (one cell per target kind,
//! one seed). The full matrix and the model-misfit study are `#[ignore]`d:
//!
//! ```text
//! cargo test --release -p ringgrid --test photometric_target_matrix -- --ignored --nocapture
//! ```
//!
//! `MATRIX_SEEDS` (default 6) sets the Monte-Carlo seeds per cell,
//! `MATRIX_KINDS` (comma-separated names) and `MATRIX_SCALES` (outer radii in
//! px) restrict the matrix, `MATRIX_DPI` overrides the board raster density,
//! `MATRIX_THREADS` sets the parallelism, `MATRIX_FIXED_RADII` turns off the
//! per-image print-radii estimate (the fits use the nominal radii, exact
//! here; the estimate's image-wide error is common to all markers of an
//! exposure and is not in a marker's covariance), `MATRIX_NOMINAL_INNER`
//! (mm, plain kinds) detects with that nominal inner radius while the scene
//! keeps the printed one, so the per-image print-radii estimate must recover
//! the print rather than the nominal value (radii and blur are still scored
//! against the print).

mod support;

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ringgrid::diagnostics::{DetectionDiagnostics, PhotometricStatus};
use ringgrid::{
    BoardFrame, CenterSource, CircleRefinementMethod, DetectConfig, DetectedMarker,
    DetectionResult, Detector, OriginDots, TargetLayout,
};
use support::blur_scene::{BoardRaster, Psf, Radiance, SceneSpec, apply_h, expose, radiance};

/// Every first-class target kind, with a representative ring geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coded hex, wide code band (the stroke width real prints show).
    CodedHex,
    CodedRect,
    PlainHexDots,
    PlainHexNoDots,
    PlainRectDots,
    PlainRectNoDots,
    /// Plain rect with a thin annulus (1 mm ring on a 5.6 mm marker).
    PlainRectThin,
}

const KINDS: [Kind; 7] = [
    Kind::CodedHex,
    Kind::CodedRect,
    Kind::PlainHexDots,
    Kind::PlainHexNoDots,
    Kind::PlainRectDots,
    Kind::PlainRectNoDots,
    Kind::PlainRectThin,
];

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::CodedHex => "coded_hex",
            Kind::CodedRect => "coded_rect",
            Kind::PlainHexDots => "plain_hex_dots",
            Kind::PlainHexNoDots => "plain_hex_nodots",
            Kind::PlainRectDots => "plain_rect_dots",
            Kind::PlainRectNoDots => "plain_rect_nodots",
            Kind::PlainRectThin => "plain_rect_thin",
        }
    }

    fn target(self) -> TargetLayout {
        self.target_with_inner(None)
    }

    /// The target, with a plain marker's inner radius replaced by `inner`
    /// (mm) when given.
    fn target_with_inner(self, inner: Option<f32>) -> TargetLayout {
        let plain = |pitch, outer, default_inner, dots| match self {
            Kind::PlainHexDots | Kind::PlainHexNoDots => {
                TargetLayout::plain_hex(pitch, 13, 12, outer, inner.unwrap_or(default_inner), dots)
            }
            _ => {
                TargetLayout::plain_rect(pitch, 12, 12, outer, inner.unwrap_or(default_inner), dots)
            }
        };
        let t = match self {
            Kind::CodedHex => TargetLayout::coded_hex(8.0, 15, 14, 4.9565, 2.966, 0.839),
            Kind::CodedRect => TargetLayout::coded_rect(14.0, 12, 12, 5.6, 3.2, 1.0),
            Kind::PlainHexDots => plain(8.0, 4.8, 2.4, OriginDots::Auto),
            Kind::PlainHexNoDots => plain(8.0, 4.8, 2.4, OriginDots::None),
            Kind::PlainRectDots => plain(14.0, 5.6, 2.8, OriginDots::Auto),
            Kind::PlainRectNoDots => plain(14.0, 5.6, 2.8, OriginDots::None),
            Kind::PlainRectThin => plain(14.0, 5.6, 4.6, OriginDots::Auto),
        };
        t.expect("valid matrix target")
    }
}

/// Scene effects beyond the model's Gaussian PSF and white noise.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Effect {
    None,
    /// Anisotropic Gaussian PSF, axis ratio ≈ 1.8.
    Anisotropic,
    /// Gaussian core plus a 6 px halo with 15 % of the energy.
    Halo,
    /// Uniform defocus disc of radius 2 σ.
    Disc,
    /// Camera response gamma 0.8.
    Gamma,
    /// Sensor noise correlated over 1 px.
    CorrelatedNoise,
    /// Anisotropic PSF, gamma 0.8 and noise correlated over 0.7 px together.
    Combined,
}

impl Effect {
    fn apply(self, spec: &mut SceneSpec, blur: f64) {
        let anisotropic = Psf::Anisotropic {
            sigma_major: 1.35 * blur,
            sigma_minor: 0.75 * blur,
            angle_deg: 30.0,
        };
        match self {
            Effect::None => {}
            Effect::Anisotropic => spec.psf = anisotropic,
            Effect::Halo => {
                spec.psf = Psf::Halo {
                    sigma: blur,
                    halo_sigma: 6.0,
                    halo_weight: 0.15,
                }
            }
            Effect::Disc => spec.psf = Psf::Disc { radius: 2.0 * blur },
            Effect::Gamma => spec.gamma = 0.8,
            Effect::CorrelatedNoise => spec.noise_corr_px = 1.0,
            Effect::Combined => {
                spec.psf = anisotropic;
                spec.gamma = 0.8;
                spec.noise_corr_px = 0.7;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    kind: Kind,
    /// Drawn outer marker radius at the image centre, px.
    outer_radius_px: f64,
    blur: f64,
    contrast: f64,
    tilt: f64,
    effect: Effect,
}

/// One marker's photometric fits across noise seeds, keyed by its true cell.
#[derive(Default)]
struct MarkerSamples {
    centers: Vec<[f64; 2]>,
    /// `(trace of the reported covariance, covariance inflation)` per seed.
    traces: Vec<(f64, Option<f64>)>,
    /// The fitted PSF per seed (`study` feature only).
    blurs: Vec<BlurSample>,
    truth: [f64; 2],
}

/// One marker's fitted PSF `(sigma, a, b)` with its reported covariance
/// (correlated-residual and white-residual).
#[derive(Debug, Clone, Copy)]
struct BlurSample {
    /// Exposure (noise seed) of the sample.
    seed: u64,
    value: [f64; 3],
    cov: [[f64; 3]; 3],
    cov_white: [[f64; 3]; 3],
}

/// Pooled PSF statistics of a cell.
#[derive(Debug, Clone, Copy)]
struct BlurSummary {
    /// Reported sd over Monte-Carlo scatter of `sigma`, `a`, `b`:
    /// `(correlated, white)` covariance.
    ratio: [(f64, f64); 3],
    /// Mean fitted `sigma`, `a`, `b` over all samples.
    mean: [f64; 3],
    /// Pooled Monte-Carlo sd of `sigma`, `a`, `b`.
    sd: [f64; 3],
    /// Fraction of samples whose covariance fell back to the white
    /// posterior (reported covariance equal to the white one).
    fallback: f64,
    /// Correlations of `(sigma, a)`, `(sigma, b)`, `(a, b)`: pooled reported
    /// and pooled empirical (within-marker deviations).
    corr: [(f64, f64); 3],
}

/// The `(sigma, a)`, `(sigma, b)`, `(a, b)` pairs.
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

#[derive(Debug, Default, Clone)]
struct CellResult {
    n_visible: usize,
    /// Detected markers, per exposure (mean).
    n_detected: f64,
    /// Markers with a board position, per exposure (mean).
    n_located: f64,
    /// Photometric centres per exposure (mean).
    n_photometric: f64,
    /// Markers with a wrong board position (all exposures).
    n_wrong: usize,
    /// Markers whose photometric metrics are missing (must be 0).
    n_silent: usize,
    rms_projective: f64,
    rms_photometric: f64,
    /// Homography RMS error at the true cell centres, first exposure.
    h_rms_projective: Option<f64>,
    h_rms_photometric: Option<f64>,
    /// Pooled reported σ / Monte-Carlo scatter: correlated, white covariance.
    scatter_ratio: Option<(f64, f64)>,
    /// Pooled reported σ / error against the truth: correlated, white.
    error_ratio: Option<(f64, f64)>,
    /// RMS over markers of the seed-mean error (bias), px.
    bias_rms: Option<f64>,
    /// The fitted PSF (`study` feature only).
    blur: Option<BlurSummary>,
    /// Mean covariance inflation over photometric centres (NaN without the
    /// `study` feature).
    inflation: f64,
    /// Mean fit residual standard deviation over photometric centres.
    residual: f64,
    statuses: BTreeMap<String, usize>,
    /// Print radii the photometric stage estimated, per exposure (`None`
    /// where it kept the nominal radii).
    radii_by_seed: Vec<Option<[f64; 4]>>,
    /// Per-exposure structure of the fitted-sigma error (`study` feature).
    exposure: Option<ExposureStats>,
    /// Pooled inner radius (mm) the fits used (`MATRIX_POOLED_RADII`), and
    /// the number of exposures it was pooled from.
    pooled_r0: Option<(f64, usize)>,
    /// Photometric centre RMS per exposure (NaN without photometric
    /// centres), for the seed-to-seed scatter of `rms_photometric`.
    rms_by_seed: Vec<f64>,
}

/// How the fitted-sigma error splits between a per-exposure common mode and
/// the markers of one exposure, and what drives the common mode.
#[derive(Debug, Clone)]
struct ExposureStats {
    n_exposures: usize,
    /// Exposures whose print radii were estimated (not nominal).
    n_estimated: usize,
    /// Share of the per-marker sigma variance across exposures that is the
    /// exposure's common mode.
    between_share: f64,
    /// Reported sd over the within-exposure scatter (the common mode
    /// removed), for `sigma`, `a`, `b`.
    within_ratio: [f64; 3],
    /// Per free radius: sd over exposures of the consensus error `dr`
    /// (mm, 0 where nominal), correlation of the common mode with `dr`, and
    /// the slope `d sigma / d r` (px/mm).
    radii: Vec<(f64, f64, f64)>,
    /// R² of the common mode on all free radii jointly.
    r2: f64,
    /// Mean sigma error against the PSF plus pixel box (px).
    mean_error: f64,
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|e| e * e).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// Ground truth of a frame: projected cell centres and the cells whose full
/// outer ring lies inside the image.
struct Truth {
    centers: Vec<[f64; 2]>,
    visible: usize,
    /// Lattice spacing in px at the image centre.
    spacing_px: f64,
}

impl Truth {
    fn new(target: &TargetLayout, h: &[[f64; 3]; 3], size: [u32; 2], px_per_mm: f64) -> Self {
        let [w, hh] = [f64::from(size[0]), f64::from(size[1])];
        let r_mm = f64::from(target.outer_draw_radius_mm());
        let mut visible = 0;
        let centers = target
            .cells()
            .iter()
            .map(|c| {
                let xy = [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])];
                let inside = (0..16).all(|k| {
                    let a = k as f64 * std::f64::consts::PI / 8.0;
                    let p = apply_h(h, [xy[0] + r_mm * a.cos(), xy[1] + r_mm * a.sin()]);
                    p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= w - 1.0 && p[1] <= hh - 1.0
                });
                visible += usize::from(inside);
                apply_h(h, xy)
            })
            .collect();
        Self {
            centers,
            visible,
            spacing_px: f64::from(target.min_center_spacing_mm()) * px_per_mm,
        }
    }

    /// The true cell of a detected marker: from its board position when it has
    /// one (`Err` when that is far from the marker: a wrong label), else the
    /// nearest projected cell.
    fn cell_of(&self, m: &DetectedMarker, h: &[[f64; 3]; 3]) -> Result<usize, ()> {
        let near = |p: [f64; 2]| {
            self.centers
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    let da = (a.1[0] - p[0]).hypot(a.1[1] - p[1]);
                    let db = (b.1[0] - p[0]).hypot(b.1[1] - p[1]);
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i)
                .ok_or(())
        };
        let idx = match m.board_xy_mm {
            Some(xy) => {
                let p = apply_h(h, xy);
                near(p)?
            }
            None => near(m.center)?,
        };
        let t = self.centers[idx];
        let err = (m.center[0] - t[0]).hypot(m.center[1] - t[1]);
        if err > 0.4 * self.spacing_px {
            return Err(());
        }
        Ok(idx)
    }
}

fn detector(
    target: &TargetLayout,
    outer_radius_px: f64,
    method: CircleRefinementMethod,
) -> Detector {
    let mut config = DetectConfig::from_target_and_marker_diameter(
        target.clone(),
        (2.0 * outer_radius_px) as f32,
    );
    config.circle_refinement = method;
    if std::env::var("MATRIX_FIXED_RADII").is_ok() {
        config.advanced.photometric.estimate_print_radii = false;
    }
    Detector::with_config(config)
}

fn spec_for(cell: &Cell, target: &TargetLayout, size: [u32; 2]) -> SceneSpec {
    let mut spec = SceneSpec::for_target(
        target,
        cell.outer_radius_px,
        cell.blur,
        cell.tilt,
        cell.contrast,
        1,
    );
    spec.image_size = size;
    cell.effect.apply(&mut spec, cell.blur);
    spec
}

/// Per fitted marker (by index), the covariance inflation (reported centre σ
/// over the white-residual σ) and the fitted PSF. Study output, empty without
/// the feature.
type Inflations = HashMap<usize, (Option<f64>, BlurSample)>;

/// Photometric detection with the markers' covariance inflations.
#[cfg(feature = "study")]
fn detect_photometric(
    det: &Detector,
    image: &image::GrayImage,
) -> (DetectionResult, DetectionDiagnostics, Inflations) {
    let (res, diag, residuals) = det
        .detect_with_photometric_residuals(image)
        .expect("detect");
    let inflations = residuals
        .iter()
        .map(|r| {
            let blur = BlurSample {
                seed: 0,
                value: [r.blur_sigma_px, r.aniso[0], r.aniso[1]],
                cov: r.blur_covariance,
                cov_white: r.blur_covariance_white,
            };
            (r.marker_index, (r.covariance_inflation, blur))
        })
        .collect();
    (res, diag, inflations)
}

/// Photometric detection; the inflations need the `study` feature.
#[cfg(not(feature = "study"))]
fn detect_photometric(
    det: &Detector,
    image: &image::GrayImage,
) -> (DetectionResult, DetectionDiagnostics, Inflations) {
    let (res, diag) = det.detect_with_diagnostics(image).expect("detect");
    (res, diag, Inflations::new())
}

/// RMS error of a result's homography at the true centres of the cells
/// imaged inside the frame; `None` without a homography in the absolute
/// board frame (the truth's).
fn homography_rms(
    res: &DetectionResult,
    target: &TargetLayout,
    truth: &Truth,
    size: [u32; 2],
) -> Option<f64> {
    if res.board_frame != Some(BoardFrame::Absolute) {
        return None;
    }
    let h = res.homography?;
    let inside = |p: [f64; 2]| {
        p[0] >= 0.0 && p[1] >= 0.0 && p[0] < f64::from(size[0]) && p[1] < f64::from(size[1])
    };
    let errs: Vec<f64> = target
        .cells()
        .iter()
        .zip(&truth.centers)
        .filter(|(_, t)| inside(**t))
        .map(|(c, t)| {
            let p = apply_h(&h, [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])]);
            (p[0] - t[0]).hypot(p[1] - t[1])
        })
        .collect();
    Some(rms(&errs))
}

/// Projective-centre RMS of one exposure.
fn projective_rms(res: &DetectionResult, truth: &Truth, h: &[[f64; 3]; 3]) -> f64 {
    let errs: Vec<f64> = res
        .detected_markers
        .iter()
        .filter_map(|m| {
            let i = truth.cell_of(m, h).ok()?;
            let t = truth.centers[i];
            Some((m.center[0] - t[0]).hypot(m.center[1] - t[1]))
        })
        .collect();
    rms(&errs)
}

/// Run one cell: `n_seeds` photometric exposures plus a projective baseline.
fn run_cell(board: &BoardRaster, cell: &Cell, n_seeds: u64, size: [u32; 2]) -> CellResult {
    let target = &board.target;
    let spec = spec_for(cell, target, size);
    let rad: Radiance = radiance(board, &spec);
    let truth = Truth::new(target, &rad.h, size, spec.px_per_mm);
    let mut pooled_r0 = None;
    let photometric = match pooled_inner_radius(target, cell, &rad, &spec, n_seeds) {
        // Radii fixed to the value pooled over the cell's exposures.
        Some((r0, n)) => {
            pooled_r0 = Some((r0, n));
            let pooled = cell.kind.target_with_inner(Some(r0 as f32));
            let mut config = DetectConfig::from_target_and_marker_diameter(
                pooled,
                (2.0 * cell.outer_radius_px) as f32,
            );
            config.circle_refinement = CircleRefinementMethod::Photometric;
            config.advanced.photometric.estimate_print_radii = false;
            Detector::with_config(config)
        }
        // A nominal inner radius off the printed one (`MATRIX_NOMINAL_INNER`).
        None => match std::env::var("MATRIX_NOMINAL_INNER") {
            Ok(inner) => detector(
                &cell
                    .kind
                    .target_with_inner(Some(inner.parse().expect("MATRIX_NOMINAL_INNER is mm"))),
                cell.outer_radius_px,
                CircleRefinementMethod::Photometric,
            ),
            Err(_) => detector(
                target,
                cell.outer_radius_px,
                CircleRefinementMethod::Photometric,
            ),
        },
    };
    let projective = detector(
        target,
        cell.outer_radius_px,
        CircleRefinementMethod::ProjectiveCenter,
    );
    let mut out = CellResult {
        n_visible: truth.visible,
        pooled_r0,
        ..CellResult::default()
    };
    let mut samples: BTreeMap<usize, MarkerSamples> = BTreeMap::new();
    let mut errors = Vec::new();
    let mut inflations = Vec::new();
    for seed in 0..n_seeds {
        let first_error = errors.len();
        let scene = expose(&rad, &spec, 1000 + seed);
        let (res, diag, inflations_of) = detect_photometric(&photometric, &scene.image);
        if seed == 0 {
            let res_p = projective.detect(&scene.image).expect("detect");
            out.rms_projective = projective_rms(&res_p, &truth, &rad.h);
            out.h_rms_projective = homography_rms(&res_p, target, &truth, size);
            out.h_rms_photometric = homography_rms(&res, target, &truth, size);
        }
        out.n_detected += res.detected_markers.len() as f64;
        out.radii_by_seed.push(diag.photometric_print_radii_mm);
        for (index, (m, d)) in res.detected_markers.iter().zip(&diag.markers).enumerate() {
            let status = d
                .photometric
                .as_ref()
                .map(|p| format!("{:?}", p.status))
                .unwrap_or_else(|| "missing".into());
            if d.photometric.is_none() {
                out.n_silent += 1;
            }
            if d.photometric.as_ref().map(|p| p.status) != Some(PhotometricStatus::NoBoardPosition)
            {
                out.n_located += 1.0;
            }
            *out.statuses.entry(status).or_default() += 1;
            let Ok(idx) = truth.cell_of(m, &rad.h) else {
                out.n_wrong += 1;
                continue;
            };
            if m.center_source != CenterSource::Photometric {
                continue;
            }
            let (Some(cov), Some(p)) = (m.center_cov, d.photometric.as_ref()) else {
                continue;
            };
            let study = inflations_of.get(&index).copied();
            let inflation = study.and_then(|s| s.0);
            let t = truth.centers[idx];
            errors.push((m.center[0] - t[0]).hypot(m.center[1] - t[1]));
            inflations.push(inflation);
            out.residual += p.residual_sigma.unwrap_or(0.0);
            out.n_photometric += 1.0;
            let s = samples.entry(idx).or_default();
            s.truth = t;
            s.centers.push(m.center);
            s.traces.push((cov[0][0] + cov[1][1], inflation));
            s.blurs.extend(study.map(|s| BlurSample { seed, ..s.1 }));
        }
        let seed_errors = &errors[first_error..];
        out.rms_by_seed.push(if seed_errors.is_empty() {
            f64::NAN
        } else {
            rms(seed_errors)
        });
    }
    out.n_detected /= n_seeds as f64;
    out.n_located /= n_seeds as f64;
    out.n_photometric /= n_seeds as f64;
    out.rms_photometric = rms(&errors);
    let known: Vec<f64> = inflations.iter().flatten().copied().collect();
    out.inflation = if known.is_empty() {
        f64::NAN
    } else {
        known.iter().sum::<f64>() / known.len() as f64
    };
    out.residual /= inflations.len().max(1) as f64;
    summarize_monte_carlo(&samples, n_seeds, &mut out);
    out.exposure = exposure_stats(
        &samples,
        &out.radii_by_seed,
        true_print_radii(target),
        if target.is_coded() { 3 } else { 1 },
        psf_moments(spec.psf, true)[0],
        n_seeds,
    );
    out
}

/// With `MATRIX_POOLED_RADII` set, on a plain target: the median over the
/// cell's exposures of the per-image inner-radius estimate (where one was
/// made) and the number of exposures it came from.
fn pooled_inner_radius(
    target: &TargetLayout,
    cell: &Cell,
    rad: &Radiance,
    spec: &SceneSpec,
    n_seeds: u64,
) -> Option<(f64, usize)> {
    if std::env::var("MATRIX_POOLED_RADII").is_err() || target.is_coded() {
        return None;
    }
    let det = detector(
        target,
        cell.outer_radius_px,
        CircleRefinementMethod::Photometric,
    );
    let mut r0: Vec<f64> = (0..n_seeds)
        .filter_map(|seed| {
            let scene = expose(rad, spec, 1000 + seed);
            let (_, diag) = det.detect_with_diagnostics(&scene.image).expect("detect");
            diag.photometric_print_radii_mm.map(|r| r[0])
        })
        .collect();
    if r0.is_empty() {
        return None;
    }
    r0.sort_by(|a, b| a.total_cmp(b));
    Some((r0[r0.len() / 2], r0.len()))
}

/// The printed radii of the rendered target (board mm), as the photometric
/// model orders them: hole edge, inner ring outer edge, outer ring inner
/// edge, outer edge.
fn true_print_radii(target: &TargetLayout) -> [f64; 4] {
    let ring = target.ring();
    let (inner, outer) = (
        f64::from(ring.inner_radius_mm),
        f64::from(ring.outer_radius_mm),
    );
    let h = f64::from(target.outer_draw_radius_mm()) - outer;
    if target.is_coded() {
        [inner - h, inner + h, outer - h, outer + h]
    } else {
        [inner, outer, outer, outer]
    }
}

/// Per-exposure structure of the fitted-sigma error over the markers fitted
/// in (almost) every exposure; `None` without blur samples.
fn exposure_stats(
    samples: &BTreeMap<usize, MarkerSamples>,
    radii_by_seed: &[Option<[f64; 4]>],
    true_radii: [f64; 4],
    n_free: usize,
    true_sigma: f64,
    n_seeds: u64,
) -> Option<ExposureStats> {
    let markers: Vec<&MarkerSamples> = samples
        .values()
        .filter(|s| s.blurs.len() >= 2 && (s.blurs.len() as u64) + 1 >= n_seeds)
        .collect();
    if markers.len() < 3 {
        return None;
    }
    let n_exp = n_seeds as usize;
    // Per marker and component: mean over its exposures.
    let means: Vec<[f64; 3]> = markers
        .iter()
        .map(|s| {
            let n = s.blurs.len() as f64;
            std::array::from_fn(|k| s.blurs.iter().map(|b| b.value[k]).sum::<f64>() / n)
        })
        .collect();
    // Exposure common mode: mean deviation of the markers present.
    let mut common = vec![[0.0f64; 3]; n_exp];
    let mut count = vec![0usize; n_exp];
    for (s, m) in markers.iter().zip(&means) {
        for b in &s.blurs {
            let e = b.seed as usize;
            count[e] += 1;
            for k in 0..3 {
                common[e][k] += b.value[k] - m[k];
            }
        }
    }
    for (c, &n) in common.iter_mut().zip(&count) {
        for v in c.iter_mut() {
            *v /= n.max(1) as f64;
        }
    }
    let used: Vec<usize> = (0..n_exp).filter(|&e| count[e] >= 3).collect();
    let mean_of =
        |f: &dyn Fn(usize) -> f64| used.iter().map(|&e| f(e)).sum::<f64>() / used.len() as f64;
    let var_of = |f: &dyn Fn(usize) -> f64| {
        let m = mean_of(f);
        used.iter().map(|&e| (f(e) - m).powi(2)).sum::<f64>() / (used.len() as f64 - 1.0)
    };
    // Total per-marker variance, within-exposure variance and reported.
    let (mut total, mut within, mut reported) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for (s, m) in markers.iter().zip(&means) {
        let n = s.blurs.len() as f64;
        for k in 0..3 {
            total[k] += s
                .blurs
                .iter()
                .map(|b| (b.value[k] - m[k]).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            let w: f64 = s
                .blurs
                .iter()
                .map(|b| {
                    let e = b.seed as usize;
                    let mk = count[e] as f64;
                    // Removing the common mode also removes 1/M of the marker's own
                    // deviation: correct for it.
                    (b.value[k] - m[k] - common[e][k]).powi(2) * mk / (mk - 1.0)
                })
                .sum();
            within[k] += w / (n - 1.0);
            reported[k] += s.blurs.iter().map(|b| b.cov[k][k]).sum::<f64>() / n;
        }
    }
    let between = var_of(&|e| common[e][0]);
    let total_sigma = total[0] / markers.len() as f64;
    let dr = |e: usize, k: usize| {
        radii_by_seed
            .get(e)
            .copied()
            .flatten()
            .map_or(0.0, |r| r[k] - true_radii[k])
    };
    let corr = |f: &dyn Fn(usize) -> f64, g: &dyn Fn(usize) -> f64| {
        let (mf, mg) = (mean_of(f), mean_of(g));
        let c = used.iter().map(|&e| (f(e) - mf) * (g(e) - mg)).sum::<f64>()
            / (used.len() as f64 - 1.0);
        c / (var_of(f) * var_of(g)).sqrt()
    };
    let y = |e: usize| common[e][0];
    let radii: Vec<(f64, f64, f64)> = (0..n_free)
        .map(|k| {
            let x = |e: usize| dr(e, k);
            let vx = var_of(&x);
            let c = corr(&y, &x);
            let slope = c * (var_of(&y) / vx).sqrt();
            (vx.sqrt(), c, slope)
        })
        .collect();
    let r2 = joint_r2(&used, &y, &|e, k| dr(e, k), n_free);
    let mean_error = markers
        .iter()
        .flat_map(|s| s.blurs.iter().map(|b| b.value[0] - true_sigma))
        .sum::<f64>()
        / markers.iter().map(|s| s.blurs.len()).sum::<usize>() as f64;
    Some(ExposureStats {
        n_exposures: used.len(),
        n_estimated: radii_by_seed.iter().filter(|r| r.is_some()).count(),
        between_share: between / total_sigma,
        within_ratio: std::array::from_fn(|k| (reported[k] / within[k]).sqrt()),
        radii,
        r2,
        mean_error,
    })
}

/// R² of the least-squares fit of `y(e)` on `x(e, 0..n)` with an intercept,
/// over the exposures `used` (NaN when singular).
fn joint_r2(
    used: &[usize],
    y: &dyn Fn(usize) -> f64,
    x: &dyn Fn(usize, usize) -> f64,
    n: usize,
) -> f64 {
    let m = n + 1;
    let row = |e: usize| -> Vec<f64> {
        std::iter::once(1.0)
            .chain((0..n).map(|k| x(e, k)))
            .collect()
    };
    let mut a = vec![vec![0.0f64; m + 1]; m];
    for &e in used {
        let r = row(e);
        for i in 0..m {
            for j in 0..m {
                a[i][j] += r[i] * r[j];
            }
            a[i][m] += r[i] * y(e);
        }
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..m {
        let p = (c..m)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        if a[c][c].abs() < 1e-18 {
            return f64::NAN;
        }
        for i in 0..m {
            if i != c {
                let f = a[i][c] / a[c][c];
                for j in c..=m {
                    a[i][j] -= f * a[c][j];
                }
            }
        }
    }
    let beta: Vec<f64> = (0..m).map(|i| a[i][m] / a[i][i]).collect();
    let ym = used.iter().map(|&e| y(e)).sum::<f64>() / used.len() as f64;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for &e in used {
        let pred: f64 = row(e).iter().zip(&beta).map(|(r, b)| r * b).sum();
        ss_res += (y(e) - pred).powi(2);
        ss_tot += (y(e) - ym).powi(2);
    }
    1.0 - ss_res / ss_tot
}

/// Pooled covariance honesty from the per-marker samples.
fn summarize_monte_carlo(
    samples: &BTreeMap<usize, MarkerSamples>,
    n_seeds: u64,
    out: &mut CellResult,
) {
    let (mut emp, mut err2, mut rep_c, mut rep_w, mut bias2, mut n_err, mut n_m) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0usize, 0usize);
    for s in samples.values() {
        let n = s.centers.len();
        if n < 2 || (n as u64) + 1 < n_seeds {
            continue;
        }
        let m = [
            s.centers.iter().map(|c| c[0]).sum::<f64>() / n as f64,
            s.centers.iter().map(|c| c[1]).sum::<f64>() / n as f64,
        ];
        emp += s
            .centers
            .iter()
            .map(|c| (c[0] - m[0]).powi(2) + (c[1] - m[1]).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        for (c, (tr, infl)) in s.centers.iter().zip(&s.traces) {
            err2 += (c[0] - s.truth[0]).powi(2) + (c[1] - s.truth[1]).powi(2);
            n_err += 1;
            rep_c += tr / n as f64;
            // White-residual trace; unknown (NaN) without the inflation.
            rep_w += infl.map_or(f64::NAN, |i| tr / (i * i)) / n as f64;
        }
        bias2 += (m[0] - s.truth[0]).powi(2) + (m[1] - s.truth[1]).powi(2);
        n_m += 1;
    }
    if n_m < 3 {
        return;
    }
    out.scatter_ratio = Some(((rep_c / emp).sqrt(), (rep_w / emp).sqrt()));
    // Mean reported trace per sample vs mean squared error per sample.
    let mean_err2 = err2 / n_err as f64;
    out.error_ratio = Some((
        (rep_c / n_m as f64 / mean_err2).sqrt(),
        (rep_w / n_m as f64 / mean_err2).sqrt(),
    ));
    out.bias_rms = Some((bias2 / n_m as f64).sqrt());
    out.blur = summarize_blur(samples, n_seeds);
}

/// Pooled PSF honesty and mean, over markers fitted in (almost) every seed.
fn summarize_blur(samples: &BTreeMap<usize, MarkerSamples>, n_seeds: u64) -> Option<BlurSummary> {
    let (mut emp, mut rep_c, mut rep_w, mut sum, mut n_s, mut n_m) =
        ([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], 0usize, 0usize);
    let mut n_fallback = 0usize;
    let (mut emp_kl, mut rep_kl) = ([[0.0; 3]; 3], [[0.0; 3]; 3]);
    for s in samples.values() {
        let n = s.blurs.len();
        if n < 2 || (n as u64) + 1 < n_seeds {
            continue;
        }
        for k in 0..3 {
            let m = s.blurs.iter().map(|b| b.value[k]).sum::<f64>() / n as f64;
            emp[k] += s
                .blurs
                .iter()
                .map(|b| (b.value[k] - m).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;
            rep_c[k] += s.blurs.iter().map(|b| b.cov[k][k]).sum::<f64>() / n as f64;
            rep_w[k] += s.blurs.iter().map(|b| b.cov_white[k][k]).sum::<f64>() / n as f64;
            sum[k] += s.blurs.iter().map(|b| b.value[k]).sum::<f64>();
        }
        let m: [f64; 3] =
            std::array::from_fn(|k| s.blurs.iter().map(|b| b.value[k]).sum::<f64>() / n as f64);
        for k in 0..3 {
            for l in 0..3 {
                emp_kl[k][l] += s
                    .blurs
                    .iter()
                    .map(|b| (b.value[k] - m[k]) * (b.value[l] - m[l]))
                    .sum::<f64>()
                    / (n - 1) as f64;
                rep_kl[k][l] += s.blurs.iter().map(|b| b.cov[k][l]).sum::<f64>() / n as f64;
            }
        }
        n_s += n;
        n_fallback += s.blurs.iter().filter(|b| b.cov == b.cov_white).count();
        n_m += 1;
    }
    (n_m >= 3).then(|| BlurSummary {
        ratio: std::array::from_fn(|k| ((rep_c[k] / emp[k]).sqrt(), (rep_w[k] / emp[k]).sqrt())),
        mean: sum.map(|v| v / n_s as f64),
        sd: emp.map(|v| (v / n_m as f64).sqrt()),
        fallback: n_fallback as f64 / n_s as f64,
        corr: PAIRS.map(|(k, l)| {
            let rho = |c: &[[f64; 3]; 3]| c[k][l] / (c[k][k] * c[l][l]).sqrt();
            (rho(&rep_kl), rho(&emp_kl))
        }),
    })
}

/// Second moments of the rendered PSF as `(sigma, a, b)` of the model's
/// parametrization (covariance `sigma^2 [[1 + a, b], [b, 1 - a]]`), with
/// `pixel_box` adding the variance 1/12 per axis of the area sampling.
fn psf_moments(psf: Psf, pixel_box: bool) -> [f64; 3] {
    let iso = |v: f64| [[v, 0.0], [0.0, v]];
    let mut c = match psf {
        Psf::Gaussian { sigma } => iso(sigma * sigma),
        Psf::Anisotropic {
            sigma_major,
            sigma_minor,
            angle_deg,
        } => {
            let (s, co) = angle_deg.to_radians().sin_cos();
            let (ma, mi) = (sigma_major * sigma_major, sigma_minor * sigma_minor);
            [
                [ma * co * co + mi * s * s, (ma - mi) * co * s],
                [(ma - mi) * co * s, ma * s * s + mi * co * co],
            ]
        }
        Psf::Halo {
            sigma,
            halo_sigma,
            halo_weight,
        } => iso((1.0 - halo_weight) * sigma * sigma + halo_weight * halo_sigma * halo_sigma),
        Psf::Disc { radius } => iso(0.25 * radius * radius),
        Psf::VaryingGaussian { .. } => unreachable!("not in the matrix"),
    };
    if pixel_box {
        c[0][0] += 1.0 / 12.0;
        c[1][1] += 1.0 / 12.0;
    }
    let tr = c[0][0] + c[1][1];
    [
        (0.5 * tr).sqrt(),
        (c[0][0] - c[1][1]) / tr,
        2.0 * c[0][1] / tr,
    ]
}

fn opt4(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{v:.4}"))
}

fn fmt_pair(p: Option<(f64, f64)>) -> String {
    p.map_or("    -       -  ".into(), |(c, w)| {
        format!("{c:6.3} {w:6.3}")
    })
}

fn print_header() {
    println!(
        "kind               R_px blur  con tilt effect          vis   det   loc  phot wrong  rms_proj rms_phot  scatter(corr white)  err(corr white)  bias   infl  resid  h_proj  h_phot"
    );
}

fn print_row(cell: &Cell, r: &CellResult) {
    println!(
        "{:<18} {:>4.0} {:>4.1} {:>4.0} {:>4.0} {:<15} {:>4} {:>5.1} {:>5.1} {:>5.1} {:>4}  {:>8.4} {:>8.4}  {}       {}   {:>6} {:>5.2} {:>5.2} {:>7} {:>7}",
        cell.kind.name(),
        cell.outer_radius_px,
        cell.blur,
        cell.contrast,
        cell.tilt,
        format!("{:?}", cell.effect),
        r.n_visible,
        r.n_detected,
        r.n_located,
        r.n_photometric,
        r.n_wrong,
        r.rms_projective,
        r.rms_photometric,
        fmt_pair(r.scatter_ratio),
        fmt_pair(r.error_ratio),
        r.bias_rms.map_or("-".into(), |b| format!("{b:.4}")),
        r.inflation,
        r.residual,
        opt4(r.h_rms_projective),
        opt4(r.h_rms_photometric),
    );
    if let Some(b) = r.blur {
        let mut spec = SceneSpec::for_target(
            &cell.kind.target(),
            cell.outer_radius_px,
            cell.blur,
            0.0,
            0.0,
            0,
        );
        cell.effect.apply(&mut spec, cell.blur);
        let psf = psf_moments(spec.psf, false);
        let eff = psf_moments(spec.psf, true);
        println!(
            "    blur: sigma {:.4} (psf {:.4} bias {:+.4} {:+.1}%, +pixel {:.4} bias {:+.4} {:+.1}%) sd {:.4} ratio {:.3}/{:.3} | a {:+.4} (true {:+.4}) sd {:.4} ratio {:.3}/{:.3} | b {:+.4} (true {:+.4}) sd {:.4} ratio {:.3}/{:.3} | fallback {:.3} | rho {:+.2}/{:+.2} {:+.2}/{:+.2} {:+.2}/{:+.2}",
            b.mean[0],
            psf[0],
            b.mean[0] - psf[0],
            100.0 * (b.mean[0] / psf[0] - 1.0),
            eff[0],
            b.mean[0] - eff[0],
            100.0 * (b.mean[0] / eff[0] - 1.0),
            b.sd[0],
            b.ratio[0].0,
            b.ratio[0].1,
            b.mean[1],
            eff[1],
            b.sd[1],
            b.ratio[1].0,
            b.ratio[1].1,
            b.mean[2],
            eff[2],
            b.sd[2],
            b.ratio[2].0,
            b.ratio[2].1,
            b.fallback,
            b.corr[0].0,
            b.corr[0].1,
            b.corr[1].0,
            b.corr[1].1,
            b.corr[2].0,
            b.corr[2].1,
        );
    }
    let estimated: Vec<[f64; 4]> = r.radii_by_seed.iter().flatten().copied().collect();
    if !estimated.is_empty() {
        let n = estimated.len() as f64;
        let stats: Vec<String> = (0..3)
            .map(|k| {
                let m = estimated.iter().map(|x| x[k]).sum::<f64>() / n;
                let sd = (estimated.iter().map(|x| (x[k] - m).powi(2)).sum::<f64>()
                    / (n - 1.0).max(1.0))
                .sqrt();
                format!("r{k} {m:.4} sd {sd:.4}")
            })
            .collect();
        println!(
            "    print radii estimated in {} exposures: {}",
            estimated.len(),
            stats.join(", ")
        );
    }
    let seeds: Vec<f64> = r
        .rms_by_seed
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if seeds.len() > 1 {
        let n = seeds.len() as f64;
        let mean = seeds.iter().sum::<f64>() / n;
        let sd = (seeds.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let list: Vec<String> = r.rms_by_seed.iter().map(|v| format!("{v:.5}")).collect();
        println!(
            "    rms by seed: mean {mean:.5} se {:.5} | {}",
            sd / n.sqrt(),
            list.join(",")
        );
    }
    if let Some((r0, n)) = r.pooled_r0 {
        println!("    pooled r0 {r0:.4} mm from {n} exposures");
    }
    if let Some(e) = &r.exposure {
        let radii: Vec<String> = e
            .radii
            .iter()
            .enumerate()
            .map(|(k, (sd, c, g))| format!("r{k} sd {sd:.4} corr {c:+.2} slope {g:+.3}"))
            .collect();
        println!(
            "    exposure: n {} estimated {} | between share {:.2} | within ratio {:.3} {:.3} {:.3} | mean err {:+.4} | {} | R2 {:.2}",
            e.n_exposures,
            e.n_estimated,
            e.between_share,
            e.within_ratio[0],
            e.within_ratio[1],
            e.within_ratio[2],
            e.mean_error,
            radii.join(", "),
            e.r2,
        );
    }
    if r.n_silent > 0 || std::env::var("MATRIX_STATUSES").is_ok() {
        println!("    statuses {:?}", r.statuses);
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn selected_kinds() -> Vec<Kind> {
    match std::env::var("MATRIX_KINDS") {
        Ok(list) => KINDS
            .into_iter()
            .filter(|k| list.split(',').any(|n| n.trim() == k.name()))
            .collect(),
        Err(_) => KINDS.to_vec(),
    }
}

/// Raster density for a marker scale: about seven raster pixels per camera
/// pixel.
fn raster_dpi(target: &TargetLayout, outer_radius_px: f64) -> f32 {
    if let Ok(dpi) = std::env::var("MATRIX_DPI") {
        return dpi.parse().expect("MATRIX_DPI is a number");
    }
    let px_per_mm = outer_radius_px / f64::from(target.outer_draw_radius_mm());
    (7.0 * 25.4 * px_per_mm).clamp(600.0, 1200.0) as f32
}

/// `f` over `items` on `MATRIX_THREADS` threads (default 8), in order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let results: Mutex<Vec<Option<R>>> = Mutex::new(items.iter().map(|_| None).collect());
    let next = AtomicUsize::new(0);
    let threads = env_or("MATRIX_THREADS", 8usize).max(1);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let r = f(item);
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("item ran"))
        .collect()
}

/// Run `cells` in parallel (they share one raster per kind and scale) and
/// print them in order.
fn run_cells(cells: &[Cell], n_seeds: u64, size: [u32; 2]) -> Vec<CellResult> {
    let mut rasters: BTreeMap<(usize, u64), BoardRaster> = BTreeMap::new();
    for c in cells {
        let k = KINDS.iter().position(|k| *k == c.kind).unwrap();
        rasters
            .entry((k, c.outer_radius_px.to_bits()))
            .or_insert_with(|| {
                let target = c.kind.target();
                let dpi = raster_dpi(&target, c.outer_radius_px);
                BoardRaster::with_dpi(target, dpi)
            });
    }
    let results = parallel_map(cells, |c| {
        let k = KINDS.iter().position(|k| *k == c.kind).unwrap();
        let board = &rasters[&(k, c.outer_radius_px.to_bits())];
        run_cell(board, c, n_seeds, size)
    });
    print_header();
    for (c, r) in cells.iter().zip(&results) {
        print_row(c, r);
    }
    results
}

/// Default run: one medium-scale cell per target kind. Every marker reports a
/// photometric status, the photometric centres cover the located markers and
/// meet 0.05 px RMS, no label is wrong, and the returned homography — refit
/// from the refined centres — is more accurate than the projective-centre one.
#[test]
fn photometric_refines_every_target_kind() {
    let cells: Vec<Cell> = KINDS
        .iter()
        .map(|&kind| Cell {
            kind,
            outer_radius_px: 17.0,
            blur: 1.5,
            contrast: 80.0,
            tilt: 20.0,
            effect: Effect::None,
        })
        .collect();
    let results = run_cells(&cells, 1, [400, 300]);
    for (c, r) in cells.iter().zip(&results) {
        let name = c.kind.name();
        assert_eq!(r.n_silent, 0, "{name}: every marker reports a status");
        assert_eq!(r.n_wrong, 0, "{name}: wrong labels");
        assert!(r.n_located >= 10.0, "{name}: {} located", r.n_located);
        assert!(
            r.n_photometric >= 0.9 * r.n_located,
            "{name}: photometric coverage {} of {}",
            r.n_photometric,
            r.n_located
        );
        assert!(
            r.rms_photometric <= 0.05,
            "{name}: photometric rms {}",
            r.rms_photometric
        );
        assert!(
            r.rms_photometric < r.rms_projective,
            "{name}: photometric {} vs projective {}",
            r.rms_photometric,
            r.rms_projective
        );
        if let (Some(proj), Some(phot)) = (r.h_rms_projective, r.h_rms_photometric) {
            assert!(phot < proj, "{name}: homography rms {phot} vs {proj}");
        }
    }
}

/// Fail closed: a fit stopped by the iteration cap before it converged
/// reports `NotConverged`, and its marker keeps the centre the pipeline
/// produced without the stage.
#[test]
fn unconverged_fits_keep_their_previous_centre() {
    let cell = Cell {
        kind: Kind::PlainRectDots,
        outer_radius_px: 17.0,
        blur: 1.5,
        contrast: 80.0,
        tilt: 20.0,
        effect: Effect::None,
    };
    let size = [400, 300];
    let target = cell.kind.target();
    let board = BoardRaster::with_dpi(target.clone(), raster_dpi(&target, cell.outer_radius_px));
    let spec = spec_for(&cell, &target, size);
    let scene = expose(&radiance(&board, &spec), &spec, 1000);

    let mut config = DetectConfig::from_target_and_marker_diameter(
        target.clone(),
        (2.0 * cell.outer_radius_px) as f32,
    );
    config.circle_refinement = CircleRefinementMethod::Photometric;
    config.advanced.photometric.max_iterations = 1;
    let (res, diag) = Detector::with_config(config)
        .detect_with_diagnostics(&scene.image)
        .expect("detect");
    let reference = detector(
        &target,
        cell.outer_radius_px,
        CircleRefinementMethod::ProjectiveCenter,
    )
    .detect(&scene.image)
    .expect("detect");

    assert!(res.detected_markers.len() >= 10, "markers detected");
    assert_eq!(res.detected_markers.len(), reference.detected_markers.len());
    for ((m, d), r) in res
        .detected_markers
        .iter()
        .zip(&diag.markers)
        .zip(&reference.detected_markers)
    {
        let status = d.photometric.as_ref().map(|p| p.status);
        assert!(
            matches!(
                status,
                Some(PhotometricStatus::NotConverged | PhotometricStatus::NoBoardPosition)
            ),
            "status {status:?}"
        );
        assert_ne!(m.center_source, CenterSource::Photometric);
        assert!(m.center_cov.is_none());
        assert_eq!(m.center, r.center, "centre kept");
    }
}

/// The full matrix: kind × scale {8, 17, 33 px} × blur {0.8, 1.5, 3} ×
/// contrast {40, 80} × tilt {0, 20, 30}, Gaussian PSF, white noise.
#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn photometric_target_matrix() {
    let n_seeds = env_or("MATRIX_SEEDS", 6u64);
    let scales: Vec<f64> = std::env::var("MATRIX_SCALES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![8.0, 17.0, 33.0]);
    let mut cells = Vec::new();
    for kind in selected_kinds() {
        for &outer_radius_px in &scales {
            for blur in [0.8, 1.5, 3.0] {
                for contrast in [40.0, 80.0] {
                    for tilt in [0.0, 20.0, 30.0] {
                        cells.push(Cell {
                            kind,
                            outer_radius_px,
                            blur,
                            contrast,
                            tilt,
                            effect: Effect::None,
                        });
                    }
                }
            }
        }
    }
    run_cells(&cells, n_seeds, [720, 540]);
}

/// Marker scale: kind × outer radius {10, 20, 33, 45, 60} px (20–120 px
/// across) × blur {0.8, 1.5, 3, 5} × tilt {0, 30°}, contrast 80, on a
/// 1280×960 frame so that large markers are detected in numbers.
///
/// Wherever the model explains the scene (mean fit residual within 1.25× the
/// noise σ of 2) no fit may stop unconverged. Where it does not — on a plain
/// board without origin dots, the relative frame models only the labelled
/// neighbours, and at 20 px, blur 3 and 30° tilt the undetected ones blur
/// into the patch — fits may fail, and must say so. In the accepted regime
/// (outer radius ≥ 17 px, blur ≤ 3 px) with at least five photometric
/// centres, their RMS error stays within 0.05 px.
#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn photometric_scale_matrix() {
    let n_seeds = env_or("MATRIX_SEEDS", 2u64);
    let mut cells = Vec::new();
    for kind in selected_kinds() {
        for outer_radius_px in [10.0, 20.0, 33.0, 45.0, 60.0] {
            for blur in [0.8, 1.5, 3.0, 5.0] {
                for tilt in [0.0, 30.0] {
                    cells.push(Cell {
                        kind,
                        outer_radius_px,
                        blur,
                        contrast: 80.0,
                        tilt,
                        effect: Effect::None,
                    });
                }
            }
        }
    }
    let results = run_cells(&cells, n_seeds, [1280, 960]);
    let not_converged = format!("{:?}", PhotometricStatus::NotConverged);
    for (c, r) in cells.iter().zip(&results) {
        let cell = format!(
            "{} R {} blur {} tilt {}",
            c.kind.name(),
            c.outer_radius_px,
            c.blur,
            c.tilt
        );
        assert_eq!(r.n_silent, 0, "{cell}: every marker reports a status");
        // Scenes are rendered with noise σ 2 (`SceneSpec`).
        let model_fits = r.residual <= 1.25 * 2.0;
        if !model_fits {
            continue;
        }
        assert_eq!(
            r.statuses.get(&not_converged).copied().unwrap_or(0),
            0,
            "{cell}: fits that did not converge"
        );
        let accepted_regime = c.outer_radius_px >= 17.0 && c.blur <= 3.0;
        if accepted_regime && r.n_photometric * n_seeds as f64 >= 5.0 {
            assert!(
                r.rms_photometric <= 0.05,
                "{cell}: photometric rms {}",
                r.rms_photometric
            );
        }
    }
}

/// Model misfit: PSFs and responses the photometric model cannot represent,
/// and correlated sensor noise, at medium scale, blur 1.5, contrast 80, tilt
/// 20°, for every target kind.
#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn photometric_misfit_matrix() {
    let n_seeds = env_or("MATRIX_SEEDS", 8u64);
    let mut cells = Vec::new();
    for kind in selected_kinds() {
        for effect in [
            Effect::None,
            Effect::Anisotropic,
            Effect::Halo,
            Effect::Disc,
            Effect::Gamma,
            Effect::CorrelatedNoise,
            Effect::Combined,
        ] {
            cells.push(Cell {
                kind,
                outer_radius_px: 17.0,
                blur: 1.5,
                contrast: 80.0,
                tilt: 20.0,
                effect,
            });
        }
    }
    run_cells(&cells, n_seeds, [720, 540]);
}

/// Fitted PSF sigma against exposure and contrast at a fixed true PSF
/// (Gaussian, blur 1.5, outer radius 17 px, tilt 20°): one radiance scaled by
/// an exposure gain (up to partial saturation of the paper), and scenes
/// rendered with different paper-minus-ink contrasts. Prints the mean and
/// spread of the accepted fits' sigma.
#[test]
#[ignore = "study; run in release with --ignored --nocapture"]
fn photometric_blur_exposure() {
    let n_seeds = env_or("MATRIX_SEEDS", 4u64);
    let size = [720, 540];
    let blur = 1.5;
    for kind in selected_kinds() {
        let target = kind.target();
        let board = BoardRaster::with_dpi(target.clone(), raster_dpi(&target, 17.0));
        let det = detector(&target, 17.0, CircleRefinementMethod::Photometric);
        let row = |label: String, rad: &Radiance, spec: &SceneSpec| {
            let (mut sigmas, mut masked, mut contrast, mut n_fit) = (Vec::new(), 0.0, 0.0, 0usize);
            for seed in 0..n_seeds {
                let scene = expose(rad, spec, 1000 + seed);
                let (res, diag) = det.detect_with_diagnostics(&scene.image).expect("detect");
                for (m, p) in res
                    .detected_markers
                    .iter()
                    .zip(&diag.markers)
                    .filter_map(|(m, d)| Some((m, d.photometric.as_ref()?)))
                {
                    if p.status == PhotometricStatus::NoBoardPosition {
                        continue;
                    }
                    n_fit += 1;
                    if p.status != PhotometricStatus::Accepted {
                        continue;
                    }
                    sigmas.push(m.blur.expect("accepted fit has a blur").sigma_px);
                    masked += p.masked_fraction;
                    contrast += p.contrast.unwrap_or(f64::NAN);
                }
            }
            let n = sigmas.len().max(1) as f64;
            let mean = sigmas.iter().sum::<f64>() / n;
            let sd = (sigmas.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n).sqrt();
            println!(
                "{:<18} {label:<22} accepted {:>4}/{:<4} sigma {mean:.4} (sd {sd:.4}, {:+.4} vs psf+pixel {:.4}) fitted contrast {:6.1} masked {:.3}",
                kind.name(),
                sigmas.len(),
                n_fit,
                mean - psf_moments(spec.psf, true)[0],
                psf_moments(spec.psf, true)[0],
                contrast / n,
                masked / n,
            );
        };
        // Exposure: one radiance (paper 150, ink 70 at the image centre,
        // ±15 % illumination) scaled by a gain.
        let spec = SceneSpec::for_target(&target, 17.0, blur, 20.0, 80.0, 1);
        let spec = SceneSpec {
            image_size: size,
            ..spec
        };
        let rad = radiance(&board, &spec);
        for gain in [0.25, 0.5, 1.0, 1.5, 1.7, 2.0] {
            let scaled = Radiance {
                buf: rad.buf.iter().map(|v| gain * v).collect(),
                h: rad.h,
            };
            row(format!("gain {gain}"), &scaled, &spec);
        }
        // Contrast at a fixed ink level (paper = 70 + contrast).
        for contrast in [20.0, 40.0, 160.0] {
            let spec = SceneSpec {
                image_size: size,
                ..SceneSpec::for_target(&target, 17.0, blur, 20.0, contrast, 1)
            };
            row(
                format!("contrast {contrast}"),
                &radiance(&board, &spec),
                &spec,
            );
        }
    }
}

/// One exposure of a radiance: a gain on it, then the camera response
/// `gamma` (on the radiance over 255), noise and 8-bit clipping.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Exposure {
    gamma: f64,
    gain: f64,
}

/// The reference exposure of a series: unit gain, linear response.
const NOMINAL_EXPOSURE: Exposure = Exposure {
    gamma: 1.0,
    gain: 1.0,
};

/// What one detection of one exposure produced.
#[derive(Debug, Default)]
struct ExposureRun {
    /// The per-image print radii (`None`: nominal radii, no consensus).
    radii: Option<[f64; 4]>,
    /// Photometric centres by true cell.
    fits: BTreeMap<usize, Fit>,
    /// Projective-centre errors (px) of the same exposure, detected without
    /// the photometric stage, by true cell.
    projective: BTreeMap<usize, f64>,
    /// Markers at a wrong board position.
    n_wrong: usize,
    /// Masked (saturated) fraction of the patch of every attempted fit.
    masked: Vec<f64>,
    statuses: BTreeMap<String, usize>,
}

/// One photometric centre of an exposure.
#[derive(Debug, Clone, Copy)]
struct Fit {
    center: [f64; 2],
    /// Fitted sigma (px), where reported.
    sigma: Option<f64>,
    /// Reported centre covariance.
    cov: [[f64; 2]; 2],
    /// Fraction of the patch masked as saturated.
    masked: f64,
}

/// The noise-free scene of an exposure-series cell.
struct SeriesScene {
    spec: SceneSpec,
    rad: Radiance,
    truth: Truth,
}

/// Detect one exposure of a scene (noise seed `seed`) with the photometric
/// detector `det` and the projective-centre baseline `projective`, and score
/// it.
fn run_exposure(
    det: &Detector,
    projective: &Detector,
    scene: &SeriesScene,
    exposure: Exposure,
    seed: u64,
) -> ExposureRun {
    let scaled = Radiance {
        buf: scene.rad.buf.iter().map(|v| exposure.gain * v).collect(),
        h: scene.rad.h,
    };
    let spec = SceneSpec {
        gamma: exposure.gamma,
        ..scene.spec.clone()
    };
    let image = expose(&scaled, &spec, seed).image;
    let (res, diag) = det.detect_with_diagnostics(&image).expect("detect");
    let mut run = ExposureRun {
        radii: diag.photometric_print_radii_mm,
        ..ExposureRun::default()
    };
    for (m, d) in res.detected_markers.iter().zip(&diag.markers) {
        let status = d
            .photometric
            .as_ref()
            .map_or_else(|| "missing".into(), |p| format!("{:?}", p.status));
        *run.statuses.entry(status).or_default() += 1;
        if let Some(p) = d.photometric.as_ref()
            && p.status != PhotometricStatus::NoBoardPosition
        {
            run.masked.push(p.masked_fraction);
        }
        let Ok(idx) = scene.truth.cell_of(m, &scene.rad.h) else {
            run.n_wrong += 1;
            continue;
        };
        if m.center_source == CenterSource::Photometric
            && let (Some(cov), Some(p)) = (m.center_cov, d.photometric.as_ref())
        {
            let fit = Fit {
                center: m.center,
                sigma: m.blur.as_ref().map(|b| b.sigma_px),
                cov,
                masked: p.masked_fraction,
            };
            run.fits.insert(idx, fit);
        }
    }
    let res_p = projective.detect(&image).expect("detect");
    run.projective = res_p
        .detected_markers
        .iter()
        .filter_map(|m| {
            let i = scene.truth.cell_of(m, &scene.rad.h).ok()?;
            let t = scene.truth.centers[i];
            Some((i, (m.center[0] - t[0]).hypot(m.center[1] - t[1])))
        })
        .collect();
    run
}

/// Whether `e` lies inside the 95 % ellipse of the covariance `cov`.
fn within_95(e: [f64; 2], cov: [[f64; 2]; 2]) -> bool {
    let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    let d2 =
        (cov[1][1] * e[0] * e[0] - 2.0 * cov[0][1] * e[0] * e[1] + cov[0][0] * e[1] * e[1]) / det;
    // χ²(2 dof) 95 % quantile.
    d2 <= 5.991
}

/// Mean and standard error of the mean.
fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// `mean±se`, `mean` alone for one value, `-` for none.
fn fmt_mean_se(v: &[f64]) -> String {
    match v.len() {
        0 => format!("{:>15}", "-"),
        1 => format!("{:+7.4}        ", v[0]),
        _ => {
            let (m, se) = mean_se(v);
            format!("{m:+7.4}±{se:.4}")
        }
    }
}

/// Ground truth an exposure series is scored against.
struct SeriesTruth<'a> {
    truth: &'a Truth,
    /// Printed radii, board mm.
    radii: [f64; 4],
    /// Radii the photometric stage estimates (from the hole edge).
    n_free: usize,
    px_per_mm: f64,
    /// PSF sigma plus pixel box, px.
    sigma: f64,
    /// Per cell, the image-frame dark-sector dipole of its code band (coded
    /// targets; empty otherwise).
    code_dipole: Vec<[f64; 2]>,
}

/// Per cell, the image-frame dipole of its code band's dark sectors: the
/// mean over 16 sector centres of ink darkness times the image direction of
/// the sector from the marker centre. Empty for plain targets.
fn code_dipoles(board: &BoardRaster, h: &[[f64; 3]; 3]) -> Vec<[f64; 2]> {
    let Some((lo, hi)) = board.target.code_band_bounds_mm() else {
        return Vec::new();
    };
    let rho = 0.5 * f64::from(lo + hi);
    board
        .target
        .cells()
        .iter()
        .map(|cell| {
            let c = [f64::from(cell.xy_mm[0]), f64::from(cell.xy_mm[1])];
            let ci = apply_h(h, c);
            let mut d = [0.0; 2];
            for k in 0..16 {
                let a = (k as f64 + 0.5) * std::f64::consts::PI / 8.0;
                let p = [c[0] + rho * a.cos(), c[1] + rho * a.sin()];
                let ink = 1.0 - board.reflectance(p);
                let q = apply_h(h, p);
                let (dx, dy) = (q[0] - ci[0], q[1] - ci[1]);
                let n = dx.hypot(dy);
                d[0] += ink * dx / n / 16.0;
                d[1] += ink * dy / n / 16.0;
            }
            d
        })
        .collect()
}

/// Print one exposure of a series against the series' nominal exposure
/// (same noise seeds, same markers).
fn print_exposure(
    exposure: Exposure,
    runs: &[ExposureRun],
    nominal: &[ExposureRun],
    t: &SeriesTruth<'_>,
) {
    let n_seeds = runs.len() as f64;
    // Print radii: error against the print (px at the image centre), and
    // change against the nominal exposure of the same seed.
    let dr = |r: &ExposureRun, k: usize| r.radii.map(|r| (r[k] - t.radii[k]) * t.px_per_mm);
    let radii: Vec<String> = (0..t.n_free)
        .map(|k| {
            let err: Vec<f64> = runs.iter().filter_map(|r| dr(r, k)).collect();
            let change: Vec<f64> = runs
                .iter()
                .zip(nominal)
                .filter_map(|(r, n)| Some(dr(r, k)? - dr(n, k)?))
                .collect();
            format!("r{k} {} Δ {}", fmt_mean_se(&err), fmt_mean_se(&change))
        })
        .collect();
    let n_consensus = runs.iter().filter(|r| r.radii.is_some()).count();
    // Centres against the truth, and the reported covariance's coverage.
    let errors: Vec<([f64; 2], [[f64; 2]; 2])> = runs
        .iter()
        .flat_map(|r| {
            r.fits.iter().map(|(&i, f)| {
                let p = t.truth.centers[i];
                ([f.center[0] - p[0], f.center[1] - p[1]], f.cov)
            })
        })
        .collect();
    let n_err = errors.len().max(1) as f64;
    let rms_err = rms(&errors
        .iter()
        .map(|(e, _)| e[0].hypot(e[1]))
        .collect::<Vec<_>>());
    let signed = [0, 1].map(|a| errors.iter().map(|(e, _)| e[a]).sum::<f64>() / n_err);
    let reported = (errors.iter().map(|(_, c)| c[0][0] + c[1][1]).sum::<f64>() / n_err).sqrt();
    let covered = errors.iter().filter(|(e, c)| within_95(*e, *c)).count() as f64 / n_err;
    let projective: Vec<f64> = runs
        .iter()
        .flat_map(|r| r.projective.values().copied())
        .collect();
    let sigmas: Vec<f64> = runs
        .iter()
        .flat_map(|r| r.fits.values().filter_map(|f| f.sigma))
        .collect();
    let sigma_bias =
        (!sigmas.is_empty()).then(|| sigmas.iter().sum::<f64>() / sigmas.len() as f64 - t.sigma);
    // Paired against the nominal exposure: per seed, the mean change over the
    // markers fitted in both; then mean ± SE over seeds.
    let (mut dx, mut dy, mut ds, mut d_all, mut n_common) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0usize);
    for (r, n) in runs.iter().zip(nominal) {
        let common: Vec<([f64; 2], Option<f64>)> = r
            .fits
            .iter()
            .filter_map(|(i, f)| {
                let g = n.fits.get(i)?;
                let d = [f.center[0] - g.center[0], f.center[1] - g.center[1]];
                Some((d, f.sigma.zip(g.sigma).map(|(a, b)| a - b)))
            })
            .collect();
        if common.is_empty() {
            continue;
        }
        n_common += common.len();
        let k = common.len() as f64;
        dx.push(common.iter().map(|(d, _)| d[0]).sum::<f64>() / k);
        dy.push(common.iter().map(|(d, _)| d[1]).sum::<f64>() / k);
        let s: Vec<f64> = common.iter().filter_map(|(_, s)| *s).collect();
        if !s.is_empty() {
            ds.push(s.iter().sum::<f64>() / s.len() as f64);
        }
        d_all.extend(common.iter().map(|(d, _)| d[0].hypot(d[1])));
    }
    // Paired centre change against the code dipole: one slope for both axes.
    let dipole_fit = (!t.code_dipole.is_empty()).then(|| {
        let pairs: Vec<([f64; 2], [f64; 2])> = runs
            .iter()
            .zip(nominal)
            .flat_map(|(r, n)| {
                r.fits.iter().filter_map(|(i, f)| {
                    let g = n.fits.get(i)?;
                    Some((
                        [f.center[0] - g.center[0], f.center[1] - g.center[1]],
                        t.code_dipole[*i],
                    ))
                })
            })
            .collect();
        let m = pairs.len().max(1) as f64;
        let md = [0, 1].map(|a| pairs.iter().map(|p| p.0[a]).sum::<f64>() / m);
        let mq = [0, 1].map(|a| pairs.iter().map(|p| p.1[a]).sum::<f64>() / m);
        let (mut sdq, mut sqq, mut sdd) = (0.0, 0.0, 0.0);
        for (d, q) in &pairs {
            for a in 0..2 {
                sdq += (d[a] - md[a]) * (q[a] - mq[a]);
                sqq += (q[a] - mq[a]).powi(2);
                sdd += (d[a] - md[a]).powi(2);
            }
        }
        let k = sdq / sqq;
        (k, sdq / (sqq * sdd).sqrt(), [k * mq[0], k * mq[1]])
    });
    if let Some((k, r, explained)) = dipole_fit.filter(|f| f.1.is_finite()) {
        println!(
            "      code dipole: slope {k:+.4} px corr {r:+.2} mean dipole explains Δ {:+.4} {:+.4}",
            explained[0], explained[1]
        );
    }
    let masked: Vec<f64> = runs.iter().flat_map(|r| r.masked.iter().copied()).collect();
    let n_phot = runs.iter().map(|r| r.fits.len()).sum::<usize>() as f64 / n_seeds;
    let n_wrong: usize = runs.iter().map(|r| r.n_wrong).sum();
    println!(
        "  γ {:<4} gain {:<4} cons {}/{} {} | σ {} Δ {} | phot {:5.1} wrong {} masked {:.3} | rms {:.4} mean {:+.4} {:+.4} rep {:.4} cov95 {:.2} | proj {:5.1} rms {:.4} | vs nominal n {:5.1} Δx {} Δy {} rmsΔ {:.4}",
        exposure.gamma,
        exposure.gain,
        n_consensus,
        runs.len(),
        radii.join(" "),
        sigma_bias.map_or("    -  ".into(), |b| format!("{b:+.4}")),
        fmt_mean_se(&ds),
        n_phot,
        n_wrong,
        masked.iter().sum::<f64>() / masked.len().max(1) as f64,
        rms_err,
        signed[0],
        signed[1],
        reported,
        covered,
        projective.len() as f64 / n_seeds,
        rms(&projective),
        n_common as f64 / n_seeds,
        fmt_mean_se(&dx),
        fmt_mean_se(&dy),
        rms(&d_all),
    );
    if std::env::var("MATRIX_STATUSES").is_ok() {
        let mut statuses: BTreeMap<&str, usize> = BTreeMap::new();
        for r in runs {
            for (s, n) in &r.statuses {
                *statuses.entry(s).or_default() += n;
            }
        }
        println!("      statuses {statuses:?}");
    }
}

/// A comma-separated list of numbers from `key`, else `default`.
fn env_list(key: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(key).map_or_else(
        |_| default.to_vec(),
        |s| {
            s.split(',')
                .map(|v| v.trim().parse().expect("a number"))
                .collect()
        },
    )
}

/// Exposure and saturation (ALGO-022): do the per-image print radii, the
/// fitted sigma and the photometric centres follow exposure?
///
/// Per cell (coded hex and plain hex with dots, `MATRIX_KINDS` overrides;
/// outer radius 17 px, contrast 80, blur `MATRIX_BLURS` {1, 2} px, tilt
/// `MATRIX_TILTS` {0, 30}°) ONE noise-free radiance (paper 150, ink 70 grey
/// levels at the image centre, illumination gradient `MATRIX_ILLUM_GRADIENT`
/// 0.3, i.e. ±15 %) is exposed at the gains `MATRIX_EXPOSURES` (default
/// 0.25 … 3: from 20 grey levels of contrast to paper clipped at 255 with
/// only the darkest ink unsaturated) under the camera responses
/// `MATRIX_GAMMAS` (default 1, 0.8, 1.25), with `MATRIX_SEEDS` noise seeds
/// (default 6; σ 2 grey levels, the same field at every exposure of a seed).
/// Per exposure it prints: the print radii against the print (px at the
/// image centre) and their change against the nominal exposure (gain 1,
/// linear) of the same seed; the fitted sigma against the PSF plus pixel box
/// and its paired change; the photometric count, wrong labels and mean
/// masked (saturated) fraction; the centre RMS, mean signed error, reported
/// σ and 95 %-ellipse coverage against the truth; the projective-centre
/// count and RMS of the same image; and the paired centre change of the
/// markers fitted in both exposures (per-seed means, ± SE over seeds). On
/// coded targets it also regresses the paired change on each marker's
/// code-band dark-sector dipole. Last, per kind, the photometric centres of
/// all exposures binned by masked fraction, against the projective centre
/// of the same marker. `MATRIX_FIXED_RADII` keeps the nominal radii.
///
/// ```text
/// cargo test --release -p ringgrid --test photometric_target_matrix \
///     photometric_exposure_series -- --ignored --nocapture
/// ```
#[test]
#[ignore = "study; run in release with --ignored --nocapture"]
fn photometric_exposure_series() {
    let n_seeds = env_or("MATRIX_SEEDS", 6u64);
    let gains = env_list("MATRIX_EXPOSURES", &[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
    let gammas = env_list("MATRIX_GAMMAS", &[1.0, 0.8, 1.25]);
    let mut exposures = vec![NOMINAL_EXPOSURE];
    for &gamma in &gammas {
        for &gain in &gains {
            let e = Exposure { gamma, gain };
            if e != NOMINAL_EXPOSURE {
                exposures.push(e);
            }
        }
    }
    let kinds = match std::env::var("MATRIX_KINDS") {
        Ok(_) => selected_kinds(),
        Err(_) => vec![Kind::CodedHex, Kind::PlainHexDots],
    };
    let (size, outer_radius_px) = ([720, 540], 17.0);
    let (blurs, tilts) = (
        env_list("MATRIX_BLURS", &[1.0, 2.0]),
        env_list("MATRIX_TILTS", &[0.0, 30.0]),
    );
    let illum_gradient: f64 = env_or("MATRIX_ILLUM_GRADIENT", 0.3);
    let mut cells = Vec::new();
    for &kind in &kinds {
        for &blur in &blurs {
            for &tilt in &tilts {
                cells.push(Cell {
                    kind,
                    outer_radius_px,
                    blur,
                    contrast: 80.0,
                    tilt,
                    effect: Effect::None,
                });
            }
        }
    }
    let rasters: Vec<BoardRaster> = kinds
        .iter()
        .map(|k| {
            let target = k.target();
            let dpi = raster_dpi(&target, outer_radius_px);
            BoardRaster::with_dpi(target, dpi)
        })
        .collect();
    let board_of = |c: &Cell| &rasters[kinds.iter().position(|k| *k == c.kind).unwrap()];
    let scenes = parallel_map(&cells, |c| {
        let board = board_of(c);
        let spec = SceneSpec {
            illum_gradient,
            ..spec_for(c, &board.target, size)
        };
        let rad = radiance(board, &spec);
        let truth = Truth::new(&board.target, &rad.h, size, spec.px_per_mm);
        SeriesScene { spec, rad, truth }
    });
    let jobs: Vec<(usize, Exposure)> = (0..cells.len())
        .flat_map(|ci| exposures.iter().map(move |&e| (ci, e)))
        .collect();
    let runs = parallel_map(&jobs, |&(ci, exposure)| {
        let target = &board_of(&cells[ci]).target;
        let det = detector(target, outer_radius_px, CircleRefinementMethod::Photometric);
        let projective = detector(
            target,
            outer_radius_px,
            CircleRefinementMethod::ProjectiveCenter,
        );
        (0..n_seeds)
            .map(|seed| run_exposure(&det, &projective, &scenes[ci], exposure, 1000 + seed))
            .collect::<Vec<_>>()
    });
    for ((cell, scene), of_cell) in cells.iter().zip(&scenes).zip(runs.chunks(exposures.len())) {
        let target = &board_of(cell).target;
        println!(
            "{} R {} blur {} tilt {} ({} visible)",
            cell.kind.name(),
            cell.outer_radius_px,
            cell.blur,
            cell.tilt,
            scene.truth.visible
        );
        let truth = SeriesTruth {
            truth: &scene.truth,
            radii: true_print_radii(target),
            n_free: if target.is_coded() { 3 } else { 1 },
            px_per_mm: scene.spec.px_per_mm,
            sigma: psf_moments(scene.spec.psf, true)[0],
            code_dipole: code_dipoles(board_of(cell), &scene.rad.h),
        };
        for (exposure, r) in exposures.iter().zip(of_cell) {
            print_exposure(*exposure, r, &of_cell[0], &truth);
        }
    }
    // Photometric centres by masked fraction, over every cell and exposure
    // of a kind.
    for &kind in &kinds {
        println!("{} photometric centres by masked fraction:", kind.name());
        // (error vector, fit, projective error of the same marker and exposure).
        let fits: Vec<([f64; 2], &Fit, Option<f64>)> = cells
            .iter()
            .zip(&scenes)
            .zip(runs.chunks(exposures.len()))
            .filter(|((c, _), _)| c.kind == kind)
            .flat_map(|((_, scene), of_cell)| {
                of_cell.iter().flatten().flat_map(|r| {
                    r.fits.iter().map(|(&i, f)| {
                        let p = scene.truth.centers[i];
                        let e = [f.center[0] - p[0], f.center[1] - p[1]];
                        (e, f, r.projective.get(&i).copied())
                    })
                })
            })
            .collect();
        let bins: [(&str, f64, f64); 7] = [
            ("none", 0.0, 1e-12),
            ("(0, 0.1)", 1e-12, 0.1),
            ("[0.1, 0.3)", 0.1, 0.3),
            ("[0.3, 0.5)", 0.3, 0.5),
            ("[0.5, 0.7)", 0.5, 0.7),
            ("[0.7, 0.8)", 0.7, 0.8),
            ("[0.8, 1]", 0.8, 1.0 + 1e-12),
        ];
        for (label, lo, hi) in bins {
            let bin: Vec<&([f64; 2], &Fit, Option<f64>)> = fits
                .iter()
                .filter(|(_, f, _)| f.masked >= lo && f.masked < hi)
                .collect();
            if bin.is_empty() {
                continue;
            }
            let n = bin.len() as f64;
            let errs: Vec<f64> = bin.iter().map(|(e, _, _)| e[0].hypot(e[1])).collect();
            let mut sorted = errs.clone();
            sorted.sort_by(f64::total_cmp);
            let reported = (bin
                .iter()
                .map(|(_, f, _)| f.cov[0][0] + f.cov[1][1])
                .sum::<f64>()
                / n)
                .sqrt();
            let covered = bin.iter().filter(|(e, f, _)| within_95(*e, f.cov)).count() as f64 / n;
            // Against the projective centre of the same marker.
            let paired: Vec<(f64, f64)> = bin
                .iter()
                .zip(&errs)
                .filter_map(|((_, _, p), e)| Some((*e, (*p)?)))
                .collect();
            let worse = paired.iter().filter(|(e, p)| e > p).count();
            println!(
                "  masked {label:<10}: n {:6} rms {:.4} p95 {:.4} max {:.4} >0.5px {:4} reported {:.4} cov95 {:.2} | same markers projective rms {:.4}, photometric worse {:.2}",
                bin.len(),
                rms(&errs),
                sorted[((0.95 * n) as usize).min(bin.len() - 1)],
                sorted[bin.len() - 1],
                errs.iter().filter(|e| **e > 0.5).count(),
                reported,
                covered,
                rms(&paired.iter().map(|p| p.1).collect::<Vec<_>>()),
                worse as f64 / paired.len().max(1) as f64,
            );
        }
    }
}
