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
//!   (`white`, recovered from `covariance_inflation`);
//! - reported σ over the error against the truth (which also contains bias,
//!   which no covariance covers), and the bias itself;
//! - the result homography's RMS error at the true cell centres for both
//!   methods (first exposure; absolute board frame only).
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
//! `MATRIX_THREADS` sets the parallelism.

mod support;

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ringgrid::diagnostics::PhotometricStatus;
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
        let t = match self {
            Kind::CodedHex => TargetLayout::coded_hex(8.0, 15, 14, 4.9565, 2.966, 0.839),
            Kind::CodedRect => TargetLayout::coded_rect(14.0, 12, 12, 5.6, 3.2, 1.0),
            Kind::PlainHexDots => TargetLayout::plain_hex(8.0, 13, 12, 4.8, 2.4, OriginDots::Auto),
            Kind::PlainHexNoDots => {
                TargetLayout::plain_hex(8.0, 13, 12, 4.8, 2.4, OriginDots::None)
            }
            Kind::PlainRectDots => {
                TargetLayout::plain_rect(14.0, 12, 12, 5.6, 2.8, OriginDots::Auto)
            }
            Kind::PlainRectNoDots => {
                TargetLayout::plain_rect(14.0, 12, 12, 5.6, 2.8, OriginDots::None)
            }
            Kind::PlainRectThin => {
                TargetLayout::plain_rect(14.0, 12, 12, 5.6, 4.6, OriginDots::Auto)
            }
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
    traces: Vec<(f64, f64)>,
    truth: [f64; 2],
}

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
    /// Mean covariance inflation over photometric centres.
    inflation: f64,
    /// Mean fit residual standard deviation over photometric centres.
    residual: f64,
    statuses: BTreeMap<String, usize>,
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
    let photometric = detector(
        target,
        cell.outer_radius_px,
        CircleRefinementMethod::Photometric,
    );
    let projective = detector(
        target,
        cell.outer_radius_px,
        CircleRefinementMethod::ProjectiveCenter,
    );
    let mut out = CellResult {
        n_visible: truth.visible,
        ..CellResult::default()
    };
    let mut samples: BTreeMap<usize, MarkerSamples> = BTreeMap::new();
    let mut errors = Vec::new();
    let mut inflations = Vec::new();
    for seed in 0..n_seeds {
        let scene = expose(&rad, &spec, 1000 + seed);
        let (res, diag) = photometric
            .detect_with_diagnostics(&scene.image)
            .expect("detect");
        if seed == 0 {
            let res_p = projective.detect(&scene.image).expect("detect");
            out.rms_projective = projective_rms(&res_p, &truth, &rad.h);
            out.h_rms_projective = homography_rms(&res_p, target, &truth, size);
            out.h_rms_photometric = homography_rms(&res, target, &truth, size);
        }
        out.n_detected += res.detected_markers.len() as f64;
        for (m, d) in res.detected_markers.iter().zip(&diag.markers) {
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
            let inflation = p.covariance_inflation.unwrap_or(1.0);
            let t = truth.centers[idx];
            errors.push((m.center[0] - t[0]).hypot(m.center[1] - t[1]));
            inflations.push(inflation);
            out.residual += p.residual_sigma.unwrap_or(0.0);
            out.n_photometric += 1.0;
            let s = samples.entry(idx).or_default();
            s.truth = t;
            s.centers.push(m.center);
            s.traces.push((cov[0][0] + cov[1][1], inflation));
        }
    }
    out.n_detected /= n_seeds as f64;
    out.n_located /= n_seeds as f64;
    out.n_photometric /= n_seeds as f64;
    out.rms_photometric = rms(&errors);
    out.inflation = inflations.iter().sum::<f64>() / inflations.len().max(1) as f64;
    out.residual /= inflations.len().max(1) as f64;
    summarize_monte_carlo(&samples, n_seeds, &mut out);
    out
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
            rep_w += tr / (infl * infl) / n as f64;
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
    let results: Mutex<Vec<Option<CellResult>>> = Mutex::new(vec![None; cells.len()]);
    let next = AtomicUsize::new(0);
    let threads = env_or("MATRIX_THREADS", 8usize).max(1);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(c) = cells.get(i) else { break };
                    let k = KINDS.iter().position(|k| *k == c.kind).unwrap();
                    let board = &rasters[&(k, c.outer_radius_px.to_bits())];
                    let r = run_cell(board, c, n_seeds, size);
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    let results: Vec<CellResult> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("cell ran"))
        .collect();
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
