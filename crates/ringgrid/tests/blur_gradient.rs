//! Does a blur gradient across a marker bias its estimated centre?
//!
//! Scenes carry an isotropic Gaussian PSF whose sigma varies linearly along
//! the image y axis, `sigma(y) = sigma0 + g (y - y0)`, rendered in two forms
//! (see `BlurForm`): *gather* (every output pixel uses the kernel of its own
//! position) and *scatter* (every scene point spreads with the kernel of its
//! own position — light-conserving, the physical form of depth-dependent
//! defocus). To first order in `g` (heat-kernel expansion, a hypothesis the
//! default tests check on the renderer):
//!
//! - scatter: the ink centroid is preserved exactly, while every edge is
//!   blurred by its local sigma and displaced by `-g_n sigma` (`g_n` the
//!   sigma gradient along the edge normal): the whole pattern moves by
//!   `-g sigma0` towards lower blur, a pure translation.
//! - gather: edges keep their 50 % point but the profile is skewed; the
//!   centroid moves by `+2 g sigma0`.
//!
//! The ignored studies measure what the detector makes of it:
//!
//! ```text
//! cargo test --release -p ringgrid --features study --test blur_gradient -- --ignored --nocapture
//! ```
//!
//! - `blur_gradient_bias_matrix`: the centre bias per method, target kind,
//!   marker radius, sigma0, gradient, tilt and form, and a fitted law.
//!   `GRADIENT_SEEDS` (default 3); `GRADIENT_KINDS` (names), `GRADIENT_RADII`,
//!   `GRADIENT_SIGMAS`, `GRADIENT_TILTS`, `GRADIENT_GS` (comma-separated
//!   lists) restrict it; `GRADIENT_THREADS`; `GRADIENT_CSV=<path>` writes the
//!   rows.
//! - `blur_gradient_residual_dipole` (`study` feature): the stacked
//!   photometric residual near the outer edge by image angle, to compare with
//!   the dipole seen on real captures.
//! - `blur_gradient_ink_centroid`: the centroid of an isolated blurred marker
//!   (no detector).
//!
//! The bias of a cell is the odd part in `g`: per marker, `(e(+g) - e(-g)) /
//! 2` with the same noise seeds in both renders, averaged over seeds and over
//! the markers whose blurred footprint lies inside the band where the sigma
//! field is linear (outside it the field is clamped). This removes every
//! `g`-independent error (the projective centre's code-band bias, the noise
//! common to both renders). The `+g` minus `g = 0` difference is reported too.

mod support;

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ringgrid::{
    CenterSource, CircleRefinementMethod, DetectConfig, Detector, OriginDots, TargetLayout,
};
use support::blur_scene::{
    BlurForm, BoardRaster, Psf, Radiance, SceneSpec, apply_h, apply_psf, expose, radiance,
    varying_sigma,
};
#[cfg(feature = "study")]
use support::residual_profile::{MarkerData, analyze, dipole};

/// Smallest sigma the linear field may reach inside the band of interest.
const SIGMA_FLOOR: f64 = 0.25;
/// Paper minus ink, grey levels (the real captures show about 60).
const CONTRAST: f64 = 60.0;
const GRADIENTS: [f64; 4] = [0.005, 0.01, 0.02, 0.04];
const FORMS: [BlurForm; 2] = [BlurForm::Gather, BlurForm::Scatter];

// ---------------------------------------------------------------------------
// Renderer sanity (default run, milliseconds)
// ---------------------------------------------------------------------------

fn field_psf(sigma0: f64, g: f64, y0: f64, form: BlurForm) -> Psf {
    Psf::VaryingGaussian {
        sigma0,
        gradient: g,
        y0,
        sigma_range: [SIGMA_FLOOR, 50.0],
        form,
    }
}

/// Centroid of a non-negative `w × h` field.
fn centroid(buf: &[f64], w: usize) -> [f64; 2] {
    let (mut m, mut mx, mut my) = (0.0, 0.0, 0.0);
    for (i, v) in buf.iter().enumerate() {
        m += v;
        mx += v * (i % w) as f64;
        my += v * (i / w) as f64;
    }
    [mx / m, my / m]
}

/// A supersampled ink disc of `radius` px at `c`.
fn disc(w: usize, h: usize, c: [f64; 2], radius: f64) -> Vec<f64> {
    let mut buf = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for s in 0..64 {
                let px = x as f64 + ((s % 8) as f64 + 0.5) / 8.0 - 0.5;
                let py = y as f64 + ((s / 8) as f64 + 0.5) / 8.0 - 0.5;
                if (px - c[0]).hypot(py - c[1]) <= radius {
                    acc += 1.0;
                }
            }
            buf[y * w + x] = acc / 64.0;
        }
    }
    buf
}

/// Scatter preserves the centroid of a blurred ink disc; gather moves it by
/// `2 g sigma0` towards the larger blur.
#[test]
fn varying_blur_centroid_follows_the_form() {
    let (w, h) = (101, 101);
    let c = [50.0, 50.0];
    let ink = disc(w, h, c, 15.0);
    let (sigma0, g) = (2.0, 0.02);
    for form in FORMS {
        let mut buf = ink.clone();
        apply_psf(&mut buf, w, h, field_psf(sigma0, g, c[1], form));
        let m = centroid(&buf, w);
        let expected = match form {
            BlurForm::Scatter => 0.0,
            BlurForm::Gather => 2.0 * g * sigma0,
        };
        println!(
            "{form:?}: centroid shift {:.5} {:.5}",
            m[0] - c[0],
            m[1] - c[1]
        );
        assert!((m[0] - c[0]).abs() < 1e-6, "{form:?} x {}", m[0]);
        assert!(
            (m[1] - c[1] - expected).abs() < 0.1 * 2.0 * g * sigma0,
            "{form:?}: y shift {} vs {expected}",
            m[1] - c[1]
        );
    }
}

/// Scatter moves a straight edge's 50 % point by `-g sigma` (towards the
/// smaller blur) whichever side the ink is on; gather keeps it.
#[test]
fn scatter_moves_edges_towards_the_smaller_blur() {
    let (w, h) = (8, 121);
    let edge_y = 60.0;
    let (sigma0, g) = (2.5, 0.02);
    for ink_below in [true, false] {
        let mut ink = vec![0.0; w * h];
        for y in 0..h {
            let below = y as f64 > edge_y;
            let v = if below == ink_below { 1.0 } else { 0.0 };
            // The pixel at the edge row is half covered.
            let v = if (y as f64 - edge_y).abs() < 1e-9 {
                0.5
            } else {
                v
            };
            ink[y * w..(y + 1) * w].iter_mut().for_each(|p| *p = v);
        }
        for form in FORMS {
            let mut buf = ink.clone();
            apply_psf(&mut buf, w, h, field_psf(sigma0, g, edge_y, form));
            let col: Vec<f64> = (0..h).map(|y| buf[y * w + 3]).collect();
            // Linear interpolation of the 0.5 crossing.
            let y50 = (0..h - 1)
                .find_map(|y| {
                    let (a, b) = (col[y] - 0.5, col[y + 1] - 0.5);
                    (a * b <= 0.0 && a != b).then(|| y as f64 + a / (a - b))
                })
                .expect("edge crossing");
            let expected = match form {
                BlurForm::Scatter => -g * sigma0,
                BlurForm::Gather => 0.0,
            };
            println!(
                "ink below {ink_below}, {form:?}: edge shift {:.4}",
                y50 - edge_y
            );
            assert!(
                (y50 - edge_y - expected).abs() < 0.15 * g * sigma0,
                "{form:?} ink below {ink_below}: shift {} vs {expected}",
                y50 - edge_y
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shared study plumbing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coded hex, wide code band.
    CodedHex,
    /// Plain rect with origin dots.
    PlainRect,
    /// Plain hex with origin dots.
    PlainHex,
}

const KINDS: [Kind; 3] = [Kind::CodedHex, Kind::PlainRect, Kind::PlainHex];

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::CodedHex => "coded_hex",
            Kind::PlainRect => "plain_rect",
            Kind::PlainHex => "plain_hex",
        }
    }

    fn target(self) -> TargetLayout {
        match self {
            Kind::CodedHex => TargetLayout::coded_hex(8.0, 15, 14, 4.9565, 2.966, 0.839),
            Kind::PlainRect => TargetLayout::plain_rect(14.0, 12, 12, 5.6, 2.8, OriginDots::Auto),
            Kind::PlainHex => TargetLayout::plain_hex(8.0, 13, 12, 4.8, 2.4, OriginDots::Auto),
        }
        .expect("valid target")
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_list(key: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(key)
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn selected_kinds() -> Vec<Kind> {
    match std::env::var("GRADIENT_KINDS") {
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
    let px_per_mm = outer_radius_px / f64::from(target.outer_draw_radius_mm());
    (7.0 * 25.4 * px_per_mm).clamp(600.0, 1200.0) as f32
}

/// Frame size for a marker scale: enough markers for the board homography
/// without spending time on the smallest scale.
fn image_size(outer_radius_px: f64) -> [u32; 2] {
    if outer_radius_px <= 10.0 {
        [320, 240]
    } else if outer_radius_px <= 20.0 {
        [480, 360]
    } else {
        [720, 540]
    }
}

/// Half-width (px, along y) of the band around `y0` where the sigma field
/// stays linear and at least [`SIGMA_FLOOR`].
fn band_half_width(sigma0: f64, g: f64, height: f64) -> f64 {
    if g == 0.0 {
        f64::INFINITY
    } else {
        ((sigma0 - SIGMA_FLOOR) / g.abs()).min(height)
    }
}

/// The PSF of a cell: uniform at `g = 0`, otherwise linear within the band
/// and clamped outside it.
fn cell_psf(sigma0: f64, g: f64, y0: f64, half_width: f64, form: BlurForm) -> Psf {
    if g == 0.0 {
        return Psf::Gaussian { sigma: sigma0 };
    }
    let span = g.abs() * half_width;
    Psf::VaryingGaussian {
        sigma0,
        gradient: g,
        y0,
        sigma_range: [sigma0 - span, sigma0 + span],
        form,
    }
}

/// Projected cell centres and outer-ring footprints (`[xmin, xmax, ymin,
/// ymax]`).
struct Truth {
    centers: Vec<[f64; 2]>,
    footprints: Vec<[f64; 4]>,
    spacing_px: f64,
}

impl Truth {
    fn new(target: &TargetLayout, h: &[[f64; 3]; 3], px_per_mm: f64) -> Self {
        let r_mm = f64::from(target.outer_draw_radius_mm());
        let mut centers = Vec::new();
        let mut footprints = Vec::new();
        for c in target.cells() {
            let xy = [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])];
            let mut fp = [f64::MAX, f64::MIN, f64::MAX, f64::MIN];
            for k in 0..32 {
                let a = k as f64 * std::f64::consts::PI / 16.0;
                let p = apply_h(h, [xy[0] + r_mm * a.cos(), xy[1] + r_mm * a.sin()]);
                fp = [
                    fp[0].min(p[0]),
                    fp[1].max(p[0]),
                    fp[2].min(p[1]),
                    fp[3].max(p[1]),
                ];
            }
            centers.push(apply_h(h, xy));
            footprints.push(fp);
        }
        Self {
            centers,
            footprints,
            spacing_px: f64::from(target.min_center_spacing_mm()) * px_per_mm,
        }
    }

    /// Cells whose footprint, widened by `margin`, lies inside the image and
    /// inside the rows `[y_lo, y_hi]`.
    fn scored(&self, size: [u32; 2], margin: f64, y_lo: f64, y_hi: f64) -> Vec<usize> {
        let (w, h) = (f64::from(size[0]) - 1.0, f64::from(size[1]) - 1.0);
        (0..self.centers.len())
            .filter(|&i| {
                let f = self.footprints[i];
                f[0] - margin >= 0.0
                    && f[1] + margin <= w
                    && f[2] - margin >= y_lo.max(0.0)
                    && f[3] + margin <= y_hi.min(h)
            })
            .collect()
    }

    /// The true cell of a detected marker with a board position, if it is
    /// labelled right.
    fn cell_of(&self, board_xy: [f64; 2], center: [f64; 2], h: &[[f64; 3]; 3]) -> Option<usize> {
        let p = apply_h(h, board_xy);
        let (idx, _) = self.centers.iter().enumerate().min_by(|a, b| {
            let da = (a.1[0] - p[0]).hypot(a.1[1] - p[1]);
            let db = (b.1[0] - p[0]).hypot(b.1[1] - p[1]);
            da.total_cmp(&db)
        })?;
        let t = self.centers[idx];
        ((center[0] - t[0]).hypot(center[1] - t[1]) <= 0.4 * self.spacing_px).then_some(idx)
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

// ---------------------------------------------------------------------------
// Q1: centre bias matrix
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Projective,
    Photometric,
}

const METHODS: [Method; 2] = [Method::Projective, Method::Photometric];

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Projective => "projective",
            Method::Photometric => "photometric",
        }
    }
    fn refinement(self) -> CircleRefinementMethod {
        match self {
            Method::Projective => CircleRefinementMethod::ProjectiveCenter,
            Method::Photometric => CircleRefinementMethod::Photometric,
        }
    }
    fn source(self) -> CenterSource {
        match self {
            Method::Projective => CenterSource::ProjectiveCenter,
            Method::Photometric => CenterSource::Photometric,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Group {
    kind: Kind,
    radius: f64,
    sigma0: f64,
    tilt: f64,
}

/// Bias of one method in one cell, px; `[x, y]` = across, along the gradient.
#[derive(Debug, Clone, Copy)]
struct Bias {
    n_markers: usize,
    odd: [f64; 2],
    odd_se: [f64; 2],
    /// `e(+g) - e(0)` along y.
    direct_y: f64,
    direct_se: f64,
}

#[derive(Debug, Clone)]
struct Row {
    group: Group,
    form: BlurForm,
    g: f64,
    /// Scored cells; 0 when the band holds no full marker.
    n_scored: usize,
    /// Scored cells each method placed at `g = 0` (first seed).
    n_zero: [usize; 2],
    bias: [Option<Bias>; 2],
}

/// Per-seed centre errors of the markers a method placed itself.
type Errors = Vec<BTreeMap<usize, [f64; 2]>>;

fn measure(
    det: &Detector,
    method: Method,
    rad: &Radiance,
    spec: &SceneSpec,
    truth: &Truth,
    n_seeds: u64,
) -> Errors {
    (0..n_seeds)
        .map(|seed| {
            let scene = expose(rad, spec, 7000 + seed);
            let res = det.detect(&scene.image).expect("detect");
            let mut out = BTreeMap::new();
            for m in &res.detected_markers {
                if m.center_source != method.source() {
                    continue;
                }
                let Some(xy) = m.board_xy_mm else { continue };
                let Some(i) = truth.cell_of(xy, m.center, &rad.h) else {
                    continue;
                };
                let t = truth.centers[i];
                out.insert(i, [m.center[0] - t[0], m.center[1] - t[1]]);
            }
            out
        })
        .collect()
}

fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    if v.len() < 2 {
        return (m, f64::NAN);
    }
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, (var / n).sqrt())
}

fn bias_of(plus: &Errors, minus: &Errors, zero: &Errors, scored: &[usize]) -> Option<Bias> {
    let (mut ox, mut oy, mut dy) = (Vec::new(), Vec::new(), Vec::new());
    for &c in scored {
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0usize);
        let (mut sd, mut nd) = (0.0, 0usize);
        for s in 0..plus.len() {
            if let (Some(p), Some(m)) = (plus[s].get(&c), minus[s].get(&c)) {
                sx += 0.5 * (p[0] - m[0]);
                sy += 0.5 * (p[1] - m[1]);
                n += 1;
            }
            if let (Some(p), Some(z)) = (plus[s].get(&c), zero[s].get(&c)) {
                sd += p[1] - z[1];
                nd += 1;
            }
        }
        if n > 0 {
            ox.push(sx / n as f64);
            oy.push(sy / n as f64);
        }
        if nd > 0 {
            dy.push(sd / nd as f64);
        }
    }
    if ox.len() < 2 {
        return None;
    }
    let (bx, sex) = mean_se(&ox);
    let (by, sey) = mean_se(&oy);
    let (d, sed) = if dy.len() >= 2 {
        mean_se(&dy)
    } else {
        (f64::NAN, f64::NAN)
    };
    Some(Bias {
        n_markers: ox.len(),
        odd: [bx, by],
        odd_se: [sex, sey],
        direct_y: d,
        direct_se: sed,
    })
}

fn run_group(board: &BoardRaster, group: &Group, gradients: &[f64], n_seeds: u64) -> Vec<Row> {
    let target = &board.target;
    let size = image_size(group.radius);
    let mut spec =
        SceneSpec::for_target(target, group.radius, group.sigma0, group.tilt, CONTRAST, 1);
    spec.image_size = size;
    let y0 = 0.5 * (f64::from(size[1]) - 1.0);
    let dets: Vec<Detector> = METHODS
        .iter()
        .map(|m| detector(target, group.radius, m.refinement()))
        .collect();
    let rad0 = radiance(board, &spec);
    let truth = Truth::new(target, &rad0.h, spec.px_per_mm);
    let zero: Vec<Errors> = METHODS
        .iter()
        .zip(&dets)
        .map(|(&m, d)| measure(d, m, &rad0, &spec, &truth, n_seeds))
        .collect();
    let mut rows = Vec::new();
    for &g in gradients {
        let half = band_half_width(group.sigma0, g, f64::from(size[1]));
        let scored = truth.scored(size, 3.0 * group.sigma0, y0 - half, y0 + half);
        for form in FORMS {
            let mut row = Row {
                group: *group,
                form,
                g,
                n_scored: scored.len(),
                n_zero: [0, 1]
                    .map(|k| scored.iter().filter(|c| zero[k][0].contains_key(c)).count()),
                bias: [None, None],
            };
            if scored.len() >= 2 {
                let render = |sign: f64| {
                    let mut s = spec.clone();
                    s.psf = cell_psf(group.sigma0, sign * g, y0, half, form);
                    (radiance(board, &s), s)
                };
                let (rp, sp) = render(1.0);
                let (rm, sm) = render(-1.0);
                for (k, (&m, d)) in METHODS.iter().zip(&dets).enumerate() {
                    let plus = measure(d, m, &rp, &sp, &truth, n_seeds);
                    let minus = measure(d, m, &rm, &sm, &truth, n_seeds);
                    row.bias[k] = bias_of(&plus, &minus, &zero[k], &scored);
                }
            }
            rows.push(row);
        }
    }
    rows
}

fn fmt_bias(b: &Option<Bias>, n_zero: usize) -> String {
    match b {
        None => format!("{:>40}", format!("- ({n_zero} placed at g = 0)")),
        Some(b) => format!(
            "{:>3} {:>+8.4}±{:.4} {:>+8.4}±{:.4} {:>+8.4}",
            b.n_markers, b.odd[0], b.odd_se[0], b.odd[1], b.odd_se[1], b.direct_y
        ),
    }
}

fn print_rows(rows: &[Row]) {
    println!(
        "bias px; x = across, y = along the gradient (+ = towards larger blur); \
         odd part (e(+g)-e(-g))/2 ± s.e. over markers; direct = e(+g)-e(0) along y"
    );
    println!(
        "{:<10} {:>3} {:>4} {:>4} {:<7} {:>6} {:>4} | {:^40} | {:^40} | {:>8}",
        "kind",
        "R",
        "sig0",
        "tilt",
        "form",
        "g",
        "scrd",
        "projective: n  x  y  direct",
        "photometric: n  x  y  direct",
        "-g*sig0"
    );
    for r in rows {
        println!(
            "{:<10} {:>3.0} {:>4.1} {:>4.0} {:<7} {:>6.3} {:>4} | {} | {} | {:>+8.4}",
            r.group.kind.name(),
            r.group.radius,
            r.group.sigma0,
            r.group.tilt,
            format!("{:?}", r.form),
            r.g,
            r.n_scored,
            fmt_bias(&r.bias[0], r.n_zero[0]),
            fmt_bias(&r.bias[1], r.n_zero[1]),
            -r.g * r.group.sigma0,
        );
    }
}

fn write_csv(rows: &[Row], path: &str) {
    let mut s = String::from(
        "kind,radius,sigma0,tilt,form,g,method,n,bias_x,se_x,bias_y,se_y,direct_y,direct_se\n",
    );
    for r in rows {
        for (k, m) in METHODS.iter().enumerate() {
            if let Some(b) = &r.bias[k] {
                s.push_str(&format!(
                    "{},{},{},{},{:?},{},{},{},{},{},{},{},{},{}\n",
                    r.group.kind.name(),
                    r.group.radius,
                    r.group.sigma0,
                    r.group.tilt,
                    r.form,
                    r.g,
                    m.name(),
                    b.n_markers,
                    b.odd[0],
                    b.odd_se[0],
                    b.odd[1],
                    b.odd_se[1],
                    b.direct_y,
                    b.direct_se
                ));
            }
        }
    }
    std::fs::write(path, s).expect("write csv");
}

/// Solve the normal equations of a small least-squares problem.
fn least_squares(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<Vec<f64>> {
    let p = x.first()?.len();
    let mut a = vec![vec![0.0; p + 1]; p];
    for ((xi, yi), wi) in x.iter().zip(y).zip(w) {
        for r in 0..p {
            for c in 0..p {
                a[r][c] += wi * xi[r] * xi[c];
            }
            a[r][p] += wi * xi[r] * yi;
        }
    }
    for col in 0..p {
        let piv = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        a.swap(col, piv);
        if a[col][col].abs() < 1e-300 {
            return None;
        }
        let pivot = a[col].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / pivot[col];
                for (v, pv) in row[col..].iter_mut().zip(&pivot[col..]) {
                    *v -= f * pv;
                }
            }
        }
    }
    Some((0..p).map(|r| a[r][p] / a[r][r]).collect())
}

/// A named explanatory variable of the power-law fit.
type LawVariable = (&'static str, fn(&Row) -> f64);

/// `b = c g sigma0` (weighted, through the origin) over `cells`: `c`, the
/// share of the sum of squares it explains, the rms misfit and chi2/dof.
fn fit_linear(cells: &[(&Row, Bias)]) -> Option<(f64, f64, f64, f64)> {
    if cells.len() < 3 {
        return None;
    }
    let w: Vec<f64> = cells
        .iter()
        .map(|(_, b)| 1.0 / (b.odd_se[1].powi(2) + 1e-6))
        .collect();
    let x: Vec<Vec<f64>> = cells
        .iter()
        .map(|(r, _)| vec![r.g * r.group.sigma0])
        .collect();
    let y: Vec<f64> = cells.iter().map(|(_, b)| b.odd[1]).collect();
    let c = least_squares(&x, &y, &w)?[0];
    let (mut ss_res, mut ss_tot, mut chi2) = (0.0, 0.0, 0.0);
    for ((xi, yi), wi) in x.iter().zip(&y).zip(&w) {
        let r = yi - c * xi[0];
        ss_res += r * r;
        ss_tot += yi * yi;
        chi2 += wi * r * r;
    }
    Some((
        c,
        1.0 - ss_res / ss_tot,
        (ss_res / cells.len() as f64).sqrt(),
        chi2 / (cells.len() - 1) as f64,
    ))
}

/// Fit the bias along the gradient per method and form:
/// (1) `b = c g sigma0`, over all cells and per target kind;
/// (2) `|b| = c g^m sigma0^a R^k` in log space over the cells with a
/// significant bias of the sign of (1) (a variable that does not vary is left
/// out).
fn fit_laws(rows: &[Row]) {
    println!("\nfitted laws (bias along the gradient, px)");
    for (k, method) in METHODS.iter().enumerate() {
        for form in FORMS {
            let cells: Vec<(&Row, Bias)> = rows
                .iter()
                .filter(|r| r.form == form)
                .filter_map(|r| r.bias[k].map(|b| (r, b)))
                .filter(|(_, b)| b.odd_se[1].is_finite())
                .collect();
            let Some((c, share, rms, chi2)) = fit_linear(&cells) else {
                continue;
            };
            println!(
                "{:<11} {:<7} all       : b = {c:+.3} g sigma0 over {:>3} cells; explains \
                 {:5.1} % of sum b^2; rms misfit {rms:.4} px; chi2/dof {chi2:.1}",
                method.name(),
                format!("{form:?}"),
                cells.len(),
                100.0 * share,
            );
            for kind in KINDS {
                let sub: Vec<(&Row, Bias)> = cells
                    .iter()
                    .filter(|(r, _)| r.group.kind == kind)
                    .copied()
                    .collect();
                for tilt in [None, Some(0.0), Some(25.0)] {
                    let sub: Vec<(&Row, Bias)> = sub
                        .iter()
                        .filter(|(r, _)| tilt.is_none_or(|t| r.group.tilt == t))
                        .copied()
                        .collect();
                    if let Some((c, share, rms, chi2)) = fit_linear(&sub) {
                        println!(
                            "{:<11} {:<7} {:<10}: b = {c:+.3} g sigma0 over {:>3} cells{}; explains \
                             {:5.1} %; rms misfit {rms:.4} px; chi2/dof {chi2:.1}",
                            method.name(),
                            format!("{form:?}"),
                            kind.name(),
                            sub.len(),
                            tilt.map_or(String::new(), |t| format!(" (tilt {t:.0})")),
                            100.0 * share,
                        );
                    }
                }
            }
            // (2) log-power law over significant cells of the same sign.
            let sig: Vec<&(&Row, Bias)> = cells
                .iter()
                .filter(|(_, b)| b.odd[1] * c > 0.0 && b.odd[1].abs() > 3.0 * b.odd_se[1])
                .collect();
            let vars: [LawVariable; 3] = [
                ("g", |r| r.g),
                ("sigma0", |r| r.group.sigma0),
                ("R", |r| r.group.radius),
            ];
            let used: Vec<usize> = (0..3)
                .filter(|&v| {
                    let f = vars[v].1;
                    sig.iter().any(|(r, _)| f(r) != f(sig[0].0))
                })
                .collect();
            if sig.len() < used.len() + 3 {
                continue;
            }
            let x: Vec<Vec<f64>> = sig
                .iter()
                .map(|(r, _)| {
                    let mut row = vec![1.0];
                    row.extend(used.iter().map(|&v| vars[v].1(r).ln()));
                    row
                })
                .collect();
            let y: Vec<f64> = sig.iter().map(|(_, b)| b.odd[1].abs().ln()).collect();
            let w: Vec<f64> = sig
                .iter()
                .map(|(_, b)| (b.odd[1] / b.odd_se[1]).powi(2).min(400.0))
                .collect();
            if let Some(p) = least_squares(&x, &y, &w) {
                let rms_log = (x
                    .iter()
                    .zip(&y)
                    .map(|(xi, yi)| {
                        let f: f64 = xi.iter().zip(&p).map(|(a, b)| a * b).sum();
                        (yi - f).powi(2)
                    })
                    .sum::<f64>()
                    / sig.len() as f64)
                    .sqrt();
                let law: String = used
                    .iter()
                    .zip(&p[1..])
                    .map(|(&v, e)| format!(" {}^{e:.2}", vars[v].0))
                    .collect();
                println!(
                    "{:<11} {:<7} power law : |b| = {:.3}{law} over {} significant cells; \
                     rms log residual {rms_log:.2} (factor {:.2})",
                    method.name(),
                    format!("{form:?}"),
                    p[0].exp(),
                    sig.len(),
                    rms_log.exp()
                );
            }
        }
    }
}

/// Q1: centre bias under a blur gradient, per method, target kind, marker
/// radius, sigma0, gradient, tilt and rendering form.
#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn blur_gradient_bias_matrix() {
    let n_seeds = env_or("GRADIENT_SEEDS", 3u64);
    let radii = env_list("GRADIENT_RADII", &[8.0, 17.0, 33.0]);
    let sigmas = env_list("GRADIENT_SIGMAS", &[0.8, 1.5, 3.0, 5.0]);
    let tilts = env_list("GRADIENT_TILTS", &[0.0, 25.0]);
    let gradients = env_list("GRADIENT_GS", &GRADIENTS);
    let mut groups = Vec::new();
    for kind in selected_kinds() {
        for &radius in &radii {
            for &sigma0 in &sigmas {
                for &tilt in &tilts {
                    groups.push(Group {
                        kind,
                        radius,
                        sigma0,
                        tilt,
                    });
                }
            }
        }
    }
    let mut rasters: BTreeMap<(usize, u64), BoardRaster> = BTreeMap::new();
    for g in &groups {
        let k = KINDS.iter().position(|k| *k == g.kind).unwrap();
        rasters.entry((k, g.radius.to_bits())).or_insert_with(|| {
            let target = g.kind.target();
            let dpi = raster_dpi(&target, g.radius);
            BoardRaster::with_dpi(target, dpi)
        });
    }
    let results: Mutex<Vec<Option<Vec<Row>>>> = Mutex::new(vec![None; groups.len()]);
    let next = AtomicUsize::new(0);
    let threads = env_or("GRADIENT_THREADS", 10usize).max(1);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(g) = groups.get(i) else { break };
                    let k = KINDS.iter().position(|k| *k == g.kind).unwrap();
                    let board = &rasters[&(k, g.radius.to_bits())];
                    let rows = run_group(board, g, &gradients, n_seeds);
                    results.lock().unwrap()[i] = Some(rows);
                }
            });
        }
    });
    let rows: Vec<Row> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .flat_map(|r| r.expect("group ran"))
        .collect();
    print_rows(&rows);
    fit_laws(&rows);
    if let Ok(path) = std::env::var("GRADIENT_CSV") {
        write_csv(&rows, &path);
    }
}

// ---------------------------------------------------------------------------
// Q3: residual signature
// ---------------------------------------------------------------------------

/// Q3: the stacked residual of the photometric fit near the outer edge, by
/// image angle, on blur-gradient scenes of a coded hex board close to the real
/// captures (17 px markers, contrast 60).
#[cfg(feature = "study")]
#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn blur_gradient_residual_dipole() {
    use ringgrid::diagnostics::PhotometricStatus;

    let radius = 17.0;
    let target = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).expect("target");
    let board = BoardRaster::with_dpi(target.clone(), raster_dpi(&target, radius));
    let det = detector(&target, radius, CircleRefinementMethod::Photometric);
    let frames = env_or("GRADIENT_FRAMES", 3u64);
    let mut cells: Vec<(f64, f64, BlurForm)> = Vec::new();
    for sigma0 in env_list("GRADIENT_SIGMAS", &[1.0, 2.0, 3.0]) {
        for g in [0.0, 0.005, 0.01, 0.02] {
            cells.push((sigma0, g, BlurForm::Scatter));
        }
        cells.push((sigma0, 0.01, BlurForm::Gather));
    }
    let size = [720u32, 540];
    let y0 = 0.5 * (f64::from(size[1]) - 1.0);
    let results: Mutex<Vec<Option<String>>> = Mutex::new(vec![None; cells.len()]);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..env_or("GRADIENT_THREADS", 10usize) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(sigma0, g, form)) = cells.get(i) else { break };
                    let half = band_half_width(sigma0, g, f64::from(size[1]));
                    let mut markers = Vec::new();
                    let mut bias_y = Vec::new();
                    for frame in 0..frames {
                        let mut spec =
                            SceneSpec::for_target(&target, radius, sigma0, 20.0, CONTRAST, 1);
                        spec.image_size = size;
                        spec.tilt_axis_deg += 50.0 * frame as f64;
                        spec.psf = cell_psf(sigma0, g, y0, half, form);
                        let rad = radiance(&board, &spec);
                        let truth = Truth::new(&target, &rad.h, spec.px_per_mm);
                        let scene = expose(&rad, &spec, 300 + frame);
                        let (res, diag, residuals) =
                            det.detect_with_photometric_residuals(&scene.image).expect("detect");
                        for r in residuals {
                            let accepted = diag.markers[r.marker_index]
                                .photometric
                                .as_ref()
                                .is_some_and(|p| p.status == PhotometricStatus::Accepted);
                            if !accepted || (r.center[1] - y0).abs() > half - radius - 3.0 * sigma0
                            {
                                continue;
                            }
                            let m = &res.detected_markers[r.marker_index];
                            if let Some(xy) = m.board_xy_mm
                                && let Some(c) = truth.cell_of(xy, m.center, &rad.h)
                            {
                                bias_y.push(m.center[1] - truth.centers[c][1]);
                            }
                            markers.push(MarkerData {
                                center: r.center,
                                blur_sigma_px: r.blur_sigma_px,
                                samples: r.samples,
                            });
                        }
                    }
                    let line = if markers.len() < 10 {
                        format!("{sigma0:>4.1} {g:>6.3} {form:<8?} too few markers ({})", markers.len())
                    } else {
                        let rep = analyze(&markers);
                        let i = dipole(&rep.angle_ink);
                        let p = dipole(&rep.angle_paper);
                        let (iu, il, ia) = (i.upper, i.lower, i.harmonic);
                        let (pu, pl, pa) = (p.upper, p.lower, p.harmonic);
                        let sp = &rep.split[3];
                        let mean_sigma = markers.iter().map(|m| m.blur_sigma_px).sum::<f64>()
                            / markers.len() as f64;
                        let (by, _) = mean_se(&bias_y);
                        format!(
                            "{sigma0:>4.1} {g:>6.3} {:<8} {:>5} {mean_sigma:>6.2} | {iu:>+6.2} {il:>+6.2} {ia:>+6.2} | {pu:>+6.2} {pl:>+6.2} {pa:>+6.2} | {:>5.1} % | {by:>+8.4}",
                            format!("{form:?}"),
                            markers.len(),
                            100.0 * sp.explained_ms / sp.total_ms,
                        )
                    };
                    results.lock().unwrap()[i] = Some(line);
                }
            });
        }
    });
    println!(
        "residual = observed - model near the outer edge (|d| <= 2 px), contrast {CONTRAST}; \
         upper = image angles -180..0 (above the centre), A = first harmonic in sin(angle)"
    );
    println!(
        "{:>4} {:>6} {:<8} {:>5} {:>6} | {:^20} | {:^20} | {:^7} | {:>8}",
        "sig0",
        "g",
        "form",
        "n",
        "sigfit",
        "ink: upper lower A",
        "paper: upper lower A",
        "expl.",
        "bias_y"
    );
    for l in results.into_inner().unwrap() {
        println!("{}", l.expect("cell ran"));
    }
    println!(
        "real captures: ink upper -0.3..-0.4, lower +0.3..+0.6; paper upper +0.4..+0.6, lower -0.6..-0.7"
    );
}

// ---------------------------------------------------------------------------
// Q5: optical centroid of an isolated marker
// ---------------------------------------------------------------------------

/// Q5: the centroid of an isolated blurred marker (ink coverage, no detector,
/// no noise) under a blur gradient, relative to the same marker at `g = 0`.
#[test]
#[ignore = "study; run in release with --ignored --nocapture"]
fn blur_gradient_ink_centroid() {
    println!(
        "centroid shift of the blurred ink (px, along y = the gradient), relative to g = 0; \
         first-order prediction: scatter 0, gather +2 g sigma0"
    );
    println!(
        "{:<10} {:>3} {:>4} {:>4} {:>6} | {:>10} {:>10} | {:>9}",
        "kind", "R", "sig0", "tilt", "g", "gather", "scatter", "2 g sig0"
    );
    for kind in selected_kinds() {
        let target = kind.target();
        let (lo, hi) = target.marker_bounds_mm().expect("bounds");
        let mid = [
            0.5 * f64::from(lo[0] + hi[0]),
            0.5 * f64::from(lo[1] + hi[1]),
        ];
        let cell = target
            .cells()
            .iter()
            .min_by(|a, b| {
                let da = (f64::from(a.xy_mm[0]) - mid[0]).hypot(f64::from(a.xy_mm[1]) - mid[1]);
                let db = (f64::from(b.xy_mm[0]) - mid[0]).hypot(f64::from(b.xy_mm[1]) - mid[1]);
                da.total_cmp(&db)
            })
            .expect("cell");
        let xy = [f64::from(cell.xy_mm[0]), f64::from(cell.xy_mm[1])];
        let r_mm = f64::from(target.outer_draw_radius_mm());
        for radius in env_list("GRADIENT_RADII", &[8.0, 17.0, 33.0]) {
            let board = BoardRaster::with_dpi(target.clone(), raster_dpi(&target, radius))
                .isolated(xy, r_mm * 1.02);
            for sigma0 in env_list("GRADIENT_SIGMAS", &[0.8, 1.5, 3.0, 5.0]) {
                for tilt in [0.0, 25.0] {
                    let mut out = Vec::new();
                    for g in [0.0, 0.005, 0.01, 0.02, 0.04] {
                        let half = band_half_width(sigma0, g, f64::INFINITY);
                        if half < 1.2 * radius + 3.0 * sigma0 {
                            out.push(None);
                            continue;
                        }
                        let reach = (1.5 * radius + 6.0 * (sigma0 + g * half.min(4.0 * radius)))
                            .ceil() as u32;
                        let side = 2 * reach + 1;
                        let mut spec = SceneSpec::for_target(&target, radius, sigma0, tilt, 1.0, 1);
                        spec.image_size = [side, side];
                        spec.board_center_mm = [xy[0] - mid[0], xy[1] - mid[1]];
                        // Render the ink coverage itself: black 1, white 0.
                        spec.white = 0.0;
                        spec.contrast = -1.0;
                        spec.illum_gradient = 0.0;
                        spec.noise_sigma = 0.0;
                        let y0 = f64::from(reach);
                        let mut shifts = [0.0; 2];
                        for (f, form) in FORMS.iter().enumerate() {
                            spec.psf = cell_psf(sigma0, g, y0, half, *form);
                            let rad = radiance(&board, &spec);
                            let c = centroid(&rad.buf, side as usize);
                            let t = apply_h(&rad.h, xy);
                            shifts[f] = c[1] - t[1];
                            debug_assert!(varying_sigma(&spec.psf, y0).is_none_or(|s| s == sigma0));
                        }
                        out.push(Some(shifts));
                    }
                    let Some(base) = out[0] else { continue };
                    for (k, g) in [0.005, 0.01, 0.02, 0.04].iter().enumerate() {
                        let cell = match out[k + 1] {
                            Some(s) => {
                                format!("{:>+10.5} {:>+10.5}", s[0] - base[0], s[1] - base[1])
                            }
                            None => format!("{:>10} {:>10}", "-", "-"),
                        };
                        println!(
                            "{:<10} {:>3.0} {:>4.1} {:>4.0} {:>6.3} | {cell} | {:>+9.4}",
                            kind.name(),
                            radius,
                            sigma0,
                            tilt,
                            g,
                            2.0 * g * sigma0
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q6: photometric confidence against the true centre error
// ---------------------------------------------------------------------------

/// Blur field of a confidence-study scene: `sigma(y) = s_min + |g| d(y)`,
/// `d` the distance (px) from the sharp image edge (top for `g > 0`, bottom
/// for `g < 0`), scatter form — one half of a tilted-sensor focus band.
fn focus_field(s_min: f64, g: f64, height: f64) -> Psf {
    if g == 0.0 {
        return Psf::Gaussian { sigma: s_min };
    }
    Psf::VaryingGaussian {
        sigma0: s_min,
        gradient: g,
        y0: if g > 0.0 { 0.0 } else { height - 1.0 },
        sigma_range: [s_min, s_min + g.abs() * height],
        form: BlurForm::Scatter,
    }
}

/// Ranks (0-based, ties averaged).
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
            j += 1;
        }
        for &k in &order[i..=j] {
            r[k] = 0.5 * (i + j) as f64;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation.
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in rx.iter().zip(&ry) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx * syy).sqrt()
}

/// ROC AUC of `score` (higher = more suspect) for the `bad` class.
fn roc_auc(score: &[f64], bad: &[bool]) -> f64 {
    let r = ranks(score);
    let nb = bad.iter().filter(|&&b| b).count() as f64;
    let ng = bad.len() as f64 - nb;
    let sum: f64 = r
        .iter()
        .zip(bad)
        .filter(|(_, b)| **b)
        .map(|(r, _)| r + 1.0)
        .sum();
    (sum - nb * (nb + 1.0) / 2.0) / (nb * ng)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Q6: photometric confidence against the true centre error, over target
/// kind × marker radius × blur field (uniform, or a focus gradient: blur
/// σ 0.6 px at one image edge growing by 0.004 or 0.008 px per row) × tilt ×
/// contrast × seed. Prints the rank correlation of confidence and error,
/// the ROC AUC for centres beyond 0.05 / 0.1 px, and per band of true blur
/// the median error and confidence and the fraction beyond 0.05 px a
/// confidence cut keeps. `GRADIENT_CONF_CSV` writes one row per accepted
/// photometric centre (error, confidence, covariance, fit diagnostics);
/// `GRADIENT_RADII`, `GRADIENT_KINDS`, `GRADIENT_SEEDS` (default 2) restrict
/// or extend the grid.
#[test]
#[ignore = "study; run in release with --ignored --nocapture"]
fn blur_gradient_confidence() {
    use ringgrid::diagnostics::PhotometricStatus;
    use std::fmt::Write as _;

    let n_seeds = env_or("GRADIENT_SEEDS", 2u64);
    let radii = env_list("GRADIENT_RADII", &[10.0, 17.0, 28.0]);
    // (s_min, g): uniform blur, then focus gradients both ways.
    let fields = [
        (0.8, 0.0),
        (1.6, 0.0),
        (2.5, 0.0),
        (3.5, 0.0),
        (0.6, 0.004),
        (0.6, 0.008),
        (0.6, -0.004),
        (0.6, -0.008),
    ];
    let size = [720u32, 540];
    let mut cells = Vec::new();
    for kind in selected_kinds() {
        for &radius in &radii {
            for &(s_min, g) in &fields {
                for tilt in [0.0, 25.0] {
                    for contrast in [35.0, 70.0] {
                        for seed in 0..n_seeds {
                            cells.push((kind, radius, s_min, g, tilt, contrast, seed));
                        }
                    }
                }
            }
        }
    }
    let mut rasters: BTreeMap<(usize, u64), BoardRaster> = BTreeMap::new();
    for &(kind, radius, ..) in &cells {
        let k = KINDS.iter().position(|x| *x == kind).unwrap();
        rasters.entry((k, radius.to_bits())).or_insert_with(|| {
            let target = kind.target();
            let dpi = raster_dpi(&target, radius);
            BoardRaster::with_dpi(target, dpi)
        });
    }
    // Per scene: CSV rows and (true local blur, error, confidence) per marker.
    type SceneRows = (String, Vec<[f64; 3]>);
    let results: Mutex<Vec<Option<SceneRows>>> = Mutex::new(vec![None; cells.len()]);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..env_or("GRADIENT_THREADS", 10usize).max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(kind, radius, s_min, g, tilt, contrast, seed)) = cells.get(i) else {
                        break;
                    };
                    let k = KINDS.iter().position(|x| *x == kind).unwrap();
                    let board = &rasters[&(k, radius.to_bits())];
                    let target = &board.target;
                    let mut spec = SceneSpec::for_target(target, radius, s_min, tilt, contrast, 1);
                    spec.image_size = size;
                    spec.roll_deg += 23.0 * seed as f64;
                    spec.tilt_axis_deg += 50.0 * seed as f64;
                    spec.psf = focus_field(s_min, g, f64::from(size[1]));
                    let rad = radiance(board, &spec);
                    let truth = Truth::new(target, &rad.h, spec.px_per_mm);
                    let scene = expose(&rad, &spec, 900 + i as u64);
                    let mut config = DetectConfig::from_target_and_marker_diameter(
                        target.clone(),
                        (2.0 * radius) as f32,
                    );
                    config.circle_refinement = CircleRefinementMethod::Photometric;
                    config.min_confidence = 0.0;
                    let (res, diag) = Detector::with_config(config)
                        .detect_with_diagnostics(&scene.image)
                        .expect("detect");
                    let mut out = String::new();
                    let mut stats = Vec::new();
                    for (m, d) in res.detected_markers.iter().zip(&diag.markers) {
                        let Some(p) = d.photometric.as_ref() else {
                            continue;
                        };
                        let (Some(xy), Some(cov), Some(e)) =
                            (m.board_xy_mm, m.center_cov, m.ellipse_outer)
                        else {
                            continue;
                        };
                        if p.status != PhotometricStatus::Accepted {
                            continue;
                        }
                        let Some(c) = truth.cell_of(xy, m.center, &rad.h) else {
                            continue;
                        };
                        let t = truth.centers[c];
                        let local_sigma = varying_sigma(&spec.psf, t[1]).unwrap_or(s_min);
                        let err = (m.center[0] - t[0]).hypot(m.center[1] - t[1]);
                        stats.push([local_sigma, err, f64::from(m.confidence)]);
                        let nan = f64::NAN;
                        writeln!(
                            out,
                            "{},{radius},{s_min},{g},{tilt},{contrast},{seed},{local_sigma},\
                             {},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                            kind.name(),
                            m.center[0] - t[0],
                            m.center[1] - t[1],
                            m.confidence,
                            cov[0][0],
                            cov[0][1],
                            cov[1][1],
                            p.blur_sigma_px.unwrap_or(nan),
                            p.residual_sigma.unwrap_or(nan),
                            p.contrast.unwrap_or(nan),
                            p.n_pixels,
                            e.mean_axis(),
                            d.fit.outer_edge_width_px.map_or(nan, f64::from),
                            t[0],
                            t[1],
                            u8::from(m.id.is_some()),
                        )
                        .unwrap();
                    }
                    results.lock().unwrap()[i] = Some((out, stats));
                }
            });
        }
    });
    let mut csv = String::from(
        "kind,radius,s_min,g,tilt,contrast,seed,local_sigma,err_x,err_y,conf,cxx,cxy,cyy,\
         blur_fit,residual_sigma,contrast_fit,n_pixels,outer_r,edge_width,true_x,true_y,decoded\n",
    );
    let mut all: Vec<[f64; 3]> = Vec::new();
    for r in results.into_inner().unwrap() {
        let (rows, stats) = r.expect("cell ran");
        csv.push_str(&rows);
        all.extend(stats);
    }
    if let Ok(path) = std::env::var("GRADIENT_CONF_CSV") {
        std::fs::write(&path, csv).expect("write csv");
    }
    let err: Vec<f64> = all.iter().map(|r| r[1]).collect();
    let conf: Vec<f64> = all.iter().map(|r| r[2]).collect();
    let suspect: Vec<f64> = conf.iter().map(|c| -c).collect();
    let auc = |t: f64| roc_auc(&suspect, &err.iter().map(|e| *e > t).collect::<Vec<_>>());
    println!(
        "{} accepted photometric centres: rank correlation (confidence, error) {:+.3}; \
         ROC AUC for error > 0.05 px {:.3}, > 0.1 px {:.3}",
        all.len(),
        spearman(&conf, &err),
        auc(0.05),
        auc(0.1)
    );
    let bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 9.0)];
    println!(
        "{:>9} {:>6} {:>8} {:>9} | kept, fraction > 0.05 px at confidence >= 0.2 / 0.4 / 0.6",
        "true blur", "n", "err med", "conf med"
    );
    for (lo, hi) in bands {
        let band: Vec<&[f64; 3]> = all.iter().filter(|r| r[0] >= lo && r[0] < hi).collect();
        let cut = |c: f64| {
            let kept: Vec<f64> = band.iter().filter(|r| r[2] >= c).map(|r| r[1]).collect();
            let bad = kept.iter().filter(|e| **e > 0.05).count();
            format!(
                "{:5} {:.3}",
                kept.len(),
                bad as f64 / kept.len().max(1) as f64
            )
        };
        println!(
            "{:>4.0}-{:<4.0} {:>6} {:>8.4} {:>9.2} | {} | {} | {}",
            lo,
            hi,
            band.len(),
            median(band.iter().map(|r| r[1]).collect()),
            median(band.iter().map(|r| r[2]).collect()),
            cut(0.2),
            cut(0.4),
            cut(0.6)
        );
    }
}
