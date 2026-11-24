//! Paired center metrology with known board-to-image projection. Run with
//! `cargo test --release -p ringgrid --features study --test center_metrology_matrix -- --ignored --nocapture`.
//! Set CENTER_MATRIX_OUT to a JSONL path (default tools/out/center_metrology/synthetic.jsonl)
//! and CENTER_MATRIX_FULL=1 for the larger phase/focus matrix.
//! Set CENTER_MATRIX_DIRECTIONAL_TRAIN=1 for signed-gradient training only.
//! It writes synthetic_directional_train.jsonl by default.
//! Set CENTER_MATRIX_FOCUS_BAND=1 for the oblique two-sided focus challenge.
#![cfg(feature = "study")]

#[path = "support/blur_scene.rs"]
mod blur_scene;

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use blur_scene::{BlurForm, Psf, Radiance, Scene, SceneSpec, expose};
use ringgrid::diagnostics::MarkerDiagnostics;
use ringgrid::photometric_residuals::{PhotometricResidualMarker, PhotometricStudyOptions};
use ringgrid::{
    BoardFrame, CenterSource, CircleRefinementMethod, DetectConfig, DetectionResult, Detector,
    MarkerCoding, OriginDots, TargetLayout,
};
use serde_json::{Value, json};

const SIZE: [u32; 2] = [236, 188];
const SUBPIXELS: u32 = 5;
const RADIUS: f64 = 14.0;

#[derive(Clone, Copy, Debug)]
struct Target {
    coded: bool,
    width_delta_mm: f32,
}

impl Target {
    fn layout(self, rendered: bool) -> TargetLayout {
        let delta = if rendered { self.width_delta_mm } else { 0.0 };
        if self.coded {
            TargetLayout::coded_rect(12.0, 4, 5, 4.5, 2.9, 0.85 + delta)
        } else {
            TargetLayout::plain_rect(12.0, 4, 5, 4.925, 2.5 - delta, OriginDots::Auto)
        }
        .expect("valid layout")
    }

    fn name(self) -> &'static str {
        if self.coded {
            "coded_rect"
        } else {
            "plain_rect"
        }
    }

    fn revision(self) -> String {
        format!("{:+.2}mm", self.width_delta_mm)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Effect {
    name: &'static str,
    camera_phase: [f64; 2],
    display_phase: Option<f64>,
    point_sensor: bool,
    anisotropy: bool,
    focus_gradient: f64,
    focus_band: Option<FocusBand>,
}

#[derive(Clone, Copy, Debug)]
struct FocusBand {
    intercept_px: f64,
    slope: f64,
}

impl FocusBand {
    const ANGLE_DEG: f64 = 82.0;
    const MIN_OPTICAL_SIGMA_PX: f64 = 0.88;

    fn for_pose(pose: usize) -> Self {
        Self {
            intercept_px: [-18.0, 10.0, -4.0, 0.0, 15.0, -12.0, 21.0, -9.0, -22.0, 8.0][pose],
            slope: [
                0.021, 0.032, 0.028, 0.0, 0.035, 0.024, 0.030, 0.037, 0.026, 0.033,
            ][pose],
        }
    }

    fn normal() -> [f64; 2] {
        let (y, x) = Self::ANGLE_DEG.to_radians().sin_cos();
        [x, y]
    }

    /// Signed normal distance from the line where optical sigma is minimum.
    fn distance(self, xy: [f64; 2]) -> f64 {
        let center = SIZE.map(|n| 0.5 * f64::from(n - 1));
        let [nx, ny] = Self::normal();
        nx * (xy[0] - center[0]) + ny * (xy[1] - center[1]) - self.intercept_px
    }

    fn sigma(self, xy: [f64; 2]) -> f64 {
        Self::MIN_OPTICAL_SIGMA_PX.hypot(self.slope * self.distance(xy))
    }

    /// Sigma and its gradient including box sensor integration.
    fn reference(self, xy: [f64; 2]) -> (f64, [f64; 2]) {
        let sigma = (self.sigma(xy).powi(2) + 1.0 / 12.0).sqrt();
        let derivative = self.slope.powi(2) * self.distance(xy) / sigma;
        (sigma, Self::normal().map(|n| n * derivative))
    }
}

fn conditions(full: bool) -> Vec<Effect> {
    let mut effects = vec![
        Effect {
            name: "zero_gradient",
            ..Effect::default()
        },
        Effect {
            name: "point_sensor",
            point_sensor: true,
            ..Effect::default()
        },
        Effect {
            name: "anisotropic",
            anisotropy: true,
            ..Effect::default()
        },
        Effect {
            name: "positive_gradient",
            focus_gradient: 0.008,
            ..Effect::default()
        },
        Effect {
            name: "negative_gradient",
            focus_gradient: -0.008,
            ..Effect::default()
        },
    ];
    for (name, phase) in [
        ("camera_phase_a", [0.25, 0.25]),
        ("camera_phase_b", [0.5, 0.75]),
        ("camera_phase_c", [0.75, 0.5]),
    ] {
        effects.push(Effect {
            name,
            camera_phase: phase,
            ..Effect::default()
        });
    }
    for (name, phase) in [
        ("display_phase_a", 0.1),
        ("display_phase_b", 0.4),
        ("display_phase_c", 0.7),
    ] {
        effects.push(Effect {
            name,
            display_phase: Some(phase),
            ..Effect::default()
        });
    }
    if full {
        effects.extend([
            Effect {
                name: "display_point_sensor",
                display_phase: Some(0.4),
                point_sensor: true,
                ..Effect::default()
            },
            Effect {
                name: "display_anisotropic",
                display_phase: Some(0.4),
                anisotropy: true,
                ..Effect::default()
            },
            Effect {
                name: "display_positive_gradient",
                display_phase: Some(0.4),
                focus_gradient: 0.008,
                ..Effect::default()
            },
            Effect {
                name: "display_negative_gradient",
                display_phase: Some(0.4),
                focus_gradient: -0.008,
                ..Effect::default()
            },
            Effect {
                name: "steeper_positive_gradient",
                focus_gradient: 0.012,
                ..Effect::default()
            },
            Effect {
                name: "steeper_negative_gradient",
                focus_gradient: -0.012,
                ..Effect::default()
            },
        ]);
    }
    effects
}

/// Ring edges are evaluated analytically in board mm. The codeword table is
/// fixture data from the external JSON reference. No detector model or target
/// renderer supplies ink values.
#[derive(Clone)]
struct InkBoard {
    origin: [f64; 2],
    codes: [Option<u16>; 20],
    radii_sq: [f64; 4],
    dots: Vec<[f64; 2]>,
    dot_radius_sq: f64,
    display: Option<(f64, Vec<f64>)>,
}

impl InkBoard {
    const PITCH: f64 = 12.0;
    const DISPLAY_PITCH: f64 = 0.38;
    const DISPLAY_SIDE: usize = 192;
    const DISPLAY_OFFSET: f64 = 10.0;

    fn new(layout: &TargetLayout) -> Self {
        let words: Value = serde_json::from_str(include_str!("../../../tools/codebook.json"))
            .expect("read codeword fixture");
        let mut codes = [None; 20];
        let (origin, _) = layout.marker_bounds_mm().unwrap();
        for cell in layout.cells() {
            let [x, y] = std::array::from_fn(|i| {
                (f64::from(cell.xy_mm[i] - origin[i]) / Self::PITCH).round() as usize
            });
            codes[y * 5 + x] = cell.id.map(|id| {
                u16::from_str_radix(
                    words["codewords"][id]
                        .as_str()
                        .unwrap()
                        .trim_start_matches("0x"),
                    16,
                )
                .unwrap()
            });
        }
        Self {
            origin: origin.map(f64::from),
            codes,
            radii_sq: nominal_radii(layout).map(|r| r * r),
            dots: layout
                .fiducial_dots_mm()
                .iter()
                .map(|p| p.map(f64::from))
                .collect(),
            dot_radius_sq: layout
                .fiducials()
                .map_or(0.0, |f| f64::from(f.dot_radius_mm).powi(2)),
            display: None,
        }
    }

    fn reflectance(&self, xy: [f64; 2]) -> f64 {
        let Some((phase, levels)) = &self.display else {
            return self.analytic_reflectance(xy);
        };
        let [x, y] =
            xy.map(|v| ((v + Self::DISPLAY_OFFSET) / Self::DISPLAY_PITCH - phase).floor() as isize);
        if x < 0 || y < 0 || x >= Self::DISPLAY_SIDE as isize || y >= Self::DISPLAY_SIDE as isize {
            return 1.0;
        }
        levels[y as usize * Self::DISPLAY_SIDE + x as usize]
    }

    fn analytic_reflectance(&self, [x, y]: [f64; 2]) -> f64 {
        let col = ((x - self.origin[0]) / Self::PITCH).round();
        let row = ((y - self.origin[1]) / Self::PITCH).round();
        if (0.0..5.0).contains(&col) && (0.0..4.0).contains(&row) {
            let dx = x - self.origin[0] - col * Self::PITCH;
            let dy = y - self.origin[1] - row * Self::PITCH;
            let r2 = dx * dx + dy * dy;
            let [r0, r1, r2_band, r3] = self.radii_sq;
            if (r0..r1).contains(&r2) || (r2_band..r3).contains(&r2) {
                return 0.0;
            }
            if (r1..r2_band).contains(&r2) {
                let word = self.codes[row as usize * 5 + col as usize].expect("code band");
                let angle = dy.atan2(dx) + std::f64::consts::PI;
                let sector = (angle * 16.0 / std::f64::consts::TAU).floor() as u32 % 16;
                return f64::from((word >> sector) & 1);
            }
        }
        if self
            .dots
            .iter()
            .any(|[a, b]| (x - a).powi(2) + (y - b).powi(2) < self.dot_radius_sq)
        {
            return 0.0;
        }
        1.0
    }

    fn displayed(&self, phase: f64) -> Self {
        const TAPS: usize = 12;
        let mut levels = Vec::with_capacity(Self::DISPLAY_SIDE * Self::DISPLAY_SIDE);
        for row in 0..Self::DISPLAY_SIDE {
            for col in 0..Self::DISPLAY_SIDE {
                let x = (col as f64 + phase) * Self::DISPLAY_PITCH - Self::DISPLAY_OFFSET;
                let y = (row as f64 + phase) * Self::DISPLAY_PITCH - Self::DISPLAY_OFFSET;
                let mut total = 0.0;
                for sy in 0..TAPS {
                    for sx in 0..TAPS {
                        let sub = |s| (s as f64 + 0.5) * Self::DISPLAY_PITCH / TAPS as f64;
                        total += self.analytic_reflectance([x + sub(sx), y + sub(sy)]);
                    }
                }
                levels.push((15.0 * total / (TAPS * TAPS) as f64).round() / 15.0);
            }
        }
        Self {
            display: Some((phase, levels)),
            ..self.clone()
        }
    }
}

fn render_capture(
    board: &InkBoard,
    target: &TargetLayout,
    effect: Effect,
    capture: usize,
) -> (Radiance, SceneSpec) {
    let scale = f64::from(SUBPIXELS);
    let tilt = [0.0, 17.0, 27.0, 23.0, 12.0, 30.0][capture % 6];
    let mut spec = SceneSpec::for_target(target, RADIUS, 1.7, tilt, 90.0, 123);
    spec.image_size = SIZE;
    spec.illum_gradient = 0.0;
    spec.roll_deg = [-12.0, 13.0, 5.0, -7.0, 9.0, -4.0][capture % 6];
    spec.tilt_axis_deg += 19.0 * capture as f64;
    // A different board pose even for captures with the same zero tilt/roll.
    let angle = 1.7 * capture as f64;
    spec.board_center_mm = [0.4 * angle.cos(), 0.4 * angle.sin()];

    // Preserve the field of view at the finer grid. Fine pixel-center x maps
    // to image coordinate (x - (q-1)/2) / q. Blur before sensor box averaging.
    let mut fine = spec.clone();
    fine.image_size = SIZE.map(|d| d * SUBPIXELS);
    fine.px_per_mm *= scale;
    fine.focal_px *= scale;
    fine.supersample = 2;
    fine.psf = if effect.focus_band.is_some() {
        Psf::Gaussian { sigma: 0.0 } // apply the two-dimensional optical PSF below
    } else if effect.anisotropy {
        Psf::Anisotropic {
            sigma_major: 2.2 * scale,
            sigma_minor: 1.0 * scale,
            angle_deg: 0.0,
        }
    } else if effect.focus_gradient != 0.0 {
        Psf::VaryingGaussian {
            sigma0: 1.7 * scale,
            gradient: effect.focus_gradient,
            y0: 0.5 * f64::from(fine.image_size[1] - 1),
            sigma_range: [0.45 * scale, 3.5 * scale],
            form: BlurForm::Scatter,
        }
    } else {
        Psf::Gaussian { sigma: 1.7 * scale }
    };

    // A shift of the fine camera principal point by changing the virtual
    // canvas would alter its size. Instead translate the sharp radiance and
    // its homography together using the phase-aware renderer.
    let mut rad = blur_scene::radiance_from_reflectance(
        target,
        &fine,
        effect.camera_phase.map(|p| p * scale),
        |xy| board.reflectance(xy),
    );
    if let Some(band) = effect.focus_band {
        let [w, h] = fine.image_size.map(|n| n as usize);
        blur_scene::scatter_gaussian_field(&mut rad.buf, w, h, |xy| {
            scale * band.sigma(xy.map(|p| (p - 0.5 * (scale - 1.0)) / scale))
        });
    }
    let mut h = rad.h;
    let denominator = h[2];
    for row in &mut h[..2] {
        for (value, w) in row.iter_mut().zip(denominator) {
            *value = (*value - 0.5 * (scale - 1.0) * w) / scale;
        }
    }
    let q = SUBPIXELS as usize;
    let w = fine.image_size[0] as usize;
    let sample = |x, y| rad.buf[y * w + x];
    let mut buf = Vec::with_capacity((SIZE[0] * SIZE[1]) as usize);
    for y in 0..SIZE[1] as usize {
        for x in 0..SIZE[0] as usize {
            let mut sum = 0.0;
            if effect.point_sensor {
                sum = sample(q * x + q / 2, q * y + q / 2);
            } else {
                for dy in 0..q {
                    for dx in 0..q {
                        sum += sample(q * x + dx, q * y + dy);
                    }
                }
                sum /= (q * q) as f64;
            }
            buf.push(sum);
        }
    }
    (Radiance { buf, h }, spec)
}

fn detector(target: &TargetLayout, method: CircleRefinementMethod, estimate: bool) -> Detector {
    let mut config =
        DetectConfig::from_target_and_marker_diameter(target.clone(), 2.0 * RADIUS as f32);
    config.circle_refinement = method;
    config.advanced.photometric.estimate_print_radii = estimate;
    Detector::with_config(config)
}

fn nominal_radii(target: &TargetLayout) -> [f64; 4] {
    let inner = f64::from(target.ring().inner_radius_mm);
    let outer = f64::from(target.ring().outer_radius_mm);
    match target.coding() {
        MarkerCoding::Coded16(spec) => {
            let half = 0.5 * f64::from(spec.ring_width_mm);
            [inner - half, inner + half, outer - half, outer + half]
        }
        MarkerCoding::Plain => [inner, outer, outer, outer],
    }
}

fn noise_seed(capture: usize, replicate: usize) -> u64 {
    20261010 + 817 * capture as u64 + replicate as u64
}

fn capture_id(target: Target, capture: usize) -> String {
    format!("{}-{}-{capture}", target.name(), target.revision())
}

struct FitObservations<'a> {
    summaries: BTreeMap<usize, &'a PhotometricResidualMarker>,
    diagnostics: &'a [MarkerDiagnostics],
}

struct Evaluation<'a> {
    target: Target,
    layout: &'a TargetLayout,
    scene: &'a Scene,
    effect: Effect,
    capture: usize,
    replicate: usize,
    train: bool,
}

impl Evaluation<'_> {
    fn rows(
        &self,
        result: &DetectionResult,
        method: &str,
        runtime_ms: f64,
        radii: Option<[f64; 4]>,
        moments: &BTreeMap<usize, [f64; 2]>,
        fitted: Option<&FitObservations<'_>>,
    ) -> Vec<Value> {
        let truth: Vec<_> = self
            .layout
            .cells()
            .iter()
            .map(|cell| self.scene.project(cell.xy_mm.map(f64::from)))
            .collect();
        let mut nearest = vec![None; truth.len()];
        let mut unmatched = 0;
        let mut duplicates = 0;
        for (index, marker) in result.detected_markers.iter().enumerate() {
            let (i, distance) = truth
                .iter()
                .enumerate()
                .map(|(i, p)| (i, (p[0] - marker.center[0]).hypot(p[1] - marker.center[1])))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("non-empty target");
            let slot = &mut nearest[i];
            if distance > RADIUS {
                unmatched += 1;
            } else {
                duplicates += usize::from(slot.is_some());
                if slot.is_none_or(|(_, prev)| distance < prev) {
                    *slot = Some((index, distance));
                }
            }
        }
        let true_radii = nominal_radii(&self.target.layout(true));
        let capture = capture_id(self.target, self.capture);
        let repeat_group = format!("{capture}/{}", self.effect.name);
        self.layout
            .cells()
            .iter()
            .zip(truth)
            .enumerate()
            .map(|(i, (cell, reference))| {
                let mut status = "missing";
                let mut center = None;
                let mut cov = None;
                let mut id = None;
                let mut raw_center = None;
                let mut blur_sigma_px = None;
                let mut source = None;
                if let Some((index, _)) = nearest[i] {
                    let marker = &result.detected_markers[index];
                    id = marker.id;
                    raw_center = Some(marker.center);
                    blur_sigma_px = marker.blur.map(|b| b.sigma_px);
                    source = Some(marker.center_source);
                    let label = match (cell.id, marker.id, result.board_frame, marker.board_xy_mm) {
                        (Some(expected), Some(actual), _, _) => Some(expected == actual),
                        (None, _, Some(BoardFrame::Absolute), Some(xy)) => Some(
                            xy.iter()
                                .zip(cell.xy_mm)
                                .all(|(a, b)| (*a - f64::from(b)).abs() < 1e-4),
                        ),
                        _ => None,
                    };
                    center = match method {
                        "projective" => Some(marker.center),
                        "moment" => moments.get(&index).copied(),
                        _ if marker.center_source == CenterSource::Photometric => {
                            cov = marker.center_cov;
                            Some(marker.center)
                        }
                        _ => None,
                    }
                    .filter(|c| c.iter().all(|x| x.is_finite()));
                    status = match (center, label) {
                        (_, Some(false)) => "wrong_id",
                        (Some(_), Some(true)) => "ok",
                        (Some(_), None) => "unverified",
                        (None, _) => "missing",
                    };
                }
                let mut row = json!({
                    "dataset": "synthetic",
                    "camera": "synthetic_0",
                    "capture": capture,
                    "scene_id": self.capture,
                    "repeat_group": repeat_group,
                    "replicate": self.replicate,
                    "split": if self.train { "train" } else { "holdout" },
                    "condition": self.effect.name,
                    "method": method,
                    "key": i,
                    "center": center,
                    "reference": reference,
                    "status": status,
                    "cov": cov,
                    "runtime_ms": runtime_ms,
                    "target": self.target.name(),
                    "revision": self.target.revision(),
                    "id": id,
                    "true_id": cell.id,
                    "source": if method == "moment" && center.is_some() { json!("moment") } else { json!(source) },
                    "raw_center": raw_center,
                    "blur_sigma_px": blur_sigma_px,
                    "reference_sigma_px": reference_sigma(self.effect, reference[1]),
                    "radii_mm": radii,
                    "true_radii_mm": true_radii,
                    "display_phase": self.effect.display_phase,
                    "camera_phase": self.effect.camera_phase,
                    "point_sensor": self.effect.point_sensor,
                    "focus_gradient": self.effect.focus_gradient,
                    "unmatched_detections": unmatched,
                    "duplicate_detections": duplicates,
                });
                if let Some(band) = self.effect.focus_band {
                    let (sigma, grad) = band.reference(reference);
                    row["reference_sigma_px"] = json!(sigma);
                    row["true_blur_gradient_px_per_px"] = json!(grad);
                    row["focus_normal_xy"] = json!(FocusBand::normal());
                    row["focus_intercept_px"] = json!(band.intercept_px);
                    row["focus_slope_px_per_px"] = json!(band.slope);
                    row["focus_signed_distance_px"] = json!(band.distance(reference));
                }
                if let Some(fitted) = fitted {
                    let index = nearest[i].map(|(index, _)| index);
                    let fit = index.and_then(|index| fitted.summaries.get(&index));
                    let fit_status = index.and_then(|index| {
                        fitted.diagnostics[index].photometric.as_ref().map(|fit| fit.status)
                    });
                    row["fitted_sigma_px"] = json!(fit.map(|f| f.blur_sigma_px));
                    row["fitted_center"] = json!(fit.map(|f| f.center));
                    row["fit_status"] = json!(fit_status);
                }
                row
            })
            .collect()
    }
}

/// Isotropic-equivalent second-moment width at the projected source point.
fn reference_sigma(effect: Effect, y: f64) -> f64 {
    let optical_variance = if effect.anisotropy {
        (2.2f64.powi(2) + 1.0) / 2.0
    } else {
        let dy = y - 0.5 * f64::from(SIZE[1] - 1);
        (1.7 + effect.focus_gradient * dy).clamp(0.45, 3.5).powi(2)
    };
    let sensor_variance = if effect.point_sensor { 0.0 } else { 1.0 / 12.0 };
    (optical_variance + sensor_variance).sqrt()
}

fn time_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn board_index(display: Option<f64>, phases: &[f64]) -> usize {
    if let Some(phase) = display {
        1 + phases
            .iter()
            .position(|&p| p == phase)
            .expect("rendered phase")
    } else {
        0
    }
}

fn evaluate(e: &Evaluation<'_>, pooled: Option<[f64; 4]>) -> (Vec<Value>, Option<[f64; 4]>) {
    let projective = CircleRefinementMethod::ProjectiveCenter;
    let photometric = CircleRefinementMethod::Photometric;
    let empty = BTreeMap::new();
    let mut rows = Vec::new();

    let start = Instant::now();
    let (per_image, diagnostics) = detector(e.layout, photometric, true)
        .detect_with_diagnostics(&e.scene.image)
        .expect("per-image detection");
    let elapsed = time_ms(start);
    let estimated = diagnostics.photometric_print_radii_mm;
    rows.extend(e.rows(&per_image, "per_image", elapsed, estimated, &empty, None));
    if e.train {
        return (rows, estimated);
    }

    let start = Instant::now();
    let baseline = detector(e.layout, projective, false)
        .detect(&e.scene.image)
        .expect("projective detection");
    rows.extend(e.rows(&baseline, "projective", time_ms(start), None, &empty, None));

    let start = Instant::now();
    let nominal = detector(e.layout, photometric, false)
        .detect(&e.scene.image)
        .expect("nominal detection");
    rows.extend(e.rows(
        &nominal,
        "nominal",
        time_ms(start),
        Some(nominal_radii(e.layout)),
        &empty,
        None,
    ));

    let start = Instant::now();
    let (moment_result, _, samples) = detector(e.layout, photometric, true)
        .detect_with_photometric_study(&e.scene.image, {
            let mut options = PhotometricStudyOptions::default();
            options.moment_centers = true;
            options
        })
        .expect("moment study detection");
    let elapsed = time_ms(start);
    let moments = samples
        .into_iter()
        .filter_map(|s| Some((s.marker_index, s.moment_center?)))
        .collect();
    rows.extend(e.rows(&moment_result, "moment", elapsed, estimated, &moments, None));

    if let Some(radii) = pooled {
        let start = Instant::now();
        let (result, _, _) = detector(e.layout, photometric, false)
            .detect_with_photometric_study(
                &e.scene.image,
                PhotometricStudyOptions::default()
                    .with_fixed_radii(e.layout, radii)
                    .expect("pooled radii describe the target"),
            )
            .expect("pooled radii detection");
        rows.extend(e.rows(&result, "pooled", time_ms(start), Some(radii), &empty, None));
    } else {
        rows.extend(e.rows(
            &DetectionResult::default(),
            "pooled",
            0.0,
            None,
            &empty,
            None,
        ));
    }
    (rows, estimated)
}

fn median_radii(samples: &[[f64; 4]]) -> Option<[f64; 4]> {
    if samples.len() < 2 {
        return None;
    }
    Some(std::array::from_fn(|i| {
        let mut v: Vec<_> = samples.iter().map(|s| s[i]).collect();
        v.sort_by(f64::total_cmp);
        (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0
    }))
}

/// Render using the same training pose indices as the original matrix.
/// Noise draws 4–5 are disjoint from the original matrix's draws 0–3.
fn directional_training(
    output: &mut impl Write,
    target: Target,
    layout: &TargetLayout,
    board: &InkBoard,
    training: usize,
    threads: usize,
) {
    let phases = [0.4];
    let boards = [board.clone(), board.displayed(phases[0])];
    let effects = conditions(true);
    let jobs: Vec<_> = effects
        .into_iter()
        .filter(|e| {
            matches!(
                e.name,
                "positive_gradient"
                    | "negative_gradient"
                    | "display_positive_gradient"
                    | "display_negative_gradient"
            )
        })
        .flat_map(|e| (0..training).map(move |capture| (e, capture)))
        .collect();
    let next = AtomicUsize::new(0);
    let result_batches = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(effect, capture)) = jobs.get(j) else {
                        break;
                    };
                    let board = &boards[board_index(effect.display_phase, &phases)];
                    let (rad, spec) = render_capture(board, layout, effect, capture);
                    let rows: Vec<_> = (4..6)
                        .flat_map(|replicate| {
                            let scene = expose(&rad, &spec, noise_seed(capture, replicate));
                            let e = Evaluation {
                                target,
                                layout,
                                scene: &scene,
                                effect,
                                capture,
                                replicate,
                                train: true,
                            };
                            photometric_study_rows(&e)
                        })
                        .collect();
                    result_batches.lock().unwrap().push((j, rows));
                }
            });
        }
    });
    let mut batches = result_batches.into_inner().unwrap();
    batches.sort_by_key(|(j, _)| *j);
    assert!(
        batches
            .iter()
            .flat_map(|(_, rows)| rows)
            .any(|r| r["status"] == "ok")
    );
    for (_, rows) in batches {
        for row in rows {
            writeln!(output, "{row}").expect("write directional training row");
        }
    }
    output.flush().expect("flush directional training rows");
    println!(
        "{} {} rendered {} directional training captures",
        target.name(),
        target.revision(),
        2 * jobs.len()
    );
}

fn photometric_study_rows(e: &Evaluation<'_>) -> Vec<Value> {
    let start = Instant::now();
    let (result, diagnostics, summaries) =
        detector(e.layout, CircleRefinementMethod::Photometric, true)
            .detect_with_photometric_study(&e.scene.image, PhotometricStudyOptions::default())
            .expect("directional training detection");
    let elapsed = time_ms(start);
    let fit_observations = FitObservations {
        summaries: summaries
            .iter()
            .map(|fit| (fit.marker_index, fit))
            .collect(),
        diagnostics: &diagnostics.markers,
    };
    e.rows(
        &result,
        "per_image",
        elapsed,
        diagnostics.photometric_print_radii_mm,
        &BTreeMap::new(),
        Some(&fit_observations),
    )
}

/// Fit only the detector measurements. The rendered focus geometry is
/// attached to observations for scoring after detection.
fn focus_band_study(destination: &Path) {
    let target = Target {
        coded: true,
        width_delta_mm: 0.0,
    };
    let layout = target.layout(false);
    let board = InkBoard::new(&target.layout(true));
    let mut output =
        BufWriter::new(std::fs::File::create(destination).expect("focus band output file"));
    for capture in [0, 1, 2, 4, 5, 6, 7, 8, 9] {
        let band = FocusBand::for_pose(capture);
        let effect = Effect {
            name: "two_sided_focus_band",
            focus_band: Some(band),
            ..Effect::default()
        };
        let (rad, spec) = render_capture(&board, &layout, effect, capture);
        for replicate in 6..8 {
            let scene = expose(&rad, &spec, noise_seed(capture, replicate));
            let evaluation = Evaluation {
                target,
                layout: &layout,
                scene: &scene,
                effect,
                capture,
                replicate,
                train: capture < 4,
            };
            let rows = photometric_study_rows(&evaluation);
            for direction in [-1.0, 1.0] {
                assert!(
                    rows.iter()
                        .filter(|r| r["status"] == "ok"
                            && r["blur_sigma_px"].is_number()
                            && r["focus_signed_distance_px"].as_f64().unwrap() * direction > 0.0)
                        .count()
                        >= 3,
                    "pose {capture}, replicate {replicate}: insufficient support on side {direction}"
                );
            }
            for row in rows {
                writeln!(output, "{row}").expect("write focus band row");
            }
        }
    }
    output.flush().expect("flush focus band rows");
    std::fs::write(
        destination.with_extension("meta.json"),
        serde_json::to_vec_pretty(&json!({
            "focus_normal_angle_deg": FocusBand::ANGLE_DEG,
            "focus_normal_xy": FocusBand::normal(),
            "optical_sigma_min_px": FocusBand::MIN_OPTICAL_SIGMA_PX,
            "train_poses": [0, 1, 2],
            "holdout_poses": [4, 5, 6, 7, 8, 9],
            "replicates": [6, 7],
            "seed_base": 20261010,
            "seed_pose_stride": 817,
            "image_size_px": SIZE,
            "fine_samples_per_image_px": SUBPIXELS,
            "renderer": "analytic coded ring boundaries, variance-interpolated normalized scatter",
        }))
        .unwrap(),
    )
    .expect("write focus band metadata");
    println!(
        "focus band observations written to {}",
        destination.display()
    );
}

fn write_metadata(
    destination: &Path,
    full: bool,
    training: usize,
    holdout: usize,
    threads: usize,
    directional_train: bool,
) {
    let git = |args: &[&str]| {
        let out = Command::new("git").args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    let status = git(&["status", "--porcelain"]);
    let mut metadata = json!({
        "git_head": git(&["rev-parse", "HEAD"]),
        "git_dirty": status.as_ref().map(|s| !s.is_empty()),
        "renderer": "analytic ring boundaries, external 16-bit codebook, fine-grid optical blur",
        "image_size_px": SIZE,
        "fine_samples_per_image_px": SUBPIXELS,
        "nominal_radius_px": RADIUS,
        "sensor_noise_sigma_levels": 2,
        "seed_base": 20261010,
        "seed_pose_stride": 817,
        "full": full,
        "training_poses_per_medium_target_revision": training,
        "holdout_poses_per_condition_target_revision": holdout,
        "holdout_replicates": if full { 4 } else { 2 },
        "runtime_worker_threads": threads.max(1),
    });
    if directional_train {
        metadata["directional_training_replicates"] = json!([4, 5]);
        metadata["directional_training_gradient_px_per_px"] = json!([-0.008, 0.008]);
        metadata["directional_training_display_phase"] = json!(0.4);
        metadata["holdout_replicates"] = json!(0);
    }
    std::fs::write(
        destination.with_extension("meta.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .expect("metadata file");
}

#[test]
#[ignore = "paired metrology study; run in release"]
fn center_metrology_matrix() {
    let full = std::env::var_os("CENTER_MATRIX_FULL").is_some();
    let directional_train = std::env::var_os("CENTER_MATRIX_DIRECTIONAL_TRAIN").is_some();
    let focus_band = std::env::var_os("CENTER_MATRIX_FOCUS_BAND").is_some();
    assert!(
        !(directional_train && focus_band),
        "select one opt-in study mode"
    );
    let filename = if focus_band {
        "synthetic_focus_band.jsonl"
    } else if directional_train {
        "synthetic_directional_train.jsonl"
    } else {
        "synthetic.jsonl"
    };
    let destination = std::env::var_os("CENTER_MATRIX_OUT").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/out/center_metrology")
                .join(filename)
        },
        PathBuf::from,
    );
    if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).expect("output directory");
    }
    if focus_band {
        focus_band_study(&destination);
        return;
    }
    let effects = conditions(full);
    let training = if full { 6 } else { 4 };
    let holdout = if full { 12 } else { 6 };
    let requested_threads = std::env::var("CENTER_MATRIX_THREADS")
        .ok()
        .and_then(|n| n.parse().ok());
    let threads = requested_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(6)
    });

    write_metadata(
        &destination,
        full,
        training,
        if directional_train { 0 } else { holdout },
        threads,
        directional_train,
    );
    let mut output = BufWriter::new(std::fs::File::create(&destination).expect("output file"));
    for coded in [true, false] {
        for &width_delta_mm in if full {
            &[0.0, 0.18, -0.18][..]
        } else {
            &[0.0, 0.18][..]
        } {
            let target = Target {
                coded,
                width_delta_mm,
            };
            let layout = target.layout(false);
            let board = InkBoard::new(&target.layout(true));
            if directional_train {
                directional_training(&mut output, target, &layout, &board, training, threads);
                continue;
            }
            let mut phases: Vec<_> = effects.iter().filter_map(|e| e.display_phase).collect();
            phases.sort_by(f64::total_cmp);
            phases.dedup();
            let mut boards = vec![board.clone()];
            for &phase in &phases {
                boards.push(board.displayed(phase));
            }

            let training_effects = [
                effects[0],
                effects
                    .iter()
                    .find(|e| e.name == "display_phase_b")
                    .copied()
                    .unwrap(),
            ];
            let mut samples = [Vec::new(), Vec::new()];
            for (medium, effect) in training_effects.into_iter().enumerate() {
                for capture in 0..training {
                    let board = &boards[board_index(effect.display_phase, &phases)];
                    let (rad, spec) = render_capture(board, &layout, effect, capture);
                    let scene = expose(&rad, &spec, noise_seed(capture, 0));
                    let eval = Evaluation {
                        target,
                        layout: &layout,
                        scene: &scene,
                        effect,
                        capture,
                        replicate: 0,
                        train: true,
                    };
                    let (rows, estimate) = evaluate(&eval, None);
                    samples[medium].extend(estimate);
                    for row in rows {
                        writeln!(output, "{row}").expect("write training row");
                    }
                }
            }
            let pools = samples.each_ref().map(|s| median_radii(s));
            println!(
                "{} {} training estimates: {:?}, pooled radii: {pools:?}",
                target.name(),
                target.revision(),
                samples.each_ref().map(Vec::len)
            );
            output.flush().expect("flush training rows");

            let jobs: Vec<_> = effects
                .iter()
                .flat_map(|&effect| {
                    (training..training + holdout).map(move |capture| (effect, capture))
                })
                .collect();
            let next = AtomicUsize::new(0);
            let output = Mutex::new(&mut output);
            let first_ok = OnceLock::new();
            std::thread::scope(|scope| {
                for _ in 0..threads.max(1) {
                    scope.spawn(|| {
                        loop {
                            let j = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(effect, capture)) = jobs.get(j) else {
                                break;
                            };
                            let board = &boards[board_index(effect.display_phase, &phases)];
                            let (rad, spec) = render_capture(board, &layout, effect, capture);
                            for replicate in 0..if full { 4 } else { 2 } {
                                let scene = expose(&rad, &spec, noise_seed(capture, replicate));
                                let eval = Evaluation {
                                    target,
                                    layout: &layout,
                                    scene: &scene,
                                    effect,
                                    capture,
                                    replicate,
                                    train: false,
                                };
                                let (rows, _) = evaluate(
                                    &eval,
                                    pools[usize::from(effect.display_phase.is_some())],
                                );
                                if rows.iter().any(|r| r["status"] == "ok") {
                                    first_ok.get_or_init(|| ());
                                }
                                let mut output = output.lock().expect("lock output");
                                for row in rows {
                                    writeln!(*output, "{row}").expect("write evaluation row");
                                }
                            }
                        }
                    });
                }
            });
            assert!(first_ok.get().is_some(), "no centers with verified labels");
            println!(
                "{} {} evaluated {} holdout scenes",
                target.name(),
                target.revision(),
                jobs.len()
            );
            output
                .into_inner()
                .expect("unlock output")
                .flush()
                .expect("flush holdout rows");
        }
    }
    println!("paired observations written to {}", destination.display());
}

#[test]
fn oblique_scatter_preserves_source_mass_and_centroid() {
    const SIDE: usize = 81;
    let sigma = |[x, y]: [f64; 2]| 0.9 + 0.010 * x + 0.019 * y;
    for center in [[29, 35], [51, 47]] {
        let mut pixels = vec![0.0; SIDE * SIDE];
        pixels[center[1] * SIDE + center[0]] = 1.0;
        blur_scene::scatter_gaussian_field(&mut pixels, SIDE, SIDE, sigma);
        let expected = sigma(center.map(|x| x as f64));
        let mass: f64 = pixels.iter().sum();
        let mut first = [0.0; 2];
        let mut second = [0.0; 2];
        for (i, weight) in pixels.into_iter().enumerate() {
            let xy = [i % SIDE, i / SIDE];
            for axis in 0..2 {
                let d = xy[axis] as f64 - center[axis] as f64;
                first[axis] += weight * d;
                second[axis] += weight * d * d;
            }
        }
        assert!((mass - 1.0).abs() < 1e-12);
        for (centroid, variance) in first.into_iter().zip(second) {
            assert!(centroid.abs() < 1e-12);
            assert!((variance.sqrt() - expected).abs() < 0.004);
        }
    }
}
