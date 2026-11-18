//! Does showing a target on a display, rather than printing it, add
//! photometric centre error?
//!
//! The ideal board pattern is rasterised onto a display pixel grid of pitch
//! `p` (aligned with the board axes, shifted by a fixed fraction of a pixel,
//! with a board pitch that is not a multiple of `p`), either binary (white
//! when the pattern at the pixel centre is) or as area coverage quantised to
//! 16 grey levels. The display pixels are then imaged as piecewise-constant
//! squares through the usual homography, camera blur, noise and 8-bit
//! pipeline (supersampled so display pixels are resolved).
//!
//! Per cell (target kind, display pitch in camera px, raster mode, blur,
//! tilt) over 3 board poses × several noise seeds:
//!
//! - `proj`, `phot`: centre RMS against the ideal (un-rasterised) centre,
//!   projective-centre and photometric methods, px;
//! - `repeat`: RMS over markers of the part of the photometric error that is
//!   the same for a marker across seeds and poses when expressed in board
//!   coordinates (mean board-frame error, scaled by the nominal px/mm, with
//!   the noise contribution to the mean removed); `rest`: the remainder of
//!   `phot` (`sqrt(phot^2 - repeat^2)`);
//! - `cen`: RMS of the directly computed offset of each rasterised marker's
//!   ink centroid from the ideal pattern's (no detector), px, over the same
//!   markers; `|r-c|`: RMS of the difference between the marker's `repeat`
//!   vector and its centroid offset;
//! - `sigma`: reported centre σ (`sqrt(trace(center_cov))`, RMS), px;
//!   `scat/s`: pose-wise noise scatter over seeds divided by `sigma`;
//!   `err/s`: `phot` over `sigma`;
//! - `ac1`: median lag-1 residual autocorrelation (mean of x and y).
//!
//! ```text
//! cargo test --release -p ringgrid --features study --test displayed_target -- --ignored --nocapture
//! ```
//!
//! `DISPLAY_SEEDS` (default 6) noise seeds per pose, `DISPLAY_THREADS`
//! (default 8), `DISPLAY_PITCHES` / `DISPLAY_BLURS` (comma-separated) restrict
//! the sweep.

mod support;

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ringgrid::{
    CenterSource, CircleRefinementMethod, DetectConfig, DetectedMarker, Detector, OriginDots,
    TargetLayout,
};
use support::blur_scene::{
    BoardRaster, DisplayRaster, DisplaySpec, SceneSpec, apply_h, expose, radiance,
};

const OUTER_RADIUS_PX: f64 = 17.0;
const CONTRAST: f64 = 80.0;
const BOARD_DPI: f32 = 1200.0;
const SIZE: [u32; 2] = [400, 300];
/// Step of the ink-centroid sampling grid, mm.
const CENTROID_STEP_MM: f64 = 0.015;

/// `(roll_deg, tilt_axis_deg, board_center_mm)` of the three board poses.
const POSES: [(f64, f64, [f64; 2]); 3] = [
    (7.0, 30.0, [0.0, 0.0]),
    (41.0, 100.0, [3.1, -2.3]),
    (-23.0, 160.0, [-4.7, 5.9]),
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    CodedHex,
    PlainRect,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::CodedHex => "coded_hex",
            Kind::PlainRect => "plain_rect",
        }
    }

    fn target(self) -> TargetLayout {
        match self {
            Kind::CodedHex => TargetLayout::coded_hex(8.0, 15, 14, 4.9565, 2.966, 0.839),
            Kind::PlainRect => TargetLayout::plain_rect(14.0, 12, 12, 5.6, 2.8, OriginDots::Auto),
        }
        .expect("valid target")
    }
}

/// One scored photometric centre.
struct Rec {
    marker: usize,
    pose: usize,
    /// Error vector in image px.
    e_px: [f64; 2],
    /// The same error mapped back to the board plane, mm.
    e_board: [f64; 2],
    /// `trace(center_cov)`, px².
    trace: f64,
    ac1: Option<f64>,
}

#[derive(Default)]
struct Job {
    recs: Vec<Rec>,
    /// Squared projective-centre errors, px².
    proj_sq: Vec<f64>,
}

struct Truth {
    centers: Vec<[f64; 2]>,
    spacing_px: f64,
}

impl Truth {
    fn marker_of(&self, m: &DetectedMarker, h: &[[f64; 3]; 3]) -> Option<usize> {
        let p = apply_h(h, m.board_xy_mm?);
        let (i, c) = self.centers.iter().enumerate().min_by(|a, b| {
            let da = (a.1[0] - p[0]).hypot(a.1[1] - p[1]);
            let db = (b.1[0] - p[0]).hypot(b.1[1] - p[1]);
            da.total_cmp(&db)
        })?;
        let near = (m.center[0] - c[0]).hypot(m.center[1] - c[1]) <= 0.4 * self.spacing_px;
        near.then_some(i)
    }
}

fn detector(target: &TargetLayout, method: CircleRefinementMethod) -> Detector {
    let mut config = DetectConfig::from_target_and_marker_diameter(
        target.clone(),
        (2.0 * OUTER_RADIUS_PX) as f32,
    );
    config.circle_refinement = method;
    Detector::with_config(config)
}

/// Inverse of the 2×2 board→image Jacobian of `h` at `xy`, applied to `e`.
fn to_board(h: &[[f64; 3]; 3], xy: [f64; 2], e: [f64; 2]) -> [f64; 2] {
    let d = 0.01;
    let col = |dx: f64, dy: f64| {
        let a = apply_h(h, [xy[0] + dx, xy[1] + dy]);
        let b = apply_h(h, [xy[0] - dx, xy[1] - dy]);
        [(a[0] - b[0]) / (2.0 * d), (a[1] - b[1]) / (2.0 * d)]
    };
    let (jx, jy) = (col(d, 0.0), col(0.0, d));
    let det = jx[0] * jy[1] - jy[0] * jx[1];
    [
        (jy[1] * e[0] - jy[0] * e[1]) / det,
        (-jx[1] * e[0] + jx[0] * e[1]) / det,
    ]
}

fn run_job(
    board: &BoardRaster,
    p_cam: Option<f64>,
    blur: f64,
    tilt: f64,
    pose: usize,
    n_seeds: u64,
) -> Job {
    let target = &board.target;
    let mut spec = SceneSpec::for_target(target, OUTER_RADIUS_PX, blur, tilt, CONTRAST, 1);
    spec.image_size = SIZE;
    let (roll, axis, center) = POSES[pose];
    spec.roll_deg = roll;
    spec.tilt_axis_deg = axis;
    spec.board_center_mm = center;
    if let Some(p) = p_cam {
        // At least ~4 samples per display pixel side.
        spec.supersample = ((4.0 / p).ceil() as usize).clamp(4, 20);
    }
    let rad = radiance(board, &spec);
    let truth = Truth {
        centers: target
            .cells()
            .iter()
            .map(|c| apply_h(&rad.h, [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])]))
            .collect(),
        spacing_px: f64::from(target.min_center_spacing_mm()) * spec.px_per_mm,
    };
    let photometric = detector(target, CircleRefinementMethod::Photometric);
    let projective = detector(target, CircleRefinementMethod::ProjectiveCenter);
    let mut job = Job::default();
    for seed in 0..n_seeds {
        let scene = expose(&rad, &spec, 1000 + seed + 100 * pose as u64);
        let res = projective.detect(&scene.image).expect("detect");
        for m in &res.detected_markers {
            if let Some(i) = truth.marker_of(m, &rad.h) {
                let t = truth.centers[i];
                job.proj_sq
                    .push((m.center[0] - t[0]).powi(2) + (m.center[1] - t[1]).powi(2));
            }
        }
        let (res, _, residuals) = photometric
            .detect_with_photometric_residuals(&scene.image)
            .expect("detect");
        // Lag-1 residual autocorrelation per fitted marker.
        let ac1: HashMap<usize, f64> = residuals
            .iter()
            .map(|r| {
                let a = &r.residual_autocorrelation;
                (r.marker_index, 0.5 * (a.x[0] + a.y[0]))
            })
            .collect();
        for (index, m) in res.detected_markers.iter().enumerate() {
            let Some(i) = truth.marker_of(m, &rad.h) else {
                continue;
            };
            if m.center_source != CenterSource::Photometric {
                continue;
            }
            let Some(cov) = m.center_cov else { continue };
            let t = truth.centers[i];
            let e = [m.center[0] - t[0], m.center[1] - t[1]];
            let c = &target.cells()[i];
            let xy = [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])];
            let ac1 = ac1.get(&index).copied();
            job.recs.push(Rec {
                marker: i,
                pose,
                e_px: e,
                e_board: to_board(&rad.h, xy, e),
                trace: cov[0][0] + cov[1][1],
                ac1,
            });
        }
    }
    job
}

/// Offset (mm, board frame) of the displayed pattern's ink centroid from the
/// ideal pattern's, over a disc around the marker cell.
fn ink_centroid_offset(board: &BoardRaster, center: [f64; 2], radius_mm: f64) -> [f64; 2] {
    let n = (radius_mm / CENTROID_STEP_MM).ceil() as i64;
    let (mut wd, mut wi) = ([0.0; 3], [0.0; 3]);
    for j in -n..=n {
        for i in -n..=n {
            let (dx, dy) = (i as f64 * CENTROID_STEP_MM, j as f64 * CENTROID_STEP_MM);
            if dx.hypot(dy) > radius_mm {
                continue;
            }
            let xy = [center[0] + dx, center[1] + dy];
            let (ink_d, ink_i) = (
                1.0 - board.reflectance(xy),
                1.0 - board.ideal_reflectance(xy),
            );
            wd[0] += ink_d;
            wd[1] += ink_d * dx;
            wd[2] += ink_d * dy;
            wi[0] += ink_i;
            wi[1] += ink_i * dx;
            wi[2] += ink_i * dy;
        }
    }
    [wd[1] / wd[0] - wi[1] / wi[0], wd[2] / wd[0] - wi[2] / wi[0]]
}

#[derive(Default)]
struct Row {
    n_markers: usize,
    proj: f64,
    phot: f64,
    repeat: f64,
    rest: f64,
    cen: Option<f64>,
    diff: Option<f64>,
    sigma: f64,
    scatter_ratio: f64,
    err_ratio: f64,
    ac1: f64,
}

fn aggregate(
    jobs: &[Job],
    scale: f64,
    centroids: Option<&HashMap<usize, [f64; 2]>>,
    min_samples: usize,
) -> Row {
    let proj_sq: Vec<f64> = jobs
        .iter()
        .flat_map(|j| j.proj_sq.iter().copied())
        .collect();
    let recs: Vec<&Rec> = jobs.iter().flat_map(|j| j.recs.iter()).collect();
    let mut by_marker: BTreeMap<usize, Vec<&Rec>> = BTreeMap::new();
    for r in &recs {
        by_marker.entry(r.marker).or_default().push(r);
    }
    by_marker.retain(|_, v| v.len() >= min_samples);
    let (mut tot, mut rep, mut cen, mut diff, mut n_tot, mut n_m) = (0.0, 0.0, 0.0, 0.0, 0, 0);
    for (idx, v) in &by_marker {
        let n = v.len() as f64;
        let mean = [
            v.iter().map(|r| r.e_board[0]).sum::<f64>() / n,
            v.iter().map(|r| r.e_board[1]).sum::<f64>() / n,
        ];
        let within = v
            .iter()
            .map(|r| (r.e_board[0] - mean[0]).powi(2) + (r.e_board[1] - mean[1]).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        tot += v
            .iter()
            .map(|r| r.e_px[0].powi(2) + r.e_px[1].powi(2))
            .sum::<f64>();
        n_tot += v.len();
        rep += scale * scale * (mean[0].powi(2) + mean[1].powi(2) - within / n);
        n_m += 1;
        if let Some(c) = centroids.and_then(|c| c.get(idx)) {
            cen += scale * scale * (c[0].powi(2) + c[1].powi(2));
            diff += scale * scale * ((mean[0] - c[0]).powi(2) + (mean[1] - c[1]).powi(2));
        }
    }
    let phot_sq = tot / n_tot.max(1) as f64;
    let repeat_sq = (rep / n_m.max(1) as f64).max(0.0);
    // Pose-wise scatter over seeds.
    let mut by_pose: BTreeMap<(usize, usize), Vec<&Rec>> = BTreeMap::new();
    for r in &recs {
        by_pose.entry((r.marker, r.pose)).or_default().push(r);
    }
    let (mut var_sum, mut n_g) = (0.0, 0);
    for v in by_pose.values().filter(|v| v.len() >= 3) {
        let n = v.len() as f64;
        let m = [
            v.iter().map(|r| r.e_px[0]).sum::<f64>() / n,
            v.iter().map(|r| r.e_px[1]).sum::<f64>() / n,
        ];
        var_sum += v
            .iter()
            .map(|r| (r.e_px[0] - m[0]).powi(2) + (r.e_px[1] - m[1]).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        n_g += 1;
    }
    let sigma = (recs.iter().map(|r| r.trace).sum::<f64>() / recs.len().max(1) as f64).sqrt();
    let mut ac: Vec<f64> = recs.iter().filter_map(|r| r.ac1).collect();
    ac.sort_by(|a, b| a.total_cmp(b));
    Row {
        n_markers: n_m,
        proj: (proj_sq.iter().sum::<f64>() / proj_sq.len().max(1) as f64).sqrt(),
        phot: phot_sq.sqrt(),
        repeat: repeat_sq.sqrt(),
        rest: (phot_sq - repeat_sq).max(0.0).sqrt(),
        cen: centroids.map(|_| (cen / n_m.max(1) as f64).sqrt()),
        diff: centroids.map(|_| (diff / n_m.max(1) as f64).sqrt()),
        sigma,
        scatter_ratio: (var_sum / n_g.max(1) as f64).sqrt() / sigma,
        err_ratio: phot_sq.sqrt() / sigma,
        ac1: ac.get(ac.len() / 2).copied().unwrap_or(f64::NAN),
    }
}

fn env_list(key: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(key)
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

/// Print `-` for a missing value.
fn opt(v: Option<f64>) -> String {
    v.map_or("    -".into(), |v| format!("{v:5.3}"))
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn displayed_target_centre_error() {
    let n_seeds: u64 = std::env::var("DISPLAY_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let threads: usize = std::env::var("DISPLAY_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let pitches = env_list("DISPLAY_PITCHES", &[0.2, 0.3, 0.4, 0.6, 0.8]);
    let blurs = env_list("DISPLAY_BLURS", &[0.5, 0.8, 1.2, 2.0, 3.0]);
    let tilts = [0.0, 20.0];
    println!(
        "kind        pitch raster  blur tilt  nmk  proj  phot repeat  rest   cen |r-c| sigma scat/s err/s   ac1"
    );
    for kind in [Kind::CodedHex, Kind::PlainRect] {
        let ideal = BoardRaster::with_dpi(kind.target(), BOARD_DPI);
        let r_mm = f64::from(ideal.target.outer_draw_radius_mm());
        let scale = OUTER_RADIUS_PX / r_mm;
        let mut groups: Vec<Option<(f64, DisplayRaster)>> = vec![None];
        for &p in &pitches {
            groups.push(Some((p, DisplayRaster::Binary)));
            groups.push(Some((p, DisplayRaster::Grey16)));
        }
        for group in groups {
            let board = match group {
                None => ideal.clone(),
                Some((p, raster)) => ideal.displayed(DisplaySpec {
                    pitch_mm: p / scale,
                    raster,
                }),
            };
            let p_cam = group.map(|g| g.0);
            let cells: Vec<(f64, f64)> = blurs
                .iter()
                .flat_map(|&b| tilts.iter().map(move |&t| (b, t)))
                .collect();
            let jobs: Vec<(usize, usize)> = (0..cells.len())
                .flat_map(|c| (0..POSES.len()).map(move |p| (c, p)))
                .collect();
            let results: Mutex<Vec<Option<Job>>> = Mutex::new(jobs.iter().map(|_| None).collect());
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..threads.max(1) {
                    scope.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(c, pose)) = jobs.get(i) else { break };
                            let (blur, tilt) = cells[c];
                            let job = run_job(&board, p_cam, blur, tilt, pose, n_seeds);
                            results.lock().unwrap()[i] = Some(job);
                        }
                    });
                }
            });
            let results: Vec<Job> = results
                .into_inner()
                .unwrap()
                .into_iter()
                .map(|j| j.expect("job ran"))
                .collect();
            // Ink centroid offsets of every marker seen, once per group.
            let mut centroids: HashMap<usize, [f64; 2]> = HashMap::new();
            if group.is_some() {
                let window = r_mm + 0.5;
                for j in &results {
                    for r in &j.recs {
                        centroids.entry(r.marker).or_insert_with(|| {
                            let c = &board.target.cells()[r.marker];
                            ink_centroid_offset(
                                &board,
                                [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])],
                                window,
                            )
                        });
                    }
                }
            }
            for (c, &(blur, tilt)) in cells.iter().enumerate() {
                let cell_jobs = &results[c * POSES.len()..(c + 1) * POSES.len()];
                let row = aggregate(
                    cell_jobs,
                    scale,
                    group.map(|_| &centroids),
                    (n_seeds as usize * POSES.len()).saturating_sub(2).max(2),
                );
                let (pitch, mode) = match group {
                    None => ("print".to_string(), "-"),
                    Some((p, DisplayRaster::Binary)) => (format!("{p:.1}"), "binary"),
                    Some((p, DisplayRaster::Grey16)) => (format!("{p:.1}"), "grey16"),
                };
                println!(
                    "{:<11} {:>5} {:<6} {:>5.1} {:>4.0} {:>4} {:5.3} {:5.3} {:6.3} {:5.3} {} {} {:5.3} {:6.2} {:5.2} {:5.2}",
                    kind.name(),
                    pitch,
                    mode,
                    blur,
                    tilt,
                    row.n_markers,
                    row.proj,
                    row.phot,
                    row.repeat,
                    row.rest,
                    opt(row.cen),
                    opt(row.diff),
                    row.sigma,
                    row.scatter_ratio,
                    row.err_ratio,
                    row.ac1,
                );
            }
        }
    }
}

#[test]
fn fitted_screen_target_preserves_ids_and_centers_across_display_phase() {
    use ringgrid::ScreenTargetOptions;

    let template = TargetLayout::default_hex();
    let pitch_mm = 25.4 / 150.0;
    let mut by_phase = Vec::new();
    for [width, height] in [[460, 520], [461, 521]] {
        let artifacts = template
            .render_fitted_screen_target(&ScreenTargetOptions::new(width, height, pitch_mm, 10.752))
            .unwrap();
        let board = BoardRaster::from_screen_artifacts(&artifacts);
        let target = &board.target;
        let mut spec = SceneSpec::for_target(target, OUTER_RADIUS_PX, 0.8, 0.0, CONTRAST, 1);
        spec.image_size = [440, 440];
        spec.supersample = 12;
        spec.roll_deg = 7.0;
        let scene = expose(&radiance(&board, &spec), &spec, 1000);
        let result = detector(target, CircleRefinementMethod::Photometric)
            .detect(&scene.image)
            .unwrap();
        let mut errors = BTreeMap::new();
        for marker in result.detected_markers {
            let id = marker.id.expect("absolute board id");
            let xy = target
                .xy_mm_of_id(id)
                .expect("id present on the exported target");
            let want = scene.project(xy.map(f64::from));
            let error = [marker.center[0] - want[0], marker.center[1] - want[1]];
            assert!(
                error[0].hypot(error[1]) < 1.0,
                "id {id} has wrong center: {error:?}"
            );
            assert!(errors.insert(id, error).is_none());
        }
        assert!(
            errors.len() * 10 >= target.n_cells() * 9,
            "{}/{} markers",
            errors.len(),
            target.n_cells()
        );
        by_phase.push(errors);
    }
    let differences: Vec<f64> = by_phase[0]
        .iter()
        .filter_map(|(id, a)| {
            by_phase[1]
                .get(id)
                .map(|b| (a[0] - b[0]).hypot(a[1] - b[1]))
        })
        .collect();
    assert!(differences.len() >= 20);
    let rms = (differences.iter().map(|d| d * d).sum::<f64>() / differences.len() as f64).sqrt();
    assert!(rms < 0.10, "phase changes centers by {rms} camera px RMS");
}
