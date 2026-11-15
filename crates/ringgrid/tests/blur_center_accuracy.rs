//! Centre accuracy under defocus blur, against exact synthetic ground truth.
//!
//! The scenes model a tilted-sensor capture: ≈33 px markers, 40 grey levels of
//! contrast, σ = 2 grey-level noise, an illumination gradient and a saturated
//! glare blob (see `support/blur_scene.rs`).
//!
//! `cargo test --release -p ringgrid --test blur_center_accuracy -- --ignored --nocapture`
//! prints the full benchmark table.

mod support;

use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};
use support::blur_scene::{BoardRaster, CenterStats, SceneSpec, render, score};

const BLURS: [f64; 5] = [0.8, 1.5, 2.5, 3.0, 4.0];

/// A print like the real boards this was measured on: the nominal default
/// board's outer and hole edges, but 0.84 mm strokes, so the code band is
/// 1.15 mm wide instead of 0.45 mm and stays decodable under heavy blur. The
/// detector is still configured with the nominal default board.
fn wide_code_print() -> TargetLayout {
    TargetLayout::coded_hex(8.0, 15, 14, 4.9565, 2.966, 0.839).expect("valid print")
}
const TILTS: [f64; 3] = [0.0, 20.0, 30.0];

fn detector(method: CircleRefinementMethod) -> Detector {
    let mut config = DetectConfig::from_target(TargetLayout::default_hex());
    config.circle_refinement = method;
    Detector::with_config(config)
}

fn method_name(method: CircleRefinementMethod) -> &'static str {
    match method {
        CircleRefinementMethod::None => "none",
        CircleRefinementMethod::ProjectiveCenter => "projective-center",
        CircleRefinementMethod::Photometric => "photometric",
    }
}

fn contrast() -> f64 {
    std::env::var("BLUR_BENCH_CONTRAST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40.0)
}

fn run_cell(
    board: &BoardRaster,
    det: &Detector,
    blur: f64,
    seeds: std::ops::Range<u64>,
    match_undecoded: bool,
) -> CenterStats {
    let mut stats = CenterStats::default();
    for seed in seeds {
        for (k, &tilt) in TILTS.iter().enumerate() {
            let mut spec = SceneSpec::with_contrast(blur, tilt, seed * 31 + k as u64, contrast());
            spec.tilt_axis_deg += 50.0 * seed as f64;
            let scene = render(board, &spec);
            let result = det.detect_adaptive(&scene.image).expect("detect");
            stats.merge(&score(
                &scene,
                &board.target,
                &result.detected_markers,
                match_undecoded,
            ));
        }
    }
    stats
}

fn print_row(label: &str, blur: f64, s: &CenterStats) {
    use ringgrid::CenterSource;
    let mut parts = Vec::new();
    for src in [
        CenterSource::OuterEllipse,
        CenterSource::ProjectiveCenter,
        CenterSource::Photometric,
    ] {
        let e: Vec<f64> = s
            .by_source
            .iter()
            .filter(|(k, _)| *k == src)
            .map(|(_, e)| *e)
            .collect();
        if !e.is_empty() {
            let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
            parts.push(format!("{src:?}:{}@{rms:.3}", e.len()));
        }
    }
    if !s.with_cov.is_empty() {
        // Error scatter versus reported covariance: sqrt(E[e^T e] / E[tr C]).
        let n = s.with_cov.len() as f64;
        let e2: f64 = s
            .with_cov
            .iter()
            .map(|(_, e)| e[0] * e[0] + e[1] * e[1])
            .sum::<f64>()
            / n;
        let tr: f64 = s
            .with_cov
            .iter()
            .map(|(c, _)| c[0][0] + c[1][1])
            .sum::<f64>()
            / n;
        parts.push(format!("err/reported-sigma {:.2}", (e2 / tr).sqrt()));
    }
    println!("    sources {}", parts.join(" "));
    println!(
        "{label:<22} {blur:>4.1}  {:>5}/{:<5} {:>3}  {:>7.4} {:>7.4} {:>7.4}",
        s.n_decoded,
        s.n_visible,
        s.n_wrong_id,
        s.rms(),
        s.quantile(0.5),
        s.quantile(0.95)
    );
}

/// One acceptance scene: centre errors of the photometric detector and the
/// ids each method decoded.
fn acceptance_case(
    board: &BoardRaster,
    blur: f64,
    contrast: f64,
    tilt: f64,
) -> (CenterStats, Vec<usize>, Vec<usize>) {
    let spec = SceneSpec::with_contrast(blur, tilt, 7, contrast);
    let scene = render(board, &spec);
    let mut config =
        DetectConfig::from_target_and_marker_diameter(TargetLayout::default_hex(), 37.0);
    config.circle_refinement = CircleRefinementMethod::ProjectiveCenter;
    let baseline = Detector::with_config(config.clone())
        .detect(&scene.image)
        .unwrap();
    config.circle_refinement = CircleRefinementMethod::Photometric;
    let refined = Detector::with_config(config).detect(&scene.image).unwrap();
    let ids = |r: &ringgrid::DetectionResult| {
        let mut v: Vec<usize> = r.detected_markers.iter().filter_map(|m| m.id).collect();
        v.sort_unstable();
        v
    };
    let stats = score(&scene, &board.target, &refined.detected_markers, false);
    (stats, ids(&baseline), ids(&refined))
}

/// Acceptance: on 33 px markers under a 20° tilt, an illumination gradient, a
/// saturated glare blob and σ = 2 noise, photometric centres reach ≤ 0.03 px
/// RMS at blur 0.8 (40 grey levels of contrast) and ≤ 0.05 px RMS at blur 3.0
/// (80 grey levels: at 40 the decoder loses blur-3 markers for every centre
/// method), and report every marker the default method reports (and those
/// whose outer-only centre the default method cannot report: photometry
/// corrects them). Debug
/// builds take about half a minute.
#[test]
fn photometric_centres_meet_accuracy_targets() {
    let board = BoardRaster::new(wide_code_print());
    for (blur, contrast, max_rms) in [(0.8, 40.0, 0.03), (3.0, 80.0, 0.05)] {
        let (stats, ids_baseline, ids_refined) = acceptance_case(&board, blur, contrast, 20.0);
        let photometric = stats
            .by_source
            .iter()
            .filter(|(s, _)| *s == ringgrid::CenterSource::Photometric)
            .count();
        eprintln!(
            "blur {blur}: {} decoded of {} visible, {photometric} photometric, rms {:.4} px, p95 {:.4} px",
            stats.n_decoded,
            stats.n_visible,
            stats.rms(),
            stats.quantile(0.95)
        );
        assert!(
            ids_baseline.iter().all(|id| ids_refined.contains(id)),
            "refinement must keep every marker the default method reports"
        );
        assert_eq!(stats.n_wrong_id, 0);
        assert!(
            photometric * 20 >= stats.n_decoded * 19,
            "photometric coverage"
        );
        assert!(
            stats.rms() <= max_rms,
            "blur {blur}: rms {} px",
            stats.rms()
        );
    }
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn blur_center_benchmark() {
    let print = match std::env::var("BLUR_BENCH_PRINT").as_deref() {
        Ok("nominal") => TargetLayout::default_hex(),
        _ => wide_code_print(),
    };
    let board = BoardRaster::new(print);
    println!("method                 blur  decoded/visible wrong  rms     median  p95   (px)");
    for method in [
        CircleRefinementMethod::None,
        CircleRefinementMethod::ProjectiveCenter,
        CircleRefinementMethod::Photometric,
    ] {
        let det = detector(method);
        for blur in BLURS {
            let s = run_cell(&board, &det, blur, 0..2, false);
            print_row(method_name(method), blur, &s);
        }
    }
}

/// Diagnostic: the same scenes with every code sector painted black. Markers
/// cannot be decoded, so detections are matched to the nearest true centre.
#[test]
#[ignore = "diagnostic; run in release with --ignored --nocapture"]
fn blur_center_without_code_band() {
    let with_code = BoardRaster::new(TargetLayout::default_hex());
    let without_code = BoardRaster::with_code_band(TargetLayout::default_hex(), false);
    let mut config = DetectConfig::from_target(TargetLayout::default_hex());
    config.advanced.use_global_filter = false;
    config.advanced.completion.enable = false;
    for method in [
        CircleRefinementMethod::None,
        CircleRefinementMethod::ProjectiveCenter,
    ] {
        config.circle_refinement = method;
        let det = Detector::with_config(config.clone());
        for blur in BLURS {
            let a = run_cell(&with_code, &det, blur, 0..1, true);
            let b = run_cell(&without_code, &det, blur, 0..1, true);
            print_row(&format!("{} code", method_name(method)), blur, &a);
            print_row(&format!("{} solid", method_name(method)), blur, &b);
        }
    }
}

/// Write one acceptance scene to `$BLUR_SCENE_DUMP` (blur from `$BLUR_SCENE_SIGMA`).
#[test]
#[ignore = "manual inspection helper"]
fn dump_scene() {
    let Ok(path) = std::env::var("BLUR_SCENE_DUMP") else {
        return;
    };
    let blur: f64 = std::env::var("BLUR_SCENE_SIGMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let board = BoardRaster::new(TargetLayout::default_hex());
    let scene = render(&board, &SceneSpec::acceptance(blur, 30.0, 1));
    scene.image.save(path).expect("write scene");
}

/// List the worst photometric centres of one scene (debug helper).
#[test]
#[ignore = "debug helper"]
fn worst_photometric_markers() {
    let blur: f64 = std::env::var("BLUR_SCENE_SIGMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8);
    let board = BoardRaster::new(wide_code_print());
    let det = detector(CircleRefinementMethod::Photometric);
    for (k, &tilt) in TILTS.iter().enumerate() {
        let spec = SceneSpec::with_contrast(blur, tilt, k as u64, contrast());
        let scene = render(&board, &spec);
        let (res, diag) = det.detect_adaptive_with_diagnostics(&scene.image).unwrap();
        let mut rows: Vec<_> = res
            .detected_markers
            .iter()
            .zip(&diag.markers)
            .filter_map(|(m, d)| {
                let gt = scene.project(m.board_xy_mm?);
                let e = [m.center[0] - gt[0], m.center[1] - gt[1]];
                Some((
                    e[0].hypot(e[1]),
                    e,
                    m.center,
                    m.center_cov,
                    d.photometric.clone(),
                ))
            })
            .collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut rs: Vec<(f64, f64)> = rows
            .iter()
            .filter_map(|r| Some((r.4.as_ref()?.residual_sigma?, r.0)))
            .collect();
        rs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let q = |f: f64| rs[((rs.len() - 1) as f64 * f) as usize].0;
        println!(
            "tilt {tilt}: residual sigma q10 {:.2} q50 {:.2} q90 {:.2} q98 {:.2}",
            q(0.1),
            q(0.5),
            q(0.9),
            q(0.98)
        );
        println!("tilt {tilt}: radii {:?}", diag.photometric_print_radii_mm);
        let mut counts = std::collections::BTreeMap::new();
        for d in &diag.markers {
            let k = d
                .photometric
                .as_ref()
                .map(|p| format!("{:?}", p.status))
                .unwrap_or("none".into());
            *counts.entry(k).or_insert(0) += 1;
        }
        println!("  statuses {counts:?}");
        let ratio = |sel: &dyn Fn(f64) -> bool| {
            let v: Vec<(f64, f64)> = rows
                .iter()
                .filter_map(|r| {
                    let c = r.3?;
                    let rs = r.4.as_ref()?.residual_sigma?;
                    sel(rs).then_some((r.0 * r.0, c[0][0] + c[1][1]))
                })
                .collect();
            let n = v.len().max(1) as f64;
            (
                v.len(),
                (v.iter().map(|x| x.0).sum::<f64>() / n).sqrt(),
                (v.iter().map(|x| x.1).sum::<f64>() / n).sqrt(),
            )
        };
        println!(
            "  clean (resid<2.3) n/err/reported {:?}; others {:?}",
            ratio(&|r| r < 2.3),
            ratio(&|r| r >= 2.3)
        );
        for r in rows
            .iter()
            .filter(|r| {
                r.4.as_ref()
                    .is_some_and(|p| p.status != ringgrid::diagnostics::PhotometricStatus::Accepted)
            })
            .take(4)
        {
            println!(
                "  rejected err {:.3} at ({:.0},{:.0}) {:?}",
                r.0, r.2[0], r.2[1], r.4
            );
        }
        for r in rows.iter().take(6) {
            println!(
                "  err {:.3} {:?} at ({:.0},{:.0}) cov {:?} {:?}",
                r.0, r.1, r.2[0], r.2[1], r.3, r.4
            );
        }
    }
}

/// Centres of one marker across noise seeds, and its summed reported covariance.
type PerMarker = (Vec<[f64; 2]>, [[f64; 2]; 2]);

/// Scene-level Monte-Carlo: the same frame rendered with independent noise
/// seeds; the scatter of each marker's centre across seeds is compared with
/// its mean reported covariance.
#[test]
#[ignore = "Monte-Carlo benchmark; run in release with --ignored --nocapture"]
fn scene_covariance_monte_carlo() {
    use std::collections::BTreeMap;
    let board = BoardRaster::new(wide_code_print());
    let mut config =
        DetectConfig::from_target_and_marker_diameter(TargetLayout::default_hex(), 37.0);
    config.circle_refinement = CircleRefinementMethod::Photometric;
    let det = Detector::with_config(config);
    for (blur, contrast) in [(0.8, 40.0), (1.5, 40.0), (3.0, 80.0)] {
        let n_seeds = 30u64;
        // id -> (centres, summed covariance)
        let mut per_id: BTreeMap<usize, PerMarker> = BTreeMap::new();
        for seed in 0..n_seeds {
            let mut spec = SceneSpec::with_contrast(blur, 20.0, 1, contrast);
            spec.seed = 1000 + seed;
            let scene = render(&board, &spec);
            for m in det.detect(&scene.image).unwrap().detected_markers {
                let (Some(id), Some(cov)) = (m.id, m.center_cov) else {
                    continue;
                };
                let e = per_id.entry(id).or_insert((Vec::new(), [[0.0; 2]; 2]));
                e.0.push(m.center);
                for (acc_row, cov_row) in e.1.iter_mut().zip(cov) {
                    for (a, c) in acc_row.iter_mut().zip(cov_row) {
                        *a += c;
                    }
                }
            }
        }
        let (mut emp, mut rep, mut n_markers) = (0.0, 0.0, 0usize);
        let mut ratios = Vec::new();
        for (centres, cov_sum) in per_id.values() {
            let n = centres.len();
            if n + 2 < n_seeds as usize {
                continue;
            }
            let m = [
                centres.iter().map(|c| c[0]).sum::<f64>() / n as f64,
                centres.iter().map(|c| c[1]).sum::<f64>() / n as f64,
            ];
            let var: f64 = centres
                .iter()
                .map(|c| (c[0] - m[0]).powi(2) + (c[1] - m[1]).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;
            let r = (cov_sum[0][0] + cov_sum[1][1]) / n as f64;
            emp += var;
            rep += r;
            ratios.push((r / var).sqrt());
            n_markers += 1;
        }
        ratios.sort_by(|a, b| a.total_cmp(b));
        println!(
            "blur {blur} contrast {contrast}: {n_markers} markers, empirical rms {:.4} px, reported rms {:.4} px, pooled reported/empirical {:.3}, per-marker ratio p10 {:.2} p50 {:.2} p90 {:.2}",
            (emp / n_markers as f64).sqrt(),
            (rep / n_markers as f64).sqrt(),
            (rep / emp).sqrt(),
            ratios[ratios.len() / 10],
            ratios[ratios.len() / 2],
            ratios[ratios.len() * 9 / 10],
        );
    }
}
