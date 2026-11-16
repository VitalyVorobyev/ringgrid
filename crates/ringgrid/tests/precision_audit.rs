//! Precision audit: hard synthetic views of every target kind, scored against
//! ground truth for wrong IDs, wrong board positions, wrong centres and wrong
//! homographies (see `support/precision.rs` for the definitions).
//!
//! - `precision_fast_subset` (default run, ~30 s in a debug build) asserts
//!   zero wrong outputs and recall floors on six scenes that failed before.
//! - `precision_full_sweep` (`--ignored`, ~2 min in release) runs the whole
//!   catalogue plus the operating-range grid and asserts the same contract
//!   with per-group recall floors:
//!   `cargo test --release -p ringgrid --test precision_audit -- --ignored precision_full_sweep --nocapture`
//! - Cross-build comparison: `export_precision_scenes` writes every scene
//!   (`image.png`, `target.json`, `gt.json`, and a `manifest.tsv` of
//!   `name  dmin  dmax`) under `$RINGGRID_PRECISION_DIR`; run any detector
//!   build's CLI over it writing `<scene>/det_<run>.json`, then
//!   `score_precision_runs` with `RINGGRID_PRECISION_RUNS=old,new` prints the
//!   tables side by side.

#[path = "support/precision.rs"]
mod precision;
mod support;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use image::GrayImage;
use precision::{Boards, Case, Score, Truth, catalogue, render_case, score, summary};
use ringgrid::{Detector, MarkerScalePrior};
use serde_json::Value;

fn detect_json(case: &Case, image: &GrayImage) -> Value {
    let result = case_detector(case).detect(image).expect("detect");
    serde_json::to_value(&result).expect("serialize result")
}

/// Per-(board, group) aggregate table plus every scene with a wrong output.
fn report(rows: &[(Truth, Score)]) -> String {
    let mut out = String::new();
    let mut groups: BTreeMap<(String, String), (usize, Score)> = BTreeMap::new();
    for (t, s) in rows {
        let e = groups
            .entry((t.board.clone(), t.group.clone()))
            .or_default();
        e.0 += 1;
        e.1.add(s);
    }
    for ((board, group), (n, s)) in &groups {
        out += &format!("{board:10} {group:10} n={n:2} {}\n", summary(s));
    }
    out += "scenes with wrong outputs:\n";
    for (t, s) in rows.iter().filter(|(_, s)| s.n_wrong() > 0) {
        out += &format!("  {:28} frame {:18} {}\n", t.name, s.frame, summary(s));
    }
    out
}

fn run_cases(cases: &[Case]) -> Vec<(Truth, Score)> {
    let mut boards = Boards::default();
    cases
        .iter()
        .map(|case| {
            let (image, truth) = render_case(case, &mut boards);
            let det = detect_json(case, &image);
            let s = score(&truth, &det);
            (truth, s)
        })
        .collect()
}

/// Wrong outputs the current detector still reports, per scene (a plain view
/// of six isolated rings labeled as neighbours). Every other single-board
/// scene must report none.
const KNOWN_WRONG: &[(&str, usize)] = &[("plainhex-few-spread-6", 5)];

/// Condition group of the scenes with a second board in view. One board per
/// image is the supported case: these scenes are scored and reported (the
/// table prints their wrong outputs) but do not gate.
const UNSUPPORTED_GROUP: &str = "foreign";

/// Measured recall floors per (board, condition group), operating range:
/// the full sweep's recall minus 0.02 (default `ProjectiveCenter`
/// refinement), so a later change cannot trade precision for recall, or
/// collapse recall, silently. Plain boards count only absolutely anchored
/// markers (their dots must be in view); `oor` scenes have no recall target.
const RECALL_FLOORS: &[(&str, &str, f64)] = &[
    ("hex", "blur", 0.70),
    ("hex", "contrast", 0.54),
    ("hex", "dark", 0.50),
    ("hex", "distortion", 0.90),
    ("hex", "grid", 0.64),
    ("hex", "occlusion", 0.96),
    ("hex", "partial", 0.93),
    ("hex", "tilt", 0.33),
    ("hexext", "blur", 0.70),
    ("hexext", "contrast", 0.47),
    ("hexext", "dark", 0.47),
    ("hexext", "distortion", 0.78),
    ("hexext", "grid", 0.67),
    ("hexext", "occlusion", 0.95),
    ("hexext", "partial", 0.93),
    ("hexext", "tilt", 0.36),
    ("plainhex", "blur", 0.41),
    ("plainhex", "dark", 0.16),
    ("plainhex", "distortion", 0.71),
    ("plainhex", "grid", 0.38),
    ("plainhex", "partial", 0.59),
    ("plainhex", "tilt", 0.26),
    ("plainrect", "blur", 0.43),
    ("plainrect", "dark", 0.15),
    ("plainrect", "distortion", 0.84),
    ("plainrect", "grid", 0.46),
    ("plainrect", "tilt", 0.24),
    ("rect", "blur", 0.67),
    ("rect", "contrast", 0.46),
    ("rect", "dark", 0.47),
    ("rect", "distortion", 0.84),
    ("rect", "few", 0.29),
    ("rect", "grid", 0.67),
    ("rect", "occlusion", 0.92),
    ("rect", "partial", 0.91),
    ("rect", "tilt", 0.37),
];

/// Assert the precision contract on scored scenes: no wrong ID, board
/// position, relative label or homography beyond [`KNOWN_WRONG`], and the
/// group recall floors.
fn assert_precision(rows: &[(Truth, Score)], floors: &[(&str, &str, f64)]) {
    for (t, s) in rows.iter().filter(|(t, _)| t.group != UNSUPPORTED_GROUP) {
        let allowed = KNOWN_WRONG
            .iter()
            .find(|(n, _)| *n == t.name)
            .map_or(0, |(_, k)| *k);
        assert!(
            s.n_wrong() <= allowed,
            "{}: {} wrong outputs (allowed {allowed}): {}",
            t.name,
            s.n_wrong(),
            summary(s)
        );
    }
    let mut groups: BTreeMap<(&str, &str), Score> = BTreeMap::new();
    for (t, s) in rows {
        groups.entry((&t.board, &t.group)).or_default().add(s);
    }
    for (board, group, floor) in floors {
        if let Some(s) = groups.get(&(*board, *group)) {
            assert!(
                s.recall() >= *floor,
                "{board} {group}: recall {:.3} below floor {floor}",
                s.recall()
            );
        }
    }
}

/// Scenes that failed on 0.15.0 or the first grid-first code, one per
/// mechanism (raw decodes without a homography, a fallback homography of
/// ID-correction-recovered IDs, completion along an extrapolated
/// homography, a second board in view, an isolated decode, a decisive
/// minority anchor), with their measured recall floors.
const FAST_SUBSET: &[(&str, f64)] = &[
    ("rect-grid-d20-b2-t30-s0", 0.52),
    ("hex-grid-d20-b2-t30-s1", 0.0),
    ("rect-tilt-t65-d32", 0.05),
    ("hex-foreign-same", 0.95),
    ("hex-few-spread-3", 0.0),
    ("rect-grid-d20-b1.5-t50-s1", 0.48),
];

#[test]
fn precision_fast_subset() {
    let all: Vec<Case> = catalogue().into_iter().chain(explore_cases()).collect();
    let cases: Vec<Case> = FAST_SUBSET
        .iter()
        .map(|(name, _)| {
            all.iter()
                .find(|c| c.name == *name)
                .unwrap_or_else(|| panic!("scene {name}"))
                .clone()
        })
        .collect();
    let rows = run_cases(&cases);
    println!("{}", report(&rows));
    assert_precision(&rows, &[]);
    for ((t, s), (_, floor)) in rows.iter().zip(FAST_SUBSET) {
        assert!(
            s.recall() >= *floor,
            "{}: recall {:.3} below floor {floor}",
            t.name,
            s.recall()
        );
    }
}

#[test]
#[ignore = "full precision sweep (minutes in release)"]
fn precision_full_sweep() {
    let cases: Vec<Case> = catalogue().into_iter().chain(explore_cases()).collect();
    let rows = run_cases(&cases);
    println!("{}", report(&rows));
    assert_precision(&rows, RECALL_FLOORS);
}

fn precision_dir() -> PathBuf {
    PathBuf::from(std::env::var("RINGGRID_PRECISION_DIR").expect("set RINGGRID_PRECISION_DIR"))
}

#[test]
#[ignore = "writes the scene catalogue to $RINGGRID_PRECISION_DIR"]
fn export_precision_scenes() {
    let dir = precision_dir();
    let mut boards = Boards::default();
    let mut manifest = String::new();
    let cases = if std::env::var("RINGGRID_PRECISION_EXPLORE").is_ok() {
        explore_cases()
    } else {
        catalogue()
    };
    for case in cases {
        let (image, truth) = render_case(&case, &mut boards);
        let sub = dir.join(&case.name);
        std::fs::create_dir_all(&sub).expect("scene dir");
        image.save(sub.join("image.png")).expect("save image");
        std::fs::write(
            sub.join("target.json"),
            case.board.layout().to_json_string(),
        )
        .expect("target json");
        std::fs::write(sub.join("gt.json"), truth.to_json().to_string()).expect("gt json");
        let [lo, hi] = case.diameter_range();
        manifest += &format!("{}\t{lo}\t{hi}\n", case.name);
    }
    std::fs::write(dir.join("manifest.tsv"), manifest).expect("manifest");
}

#[test]
#[ignore = "scores det_<run>.json outputs under $RINGGRID_PRECISION_DIR"]
fn score_precision_runs() {
    let dir = precision_dir();
    let runs = std::env::var("RINGGRID_PRECISION_RUNS").unwrap_or_else(|_| "old,new".into());
    let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).expect("manifest");
    let runs: Vec<&str> = runs.split(',').collect();
    let mut all: Vec<Vec<(Truth, Score)>> = Vec::new();
    for run in &runs {
        let mut rows = Vec::new();
        for line in manifest.lines() {
            let name = line.split('\t').next().expect("name");
            let sub = dir.join(name);
            let read = |f: &str| -> Value {
                let text = std::fs::read_to_string(sub.join(f))
                    .unwrap_or_else(|e| panic!("{name}/{f}: {e}"));
                serde_json::from_str(&text).expect("json")
            };
            let truth = Truth::from_json(&read("gt.json"));
            let s = score(&truth, &read(&format!("det_{run}.json")));
            if std::env::var("RINGGRID_PRECISION_CSV").is_ok() {
                println!(
                    "CSV,{run},{name},{},{},{},{},{}",
                    s.n_wrong_id,
                    s.n_correct,
                    s.n_visible,
                    u8::from(s.h_reported),
                    u8::from(s.h_wrong)
                );
            }
            rows.push((truth, s));
        }
        if std::env::var("RINGGRID_PRECISION_FULL").is_ok() {
            println!("=== run {run} ===\n{}", report(&rows));
        }
        all.push(rows);
    }
    println!("{}", compare(&runs, &all));
}

/// Side-by-side table of several runs over the same scenes: per (board,
/// group) `wrongID/pos+label/ctr-loose/H-wrong recall` for each run, then
/// every scene where any run reports a wrong output.
fn compare(runs: &[&str], all: &[Vec<(Truth, Score)>]) -> String {
    let cell = |s: &Score, n_h: usize| {
        format!(
            "{:3}/{:2}/{:2}/{:1} {:.3}",
            s.n_wrong_id,
            s.n_wrong_pos + s.n_label_err,
            s.n_centre_loose,
            n_h,
            s.recall()
        )
    };
    let mut out = format!("{:21}", "board group");
    for r in runs {
        out += &format!(" | {r:^20}");
    }
    out += "\n";
    let mut groups: BTreeMap<(String, String), Vec<(Score, usize)>> = BTreeMap::new();
    for (k, rows) in all.iter().enumerate() {
        for (t, s) in rows {
            let e = groups
                .entry((t.board.clone(), t.group.clone()))
                .or_insert_with(|| vec![(Score::default(), 0); runs.len()]);
            e[k].0.add(s);
            e[k].1 += usize::from(s.h_wrong);
        }
    }
    for ((board, group), v) in &groups {
        out += &format!("{board:10} {group:10}");
        for (s, nh) in v {
            out += &format!(" | {}", cell(s, *nh));
        }
        out += "\n";
    }
    out += "scenes with a wrong output in any run (wrongID/pos+label/ctr-loose/H recall):\n";
    for i in 0..all[0].len() {
        if all.iter().all(|rows| rows[i].1.n_wrong() == 0) {
            continue;
        }
        out += &format!("  {:28}", all[0][i].0.name);
        for rows in all {
            let s = &rows[i].1;
            out += &format!(" | {}", cell(s, usize::from(s.h_wrong)));
        }
        out += "\n";
    }
    out
}

fn explore_cases() -> Vec<Case> {
    precision::BOARDS
        .into_iter()
        .flat_map(precision::grid_cases)
        .collect()
}

#[test]
#[ignore = "exploratory blur/scale/tilt grid; prints scenes with wrong outputs"]
fn precision_explore() {
    let cases = explore_cases();
    let rows = run_cases(&cases);
    let mut total = Score::default();
    for (t, s) in &rows {
        total.add(s);
        if s.n_wrong() > 0 {
            println!("{:24} {}", t.name, summary(s));
        }
    }
    println!("total {}", summary(&total));
}

#[test]
#[ignore = "prints one scene's markers with decode diagnostics ($RINGGRID_PRECISION_SCENE)"]
fn precision_inspect() {
    let name = std::env::var("RINGGRID_PRECISION_SCENE").expect("set RINGGRID_PRECISION_SCENE");
    let case = catalogue()
        .into_iter()
        .chain(explore_cases())
        .find(|c| c.name == name)
        .expect("scene");
    let (image, truth) = render_case(&case, &mut Boards::default());
    if let Ok(dir) = std::env::var("RINGGRID_PRECISION_DIR") {
        let path = PathBuf::from(dir).join(&name).join("image.png");
        if let Ok(saved) = image::open(&path) {
            println!("exported image identical: {}", saved.to_luma8() == image);
        }
    }
    let [lo, hi] = case.diameter_range();
    let detector = Detector::with_marker_scale(
        case.board.layout(),
        MarkerScalePrior::new(lo as f32, hi as f32),
    );
    let (result, diag) = detector.detect_with_diagnostics(&image).expect("detect");
    let det = serde_json::to_value(&result).expect("json");
    println!("{name}: {}", summary(&score(&truth, &det)));
    for (m, d) in result.detected_markers.iter().zip(&diag.markers) {
        let near = truth.cells.iter().filter(|c| c.present).min_by(|a, b| {
            let da = (a.px[0] - m.center[0]).hypot(a.px[1] - m.center[1]);
            let db = (b.px[0] - m.center[0]).hypot(b.px[1] - m.center[1]);
            da.total_cmp(&db)
        });
        let (true_id, err) = near.map_or((None, f64::NAN), |c| {
            (c.id, (c.px[0] - m.center[0]).hypot(c.px[1] - m.center[1]))
        });
        let dec = d.decode.as_ref().map(|x| {
            format!(
                "dec {} d{} m{} c{:.2}",
                x.best_id, x.best_dist, x.margin, x.decode_confidence
            )
        });
        println!(
            "  id {:?} true {:?} err {err:.2} src {:?} grid {:?} {}",
            m.id,
            true_id,
            d.source,
            m.grid_coord,
            dec.unwrap_or_default()
        );
    }
}

/// Detector of a case with its scale prior.
fn case_detector(case: &Case) -> Detector {
    let [lo, hi] = case.diameter_range();
    Detector::with_marker_scale(
        case.board.layout(),
        MarkerScalePrior::new(lo as f32, hi as f32),
    )
}

/// The present true cell a reported centre matches (within half a spacing).
fn matched_cell(truth: &Truth, center: [f64; 2]) -> Option<&precision::GtCell> {
    let dist = |c: &precision::GtCell| (c.px[0] - center[0]).hypot(c.px[1] - center[1]);
    truth
        .cells
        .iter()
        .filter(|c| c.present)
        .min_by(|a, b| dist(a).total_cmp(&dist(b)))
        .filter(|c| dist(c) <= 0.5 * c.spacing_px)
}

/// Centre error by centre source, per condition group, over the in-range
/// catalogue and grid (`oor` excluded): how biased are uncorrected
/// outer-ellipse centres, and how many wrong centres do they account for.
#[test]
#[ignore = "centre-source study (minutes in release)"]
fn precision_centre_sources() {
    let cases: Vec<Case> = catalogue()
        .into_iter()
        .chain(explore_cases())
        .filter(|c| c.group != "oor")
        .collect();
    let mut boards = Boards::default();
    // (group, source) -> (errors, count beyond the loose gate)
    let mut table: BTreeMap<(String, String), (Vec<f64>, usize)> = BTreeMap::new();
    for case in &cases {
        let (image, truth) = render_case(case, &mut boards);
        let result = case_detector(case).detect(&image).expect("detect");
        for m in &result.detected_markers {
            let Some(cell) = matched_cell(&truth, m.center) else {
                continue;
            };
            let err = (cell.px[0] - m.center[0]).hypot(cell.px[1] - m.center[1]);
            let loose =
                precision::CENTRE_LOOSE_MIN_PX.max(precision::CENTRE_LOOSE_FRACTION * cell.diam_px);
            let key = (
                format!("{}-{}", truth.board, truth.group),
                format!("{:?}", m.center_source),
            );
            let e = table.entry(key).or_default();
            e.0.push(err);
            e.1 += usize::from(err > loose);
        }
    }
    for ((group, source), (mut errs, n_loose)) in table {
        errs.sort_by(f64::total_cmp);
        let q = |p: f64| errs[((errs.len() - 1) as f64 * p).round() as usize];
        let n_tight = errs
            .iter()
            .filter(|&&e| e > precision::CENTRE_TIGHT_PX)
            .count();
        println!(
            "{group:22} {source:16} n {:5} median {:.3} p95 {:.3} >1px {:4} >loose {:3}",
            errs.len(),
            q(0.5),
            q(0.95),
            n_tight,
            n_loose
        );
    }
}

/// Outer-ellipse shape of a reported marker against (a) the ellipse the
/// result's homography predicts for its cell (local Jacobian × circle) and
/// (b) the median of its lattice neighbours: `[size ratio vs board median,
/// axis-ratio difference, orientation difference (rad, NaN when the
/// predicted ellipse is near-circular), neighbour axis-ratio difference,
/// local-extremum excess along a lattice line (the literal "monotone along a
/// line" form: how far the axis ratio overshoots both line neighbours)]`.
fn shape_residuals(result: &ringgrid::DetectionResult) -> Vec<Option<[f64; 5]>> {
    let Some(h) = result.homography else {
        return vec![None; result.detected_markers.len()];
    };
    let project = |p: [f64; 2]| {
        let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
        [
            (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
            (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
        ]
    };
    let ratio = |e: &ringgrid::Ellipse| e.b / e.a;
    // Predicted (axis ratio, major-axis angle, scale) per marker.
    let pred: Vec<Option<(f64, f64, f64)>> = result
        .detected_markers
        .iter()
        .map(|m| {
            let xy = m.board_xy_mm?;
            let o = project(xy);
            let px = project([xy[0] + 1.0, xy[1]]);
            let py = project([xy[0], xy[1] + 1.0]);
            let j = nalgebra::Matrix2::new(px[0] - o[0], py[0] - o[0], px[1] - o[1], py[1] - o[1]);
            let svd = j.svd(true, false);
            let (s, u) = (svd.singular_values, svd.u?);
            let (i, k) = if s[0] >= s[1] { (0, 1) } else { (1, 0) };
            Some((
                s[k] / s[i],
                u[(1, i)].atan2(u[(0, i)]),
                (s[0] * s[1]).sqrt(),
            ))
        })
        .collect();
    let sizes: Vec<f64> = result
        .detected_markers
        .iter()
        .zip(&pred)
        .filter_map(|(m, p)| {
            let e = m.ellipse_outer.as_ref()?;
            Some((e.a * e.b).sqrt() / p.as_ref()?.2)
        })
        .collect();
    let mut sorted = sizes.clone();
    sorted.sort_by(f64::total_cmp);
    let size_median = sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN);
    let by_coord: HashMap<[i32; 2], f64> = result
        .detected_markers
        .iter()
        .filter_map(|m| Some((m.grid_coord?, ratio(m.ellipse_outer.as_ref()?))))
        .collect();
    let steps: &[[i32; 2]] = &[[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];
    result
        .detected_markers
        .iter()
        .zip(&pred)
        .map(|(m, p)| {
            let (pr, pa, ps) = (*p)?;
            let e = m.ellipse_outer.as_ref()?;
            let size = (e.a * e.b).sqrt() / ps / size_median;
            let dr = ratio(e) - pr;
            let da = if pr < 0.9 {
                let d = (e.angle - pa).rem_euclid(std::f64::consts::PI);
                d.min(std::f64::consts::PI - d)
            } else {
                f64::NAN
            };
            let g = m.grid_coord?;
            let mut nb: Vec<f64> = steps
                .iter()
                .filter_map(|s| by_coord.get(&[g[0] + s[0], g[1] + s[1]]).copied())
                .collect();
            nb.sort_by(f64::total_cmp);
            let dn = if nb.len() >= 3 {
                ratio(e) - nb[nb.len() / 2]
            } else {
                f64::NAN
            };
            let r = ratio(e);
            let mut ext = 0.0f64;
            for s in [[1, 0], [0, 1], [1, -1]] {
                let prev = by_coord.get(&[g[0] - s[0], g[1] - s[1]]);
                let next = by_coord.get(&[g[0] + s[0], g[1] + s[1]]);
                if let (Some(&p), Some(&n)) = (prev, next) {
                    ext = ext.max((r - p.max(n)).max(p.min(n) - r));
                }
            }
            Some([size, dr, da, dn, ext])
        })
        .collect()
}

/// Ellipse-shape consistency as a rejection test: residual distributions of
/// good vs bad reported markers (bad = wrong ID, wrong position or centre
/// beyond the loose tolerance), and catch / false-rejection rates of
/// thresholds on each residual, with and without uncorrected distortion.
#[test]
#[ignore = "ellipse-shape study (minutes in release)"]
fn precision_shape_consistency() {
    let cases: Vec<Case> = catalogue()
        .into_iter()
        .chain(explore_cases())
        .filter(|c| c.group != "oor")
        .collect();
    let mut boards = Boards::default();
    // (distorted, bad) -> residual rows
    let mut rows: BTreeMap<(bool, bool), Vec<[f64; 5]>> = BTreeMap::new();
    for case in &cases {
        let (image, truth) = render_case(case, &mut boards);
        let result = case_detector(case).detect(&image).expect("detect");
        let res = shape_residuals(&result);
        // Per-image robust centre and spread of the axis-ratio residual.
        let mut drs: Vec<f64> = res.iter().flatten().map(|r| r[1]).collect();
        drs.sort_by(f64::total_cmp);
        let med = drs.get(drs.len() / 2).copied().unwrap_or(0.0);
        let mut dev: Vec<f64> = drs.iter().map(|d| (d - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        let mad = dev.get(dev.len() / 2).copied().unwrap_or(0.0);
        for (m, r) in result.detected_markers.iter().zip(res) {
            let Some(mut r) = r else { continue };
            // Slot 4 (line extremum) is replaced by the image-normalised
            // axis-ratio residual in units of max(0.02, 1.4826 MAD).
            if std::env::var("RINGGRID_SHAPE_NORM").is_ok() {
                r[4] = (r[1] - med) / (1.4826 * mad).max(0.02);
            }
            let bad = match matched_cell(&truth, m.center) {
                None => true,
                Some(c) => {
                    let err = (c.px[0] - m.center[0]).hypot(c.px[1] - m.center[1]);
                    let loose = precision::CENTRE_LOOSE_MIN_PX
                        .max(precision::CENTRE_LOOSE_FRACTION * c.diam_px);
                    (m.id.is_some() && m.id != c.id)
                        || m.board_xy_mm
                            .is_some_and(|xy| (xy[0] - c.xy_mm[0]).hypot(xy[1] - c.xy_mm[1]) > 0.01)
                        || err > loose
                }
            };
            rows.entry((case.spec.radial_lambda != 0.0, bad))
                .or_default()
                .push(r);
        }
    }
    let names = ["size", "axis-ratio", "angle", "nbr-ratio", "line-extr"];
    let thresholds: [&[f64]; 5] = [
        &[0.05, 0.1, 0.15, 0.2],
        &[0.03, 0.05, 0.08, 0.12],
        &[0.1, 0.2, 0.35, 0.5],
        &[0.03, 0.05, 0.08, 0.12],
        if std::env::var("RINGGRID_SHAPE_NORM").is_ok() {
            &[3.0, 4.0, 6.0, 8.0]
        } else {
            &[0.01, 0.02, 0.04, 0.08]
        },
    ];
    for ((distorted, bad), v) in &rows {
        println!("distorted {distorted} bad {bad}: {} markers", v.len());
    }
    for (k, name) in names.iter().enumerate() {
        for &t in thresholds[k] {
            let flagged = |r: &[f64; 5]| {
                let x = if k == 0 { r[0] - 1.0 } else { r[k] };
                x.abs() > t
            };
            let mut line = format!("{name:10} > {t:5.2}:");
            for distorted in [false, true] {
                let rate = |bad: bool| {
                    let v = rows.get(&(distorted, bad)).map_or(&[][..], |v| &v[..]);
                    let n = v.iter().filter(|r| flagged(r)).count();
                    format!("{n}/{}", v.len())
                };
                line += &format!(
                    "  [{}] catch {} false {}",
                    if distorted { "distorted" } else { "pinhole" },
                    rate(true),
                    rate(false)
                );
            }
            println!("{line}");
        }
    }
}

/// Board cells a result reports correctly: matched ring, right ID (coded) or
/// right absolute position (plain).
fn correct_cells(truth: &Truth, result: &ringgrid::DetectionResult) -> HashSet<[i32; 2]> {
    let absolute = result.board_frame == Some(ringgrid::BoardFrame::Absolute);
    result
        .detected_markers
        .iter()
        .filter_map(|m| {
            let c = matched_cell(truth, m.center)?;
            let ok = if truth.coded {
                m.id.is_some() && m.id == c.id
            } else {
                absolute
                    && m.board_xy_mm
                        .is_some_and(|xy| (xy[0] - c.xy_mm[0]).hypot(xy[1] - c.xy_mm[1]) < 0.01)
            };
            ok.then_some(c.coord)
        })
        .collect()
}

/// Distance (px) from a reported centre to its board position projected by
/// the result's homography; NaN without either.
fn h_offset_px(result: &ringgrid::DetectionResult, m: &ringgrid::DetectedMarker) -> f64 {
    let (Some(h), Some(xy)) = (result.homography, m.board_xy_mm) else {
        return f64::NAN;
    };
    let w = h[2][0] * xy[0] + h[2][1] * xy[1] + h[2][2];
    let px = (h[0][0] * xy[0] + h[0][1] * xy[1] + h[0][2]) / w;
    let py = (h[1][0] * xy[0] + h[1][1] * xy[1] + h[1][2]) / w;
    (px - m.center[0]).hypot(py - m.center[1])
}

/// Why a visible cell surrounded by correctly reported neighbours is missing,
/// from ablated runs of the same detector on the same image (first match):
/// reported without the corrected-centre contract → `uncorrected` (split by
/// whether that ring has an inner ellipse: `inner fitted` or `no
/// inner ellipse`);
/// reported without geometric verify → `verify` (split by the reported
/// centre's distance to the result homography's projection: `position`
/// beyond 2 px, else `shape`); reported with neither → `uncorrected+verify`;
/// a fitted ring there (fit/decode/dedup + axis-ratio filter, no labeling)
/// whose decode names another ID → `decode contradicts cell`, any other
/// fitted ring → `fitted, not labeled/completed`; a proposal within a
/// quarter diameter → `proposed, no outer fit`; else `never proposed`.
fn classify_missing(
    case: &Case,
    image: &GrayImage,
    truth: &Truth,
    missing: &[&precision::GtCell],
) -> Vec<&'static str> {
    use ringgrid::CircleRefinementMethod;
    let run = |refine: Option<CircleRefinementMethod>, verify: bool, global: bool| {
        let mut d = case_detector(case);
        if let Some(r) = refine {
            d.config_mut().circle_refinement = r;
        }
        d.config_mut().advanced.geometric_verify = verify;
        d.config_mut().advanced.use_global_filter = global;
        d.detect(image).expect("detect")
    };
    let none = Some(CircleRefinementMethod::None);
    let uncorrected_result = run(none, true, true);
    let uncorrected = correct_cells(truth, &uncorrected_result);
    let no_verify_result = run(None, false, true);
    let no_verify = correct_cells(truth, &no_verify_result);
    let neither = correct_cells(truth, &run(none, false, true));
    let raw = run(none, true, false);
    let proposals = case_detector(case).propose(image);
    let near = |p: [f64; 2], c: &precision::GtCell| {
        (p[0] - c.px[0]).hypot(p[1] - c.px[1]) <= 0.25 * c.diam_px
    };
    missing
        .iter()
        .map(|c| {
            if uncorrected.contains(&c.coord) {
                let has_inner = uncorrected_result
                    .detected_markers
                    .iter()
                    .find(|m| near(m.center, c))
                    .is_some_and(|m| m.ellipse_inner.is_some());
                return if has_inner {
                    "uncorrected: inner fitted"
                } else {
                    "uncorrected: no inner ellipse"
                };
            }
            if no_verify.contains(&c.coord) {
                let off = no_verify_result
                    .detected_markers
                    .iter()
                    .find(|m| near(m.center, c))
                    .map_or(f64::NAN, |m| h_offset_px(&no_verify_result, m));
                return if off > 2.0 {
                    "verify: position"
                } else {
                    "verify: shape"
                };
            }
            if neither.contains(&c.coord) {
                return "uncorrected+verify";
            }
            if let Some(m) = raw.detected_markers.iter().find(|m| near(m.center, c)) {
                return if m.id.is_some() && m.id != c.id {
                    "decode contradicts cell"
                } else {
                    "fitted, not labeled/completed"
                };
            }
            if proposals
                .iter()
                .any(|p| near([f64::from(p.x), f64::from(p.y)], c))
            {
                "proposed, no outer fit"
            } else {
                "never proposed"
            }
        })
        .collect()
}

/// Visible cells missing from the result while most of their lattice
/// neighbours are reported correctly (hex: ≥ 4 of 6; rect: ≥ 6 of 8), over
/// the in-range catalogue and grid, classified by [`classify_missing`].
#[test]
#[ignore = "surrounded-missing-cell study (tens of minutes in release)"]
fn precision_surrounded_missing() {
    let mut boards = Boards::default();
    // (board, cause) -> count
    let mut table: BTreeMap<(String, &'static str), usize> = BTreeMap::new();
    let mut n_missing = 0usize;
    for case in &in_range_cases() {
        let (image, truth) = render_case(case, &mut boards);
        let result = case_detector(case).detect(&image).expect("detect");
        let missing = surrounded_missing(&truth, &correct_cells(&truth, &result));
        if missing.is_empty() {
            continue;
        }
        n_missing += missing.len();
        for cause in classify_missing(case, &image, &truth, &missing) {
            *table.entry((truth.board.clone(), cause)).or_default() += 1;
        }
    }
    println!("surrounded missing cells: {n_missing}");
    for ((board, cause), n) in &table {
        println!("{board:10} {cause:32} {n:5}");
    }
}

/// The in-range catalogue and grid (`oor` excluded).
fn in_range_cases() -> Vec<Case> {
    catalogue()
        .into_iter()
        .chain(explore_cases())
        .filter(|c| c.group != "oor")
        .collect()
}

/// Visible cells missing from `ok` while most of their lattice neighbours
/// are in it (hex: ≥ 4 of 6; rect: ≥ 6 of 8).
fn surrounded_missing<'a>(truth: &'a Truth, ok: &HashSet<[i32; 2]>) -> Vec<&'a precision::GtCell> {
    const HEX: &[[i32; 2]] = &[[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];
    const RECT: &[[i32; 2]] = &[
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ];
    let (steps, need) = if truth.hex { (HEX, 4) } else { (RECT, 6) };
    truth
        .cells
        .iter()
        .filter(|c| c.visible && !ok.contains(&c.coord))
        .filter(|c| {
            let k = c.coord;
            steps
                .iter()
                .filter(|s| ok.contains(&[k[0] + s[0], k[1] + s[1]]))
                .count()
                >= need
        })
        .collect()
}

/// Condition bucket of a cell: ring diameter, blur, tilt, distortion.
fn condition_bucket(case: &Case, cell: &precision::GtCell) -> String {
    let d = match cell.diam_px {
        x if x < 26.0 => "d<26",
        x if x < 38.0 => "d26-38",
        _ => "d>=38",
    };
    let blur = match case.spec.psf {
        support::blur_scene::Psf::Gaussian { sigma } if sigma < 1.2 => "b<1.2",
        support::blur_scene::Psf::Gaussian { sigma } if sigma < 1.8 => "b1.2-1.8",
        _ => "b>=1.8",
    };
    let tilt = match case.spec.tilt_deg {
        t if t < 25.0 => "t<25",
        t if t < 45.0 => "t25-45",
        _ => "t>=45",
    };
    let dist = if case.spec.radial_lambda != 0.0 {
        " dist"
    } else {
        ""
    };
    format!("{d} {blur} {tilt}{dist}")
}

/// Inner-ellipse coverage of surrounded missing cells. For a ring the
/// refinement-free run reports there: without an inner ellipse, the inner-fit
/// reason and theta consistency, by ring size / blur / tilt; with one, whether
/// the default run reports it once geometric verify is off (a two-conic
/// centre the verify then removed) or not at all.
#[test]
#[ignore = "inner-coverage study (tens of minutes in release)"]
fn precision_inner_coverage() {
    let mut boards = Boards::default();
    let mut reasons: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut by_bucket: BTreeMap<(String, &'static str), usize> = BTreeMap::new();
    for case in &in_range_cases() {
        let (image, truth) = render_case(case, &mut boards);
        let result = case_detector(case).detect(&image).expect("detect");
        let missing = surrounded_missing(&truth, &correct_cells(&truth, &result));
        if missing.is_empty() {
            continue;
        }
        let mut d = case_detector(case);
        d.config_mut().circle_refinement = ringgrid::CircleRefinementMethod::None;
        let (raw, diag) = d.detect_with_diagnostics(&image).expect("detect");
        let mut d = case_detector(case);
        d.config_mut().advanced.geometric_verify = false;
        let no_verify = correct_cells(&truth, &d.detect(&image).expect("detect"));
        let near = |p: [f64; 2], c: &precision::GtCell| {
            (p[0] - c.px[0]).hypot(p[1] - c.px[1]) <= 0.25 * c.diam_px
        };
        for c in missing {
            let Some(i) = raw.detected_markers.iter().position(|m| near(m.center, c)) else {
                continue;
            };
            let kind = if raw.detected_markers[i].ellipse_inner.is_some() {
                if no_verify.contains(&c.coord) {
                    "inner; verify removed"
                } else {
                    "inner; lost elsewhere"
                }
            } else {
                let fit = &diag.markers[i].fit;
                let theta = fit.inner_theta_consistency.map_or("-".to_string(), |t| {
                    format!("{:.1}", (t * 10.0).floor() / 10.0)
                });
                let reason = format!("{:?} (theta {theta})", fit.inner_fit_reason);
                *reasons.entry((truth.board.clone(), reason)).or_default() += 1;
                "no inner"
            };
            *by_bucket
                .entry((condition_bucket(case, c), kind))
                .or_default() += 1;
        }
    }
    println!("inner-fit failure (no inner ellipse), by board and reason:");
    for ((board, reason), n) in &reasons {
        println!("  {board:10} {reason:48} {n:5}");
    }
    println!("by condition:");
    for ((bucket, kind), n) in &by_bucket {
        println!("  {bucket:32} {kind:22} {n:5}");
    }
}

/// Bilinear grey level at `p` (NaN off the image).
fn bilinear(img: &GrayImage, p: [f64; 2]) -> f64 {
    let (w, h) = img.dimensions();
    if !(p[0] >= 0.0 && p[1] >= 0.0 && p[0] < f64::from(w - 1) && p[1] < f64::from(h - 1)) {
        return f64::NAN;
    }
    let (x0, y0) = (p[0].floor(), p[1].floor());
    let (fx, fy) = (p[0] - x0, p[1] - y0);
    let g = |x: f64, y: f64| f64::from(img.get_pixel(x as u32, y as u32)[0]);
    (1.0 - fy) * ((1.0 - fx) * g(x0, y0) + fx * g(x0 + 1.0, y0))
        + fy * ((1.0 - fx) * g(x0, y0 + 1.0) + fx * g(x0 + 1.0, y0 + 1.0))
}

/// Outer-edge photometry of an ellipse, from 32 rays through its rim:
/// `[step, edge width px, contrast-to-noise]` — step = median paper-minus-ink
/// level (0.25 a outside vs 0.12 a inside the rim), edge width = step /
/// median peak radial gradient, noise = robust spread of the paper level
/// between neighbouring rays.
fn edge_photometry(img: &GrayImage, e: &ringgrid::Ellipse) -> [f64; 3] {
    let n = 32;
    let (mut steps, mut grads, mut outside) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..n {
        let t = std::f64::consts::TAU * k as f64 / n as f64;
        let (s, c) = (e.angle.sin(), e.angle.cos());
        let rim = [
            e.cx + c * e.a * t.cos() - s * e.b * t.sin(),
            e.cy + s * e.a * t.cos() + c * e.b * t.sin(),
        ];
        let (dx, dy) = (rim[0] - e.cx, rim[1] - e.cy);
        let len = dx.hypot(dy).max(1e-9);
        let dir = [dx / len, dy / len];
        let at = |d: f64| bilinear(img, [rim[0] + dir[0] * d, rim[1] + dir[1] * d]);
        let (o, i) = (at(0.25 * e.a), at(-0.12 * e.a));
        let mut g = f64::NEG_INFINITY;
        let mut d = -0.12 * e.a;
        while d <= 0.25 * e.a {
            g = g.max((at(d + 0.25) - at(d - 0.25)) / 0.5);
            d += 0.25;
        }
        if o.is_finite() && i.is_finite() && g.is_finite() {
            steps.push(o - i);
            grads.push(g);
            outside.push(o);
        }
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    };
    let diffs: Vec<f64> = outside.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let noise = (1.4826 * med(diffs) / std::f64::consts::SQRT_2).max(0.5);
    let step = med(steps);
    let grad = med(grads);
    [step, step / grad.max(1e-6), step / noise]
}

/// Per-marker candidate quality predictors and centre errors over the
/// in-range catalogue (dark scenes included) and grid, written as CSV to
/// `$RINGGRID_QUALITY_CSV` (markers) and `…_scenes` (visible cells per
/// scene), for the confidence study.
#[test]
#[ignore = "quality-predictor dump (minutes in release)"]
fn precision_quality_dump() {
    use std::fmt::Write as _;
    let path = std::env::var("RINGGRID_QUALITY_CSV").expect("set RINGGRID_QUALITY_CSV");
    let mut boards = Boards::default();
    let mut rows = String::from(
        "scene,board,group,err,correct,visible,conf,best_dist,margin,gap_outer,gap_inner,\
         inlier_outer,inlier_inner,rms_outer,rms_inner,cover_outer,n_inner,theta_inner,\
         radii_std_rel,h_reproj,border_r,step,blur_px,blur_rel,cnr,shape_res,ratio_dev,\
         shift_rel,a_px,axis_ratio,nbr_ratio\n",
    );
    let mut scenes = String::from("scene,name,board,group,visible\n");
    for (si, case) in in_range_cases().iter().enumerate() {
        let (image, truth) = render_case(case, &mut boards);
        let (result, diag) = case_detector(case)
            .detect_with_diagnostics(&image)
            .expect("detect");
        let n_vis = truth.cells.iter().filter(|c| c.visible).count();
        writeln!(
            scenes,
            "{si},{},{},{},{n_vis}",
            case.name, truth.board, truth.group
        )
        .unwrap();
        let ok = correct_cells(&truth, &result);
        let shape = shape_residuals(&result);
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let shape_med = med(shape.iter().flatten().map(|r| r[1]).collect());
        let ratio = |m: &ringgrid::DetectedMarker| {
            Some(m.ellipse_inner?.mean_axis() / m.ellipse_outer?.mean_axis())
        };
        let ratio_med = med(result.detected_markers.iter().filter_map(ratio).collect());
        let (w, h) = image.dimensions();
        for ((m, d), sh) in result
            .detected_markers
            .iter()
            .zip(&diag.markers)
            .zip(&shape)
        {
            let (Some(c), Some(e)) = (matched_cell(&truth, m.center), m.ellipse_outer) else {
                continue;
            };
            let err = (c.px[0] - m.center[0]).hypot(c.px[1] - m.center[1]);
            let f = &d.fit;
            let o = |x: Option<f64>| x.unwrap_or(f64::NAN);
            let of = |x: Option<f32>| x.map_or(f64::NAN, f64::from);
            let border = m.center[0]
                .min(m.center[1])
                .min(f64::from(w - 1) - m.center[0])
                .min(f64::from(h - 1) - m.center[1])
                / e.a;
            let [step, blur_px, cnr] = edge_photometry(&image, &e);
            let shift = (m.center[0] - e.cx).hypot(m.center[1] - e.cy) / e.a;
            writeln!(
                rows,
                "{si},{},{},{err},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{border},{step},\
                 {blur_px},{},{cnr},{},{},{shift},{},{},{}",
                truth.board,
                truth.group,
                u8::from(ok.contains(&c.coord)),
                u8::from(c.visible),
                m.confidence,
                d.decode
                    .as_ref()
                    .map_or(f64::NAN, |x| f64::from(x.best_dist)),
                d.decode.as_ref().map_or(f64::NAN, |x| f64::from(x.margin)),
                o(f.max_angular_gap_outer),
                o(f.max_angular_gap_inner),
                of(f.ransac_inlier_ratio_outer),
                of(f.ransac_inlier_ratio_inner),
                o(f.rms_residual_outer),
                o(f.rms_residual_inner),
                f.n_points_outer as f64 / f.n_angles_total.max(1) as f64,
                f.n_points_inner,
                of(f.inner_theta_consistency),
                of(f.radii_std_outer_px) / e.a,
                of(f.h_reproj_err_px),
                blur_px / e.a,
                sh.map_or(f64::NAN, |r| (r[1] - shape_med).abs()),
                ratio(m).map_or(f64::NAN, |r| (r - ratio_med).abs()),
                e.a,
                e.b / e.a,
                of(f.neighbor_radius_ratio),
            )
            .unwrap();
        }
    }
    std::fs::write(&path, rows).expect("write csv");
    std::fs::write(format!("{path}_scenes"), scenes).expect("write scenes");
}
