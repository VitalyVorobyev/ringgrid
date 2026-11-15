//! Precision audit: hard synthetic views of every target kind, scored against
//! ground truth for wrong IDs, wrong board positions, wrong centres and wrong
//! homographies (see `support/precision.rs` for the definitions).
//!
//! - `precision_fast_subset` (default run) asserts zero wrong outputs and
//!   recall floors on the scenes that have failed before.
//! - `precision_full_sweep` (`--ignored`) runs the whole catalogue:
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

use std::collections::BTreeMap;
use std::path::PathBuf;

use image::GrayImage;
use precision::{Boards, Case, Score, Truth, catalogue, render_case, score, summary};
use ringgrid::{Detector, MarkerScalePrior};
use serde_json::Value;

fn detect_json(case: &Case, image: &GrayImage) -> Value {
    let [lo, hi] = case.diameter_range();
    let detector = Detector::with_marker_scale(
        case.board.layout(),
        MarkerScalePrior::new(lo as f32, hi as f32),
    );
    let result = detector.detect(image).expect("detect");
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

#[test]
#[ignore = "full precision sweep (minutes in release)"]
fn precision_full_sweep() {
    let rows = run_cases(&catalogue());
    println!("{}", report(&rows));
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
