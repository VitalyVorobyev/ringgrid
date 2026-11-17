//! Golden snapshots of every public detection entry point.
//!
//! The detector API and config are being restructured under a promise of
//! identical detection output. This test is how that promise is checked: each
//! *case* (a scene, a refinement method and an entry point) is run, and its
//! output is compared with a committed snapshot under
//! `tests/fixtures/golden/<platform>/<case>.json`.
//!
//! A snapshot holds two things:
//!
//! * `digest` — a compact, **shape-independent** summary built only from the
//!   facts a caller cares about (marker ids, lattice labels, centres, board
//!   positions, confidences, homography, board frame and completeness). It
//!   must stay bit-identical through any API restructuring, even when result
//!   fields move between types.
//! * `full` — the serialized result and diagnostics (timings stripped), kept
//!   for a few representative cases. It is a secondary, shape-aware check and
//!   may legitimately be re-blessed when a field moves.
//!
//! Cases are named by what they exercise (`real_adaptive_hint32_projective`),
//! never by the Rust method, so a case stays comparable when its method is
//! renamed. The only code that knows how to run a case with the current API
//! is the small `run_*` functions below; the snapshot and comparison logic
//! does not touch the detector API.
//!
//! Floats are compared exactly (the snapshot text must match byte for byte).
//!
//! **One snapshot set per platform.** The detector is deterministic on a
//! given platform, but not across platforms: `sin`, `cos`, `atan2`, `exp`,
//! `ln`, `hypot`, `cbrt` come from the platform's math library, whose last
//! bits differ between glibc, Apple's libm and the MSVC CRT. The detector's
//! discrete decisions (edge-sample selection, inlier gates, tier merging,
//! completion) amplify those last-bit differences: between x86_64 Linux and
//! aarch64 macOS most values agree to ~1e-13, but individual markers move by
//! up to ~0.6 px and a few low-confidence markers are kept on one platform
//! and dropped on the other. That spread is far above the shifts this test
//! must catch (a real change moved centres by 0.0012 px RMS), so no numeric
//! tolerance can make one snapshot set serve every platform. Instead each
//! platform in [`SNAPSHOT_PLATFORMS`] keeps its own exact set under
//! `<target_arch>-<target_os>/`; on any other platform the test is skipped
//! with a note.
//!
//! The snapshots are blessed from an optimized build and the test only runs
//! in one: an unoptimized build takes 15+ minutes (photometric refinement) and
//! its floating-point results differ from the optimized ones in the last
//! digits, so a debug run cannot be compared with the snapshots. Plain
//! `cargo test` therefore skips it with a note; run
//! `cargo test --release -p ringgrid --test golden_entry_points` after any
//! change that can touch detection output. `RINGGRID_GOLDEN_ALL=1` forces a
//! run in any profile. CI runs it on every platform in
//! [`SNAPSHOT_PLATFORMS`].
//!
//! **Blessing.** `RINGGRID_BLESS=1 cargo test --release -p ringgrid --test
//! golden_entry_points` rewrites the current platform's set. On a mismatch
//! the test also writes the platform's complete fresh set to
//! `target/tmp/golden-actual/<platform>/` (CI uploads it as the
//! `golden-actual-<os>` artifact), so a change that moves detection output
//! is blessed for a platform you are not on by copying that set over
//! `tests/fixtures/golden/<platform>/` after reviewing the diff.
//!
//! Coverage restrictions, chosen to bound runtime: the photometric refinement
//! is slow, so the four-tier multiscale entry point is run with it only on the
//! synthetic coded scene, and the plain scene is not run through the mapper or
//! self-undistort entry points (it has no lens distortion).

mod support;

use image::{GrayImage, Luma};
use ringgrid::diagnostics::DetectionDiagnostics;
use ringgrid::{
    BoardFrame, CameraIntrinsics, CameraModel, CircleRefinementMethod, DetectConfig,
    DetectionResult, Detector, OriginFiducials, RadialTangentialDistortion, RectGeometry,
    ScaleTiers, TargetLayout,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use support::blur_scene::{BoardRaster, Scene, SceneSpec, render};

// ── Scenes ──────────────────────────────────────────────────────────────────

/// An image with the target it shows and the camera that distorted it, if any.
struct SceneData {
    target: TargetLayout,
    image: GrayImage,
    camera: Option<CameraModel>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The checked-in real image (`testdata/target_3_split_00.png`) and its target.
fn real_scene() -> SceneData {
    let root = repo_root();
    let image = image::open(root.join("testdata/target_3_split_00.png"))
        .expect("open real fixture image")
        .to_luma8();
    let target = TargetLayout::from_json_file(&root.join("testdata/board_ringgrid.json"))
        .expect("load real fixture target");
    SceneData {
        target,
        image,
        camera: None,
    }
}

/// Remap `ideal` (a pinhole image) through the lens: output pixel `d` shows
/// the ideal pixel that `camera` distorts onto `d`.
fn distort_image(ideal: &GrayImage, camera: &CameraModel) -> GrayImage {
    let (w, h) = ideal.dimensions();
    let at = |x: i64, y: i64| -> f64 {
        let x = x.clamp(0, i64::from(w) - 1) as u32;
        let y = y.clamp(0, i64::from(h) - 1) as u32;
        f64::from(ideal.get_pixel(x, y).0[0])
    };
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let Some([u, v]) = camera.undistort_pixel([f64::from(x), f64::from(y)]) else {
                continue;
            };
            let (x0, y0) = (u.floor(), v.floor());
            let (fx, fy) = (u - x0, v - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let value = at(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + at(x0 + 1, y0) * fx * (1.0 - fy)
                + at(x0, y0 + 1) * (1.0 - fx) * fy
                + at(x0 + 1, y0 + 1) * fx * fy;
            out.put_pixel(x, y, Luma([value.round().clamp(0.0, 255.0) as u8]));
        }
    }
    out
}

fn synthetic_camera(image_size: [u32; 2], focal_px: f64) -> CameraModel {
    CameraModel {
        intrinsics: CameraIntrinsics {
            fx: focal_px,
            fy: focal_px,
            cx: 0.5 * f64::from(image_size[0]),
            cy: 0.5 * f64::from(image_size[1]),
        },
        distortion: RadialTangentialDistortion {
            k1: -0.25,
            k2: 0.05,
            p1: 0.001,
            p2: -0.0008,
            k3: 0.0,
        },
    }
}

/// A coded hex target seen under 35 degrees of perspective, blurred, noisy and
/// then distorted by a radial-tangential lens. Markers are 32 px across.
fn synthetic_coded_scene() -> SceneData {
    let target = TargetLayout::coded_hex(8.0, 13, 14, 4.8, 3.2, 0.8).expect("valid coded hex");
    let spec = SceneSpec::for_target(&target, 16.0, 0.9, 35.0, 70.0, 7);
    let Scene { image, .. } = render(&BoardRaster::new(target.clone()), &spec);
    let camera = synthetic_camera(spec.image_size, spec.focal_px);
    SceneData {
        target,
        image: distort_image(&image, &camera),
        camera: Some(camera),
    }
}

/// A plain rect target with the L-shaped origin-dot triple, tilted and blurred
/// (no lens distortion). Markers are 28 px across.
fn synthetic_plain_scene() -> SceneData {
    let target = TargetLayout::new(
        "golden_plain_rect_dots",
        ringgrid::LatticeGeometry::Rect(RectGeometry {
            rows: 8,
            cols: 8,
            pitch_mm: 14.0,
        }),
        ringgrid::RingGeometry {
            outer_radius_mm: 5.6,
            inner_radius_mm: 2.8,
        },
        ringgrid::MarkerCoding::Plain,
        Some(OriginFiducials { dot_radius_mm: 1.4 }),
    )
    .expect("valid plain rect target");
    let spec = SceneSpec::for_target(&target, 14.0, 0.8, 25.0, 70.0, 11);
    let Scene { image, .. } = render(&BoardRaster::new(target.clone()), &spec);
    SceneData {
        target,
        image,
        camera: None,
    }
}

// ── How each entry point is run with the current API ────────────────────────
//
// One function per entry point. Each takes the scene and the refinement
// method and returns the result, plus diagnostics when the entry point has
// them. These are the only functions later API steps need to rewrite.

type Run = (DetectionResult, Option<DetectionDiagnostics>);

fn detector(scene: &SceneData, refine: CircleRefinementMethod, self_undistort: bool) -> Detector {
    let mut config = DetectConfig::from_target(scene.target.clone());
    config.circle_refinement = refine;
    config.self_undistort.enable = self_undistort;
    Detector::with_config(config)
}

fn run_single(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let det = detector(scene, refine, false);
    let (result, diagnostics) = det
        .detect_with_diagnostics(&scene.image)
        .expect("detect_with_diagnostics");
    // `detect` is the same pipeline without the diagnostics half.
    let plain = det.detect(&scene.image).expect("detect");
    assert_eq!(
        serde_json::to_string(&plain).unwrap(),
        serde_json::to_string(&result).unwrap(),
        "detect and detect_with_diagnostics disagree"
    );
    (result, Some(diagnostics))
}

fn run_adaptive(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let det = detector(scene, refine, false);
    let (result, diagnostics) = det
        .detect_adaptive_with_diagnostics(&scene.image)
        .expect("detect_adaptive_with_diagnostics");
    let plain = det.detect_adaptive(&scene.image).expect("detect_adaptive");
    assert_eq!(
        serde_json::to_string(&plain).unwrap(),
        serde_json::to_string(&result).unwrap(),
        "detect_adaptive and its diagnostics variant disagree"
    );
    (result, Some(diagnostics))
}

fn run_adaptive_hint32(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let result = detector(scene, refine, false)
        .detect_adaptive_with_hint(&scene.image, Some(32.0))
        .expect("detect_adaptive_with_hint");
    (result, None)
}

fn run_multiscale_two_tier(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let result = detector(scene, refine, false)
        .detect_multiscale(&scene.image, &ScaleTiers::two_tier_standard())
        .expect("detect_multiscale");
    (result, None)
}

fn run_multiscale_four_tier(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let result = detector(scene, refine, false)
        .detect_multiscale(&scene.image, &ScaleTiers::four_tier_wide())
        .expect("detect_multiscale");
    (result, None)
}

fn run_mapper(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let camera = scene.camera.expect("mapper case needs a camera");
    let (result, diagnostics) = detector(scene, refine, false)
        .detect_with_mapper_diagnostics(&scene.image, &camera)
        .expect("detect_with_mapper_diagnostics");
    (result, Some(diagnostics))
}

fn run_self_undistort(scene: &SceneData, refine: CircleRefinementMethod) -> Run {
    let (result, diagnostics) = detector(scene, refine, true)
        .detect_with_diagnostics(&scene.image)
        .expect("detect_with_diagnostics (self-undistort)");
    (result, Some(diagnostics))
}

// ── The case table ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum SceneId {
    Real,
    SyntheticCoded,
    SyntheticPlain,
}

impl SceneId {
    fn prefix(self) -> &'static str {
        match self {
            SceneId::Real => "real",
            SceneId::SyntheticCoded => "synth_coded",
            SceneId::SyntheticPlain => "synth_plain",
        }
    }

    fn build(self) -> SceneData {
        match self {
            SceneId::Real => real_scene(),
            SceneId::SyntheticCoded => synthetic_coded_scene(),
            SceneId::SyntheticPlain => synthetic_plain_scene(),
        }
    }
}

type Runner = fn(&SceneData, CircleRefinementMethod) -> Run;

/// Every entry point: its case-name infix and how to run it.
const ENTRY_POINTS: [(&str, Runner); 7] = [
    ("single", run_single),
    ("adaptive", run_adaptive),
    ("adaptive_hint32", run_adaptive_hint32),
    ("multiscale_two_tier", run_multiscale_two_tier),
    ("multiscale_four_tier", run_multiscale_four_tier),
    ("mapper", run_mapper),
    ("self_undistort", run_self_undistort),
];

const REFINEMENTS: [(&str, CircleRefinementMethod); 3] = [
    ("none", CircleRefinementMethod::None),
    ("projective", CircleRefinementMethod::ProjectiveCenter),
    ("photometric", CircleRefinementMethod::Photometric),
];

struct Case {
    name: String,
    scene: SceneId,
    runner: Runner,
    refine: CircleRefinementMethod,
    /// Store the full result + diagnostics JSON (not only the digest).
    full: bool,
}

fn is_meaningful(scene: SceneId, entry: &str, refine: &str) -> bool {
    let needs_camera = matches!(entry, "mapper" | "self_undistort");
    if needs_camera && scene != SceneId::SyntheticCoded {
        // The real image and the plain scene carry no known distortion.
        return false;
    }
    // Photometric is slow; the four-tier pass runs it only on the synthetic
    // coded scene (where it is still exercised end to end).
    !(refine == "photometric"
        && entry == "multiscale_four_tier"
        && scene != SceneId::SyntheticCoded)
}

/// Cases whose full result + diagnostics are stored. The rest keep the digest
/// only, which keeps the fixtures small.
fn stores_full(scene: SceneId, entry: &str, refine: &str) -> bool {
    matches!(
        (scene, entry, refine),
        (SceneId::Real, "single", "projective")
            | (SceneId::SyntheticCoded, "mapper", "projective")
            | (SceneId::SyntheticCoded, "self_undistort", "projective")
            | (SceneId::SyntheticPlain, "single", "photometric")
    )
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for scene in [
        SceneId::Real,
        SceneId::SyntheticCoded,
        SceneId::SyntheticPlain,
    ] {
        for (entry, runner) in ENTRY_POINTS {
            for (refine_name, refine) in REFINEMENTS {
                if !is_meaningful(scene, entry, refine_name) {
                    continue;
                }
                cases.push(Case {
                    name: format!("{}_{entry}_{refine_name}", scene.prefix()),
                    scene,
                    runner,
                    refine,
                    full: stores_full(scene, entry, refine_name),
                });
            }
        }
    }
    cases
}

// ── Snapshot construction ───────────────────────────────────────────────────

fn frame_name(frame: Option<BoardFrame>) -> Value {
    serde_json::to_value(frame).expect("serialize board frame")
}

/// The shape-independent digest. Reads only facts that every version of the
/// result must carry; sorted so marker order cannot matter.
fn digest(result: &DetectionResult) -> Value {
    let mut markers: Vec<_> = result.detected_markers.iter().collect();
    markers.sort_by(|a, b| {
        a.id.cmp(&b.id)
            .then(a.grid_coord.cmp(&b.grid_coord))
            .then(a.center[0].total_cmp(&b.center[0]))
            .then(a.center[1].total_cmp(&b.center[1]))
    });
    let markers: Vec<Value> = markers
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "grid_coord": m.grid_coord,
                "center": m.center,
                "board_xy_mm": m.board_xy_mm,
                "confidence": m.confidence,
            })
        })
        .collect();
    let homography = result
        .homography
        .map(|h| h.iter().flatten().copied().collect::<Vec<f64>>());
    json!({
        "marker_count": markers.len(),
        "markers": markers,
        "homography": homography,
        "board_frame": frame_name(result.board_frame),
        "board_complete": result.board_complete,
    })
}

/// Result and diagnostics as JSON with every timing field removed.
fn full_json(result: &DetectionResult, diagnostics: Option<&DetectionDiagnostics>) -> Value {
    let mut diagnostics = diagnostics.map(|d| serde_json::to_value(d).expect("serialize"));
    if let Some(Value::Object(map)) = diagnostics.as_mut() {
        map.remove("timings");
    }
    json!({
        "result": serde_json::to_value(result).expect("serialize"),
        "diagnostics": diagnostics,
    })
}

fn snapshot_text(case: &Case, (result, diagnostics): &Run) -> String {
    let digest = serde_json::to_string(&digest(result)).unwrap();
    if case.full {
        let full = serde_json::to_string(&full_json(result, diagnostics.as_ref())).unwrap();
        format!("{{\n\"digest\": {digest},\n\"full\": {full}\n}}\n")
    } else {
        format!("{{\n\"digest\": {digest}\n}}\n")
    }
}

// ── Comparison ──────────────────────────────────────────────────────────────

/// Platforms (`<target_arch>-<target_os>`) that keep a snapshot set: the one
/// the release workflow runs on and the maintainers' development machines.
const SNAPSHOT_PLATFORMS: [&str; 2] = ["x86_64-linux", "aarch64-macos"];

fn platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn golden_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden")
}

/// Where a mismatching run leaves the platform's complete fresh set.
fn actual_dir(platform: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("golden-actual")
        .join(platform)
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 60 {
        format!("{}...", &s[..57])
    } else {
        s
    }
}

/// Collect `(json path, old, new)` triples where `old` and `new` differ.
fn diff_values(path: &str, old: &Value, new: &Value, out: &mut Vec<String>) {
    if old == new {
        return;
    }
    match (old, new) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: BTreeSet<_> = a.keys().chain(b.keys()).collect();
            for k in keys {
                let p = format!("{path}.{k}");
                match (a.get(k), b.get(k)) {
                    (Some(x), Some(y)) => diff_values(&p, x, y, out),
                    (Some(x), None) => out.push(format!("{p}: {} -> <absent>", short(x))),
                    (None, Some(y)) => out.push(format!("{p}: <absent> -> {}", short(y))),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                out.push(format!("{path}: length {} -> {}", a.len(), b.len()));
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                diff_values(&format!("{path}[{i}]"), x, y, out);
            }
        }
        _ => out.push(format!("{path}: {} -> {}", short(old), short(new))),
    }
}

/// Human-readable differences between the committed and fresh snapshot.
fn describe_mismatch(case: &str, old: &str, new: &str) -> String {
    let (Ok(old), Ok(new)) = (
        serde_json::from_str::<Value>(old),
        serde_json::from_str::<Value>(new),
    ) else {
        return format!("{case}: snapshot is not valid JSON");
    };
    let mut diffs = Vec::new();
    diff_values("$", &old, &new, &mut diffs);
    if diffs.is_empty() {
        return format!("{case}: text differs but parsed values are equal (formatting only)");
    }
    let shown = diffs.iter().take(12).map(|d| format!("    {d}"));
    let more = diffs.len().saturating_sub(12);
    let mut lines = vec![format!("{case}: {} differing value(s)", diffs.len())];
    lines.extend(shown);
    if more > 0 {
        lines.push(format!("    ... and {more} more"));
    }
    lines.join("\n")
}

/// Entries of the golden root that are not a snapshot set of a platform in
/// [`SNAPSHOT_PLATFORMS`]: a set nobody runs is a stale promise.
fn stray_snapshot_sets(root: &Path) -> Vec<String> {
    let mut stray = Vec::new();
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_dir() {
            if !SNAPSHOT_PLATFORMS.contains(&name.as_str()) {
                stray.push(format!(
                    "{name}/: snapshot set of a platform not in SNAPSHOT_PLATFORMS"
                ));
            }
        } else if name.ends_with(".json") {
            stray.push(format!("{name}: snapshot outside a platform set"));
        }
    }
    stray
}

/// Replace `dir` with the complete fresh snapshot set.
fn write_set(dir: &Path, fresh: &[(String, String)]) {
    if dir.exists() {
        std::fs::remove_dir_all(dir).expect("clear snapshot dir");
    }
    std::fs::create_dir_all(dir).expect("create snapshot dir");
    for (name, text) in fresh {
        std::fs::write(dir.join(format!("{name}.json")), text).expect("write snapshot");
    }
}

#[test]
fn entry_points_match_golden_snapshots() {
    if cfg!(debug_assertions) && std::env::var_os("RINGGRID_GOLDEN_ALL").is_none() {
        eprintln!("golden_entry_points: skipped in an unoptimized build; use --release");
        return;
    }
    let root = golden_root();
    let mut failures = stray_snapshot_sets(&root);
    let platform = platform();
    if !SNAPSHOT_PLATFORMS.contains(&platform.as_str()) {
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        eprintln!(
            "golden_entry_points: skipped on {platform}; snapshot sets exist for {SNAPSHOT_PLATFORMS:?}"
        );
        return;
    }
    let bless = std::env::var_os("RINGGRID_BLESS").is_some();
    let dir = root.join(&platform);
    let cases = cases();

    let mut fresh: Vec<(String, String)> = Vec::new();
    let mut scenes: Vec<(SceneId, SceneData)> = Vec::new();
    for case in &cases {
        if !scenes.iter().any(|(id, _)| *id == case.scene) {
            scenes.push((case.scene, case.scene.build()));
        }
        let scene = &scenes.iter().find(|(id, _)| *id == case.scene).unwrap().1;
        let started = std::time::Instant::now();
        let run = (case.runner)(scene, case.refine);
        let text = snapshot_text(case, &run);
        eprintln!(
            "{:<48} {:>4} markers {:>7.2}s",
            case.name,
            run.0.detected_markers.len(),
            started.elapsed().as_secs_f64()
        );
        if !bless {
            let path = dir.join(format!("{}.json", case.name));
            match std::fs::read_to_string(&path) {
                Ok(old) if old == text => {}
                Ok(old) => failures.push(describe_mismatch(&case.name, &old, &text)),
                Err(_) => failures.push(format!(
                    "{}: no snapshot at {} (run with RINGGRID_BLESS=1)",
                    case.name,
                    path.display()
                )),
            }
        }
        fresh.push((case.name.clone(), text));
    }

    if bless {
        // Writing the whole set also drops snapshots of cases that are gone.
        write_set(&dir, &fresh);
        return;
    }

    // A snapshot nobody runs any more is a stale promise.
    let known: BTreeSet<String> = cases.iter().map(|c| format!("{}.json", c.name)).collect();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        if !known.contains(&file) {
            failures.push(format!("{platform}/{file}: snapshot has no matching case"));
        }
    }

    if !failures.is_empty() {
        let actual = actual_dir(&platform);
        write_set(&actual, &fresh);
        failures.push(format!(
            "the complete fresh {platform} set is in {}; after reviewing the diff, \
             bless it by copying it over {}",
            actual.display(),
            dir.display()
        ));
    }
    assert!(
        failures.is_empty(),
        "golden snapshots ({platform}) changed:\n{}",
        failures.join("\n")
    );
}
