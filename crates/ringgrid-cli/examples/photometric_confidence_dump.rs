//! Per-marker dump of photometric confidences on strip captures of the
//! default hex board.
//!
//! Every tile of every strip is detected with `detect_adaptive` (photometric
//! refinement, or projective with `--projective`; `min_confidence = 0` so a
//! confidence cut can be simulated offline); each reported marker is written
//! as one JSON line with its centre, board position, confidence, covariance
//! and photometric diagnostics. `tools/rtv3d_confidence_report.py` scores the
//! confidences against leave-one-out centre-error proxies (ADR-029).
//!
//! ```text
//! cargo run --release -p ringgrid-cli --example photometric_confidence_dump -- \
//!     [--tile-width 720] [--projective] --out dump.jsonl strip0.png strip1.png ...
//! ```

use std::io::Write;
use std::path::PathBuf;

use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tile_width = 720u32;
    let mut out = None;
    let mut method = CircleRefinementMethod::Photometric;
    let mut images = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tile-width" => tile_width = it.next().ok_or("--tile-width")?.parse()?,
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out")?)),
            "--projective" => method = CircleRefinementMethod::ProjectiveCenter,
            _ => images.push(PathBuf::from(a)),
        }
    }
    let out = out.ok_or("--out is required")?;
    let mut config = DetectConfig::from_target(TargetLayout::default_hex());
    config.circle_refinement = method;
    config.min_confidence = 0.0;
    let detector = Detector::with_config(config);
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    for path in &images {
        let gray = image::open(path)?.to_luma8();
        let (w, h) = gray.dimensions();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        for cam in 0..w / tile_width {
            let tile =
                image::imageops::crop_imm(&gray, cam * tile_width, 0, tile_width, h).to_image();
            let (result, diag) = detector.detect_adaptive_with_diagnostics(&tile)?;
            for (m, d) in result.detected_markers.iter().zip(&diag.markers) {
                let line = serde_json::json!({
                    "tile": format!("{stem}_{cam:02}"),
                    "height": h,
                    "id": m.id,
                    "center": m.center,
                    "board_xy_mm": m.board_xy_mm,
                    "confidence": m.confidence,
                    "center_source": m.center_source,
                    "center_cov": m.center_cov,
                    "outer_mean_axis": m.ellipse_outer.map(|e| e.mean_axis()),
                    "edge_width": d.fit.outer_edge_width_px,
                    "photometric": d.photometric,
                });
                writeln!(file, "{line}")?;
            }
        }
        eprintln!("{stem}");
    }
    Ok(())
}
