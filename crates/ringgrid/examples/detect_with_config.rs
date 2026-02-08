use image::ImageReader;
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetSpec};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <target.json> <image.png> <marker_diameter_px>",
            args[0]
        );
        std::process::exit(2);
    }

    let target = TargetSpec::from_json_file(Path::new(&args[1]))?;
    let image = ImageReader::open(&args[2])?.decode()?.to_luma8();
    let marker_diameter_px: f32 = args[3].parse()?;

    let mut cfg =
        DetectConfig::from_target_and_marker_diameter(target.board().clone(), marker_diameter_px);
    cfg.circle_refinement = CircleRefinementMethod::ProjectiveCenter;
    cfg.self_undistort.enable = true;
    cfg.self_undistort.min_markers = 12;

    let detector = Detector::with_config(cfg);
    let result = detector.detect_with_self_undistort(&image);
    if let Some(su) = result.self_undistort.as_ref() {
        println!(
            "Self-undistort: lambda={:.3e}, applied={}, objective {:.4} -> {:.4}",
            su.model.lambda, su.applied, su.objective_at_zero, su.objective_at_lambda
        );
    }
    println!("Detected {} markers.", result.detected_markers.len());
    Ok(())
}
