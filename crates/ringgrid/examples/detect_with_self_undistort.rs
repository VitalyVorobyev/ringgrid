use image::ImageReader;
use ringgrid::{BoardLayout, CircleRefinementMethod, DetectConfig, Detector, MarkerScalePrior};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <target.json> <image.png>", args[0]);
        std::process::exit(2);
    }

    let board = BoardLayout::from_json_file(Path::new(&args[1]))?;
    let image = ImageReader::open(&args[2])?.decode()?.to_luma8();

    let mut cfg = DetectConfig::from_target(board);
    // Optional explicit scale-search prior override.
    cfg.set_marker_scale_prior(MarkerScalePrior::new(20.0, 56.0));
    cfg.circle_refinement = CircleRefinementMethod::ProjectiveCenter;
    cfg.self_undistort.enable = true;
    cfg.self_undistort.min_markers = 12;

    let detector = Detector::with_config(cfg);
    let result = detector.detect(&image);
    if let Some(su) = result.self_undistort.as_ref() {
        println!(
            "Self-undistort: lambda={:.3e}, applied={}, objective {:.4} -> {:.4}",
            su.model.lambda, su.applied, su.objective_at_zero, su.objective_at_lambda
        );
    }
    println!("Detected {} markers.", result.detected_markers.len());
    Ok(())
}
