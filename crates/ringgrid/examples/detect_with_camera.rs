use image::ImageReader;
use ringgrid::{BoardLayout, CameraIntrinsics, CameraModel, Detector, RadialTangentialDistortion};
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
    let (w, h) = image.dimensions();

    // Example camera parameters; replace with calibrated values.
    let camera = CameraModel {
        intrinsics: CameraIntrinsics {
            fx: 900.0,
            fy: 900.0,
            cx: w as f64 * 0.5,
            cy: h as f64 * 0.5,
        },
        distortion: RadialTangentialDistortion {
            k1: -0.15,
            k2: 0.05,
            p1: 0.001,
            p2: -0.001,
            k3: 0.0,
        },
    };

    let detector = Detector::new(board);
    let result = detector.detect_with_mapper(&image, &camera);
    println!("Detected {} markers.", result.detected_markers.len());
    Ok(())
}
