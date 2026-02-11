use image::ImageReader;
use ringgrid::{BoardLayout, Detector};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <target.json> <image.png> [out.json]", args[0]);
        std::process::exit(2);
    }

    let board = BoardLayout::from_json_file(Path::new(&args[1]))?;
    let image = ImageReader::open(&args[2])?.decode()?.to_luma8();

    let detector = Detector::new(board);
    let result = detector.detect(&image);

    let n_with_id = result
        .detected_markers
        .iter()
        .filter(|m| m.id.is_some())
        .count();
    println!(
        "Detected {} markers ({} with ID).",
        result.detected_markers.len(),
        n_with_id
    );

    if let Some(out_path) = args.get(3) {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(out_path, json)?;
        println!("Wrote {out_path}");
    }
    Ok(())
}
