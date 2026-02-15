# ringgrid

`ringgrid` is a pure-Rust detector for dense coded ring calibration markers arranged on a hex lattice.
It detects markers, decodes IDs, estimates board homography, and returns structured results.

## Installation

```toml
[dependencies]
ringgrid = "0.1.0"
```

## Minimal Example

```rust,no_run
use image::ImageReader;
use ringgrid::{BoardLayout, Detector};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let board = BoardLayout::from_json_file(Path::new("target.json"))?;
    let image = ImageReader::open("frame.png")?.decode()?.to_luma8();

    let detector = Detector::new(board);
    let result = detector.detect(&image);
    println!("detected markers: {}", result.detected_markers.len());
    Ok(())
}
```

## Coordinate Frames

- `DetectedMarker.center` is always in image coordinates.
- `DetectedMarker.center_mapped` is present only when a mapper-driven pass is active.
- `DetectionResult.center_frame` and `DetectionResult.homography_frame` explicitly describe output-frame conventions.

## Related Tools

The repository contains:

- `ringgrid-cli` binary (`cargo run -p ringgrid-cli -- ...`).
- Synthetic data generation, scoring, and visualization scripts under `tools/`.
- Board/codebook generators and examples.

Repository: <https://github.com/VitalyVorobyev/ringgrid>

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
