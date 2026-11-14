//! Where does the photometric intensity model fail to fit real images?
//!
//! Every image (or every camera tile of a horizontal strip, with
//! `--tile-width`) is detected with photometric centre refinement; the
//! per-pixel fit residual (observed minus model grey level) of every accepted
//! marker is then aggregated into tables (no per-image output):
//!
//! - (a) residual vs normalised radius, per blur tercile;
//! - (b) residual vs signed distance to the nearest model edge, for inner-ring,
//!   outer-ring and code-sector edges, per blur tercile;
//! - (c) residual vs image-frame angle near the outer edge (anisotropic PSF,
//!   illumination direction);
//! - (d) the share of the residual variance the across-marker mean profile of
//!   (a)+(b) explains (a common, deterministic misfit), and the lag-1
//!   autocorrelation of the remainder;
//! - (e) residual vs observed grey level (response nonlinearity, saturation);
//! - (f) noise in flat regions, far from every edge.
//!
//! The aggregation lives in `crates/ringgrid/tests/support/residual_profile.rs`,
//! shared with the synthetic sanity test. All residual maps are kept in
//! memory (about 50 bytes per pixel).
//!
//! ```text
//! cargo run --release -p ringgrid-cli --example photometric_residual_profile -- \
//!     --hex 8,15,14,4.8,3.2,1.152 [--tile-width 720] [--json out.json] img0.png img1.png ...
//! ```

use std::path::PathBuf;

use ringgrid::diagnostics::PhotometricStatus;
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};

#[path = "../../ringgrid/tests/support/residual_profile.rs"]
mod residual_profile;
use residual_profile::MarkerData;

struct Args {
    hex: [f32; 6],
    tile_width: Option<u32>,
    json: Option<PathBuf>,
    images: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut hex = None;
    let mut tile_width = None;
    let mut json = None;
    let mut images = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hex" => {
                let v = it.next().ok_or("--hex needs a value")?;
                let parts: Vec<f32> = v
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?;
                if parts.len() != 6 {
                    return Err(
                        "--hex takes pitch,rows,long_row_cols,outer,inner,ring_width".into(),
                    );
                }
                hex = Some([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]);
            }
            "--tile-width" => {
                let v = it.next().ok_or("--tile-width needs a value")?;
                tile_width = Some(v.parse::<u32>().map_err(|e| e.to_string())?);
            }
            "--json" => json = Some(PathBuf::from(it.next().ok_or("--json needs a path")?)),
            _ if a.starts_with("--") => return Err(format!("unknown flag {a}")),
            _ => images.push(PathBuf::from(a)),
        }
    }
    if images.is_empty() {
        return Err("no images given".into());
    }
    Ok(Args {
        hex: hex.ok_or("--hex is required")?,
        tile_width,
        json,
        images,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let [pitch, rows, cols, outer, inner, width] = args.hex;
    let target = TargetLayout::coded_hex(pitch, rows as usize, cols as usize, outer, inner, width)?;
    let mut config = DetectConfig::from_target(target);
    config.circle_refinement = CircleRefinementMethod::Photometric;
    let detector = Detector::with_config(config);

    let mut markers: Vec<MarkerData> = Vec::new();
    let mut n_tiles = 0usize;
    for path in &args.images {
        let gray = image::open(path)?.to_luma8();
        let (w, h) = gray.dimensions();
        let tiles = args.tile_width.map_or(1, |tw| (w / tw).max(1));
        for cam in 0..tiles {
            let tile = match args.tile_width {
                Some(tw) => image::imageops::crop_imm(&gray, cam * tw, 0, tw.min(w), h).to_image(),
                None => gray.clone(),
            };
            let (_result, diag, residuals) =
                detector.detect_adaptive_with_photometric_residuals(&tile)?;
            n_tiles += 1;
            for r in residuals {
                let accepted = diag.markers[r.marker_index]
                    .photometric
                    .as_ref()
                    .is_some_and(|p| p.status == PhotometricStatus::Accepted);
                if accepted {
                    markers.push(MarkerData {
                        center: r.center,
                        blur_sigma_px: r.blur_sigma_px,
                        samples: r.samples,
                    });
                }
            }
        }
    }
    eprintln!("{} images, {} tiles", args.images.len(), n_tiles);
    if markers.is_empty() {
        return Err("no accepted photometric fits".into());
    }

    let report = residual_profile::analyze(&markers);
    report.print();
    if let Some(path) = args.json {
        std::fs::write(path, serde_json::to_string_pretty(&report.to_json())?)?;
    }
    Ok(())
}
