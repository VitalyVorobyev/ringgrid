//! Image-level runner for the paired real-image study in tools/metrology_real.py.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use ringgrid::photometric_residuals::PhotometricStudyOptions;
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, MarkerScalePrior};
use serde_json::json;

#[derive(Clone, ValueEnum)]
enum Method {
    Projective,
    PerImage,
    Nominal,
    Pooled,
}

#[derive(Parser)]
struct Args {
    image: PathBuf,
    target: PathBuf,
    #[arg(value_enum)]
    method: Method,
    #[arg(long, num_args = 4)]
    radii: Vec<f64>,
    #[arg(long, num_args = 2)]
    scale: Vec<f32>,
    #[arg(long)]
    no_moment: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let image = image::open(args.image)?.to_luma8();
    let target = ringgrid::cli::load_target(&args.target)?;
    let mut config = DetectConfig::from_target(target);
    let fixed_scale = !args.scale.is_empty();
    if fixed_scale {
        config.set_marker_scale_prior(MarkerScalePrior::new(args.scale[0], args.scale[1]));
    }
    let projective = matches!(args.method, Method::Projective);
    config.circle_refinement = if projective {
        CircleRefinementMethod::ProjectiveCenter
    } else {
        CircleRefinementMethod::Photometric
    };
    config.advanced.photometric.estimate_print_radii = matches!(args.method, Method::PerImage);
    let detector = Detector::with_config(config);
    let start = Instant::now();
    let (result, diagnostics, fits) = if projective {
        let (result, diagnostics) = if fixed_scale {
            detector.detect_with_diagnostics(&image)?
        } else {
            detector.detect_adaptive_with_diagnostics(&image)?
        };
        (result, diagnostics, Vec::new())
    } else {
        let mut options = PhotometricStudyOptions::default();
        options.moment_centers = !args.no_moment && matches!(args.method, Method::PerImage);
        if matches!(args.method, Method::Pooled) {
            let radii: [f64; 4] = args
                .radii
                .try_into()
                .map_err(|_| "pooled requires four radii")?;
            options = options
                .with_fixed_radii(&detector.config().target, radii)
                .ok_or("pooled radii do not describe the target's marker")?;
        }
        if fixed_scale {
            detector.detect_with_photometric_study(&image, options)?
        } else {
            detector.detect_adaptive_with_photometric_study(&image, options)?
        }
    };
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut fit_by_index = vec![None; result.detected_markers.len()];
    for fit in fits {
        let i = fit.marker_index;
        fit_by_index[i] = Some(fit);
    }
    let markers = result
        .detected_markers
        .iter()
        .zip(&diagnostics.markers)
        .zip(&fit_by_index)
        .map(|((m, d), fit)| {
            json!({
                "id": m.id,
                "coord": m.grid_coord,
                "board_xy_mm": m.board_xy_mm,
                "center": m.center,
                "source": m.center_source,
                "cov": m.center_cov,
                "diameter_px": m.ellipse_outer.as_ref().map(|e| 2.0 * e.mean_axis()),
                "contrast": d.photometric.as_ref().and_then(|p| p.contrast),
                "status": d.photometric.as_ref().map(|p| p.status),
                "blur_sigma_px": fit.as_ref().map(|f| f.blur_sigma_px),
                "radii_mm": fit.as_ref().map(|f| f.radii_mm),
                "moment_center": fit.as_ref().and_then(|f| f.moment_center),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        json!({
            "board_frame": result.board_frame,
            "estimated_radii_mm": diagnostics.photometric_print_radii_mm,
            "runtime_ms": runtime_ms,
            "markers": markers,
        })
    );
    Ok(())
}
