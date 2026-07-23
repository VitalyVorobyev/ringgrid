//! Generate every supported target combination — one constructor per row of the
//! target matrix — and write the printable artifacts for each.
//!
//! ```bash
//! cargo run --example generate_targets -- ./out/targets
//! ```
//!
//! Each target writes four files into its own subdirectory: the canonical
//! `target_spec.json` the detector reads, plus a printable `.svg` / `.png` and a
//! `.dxf` for laser/CNC fabrication.

use ringgrid::{OriginDots, PngTargetOptions, SvgTargetOptions, TargetLayout};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "out".to_string());
    let out_dir = Path::new(&out_dir);

    // Markers must be identifiable. That comes from codes, from origin dots, or
    // from detecting the complete board — these six calls are every valid way
    // to combine those. Coded targets take no dots: decoded IDs already anchor
    // the board, so coded-plus-dots is rejected at construction.
    let targets = [
        (
            "hex_coded",
            TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)?,
        ),
        (
            "rect_coded",
            TargetLayout::coded_rect(14.0, 20, 20, 4.8, 3.2, 1.152)?,
        ),
        (
            "hex_plain_dots",
            TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::Auto)?,
        ),
        (
            "hex_plain_nodots",
            TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::None)?,
        ),
        (
            "rect_plain_dots",
            TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)?,
        ),
        (
            "rect_plain_nodots",
            TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::None)?,
        ),
    ];

    let margin_mm = 5.0;
    for (name, target) in targets {
        // Constructors derive a deterministic geometry-based name; override it
        // when you want something you'll recognize on a printed sheet.
        let target = target.with_name(name)?;
        let dir = out_dir.join(name);

        target.write_json_file(&dir.join("target_spec.json"))?;
        target.write_target_svg(
            &dir.join("target_print.svg"),
            &SvgTargetOptions {
                margin_mm,
                include_scale_bar: true,
            },
        )?;
        target.write_target_png(
            &dir.join("target_print.png"),
            &PngTargetOptions {
                dpi: 300.0,
                margin_mm,
                include_scale_bar: true,
            },
        )?;
        target.write_target_dxf(&dir.join("target_print.dxf"))?;

        // Check the physical size before committing to a print run — the plain
        // rect defaults above make a board larger than A3.
        let side_mm = target.print_side_mm(margin_mm);
        let dots = target.fiducial_dots_mm().len();
        println!(
            "{name}: {} markers, {dots} origin dots, {side_mm:.1} x {side_mm:.1} mm -> {}",
            target.n_cells(),
            dir.display()
        );
    }

    println!("\nPrint at 100% scale, then check the printed scale bar with a ruler.");
    Ok(())
}
