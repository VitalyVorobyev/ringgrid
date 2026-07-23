//! `ringgrid` — the published command-line tool.
//!
//! Generate printable calibration targets from a recipe, and detect them in
//! single images or whole directories. Built only with `--features cli`.
//!
//! ```text
//! ringgrid gen     <recipe.toml>  --out DIR         # target artifacts
//! ringgrid detect  --image P --target T --out J     # one image
//! ringgrid batch   --images DIR --target T --out-dir D
//! ringgrid example --list | --name NAME [--out FILE]
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use ringgrid::cli::{
    self, BatchOutcome, Format, TargetRecipe, build_detector, decoded_count, detect_file,
    image_files_in_dir, load_target, run_batch, write_target_artifacts,
};

#[derive(Parser)]
#[command(
    name = "ringgrid",
    version,
    about = "Generate ringgrid calibration targets and detect them in images"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate target artifacts (JSON spec + SVG/PNG/DXF) from a recipe.
    Gen(GenArgs),
    /// Detect markers in a single image.
    Detect(DetectArgs),
    /// Detect markers across every image in a directory.
    Batch(BatchArgs),
    /// Write or list the built-in example recipes.
    Example(ExampleArgs),
}

#[derive(Args)]
struct GenArgs {
    /// Recipe file (`.toml` or `.json`). See `ringgrid example`.
    recipe: PathBuf,
    /// Output directory (created if absent).
    #[arg(long, default_value = "out")]
    out: PathBuf,
    /// Basename for printable artifacts (svg/png/dxf).
    #[arg(long, default_value = "target_print")]
    basename: String,
    /// Override the target name.
    #[arg(long)]
    name: Option<String>,
    /// Override the lattice pitch (mm).
    #[arg(long)]
    pitch_mm: Option<f32>,
    /// Override the PNG resolution (dpi).
    #[arg(long)]
    dpi: Option<f32>,
    /// Override the print margin (mm).
    #[arg(long)]
    margin_mm: Option<f32>,
    /// Override the emitted formats (comma-separated: json,svg,png,dxf).
    #[arg(long, value_delimiter = ',')]
    formats: Option<Vec<Format>>,
}

#[derive(Args)]
struct DetectArgs {
    /// Input image.
    #[arg(long)]
    image: PathBuf,
    /// Target spec (`target_spec.json`) or recipe (`.toml`/`.json`).
    #[arg(long)]
    target: PathBuf,
    /// Output JSON path (stdout if omitted).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Approximate marker diameter (px) for focused single-pass detection.
    #[arg(long)]
    marker_diameter: Option<f32>,
    /// Detection-config overlay (`.json`/`.toml`).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Require the complete board: fail if any cell is missing.
    #[arg(long)]
    strict: bool,
}

#[derive(Args)]
struct BatchArgs {
    /// Directory of input images.
    #[arg(long)]
    images: PathBuf,
    /// Target spec or recipe.
    #[arg(long)]
    target: PathBuf,
    /// Directory for per-image `<stem>.json` results (created if absent).
    #[arg(long)]
    out_dir: PathBuf,
    /// Aggregate summary path (defaults to `<out_dir>/summary.json`).
    #[arg(long)]
    summary: Option<PathBuf>,
    /// Approximate marker diameter (px) for focused single-pass detection.
    #[arg(long)]
    marker_diameter: Option<f32>,
    /// Detection-config overlay (`.json`/`.toml`).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Require the complete board on every image.
    #[arg(long)]
    strict: bool,
}

#[derive(Args)]
struct ExampleArgs {
    /// List available example recipes and exit.
    #[arg(long)]
    list: bool,
    /// Name of the example to emit (see `--list`).
    #[arg(long)]
    name: Option<String>,
    /// Write the example to this file (stdout if omitted).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .ok();

    let cli = Cli::parse();
    let result = match cli.command {
        Command::Gen(args) => run_gen(args),
        Command::Detect(args) => run_detect(args),
        Command::Batch(args) => run_batch_cmd(args),
        Command::Example(args) => run_example(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run_gen(args: GenArgs) -> Result<(), String> {
    let mut recipe = load_recipe(&args.recipe)?;

    // CLI overrides take precedence over recipe fields.
    if let Some(name) = args.name {
        recipe.name = name;
    }
    if let Some(pitch) = args.pitch_mm {
        recipe.set_pitch_mm(pitch);
    }
    if let Some(dpi) = args.dpi {
        recipe.render.dpi = dpi;
    }
    if let Some(margin) = args.margin_mm {
        recipe.render.margin_mm = margin;
    }
    if let Some(formats) = args.formats {
        recipe.render.formats = formats;
    }

    let target = recipe.to_target().map_err(|e| e.to_string())?;
    let written = write_target_artifacts(&target, &args.out, &args.basename, &recipe.render)
        .map_err(|e| e.to_string())?;

    for path in &written {
        println!("wrote {}", path.display());
    }
    print_target_summary(&target, &recipe.render);
    Ok(())
}

/// Paper sizes a target might be printed on, smallest first (mm, portrait).
const PAPER_SIZES: &[(&str, f32, f32)] = &[
    ("A4", 210.0, 297.0),
    ("Letter", 215.9, 279.4),
    ("A3", 297.0, 420.0),
];

/// Describe what `gen` just produced: geometry, marker/dot counts, and the
/// physical print size — the facts you need before sending a target to a
/// printer, none of which are obvious from the four output paths alone.
fn print_target_summary(target: &ringgrid::TargetLayout, render: &cli::RenderRecipe) {
    let lattice = match target.lattice() {
        ringgrid::LatticeGeometry::Hex(hex) => {
            format!("hex {} rows x {} cols", hex.rows, hex.long_row_cols)
        }
        ringgrid::LatticeGeometry::Rect(rect) => {
            format!("rect {}x{}", rect.rows, rect.cols)
        }
    };
    let coding = if target.is_coded() { "coded" } else { "plain" };
    let dots = match target.fiducials() {
        Some(_) => format!("{} origin dots", target.fiducial_dots_mm().len()),
        None => "no origin dots".to_string(),
    };
    eprintln!(
        "target: {} — {lattice}, pitch {:.1} mm, {coding}, {dots}",
        target.name(),
        target.pitch_mm()
    );
    eprintln!("markers: {}", target.n_cells());

    let side_mm = target.print_side_mm(render.margin_mm);
    let fits = PAPER_SIZES
        .iter()
        .find(|(_, w, h)| side_mm <= *w && side_mm <= *h);
    let note = match fits {
        Some((name, _, _)) => format!("fits {name}"),
        None => "exceeds A3 297x420 mm — use a plotter or tile the print".to_string(),
    };
    eprintln!("print size: {side_mm:.1} x {side_mm:.1} mm ({note})");

    if render.formats.contains(&cli::Format::Png) {
        let px = (f64::from(side_mm) * f64::from(render.dpi) / 25.4)
            .round()
            .max(1.0) as u32;
        eprintln!("png: {px} x {px} px @ {:.0} dpi", render.dpi);
    }
    eprintln!("print at 100% scale, then verify the printed scale bar with a ruler");
}

fn run_detect(args: DetectArgs) -> Result<(), String> {
    let target = load_target(&args.target).map_err(|e| e.to_string())?;
    let overlay = args.config.as_deref().map(load_overlay).transpose()?;
    let detector = build_detector(target, args.marker_diameter, args.strict, overlay)
        .map_err(|e| e.to_string())?;

    let result = detect_file(&detector, &args.image, args.marker_diameter.is_some())
        .map_err(|e| e.to_string())?;

    let json = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
    match &args.out {
        Some(path) => {
            std::fs::write(path, &json).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!(
                "detected {} markers ({} decoded){} -> {}",
                result.detected_markers.len(),
                decoded_count(&result),
                complete_note(result.board_complete),
                path.display()
            );
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn run_batch_cmd(args: BatchArgs) -> Result<(), String> {
    let target = load_target(&args.target).map_err(|e| e.to_string())?;
    let overlay = args.config.as_deref().map(load_overlay).transpose()?;
    let detector = build_detector(target, args.marker_diameter, args.strict, overlay)
        .map_err(|e| e.to_string())?;

    let images = image_files_in_dir(&args.images).map_err(|e| e.to_string())?;
    if images.is_empty() {
        return Err(format!("no images found in {}", args.images.display()));
    }
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("{}: {e}", args.out_dir.display()))?;

    let outcomes = run_batch(&detector, &images, args.marker_diameter.is_some());

    let mut n_ok = 0usize;
    let mut n_complete = 0usize;
    let mut items = Vec::with_capacity(outcomes.len());
    for outcome in &outcomes {
        let stem = outcome
            .image
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let item = summarize_outcome(outcome, &args.out_dir, stem)?;
        if outcome.result.is_some() {
            n_ok += 1;
        }
        if outcome
            .result
            .as_ref()
            .and_then(|r| r.board_complete)
            .unwrap_or(false)
        {
            n_complete += 1;
        }
        items.push(item);
    }

    let summary = serde_json::json!({
        "n_images": outcomes.len(),
        "n_ok": n_ok,
        "n_complete": n_complete,
        "items": items,
    });
    let summary_path = args
        .summary
        .unwrap_or_else(|| args.out_dir.join("summary.json"));
    let summary_json = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    std::fs::write(&summary_path, summary_json)
        .map_err(|e| format!("{}: {e}", summary_path.display()))?;

    eprintln!(
        "batch: {} images, {n_ok} detected, {n_complete} complete -> {}",
        outcomes.len(),
        summary_path.display()
    );
    Ok(())
}

/// Write one batch outcome's result to `<out_dir>/<stem>.json` and return its
/// compact summary entry.
fn summarize_outcome(
    outcome: &BatchOutcome,
    out_dir: &Path,
    stem: &str,
) -> Result<serde_json::Value, String> {
    match &outcome.result {
        Some(result) => {
            let path = out_dir.join(format!("{stem}.json"));
            let json = serde_json::to_string_pretty(result).map_err(|e| e.to_string())?;
            std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(serde_json::json!({
                "image": outcome.image.display().to_string(),
                "ok": true,
                "markers": result.detected_markers.len(),
                "decoded": decoded_count(result),
                "board_complete": result.board_complete,
                "output": path.display().to_string(),
            }))
        }
        None => Ok(serde_json::json!({
            "image": outcome.image.display().to_string(),
            "ok": false,
            "error": outcome.error,
        })),
    }
}

fn run_example(args: ExampleArgs) -> Result<(), String> {
    if args.list || args.name.is_none() {
        eprintln!("available example recipes:");
        for (name, _) in cli::EXAMPLE_RECIPES {
            println!("{name}");
        }
        return Ok(());
    }
    let name = args.name.expect("checked above");
    let text = cli::example_recipe(&name)
        .ok_or_else(|| format!("unknown example '{name}' (see --list)"))?;
    match &args.out {
        Some(path) => {
            std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!("wrote {} -> {}", name, path.display());
        }
        None => print!("{text}"),
    }
    Ok(())
}

fn load_recipe(path: &Path) -> Result<TargetRecipe, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
    {
        TargetRecipe::parse_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    } else {
        TargetRecipe::parse_toml(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

fn load_overlay(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
    {
        toml::from_str::<serde_json::Value>(&text).map_err(|e| format!("{}: {e}", path.display()))
    } else {
        serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

fn complete_note(board_complete: Option<bool>) -> &'static str {
    match board_complete {
        Some(true) => ", board complete",
        Some(false) => ", board INCOMPLETE",
        None => "",
    }
}
