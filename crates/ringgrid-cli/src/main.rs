//! ringgrid CLI — command-line interface for ring marker detection.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

type CliError = Box<dyn std::error::Error>;
type CliResult<T> = Result<T, CliError>;

#[derive(Parser)]
#[command(name = "ringgrid")]
#[command(about = "Detect circle/ring calibration targets in images (hex lattice, 16-sector coded rings)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect markers in an image.
    Detect {
        /// Path to the input image.
        #[arg(long)]
        image: PathBuf,

        /// Path to write detection results (JSON).
        #[arg(long)]
        out: PathBuf,

        /// Path to write debug information (JSON).
        #[arg(long)]
        debug: Option<PathBuf>,

        /// Expected marker outer diameter in pixels (for parameter tuning).
        #[arg(long, default_value = "32.0")]
        marker_diameter: f64,
    },

    /// Print embedded codebook statistics.
    CodebookInfo,

    /// Decode a 16-bit word against the embedded codebook.
    DecodeTest {
        /// Observed 16-bit word (hex, e.g. 0xABCD).
        #[arg(long)]
        word: String,
    },
}

fn main() -> CliResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Detect {
            image: image_path,
            out,
            debug,
            marker_diameter,
        } => run_detect(&image_path, &out, debug.as_deref(), marker_diameter),

        Commands::CodebookInfo => run_codebook_info(),

        Commands::DecodeTest { word } => run_decode_test(&word),
    }
}

// ── codebook-info ──────────────────────────────────────────────────────

fn run_codebook_info() -> CliResult<()> {
    use ringgrid_core::codebook::*;

    println!("ringgrid embedded codebook");
    println!("  bits per codeword:    {}", CODEBOOK_BITS);
    println!("  number of codewords:  {}", CODEBOOK_N);
    println!("  min cyclic Hamming:   {}", CODEBOOK_MIN_CYCLIC_DIST);
    println!("  generator seed:       {}", CODEBOOK_SEED);

    if CODEBOOK_N > 0 {
        println!("  first codeword:       0x{:04X}", CODEBOOK[0]);
        println!("  last codeword:        0x{:04X}", CODEBOOK[CODEBOOK_N - 1]);
    }

    Ok(())
}

// ── decode-test ────────────────────────────────────────────────────────

fn run_decode_test(word_str: &str) -> CliResult<()> {
    use ringgrid_core::codec::{Codebook, Match};

    let word_str = word_str.trim().trim_start_matches("0x").trim_start_matches("0X");
    let word = u16::from_str_radix(word_str, 16)
        .map_err(|e| -> CliError { format!("invalid hex word: {}", e).into() })?;

    let cb = Codebook::default();
    let m: Match = cb.match_word(word);

    println!("Input word:   0x{:04X} (binary: {:016b})", word, word);
    println!("Best match:");
    println!("  id:         {}", m.id);
    println!("  codeword:   0x{:04X}", cb.word(m.id).unwrap_or(0));
    println!("  rotation:   {} sectors", m.rotation);
    println!("  distance:   {} bits", m.dist);
    println!("  margin:     {} bits", m.margin);
    println!("  confidence: {:.3}", m.confidence);

    Ok(())
}

// ── detect ─────────────────────────────────────────────────────────────

fn run_detect(
    image_path: &std::path::Path,
    out_path: &std::path::Path,
    debug_path: Option<&std::path::Path>,
    marker_diameter: f64,
) -> CliResult<()> {
    tracing::info!("Loading image: {}", image_path.display());

    let img = image::open(image_path).map_err(|e| -> CliError {
        format!("Failed to open image {}: {}", image_path.display(), e).into()
    })?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();

    tracing::info!("Image size: {}x{}", w, h);

    // Configure detection parameters from marker_diameter
    let r_outer = marker_diameter as f32 / 2.0;
    let mut config = ringgrid_core::ring::DetectConfig::default();

    // Scale proposal search radii
    config.proposal.r_min = (r_outer * 0.4).max(2.0);
    config.proposal.r_max = r_outer * 1.7;
    config.proposal.nms_radius = r_outer * 0.8;

    // Scale edge sampling range
    config.edge_sample.r_max = r_outer * 2.0;
    config.edge_sample.r_min = 1.5;

    // Scale ellipse validation
    config.min_semi_axis = (r_outer as f64 * 0.3).max(2.0);
    config.max_semi_axis = r_outer as f64 * 2.5;

    // Run detection pipeline
    let result = ringgrid_core::ring::detect_rings(&gray, &config);
    tracing::info!("Detected {} markers", result.detected_markers.len());

    // Write results
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(out_path, &json)?;
    tracing::info!("Results written to {}", out_path.display());

    // Write debug output (same data, full detail)
    if let Some(debug_path) = debug_path {
        let debug_json = serde_json::to_string_pretty(&result)?;
        std::fs::write(debug_path, &debug_json)?;
        tracing::info!("Debug info written to {}", debug_path.display());
    }

    Ok(())
}
