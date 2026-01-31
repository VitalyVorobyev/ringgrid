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

        /// Expected marker outer diameter in pixels (for band-pass tuning).
        #[arg(long, default_value = "20.0")]
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
            marker_diameter: _marker_diameter,
        } => run_detect(&image_path, &out, debug.as_deref()),

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
) -> CliResult<()> {
    tracing::info!("Loading image: {}", image_path.display());

    let img = image::open(image_path).map_err(|e| -> CliError {
        format!("Failed to open image {}: {}", image_path.display(), e).into()
    })?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();

    tracing::info!("Image size: {}x{}", w, h);

    // ── Placeholder pipeline ───────────────────────────────────────────
    // TODO Milestone 2: replace with real pipeline
    //
    // Real pipeline stages:
    // 1. preprocess::normalize_illumination
    // 2. preprocess::bandpass_filter
    // 3. edges::detect_edges
    // 4. edges::group_arcs
    // 5. conic::fit_ellipse_ransac (per arc group)
    // 6. lattice::build_neighbor_graph
    // 7. lattice::estimate_vanishing_line
    // 8. lattice::affine_rectification_homography
    // 9. refine::refine_marker (per marker)
    // 10. codec::decode_marker_id (per marker)

    let result = ringgrid_core::DetectionResult::empty(w, h);
    tracing::info!("Detected {} markers (pipeline is stub)", result.markers.len());

    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(out_path, &json)?;
    tracing::info!("Results written to {}", out_path.display());

    if let Some(debug_path) = debug_path {
        let debug_info = serde_json::json!({
            "image_path": image_path.to_string_lossy(),
            "image_size": [w, h],
            "pipeline_stages": [
                {"name": "preprocess", "status": "stub"},
                {"name": "edges", "status": "stub"},
                {"name": "conic_fit", "status": "stub"},
                {"name": "lattice", "status": "stub"},
                {"name": "refine", "status": "stub"},
                {"name": "codec", "status": "stub"},
            ],
            "detected_markers": [],
        });
        let debug_json = serde_json::to_string_pretty(&debug_info)?;
        std::fs::write(debug_path, &debug_json)?;
        tracing::info!("Debug info written to {}", debug_path.display());
    }

    Ok(())
}
