//! ringgrid CLI — command-line interface for ring marker detection.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

type CliError = Box<dyn std::error::Error>;
type CliResult<T> = Result<T, CliError>;

#[derive(Parser)]
#[command(name = "ringgrid")]
#[command(
    about = "Detect circle/ring calibration targets in images (hex lattice, 16-sector coded rings)"
)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Detect markers in an image.
    Detect(CliDetectArgs),

    /// Print embedded codebook statistics.
    CodebookInfo,

    /// Print embedded board specification.
    BoardInfo,

    /// Decode a 16-bit word against the embedded codebook.
    DecodeTest {
        /// Observed 16-bit word (hex, e.g. 0xABCD).
        #[arg(long)]
        word: String,
    },
}

#[derive(Debug, Clone, Args)]
struct CliDetectArgs {
    /// Path to the input image.
    #[arg(long)]
    image: PathBuf,

    /// Path to write detection results (JSON).
    #[arg(long)]
    out: PathBuf,

    /// DEPRECATED: use --debug-json. Writes a versioned debug dump (JSON).
    #[arg(long)]
    debug: Option<PathBuf>,

    /// Path to write a comprehensive versioned debug dump (JSON).
    #[arg(long)]
    debug_json: Option<PathBuf>,

    /// Include edge point arrays in debug dump (can get large).
    #[arg(long)]
    debug_store_points: bool,

    /// Maximum number of candidates to record in the debug dump.
    #[arg(long, default_value = "300")]
    debug_max_candidates: usize,

    /// Expected marker outer diameter in pixels (for parameter tuning).
    #[arg(long, default_value = "32.0")]
    marker_diameter: f64,

    /// RANSAC inlier threshold in pixels for homography fitting.
    #[arg(long, default_value = "5.0")]
    ransac_thresh_px: f64,

    /// Maximum RANSAC iterations for homography.
    #[arg(long, default_value = "2000")]
    ransac_iters: usize,

    /// Disable global homography filtering.
    #[arg(long)]
    no_global_filter: bool,

    /// Disable refinement using fitted homography.
    #[arg(long)]
    no_refine: bool,

    /// Disable homography-guided completion (fitting missing IDs at H-projected locations).
    #[arg(long)]
    no_complete: bool,

    /// Completion reprojection gate in pixels (tight).
    #[arg(long, default_value = "3.0")]
    complete_reproj_gate: f64,

    /// Minimum completion fit confidence in [0, 1].
    #[arg(long, default_value = "0.45")]
    complete_min_conf: f32,

    /// Completion ROI radius in pixels for edge sampling (default: 0.75 * marker_diameter, clamped).
    #[arg(long)]
    complete_roi_radius: Option<f64>,

    /// Disable per-marker board-plane center refinement.
    #[arg(long)]
    no_nl_refine: bool,

    /// Circle refinement method after local fits are accepted.
    #[arg(long, value_enum, default_value_t = CircleRefineMethodArg::ProjectiveCenter)]
    circle_refine_method: CircleRefineMethodArg,

    /// Projective center gate: maximum allowed correction shift (px).
    /// Defaults to marker_diameter when omitted.
    #[arg(long)]
    proj_center_max_shift_px: Option<f64>,

    /// Projective center gate: reject candidates with residual above this value.
    #[arg(long, default_value = "0.25")]
    proj_center_max_residual: f64,

    /// Projective center gate: reject candidates with eigen-separation below this value.
    #[arg(long, default_value = "1e-6")]
    proj_center_min_eig_sep: f64,

    /// NL refine: maximum solver iterations.
    #[arg(long, default_value = "20")]
    nl_max_iters: usize,

    /// NL refine: Huber delta in board units (mm).
    #[arg(long, default_value = "0.2")]
    nl_huber_delta_mm: f64,

    /// NL refine: minimum number of edge points required.
    #[arg(long, default_value = "6")]
    nl_min_points: usize,

    /// NL refine: reject if refined center shifts more than this (mm) in board coordinates.
    #[arg(long, default_value = "1.0")]
    nl_reject_shift_mm: f64,

    /// NL refine: solver backend for fixed-radius circle center optimization.
    #[arg(long, value_enum, default_value_t = NlSolverArg::Lm)]
    nl_solver: NlSolverArg,

    /// NL refine: enable a single homography refit from refined centers.
    #[arg(long, default_value = "true")]
    nl_h_refit: bool,

    /// NL refine: disable homography refit from refined centers.
    #[arg(long, conflicts_with = "nl_h_refit")]
    no_nl_h_refit: bool,

    #[command(flatten)]
    camera: CliCameraArgs,
}

#[derive(Debug, Clone, Args, Default)]
struct CliCameraArgs {
    /// Camera intrinsic fx (pixels). If set, fy/cx/cy are required too.
    #[arg(long)]
    cam_fx: Option<f64>,
    /// Camera intrinsic fy (pixels). If set, fx/cx/cy are required too.
    #[arg(long)]
    cam_fy: Option<f64>,
    /// Camera principal point cx (pixels). If set, fx/fy/cy are required too.
    #[arg(long)]
    cam_cx: Option<f64>,
    /// Camera principal point cy (pixels). If set, fx/fy/cx are required too.
    #[arg(long)]
    cam_cy: Option<f64>,
    /// Radial distortion coefficient k1.
    #[arg(long, default_value_t = 0.0)]
    cam_k1: f64,
    /// Radial distortion coefficient k2.
    #[arg(long, default_value_t = 0.0)]
    cam_k2: f64,
    /// Tangential distortion coefficient p1.
    #[arg(long, default_value_t = 0.0)]
    cam_p1: f64,
    /// Tangential distortion coefficient p2.
    #[arg(long, default_value_t = 0.0)]
    cam_p2: f64,
    /// Radial distortion coefficient k3.
    #[arg(long, default_value_t = 0.0)]
    cam_k3: f64,
}

impl CliCameraArgs {
    fn to_core(&self) -> CliResult<Option<ringgrid_core::camera::CameraModel>> {
        let intr = [self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy];
        let any_intr = intr.iter().any(Option::is_some);
        if !any_intr {
            return Ok(None);
        }
        if intr.iter().any(Option::is_none) {
            return Err(
                "camera intrinsics are partial; provide all of --cam-fx --cam-fy --cam-cx --cam-cy"
                    .to_string()
                    .into(),
            );
        }

        let model = ringgrid_core::camera::CameraModel {
            intrinsics: ringgrid_core::camera::CameraIntrinsics {
                fx: self.cam_fx.expect("validated"),
                fy: self.cam_fy.expect("validated"),
                cx: self.cam_cx.expect("validated"),
                cy: self.cam_cy.expect("validated"),
            },
            distortion: ringgrid_core::camera::RadialTangentialDistortion {
                k1: self.cam_k1,
                k2: self.cam_k2,
                p1: self.cam_p1,
                p2: self.cam_p2,
                k3: self.cam_k3,
            },
        };
        if !model.intrinsics.is_valid() {
            return Err("invalid camera intrinsics: fx/fy must be finite and non-zero".into());
        }
        Ok(Some(model))
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CircleRefineMethodArg {
    None,
    ProjectiveCenter,
    NlBoard,
}

impl CircleRefineMethodArg {
    fn to_core(self) -> ringgrid_core::ring::CircleRefinementMethod {
        match self {
            Self::None => ringgrid_core::ring::CircleRefinementMethod::None,
            Self::ProjectiveCenter => ringgrid_core::ring::CircleRefinementMethod::ProjectiveCenter,
            Self::NlBoard => ringgrid_core::ring::CircleRefinementMethod::NlBoard,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum NlSolverArg {
    Irls,
    Lm,
}

impl NlSolverArg {
    fn to_core(self) -> ringgrid_core::refine::CircleCenterSolver {
        match self {
            Self::Irls => ringgrid_core::refine::CircleCenterSolver::Irls,
            Self::Lm => ringgrid_core::refine::CircleCenterSolver::Lm,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DetectPreset {
    marker_diameter_px: f32,
}

#[derive(Debug, Clone)]
struct DetectOverrides {
    use_global_filter: bool,
    refine_with_h: bool,
    ransac_thresh_px: f64,
    ransac_iters: usize,
    completion_enable: bool,
    completion_reproj_gate_px: f32,
    completion_min_fit_confidence: f32,
    completion_roi_radius_px: Option<f32>,
    camera: Option<ringgrid_core::camera::CameraModel>,
    circle_refinement: ringgrid_core::ring::CircleRefinementMethod,
    projective_center_max_shift_px: Option<f64>,
    projective_center_max_residual: f64,
    projective_center_min_eig_sep: f64,
    nl_max_iters: usize,
    nl_huber_delta_mm: f64,
    nl_min_points: usize,
    nl_reject_shift_mm: f64,
    nl_solver: ringgrid_core::refine::CircleCenterSolver,
    nl_enable_h_refit: bool,
}

impl CliDetectArgs {
    fn to_preset(&self) -> DetectPreset {
        DetectPreset {
            marker_diameter_px: self.marker_diameter as f32,
        }
    }

    fn to_overrides(&self) -> CliResult<DetectOverrides> {
        let mut circle_refinement = self.circle_refine_method.to_core();
        if self.no_nl_refine
            && circle_refinement == ringgrid_core::ring::CircleRefinementMethod::NlBoard
        {
            circle_refinement = ringgrid_core::ring::CircleRefinementMethod::None;
        }

        Ok(DetectOverrides {
            use_global_filter: !self.no_global_filter,
            refine_with_h: !self.no_refine,
            ransac_thresh_px: self.ransac_thresh_px,
            ransac_iters: self.ransac_iters,
            completion_enable: !self.no_complete,
            completion_reproj_gate_px: self.complete_reproj_gate as f32,
            completion_min_fit_confidence: self.complete_min_conf,
            completion_roi_radius_px: self.complete_roi_radius.map(|v| v as f32),
            camera: self.camera.to_core()?,
            circle_refinement,
            projective_center_max_shift_px: self.proj_center_max_shift_px,
            projective_center_max_residual: self.proj_center_max_residual,
            projective_center_min_eig_sep: self.proj_center_min_eig_sep,
            nl_max_iters: self.nl_max_iters,
            nl_huber_delta_mm: self.nl_huber_delta_mm,
            nl_min_points: self.nl_min_points,
            nl_reject_shift_mm: self.nl_reject_shift_mm,
            nl_solver: self.nl_solver.to_core(),
            nl_enable_h_refit: self.nl_h_refit && !self.no_nl_h_refit,
        })
    }
}

fn build_detect_config(
    preset: DetectPreset,
    overrides: &DetectOverrides,
) -> ringgrid_core::ring::DetectConfig {
    // Configure detection parameters from marker_diameter
    let r_outer = preset.marker_diameter_px / 2.0;
    let mut config = ringgrid_core::ring::DetectConfig {
        marker_diameter_px: preset.marker_diameter_px,
        ..Default::default()
    };

    // Scale proposal search radii
    config.proposal.r_min = (r_outer * 0.4).max(2.0);
    config.proposal.r_max = r_outer * 1.7;
    config.proposal.nms_radius = r_outer * 0.8;

    // Scale edge sampling range
    config.edge_sample.r_max = r_outer * 2.0;
    config.edge_sample.r_min = 1.5;
    config.outer_estimation.theta_samples = config.edge_sample.n_rays;

    // Scale ellipse validation
    config.min_semi_axis = (r_outer as f64 * 0.3).max(2.0);
    config.max_semi_axis = r_outer as f64 * 2.5;

    // Global filter and refinement options
    config.use_global_filter = overrides.use_global_filter;
    config.refine_with_h = overrides.refine_with_h;
    config.ransac_homography.inlier_threshold = overrides.ransac_thresh_px;
    config.ransac_homography.max_iters = overrides.ransac_iters;

    // Homography-guided completion options
    config.completion.enable = overrides.completion_enable;
    config.completion.reproj_gate_px = overrides.completion_reproj_gate_px;
    config.completion.min_fit_confidence = overrides.completion_min_fit_confidence;
    config.completion.roi_radius_px = overrides
        .completion_roi_radius_px
        .unwrap_or(((preset.marker_diameter_px as f64 * 0.75).clamp(24.0, 80.0)) as f32);
    config.camera = overrides.camera;

    // Center refinement method
    config.circle_refinement = overrides.circle_refinement;
    config.projective_center.enable = config.circle_refinement.uses_projective_center();
    config.projective_center.max_center_shift_px = Some(
        overrides
            .projective_center_max_shift_px
            .unwrap_or(preset.marker_diameter_px as f64),
    );
    config.projective_center.max_selected_residual = Some(overrides.projective_center_max_residual);
    config.projective_center.min_eig_separation = Some(overrides.projective_center_min_eig_sep);

    // Non-linear refinement options (board-plane circle fit)
    config.nl_refine.enabled = config.circle_refinement.uses_nl_refine();
    config.nl_refine.max_iters = overrides.nl_max_iters;
    config.nl_refine.huber_delta_mm = overrides.nl_huber_delta_mm;
    config.nl_refine.min_points = overrides.nl_min_points;
    config.nl_refine.reject_thresh_mm = overrides.nl_reject_shift_mm;
    config.nl_refine.solver = overrides.nl_solver;
    config.nl_refine.enable_h_refit = overrides.nl_enable_h_refit;

    config
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
        Commands::Detect(args) => run_detect(&args),
        Commands::CodebookInfo => run_codebook_info(),

        Commands::BoardInfo => run_board_info(),

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

// ── board-info ────────────────────────────────────────────────────────

fn run_board_info() -> CliResult<()> {
    use ringgrid_core::board_spec::*;

    println!("ringgrid embedded board specification");
    println!("  name:           {}", BOARD_NAME);
    println!("  markers:        {}", BOARD_N);
    println!("  pitch:          {} mm", BOARD_PITCH_MM);
    println!(
        "  board size:     {}x{} mm",
        BOARD_SIZE_MM[0], BOARD_SIZE_MM[1]
    );

    if BOARD_N > 0 {
        println!(
            "  marker 0:       ({:.1}, {:.1}) mm  [q={}, r={}]",
            BOARD_XY_MM[0][0], BOARD_XY_MM[0][1], BOARD_QR[0][0], BOARD_QR[0][1]
        );
        println!(
            "  marker {}:    ({:.1}, {:.1}) mm  [q={}, r={}]",
            BOARD_N - 1,
            BOARD_XY_MM[BOARD_N - 1][0],
            BOARD_XY_MM[BOARD_N - 1][1],
            BOARD_QR[BOARD_N - 1][0],
            BOARD_QR[BOARD_N - 1][1]
        );
    }

    Ok(())
}

// ── decode-test ────────────────────────────────────────────────────────

fn run_decode_test(word_str: &str) -> CliResult<()> {
    use ringgrid_core::codec::{Codebook, Match};

    let word_str = word_str
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
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

fn run_detect(args: &CliDetectArgs) -> CliResult<()> {
    tracing::info!("Loading image: {}", args.image.display());

    let img = image::open(&args.image).map_err(|e| -> CliError {
        format!("Failed to open image {}: {}", args.image.display(), e).into()
    })?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();

    tracing::info!("Image size: {}x{}", w, h);

    let preset = args.to_preset();
    let overrides = args.to_overrides()?;
    let config = build_detect_config(preset, &overrides);

    // Run detection pipeline (optionally with debug dump)
    let deprecated_debug_path = args.debug.as_deref();
    let debug_out_path = args.debug_json.as_deref().or(deprecated_debug_path);

    if deprecated_debug_path.is_some() && args.debug_json.is_none() {
        tracing::warn!("--debug is deprecated; use --debug-json instead");
    }

    let (result, debug_dump) = if debug_out_path.is_some() {
        let dbg_cfg = ringgrid_core::ring::DebugCollectConfig {
            image_path: Some(args.image.display().to_string()),
            marker_diameter_px: args.marker_diameter,
            max_candidates: args.debug_max_candidates,
            store_points: args.debug_store_points,
        };
        let (r, d) = ringgrid_core::ring::detect_rings_with_debug(&gray, &config, &dbg_cfg);
        (r, Some(d))
    } else {
        (ringgrid_core::ring::detect_rings(&gray, &config), None)
    };

    let n_with_id = result
        .detected_markers
        .iter()
        .filter(|m| m.id.is_some())
        .count();
    tracing::info!(
        "Detected {} markers ({} with ID)",
        result.detected_markers.len(),
        n_with_id,
    );

    if let Some(ref stats) = result.ransac {
        tracing::info!(
            "Homography: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
            stats.n_inliers,
            stats.n_candidates,
            stats.mean_err_px,
            stats.p95_err_px,
        );
    }

    // Write results
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.out, &json)?;
    tracing::info!("Results written to {}", args.out.display());

    // Write debug dump (versioned schema)
    if let Some(debug_path) = debug_out_path {
        let dump = debug_dump.expect("debug dump present when debug_out_path is set");
        let debug_json = serde_json::to_string_pretty(&dump)?;
        std::fs::write(debug_path, &debug_json)?;
        tracing::info!("Debug dump written to {}", debug_path.display());
    }

    Ok(())
}
