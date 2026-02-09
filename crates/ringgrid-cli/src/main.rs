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

    /// Print default board specification.
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
    /// Path to a board layout JSON file (target specification).
    /// When omitted, uses the built-in default board layout.
    #[arg(long)]
    target: Option<PathBuf>,

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

    /// Fixed marker outer diameter in pixels (legacy compatibility path).
    ///
    /// When set, this overrides `--marker-diameter-min` / `--marker-diameter-max`
    /// and uses a fixed-scale prior.
    #[arg(long)]
    marker_diameter: Option<f64>,

    /// Minimum marker outer diameter in pixels for scale search.
    #[arg(long, default_value = "20.0")]
    marker_diameter_min: f64,

    /// Maximum marker outer diameter in pixels for scale search.
    #[arg(long, default_value = "56.0")]
    marker_diameter_max: f64,

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

    /// Completion ROI radius in pixels for edge sampling.
    ///
    /// Default is derived from nominal diameter of the selected marker scale prior.
    #[arg(long)]
    complete_roi_radius: Option<f64>,

    /// Circle refinement method after local fits are accepted.
    #[arg(long, value_enum, default_value_t = CircleRefineMethodArg::ProjectiveCenter)]
    circle_refine_method: CircleRefineMethodArg,

    /// Projective center gate: maximum allowed correction shift (px).
    ///
    /// Default is derived from nominal diameter of the selected marker scale prior.
    #[arg(long)]
    proj_center_max_shift_px: Option<f64>,

    /// Projective center gate: reject candidates with residual above this value.
    #[arg(long, default_value = "0.25")]
    proj_center_max_residual: f64,

    /// Projective center gate: reject candidates with eigen-separation below this value.
    #[arg(long, default_value = "1e-6")]
    proj_center_min_eig_sep: f64,

    /// Enable self-undistort: estimate a 1-parameter division-model distortion
    /// from detected markers, then re-run detection with that model.
    #[arg(long)]
    self_undistort: bool,

    /// Self-undistort: minimum lambda search bound.
    #[arg(long, default_value = "-8e-7")]
    self_undistort_lambda_min: f64,

    /// Self-undistort: maximum lambda search bound.
    #[arg(long, default_value = "8e-7")]
    self_undistort_lambda_max: f64,

    /// Self-undistort: minimum number of markers with inner+outer edge points.
    #[arg(long, default_value = "6")]
    self_undistort_min_markers: usize,

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
    fn to_core(&self) -> CliResult<Option<ringgrid::CameraModel>> {
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

        let model = ringgrid::CameraModel {
            intrinsics: ringgrid::CameraIntrinsics {
                fx: self.cam_fx.expect("validated"),
                fy: self.cam_fy.expect("validated"),
                cx: self.cam_cx.expect("validated"),
                cy: self.cam_cy.expect("validated"),
            },
            distortion: ringgrid::RadialTangentialDistortion {
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
}

impl CircleRefineMethodArg {
    fn to_core(self) -> ringgrid::CircleRefinementMethod {
        match self {
            Self::None => ringgrid::CircleRefinementMethod::None,
            Self::ProjectiveCenter => ringgrid::CircleRefinementMethod::ProjectiveCenter,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DetectPreset {
    marker_scale: ringgrid::MarkerScalePrior,
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
    camera: Option<ringgrid::CameraModel>,
    circle_refinement: ringgrid::CircleRefinementMethod,
    projective_center_max_shift_px: Option<f64>,
    projective_center_max_residual: f64,
    projective_center_min_eig_sep: f64,
    self_undistort_enable: bool,
    self_undistort_lambda_range: [f64; 2],
    self_undistort_min_markers: usize,
}

impl CliDetectArgs {
    fn to_preset(&self) -> DetectPreset {
        let marker_scale = if let Some(d) = self.marker_diameter {
            ringgrid::MarkerScalePrior::from_nominal_diameter_px(d as f32)
        } else {
            ringgrid::MarkerScalePrior::new(
                self.marker_diameter_min as f32,
                self.marker_diameter_max as f32,
            )
        };
        DetectPreset { marker_scale }
    }

    fn to_overrides(&self) -> CliResult<DetectOverrides> {
        let circle_refinement = self.circle_refine_method.to_core();

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
            self_undistort_enable: self.self_undistort,
            self_undistort_lambda_range: [
                self.self_undistort_lambda_min,
                self.self_undistort_lambda_max,
            ],
            self_undistort_min_markers: self.self_undistort_min_markers,
        })
    }
}

fn build_detect_config(
    board: ringgrid::BoardLayout,
    preset: DetectPreset,
    overrides: &DetectOverrides,
) -> ringgrid::DetectConfig {
    let mut config =
        ringgrid::DetectConfig::from_target_and_scale_prior(board, preset.marker_scale);

    // Global filter and refinement options
    config.use_global_filter = overrides.use_global_filter;
    config.refine_with_h = overrides.refine_with_h;
    config.ransac_homography.inlier_threshold = overrides.ransac_thresh_px;
    config.ransac_homography.max_iters = overrides.ransac_iters;

    // Homography-guided completion options
    config.completion.enable = overrides.completion_enable;
    config.completion.reproj_gate_px = overrides.completion_reproj_gate_px;
    config.completion.min_fit_confidence = overrides.completion_min_fit_confidence;
    if let Some(roi) = overrides.completion_roi_radius_px {
        config.completion.roi_radius_px = roi;
    }
    config.camera = overrides.camera;

    // Center refinement method
    config.circle_refinement = overrides.circle_refinement;
    config.projective_center.enable = config.circle_refinement.uses_projective_center();
    if let Some(shift) = overrides.projective_center_max_shift_px {
        config.projective_center.max_center_shift_px = Some(shift);
    }
    config.projective_center.max_selected_residual = Some(overrides.projective_center_max_residual);
    config.projective_center.min_eig_separation = Some(overrides.projective_center_min_eig_sep);

    // Self-undistort options
    config.self_undistort.enable = overrides.self_undistort_enable;
    config.self_undistort.lambda_range = overrides.self_undistort_lambda_range;
    config.self_undistort.min_markers = overrides.self_undistort_min_markers;

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
    use ringgrid::codebook::*;

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
    let board = ringgrid::BoardLayout::default();

    println!("ringgrid default board specification");
    println!("  name:           {}", board.name);
    println!("  markers:        {}", board.n_markers());
    println!("  pitch:          {} mm", board.pitch_mm);
    println!("  rows:           {}", board.rows);
    println!("  long row cols:  {}", board.long_row_cols);
    if let Some(span) = board.marker_span_mm() {
        println!("  marker span:    {:.3}x{:.3} mm", span[0], span[1]);
    }

    if board.n_markers() > 0 {
        let first = &board.markers[0];
        let last = &board.markers[board.n_markers() - 1];
        println!(
            "  marker 0:       ({:.1}, {:.1}) mm  [q={}, r={}]",
            first.xy_mm[0],
            first.xy_mm[1],
            first.q.unwrap_or_default(),
            first.r.unwrap_or_default()
        );
        println!(
            "  marker {}:    ({:.1}, {:.1}) mm  [q={}, r={}]",
            board.n_markers() - 1,
            last.xy_mm[0],
            last.xy_mm[1],
            last.q.unwrap_or_default(),
            last.r.unwrap_or_default()
        );
    }

    Ok(())
}

// ── decode-test ────────────────────────────────────────────────────────

fn run_decode_test(word_str: &str) -> CliResult<()> {
    use ringgrid::codec::{Codebook, Match};

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
    if args.marker_diameter.is_some() {
        tracing::warn!(
            "--marker-diameter is legacy fixed-size mode; prefer --marker-diameter-min/--marker-diameter-max"
        );
    }
    let overrides = args.to_overrides()?;

    let board = if let Some(target_path) = &args.target {
        let board =
            ringgrid::BoardLayout::from_json_file(target_path).map_err(|e| -> CliError {
                format!(
                    "Failed to load target spec {}: {}",
                    target_path.display(),
                    e
                )
                .into()
            })?;
        tracing::info!(
            "Loaded board layout '{}' with {} markers",
            board.name,
            board.n_markers()
        );
        board
    } else {
        ringgrid::BoardLayout::default()
    };

    let config = build_detect_config(board, preset, &overrides);

    let deprecated_debug_path = args.debug.as_deref();
    let debug_out_path = args.debug_json.as_deref().or(deprecated_debug_path);

    if deprecated_debug_path.is_some() && args.debug_json.is_none() {
        tracing::warn!("--debug is deprecated; use --debug-json instead");
    }

    let detector = ringgrid::Detector::with_config(config.clone());
    let (result, debug_dump) = if debug_out_path.is_some() {
        let dbg_cfg = ringgrid::DebugCollectConfig {
            image_path: Some(args.image.display().to_string()),
            marker_diameter_px: preset.marker_scale.nominal_diameter_px() as f64,
            max_candidates: args.debug_max_candidates,
            store_points: args.debug_store_points,
        };
        let (r, d) = detector.detect_with_debug(&gray, &dbg_cfg);
        (r, Some(d))
    } else if config.self_undistort.enable {
        (detector.detect_with_self_undistort(&gray), None)
    } else {
        (detector.detect(&gray), None)
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

    if let Some(ref su) = result.self_undistort {
        tracing::info!(
            "Self-undistort: lambda={:.3e}, obj {:.6} -> {:.6}, {} markers, applied={}",
            su.model.lambda,
            su.objective_at_zero,
            su.objective_at_lambda,
            su.n_markers_used,
            su.applied,
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
