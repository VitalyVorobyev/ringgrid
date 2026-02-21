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

    /// Include pass-1 proposal diagnostics in output JSON.
    ///
    /// Adds `proposals`, `proposal_frame`, and `proposal_count` fields.
    #[arg(long)]
    include_proposals: bool,

    /// Fixed marker outer diameter in pixels (legacy compatibility path).
    ///
    /// When set, this overrides `--marker-diameter-min` / `--marker-diameter-max`
    /// and uses a fixed-scale prior.
    #[arg(long)]
    marker_diameter: Option<f64>,

    /// Minimum marker outer diameter in pixels for scale search.
    ///
    /// Default: 20.0 (built-in). Can also be set via `--config`
    /// (`marker_scale.diameter_min_px`). CLI takes precedence over config file.
    #[arg(long)]
    marker_diameter_min: Option<f64>,

    /// Maximum marker outer diameter in pixels for scale search.
    ///
    /// Default: 56.0 (built-in). Can also be set via `--config`
    /// (`marker_scale.diameter_max_px`). CLI takes precedence over config file.
    #[arg(long)]
    marker_diameter_max: Option<f64>,

    /// RANSAC inlier threshold in pixels for homography fitting.
    #[arg(long, default_value = "5.0")]
    ransac_thresh_px: f64,

    /// Maximum RANSAC iterations for homography.
    #[arg(long, default_value = "2000")]
    ransac_iters: usize,

    /// Disable global homography filtering.
    #[arg(long)]
    no_global_filter: bool,

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

    /// Require a perfect decode (dist=0, margin≥2) for completion markers.
    ///
    /// Recommended for high-distortion setups without a calibrated camera model
    /// (e.g. Scheimpflug cameras), where H-projected seeds may be inaccurate and
    /// geometry gates alone are insufficient to reject bad fits.
    #[arg(long)]
    complete_require_perfect_decode: bool,

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

    /// Maximum angular gap (degrees) allowed between consecutive edge points
    /// for both outer and inner ellipse fits. Fits with larger gaps are rejected.
    /// Default: 90 degrees.
    #[arg(long)]
    max_angular_gap_deg: Option<f64>,

    /// Require both inner and outer ellipses for every detected marker.
    /// When set, markers without a valid inner ellipse are rejected entirely.
    #[arg(long)]
    require_inner_fit: bool,

    /// Minimum number of outer edge points required for RANSAC ellipse fitting.
    /// Below this threshold, direct fit is attempted; below 6 points the fit
    /// is rejected entirely. Default: 8.
    #[arg(long)]
    min_outer_edge_points: Option<usize>,

    /// Minimum number of inner edge points required to attempt inner ellipse
    /// fitting. Default: 20.
    #[arg(long)]
    min_inner_edge_points: Option<usize>,

    /// Minimum theta consistency required for outer-radius hypothesis selection.
    #[arg(long)]
    outer_min_theta_consistency: Option<f32>,

    /// Relative threshold for accepting a second outer-radius hypothesis.
    #[arg(long)]
    outer_second_peak_min_rel: Option<f32>,

    /// Minimum decode margin for accepted marker IDs.
    #[arg(long)]
    decode_min_margin: Option<u8>,

    /// Maximum decode Hamming distance for accepted marker IDs.
    #[arg(long)]
    decode_max_dist: Option<u8>,

    /// Minimum decode confidence for accepted marker IDs.
    #[arg(long)]
    decode_min_confidence: Option<f32>,

    /// Size-term weight in outer-hypothesis scoring.
    #[arg(long)]
    outer_size_score_weight: Option<f32>,

    /// Enable structural ID verification and correction using hex neighborhood consensus.
    ///
    /// After fit-decode, each marker's ID is checked against its detected neighbors.
    /// Wrong IDs are corrected; unverified IDs are cleared (or the marker removed if
    /// `id_correction.remove_unverified = true` in the config file).
    /// Fine-tuning via `--config` section `"id_correction"`.
    #[arg(long)]
    id_correct: bool,

    /// Disable inner-as-outer recovery.
    ///
    /// By default, the detector re-attempts the outer fit for any marker whose
    /// outer radius is anomalously small compared to its neighbors (a sign that
    /// the inner edge was mistakenly fitted as the outer ellipse).
    #[arg(long)]
    no_inner_as_outer_recovery: bool,

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

    /// Path to a JSON configuration file with detection parameters.
    ///
    /// Present sections replace the defaults for that sub-config. CLI flags
    /// always take precedence over values in the file. Use `--dump-config` to
    /// print a complete template with all current defaults.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Print the default detection configuration as JSON and exit.
    ///
    /// The output is a valid `--config` template. Sections or individual
    /// fields can be removed; missing fields revert to built-in defaults.
    #[arg(long)]
    dump_config: bool,

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

/// JSON-loadable detection configuration overlay.
///
/// Each section is optional. When present, the entire sub-config is replaced
/// with the provided value; missing fields within the section revert to the
/// sub-config's own defaults (all sub-configs carry `#[serde(default)]`).
/// CLI flags are always applied on top and take precedence.
///
/// Use `--dump-config` to print a complete template.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct DetectConfigFile {
    marker_scale: Option<ringgrid::MarkerScalePrior>,
    inner_fit: Option<ringgrid::InnerFitConfig>,
    outer_fit: Option<ringgrid::OuterFitConfig>,
    completion: Option<ringgrid::CompletionParams>,
    projective_center: Option<ringgrid::ProjectiveCenterParams>,
    seed_proposals: Option<ringgrid::SeedProposalParams>,
    proposal: Option<ringgrid::ProposalConfig>,
    edge_sample: Option<ringgrid::EdgeSampleConfig>,
    decode: Option<ringgrid::DecodeConfig>,
    marker_spec: Option<ringgrid::MarkerSpec>,
    outer_estimation: Option<ringgrid::OuterEstimationConfig>,
    ransac_homography: Option<ringgrid::RansacHomographyConfig>,
    self_undistort: Option<ringgrid::SelfUndistortConfig>,
    id_correction: Option<ringgrid::IdCorrectionConfig>,
    inner_as_outer_recovery: Option<ringgrid::InnerAsOuterRecoveryConfig>,
    dedup_radius: Option<f64>,
    max_aspect_ratio: Option<f64>,
    use_global_filter: Option<bool>,
}

/// Serializable snapshot of all tunable detection parameters, excluding
/// board layout and marker scale (which are set via CLI or board JSON).
///
/// Used for `--dump-config` output.
#[derive(serde::Serialize)]
struct DetectConfigDump {
    marker_scale: ringgrid::MarkerScalePrior,
    inner_fit: ringgrid::InnerFitConfig,
    outer_fit: ringgrid::OuterFitConfig,
    completion: ringgrid::CompletionParams,
    projective_center: ringgrid::ProjectiveCenterParams,
    seed_proposals: ringgrid::SeedProposalParams,
    proposal: ringgrid::ProposalConfig,
    edge_sample: ringgrid::EdgeSampleConfig,
    decode: ringgrid::DecodeConfig,
    marker_spec: ringgrid::MarkerSpec,
    outer_estimation: ringgrid::OuterEstimationConfig,
    ransac_homography: ringgrid::RansacHomographyConfig,
    self_undistort: ringgrid::SelfUndistortConfig,
    id_correction: ringgrid::IdCorrectionConfig,
    inner_as_outer_recovery: ringgrid::InnerAsOuterRecoveryConfig,
    dedup_radius: f64,
    max_aspect_ratio: f64,
    use_global_filter: bool,
}

#[derive(Debug, Clone, Copy)]
struct DetectPreset {
    marker_scale: ringgrid::MarkerScalePrior,
}

#[derive(Debug, Clone)]
struct DetectOverrides {
    use_global_filter: bool,
    ransac_thresh_px: f64,
    ransac_iters: usize,
    completion_enable: bool,
    completion_reproj_gate_px: f32,
    completion_min_fit_confidence: f32,
    completion_roi_radius_px: Option<f32>,
    completion_require_perfect_decode: bool,
    camera: Option<ringgrid::CameraModel>,
    circle_refinement: ringgrid::CircleRefinementMethod,
    projective_center_max_shift_px: Option<f64>,
    projective_center_max_residual: f64,
    projective_center_min_eig_sep: f64,
    max_angular_gap_rad: Option<f64>,
    require_inner_fit: bool,
    min_outer_edge_points: Option<usize>,
    min_inner_edge_points: Option<usize>,
    outer_min_theta_consistency: Option<f32>,
    outer_second_peak_min_rel: Option<f32>,
    decode_min_margin: Option<u8>,
    decode_max_dist: Option<u8>,
    decode_min_confidence: Option<f32>,
    outer_size_score_weight: Option<f32>,
    id_correct_enable: bool,
    inner_as_outer_recovery_enable: bool,
    self_undistort_enable: bool,
    self_undistort_lambda_range: [f64; 2],
    self_undistort_min_markers: usize,
}

impl CliDetectArgs {
    /// Resolve marker scale prior.
    ///
    /// Precedence (highest to lowest):
    /// 1. `--marker-diameter` (fixed single value)
    /// 2. `--marker-diameter-min` / `--marker-diameter-max` (explicitly set)
    /// 3. `marker_scale` section in the JSON config file
    /// 4. Built-in defaults (20 – 56 px)
    fn to_preset(&self, config_file: Option<&DetectConfigFile>) -> DetectPreset {
        let marker_scale = if let Some(d) = self.marker_diameter {
            ringgrid::MarkerScalePrior::from_nominal_diameter_px(d as f32)
        } else if self.marker_diameter_min.is_some() || self.marker_diameter_max.is_some() {
            ringgrid::MarkerScalePrior::new(
                self.marker_diameter_min.unwrap_or(20.0) as f32,
                self.marker_diameter_max.unwrap_or(56.0) as f32,
            )
        } else {
            config_file.and_then(|f| f.marker_scale).unwrap_or_default()
        };
        DetectPreset { marker_scale }
    }

    fn to_overrides(&self) -> CliResult<DetectOverrides> {
        let circle_refinement = self.circle_refine_method.to_core();

        Ok(DetectOverrides {
            use_global_filter: !self.no_global_filter,
            ransac_thresh_px: self.ransac_thresh_px,
            ransac_iters: self.ransac_iters,
            completion_enable: !self.no_complete,
            completion_reproj_gate_px: self.complete_reproj_gate as f32,
            completion_min_fit_confidence: self.complete_min_conf,
            completion_roi_radius_px: self.complete_roi_radius.map(|v| v as f32),
            completion_require_perfect_decode: self.complete_require_perfect_decode,
            camera: self.camera.to_core()?,
            circle_refinement,
            projective_center_max_shift_px: self.proj_center_max_shift_px,
            projective_center_max_residual: self.proj_center_max_residual,
            projective_center_min_eig_sep: self.proj_center_min_eig_sep,
            max_angular_gap_rad: self.max_angular_gap_deg.map(|deg| deg.to_radians()),
            require_inner_fit: self.require_inner_fit,
            min_outer_edge_points: self.min_outer_edge_points,
            min_inner_edge_points: self.min_inner_edge_points,
            outer_min_theta_consistency: self.outer_min_theta_consistency,
            outer_second_peak_min_rel: self.outer_second_peak_min_rel,
            decode_min_margin: self.decode_min_margin,
            decode_max_dist: self.decode_max_dist,
            decode_min_confidence: self.decode_min_confidence,
            outer_size_score_weight: self.outer_size_score_weight,
            id_correct_enable: self.id_correct,
            inner_as_outer_recovery_enable: !self.no_inner_as_outer_recovery,
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
    config_file: Option<&DetectConfigFile>,
    overrides: &DetectOverrides,
) -> ringgrid::DetectConfig {
    let mut config =
        ringgrid::DetectConfig::from_target_and_scale_prior(board, preset.marker_scale);

    // Apply JSON config file sections. Each present section replaces the
    // corresponding sub-config that was derived from the scale prior. CLI
    // flags applied below always take precedence.
    if let Some(file) = config_file {
        if let Some(v) = file.inner_fit.clone() {
            config.inner_fit = v;
        }
        if let Some(v) = file.outer_fit.clone() {
            config.outer_fit = v;
        }
        if let Some(v) = file.completion.clone() {
            config.completion = v;
        }
        if let Some(v) = file.projective_center.clone() {
            config.projective_center = v;
        }
        if let Some(v) = file.seed_proposals.clone() {
            config.seed_proposals = v;
        }
        if let Some(v) = file.proposal.clone() {
            config.proposal = v;
        }
        if let Some(v) = file.edge_sample.clone() {
            config.edge_sample = v;
        }
        if let Some(v) = file.decode.clone() {
            config.decode = v;
        }
        if let Some(v) = file.marker_spec.clone() {
            config.marker_spec = v;
        }
        if let Some(v) = file.outer_estimation.clone() {
            config.outer_estimation = v;
        }
        if let Some(v) = file.ransac_homography.clone() {
            config.ransac_homography = v;
        }
        if let Some(v) = file.self_undistort.clone() {
            config.self_undistort = v;
        }
        if let Some(v) = file.dedup_radius {
            config.dedup_radius = v;
        }
        if let Some(v) = file.max_aspect_ratio {
            config.max_aspect_ratio = v;
        }
        if let Some(v) = file.id_correction.clone() {
            config.id_correction = v;
        }
        if let Some(v) = file.inner_as_outer_recovery.clone() {
            config.inner_as_outer_recovery = v;
        }
        if let Some(v) = file.use_global_filter {
            config.use_global_filter = v;
        }
    }

    // Global filter options
    config.use_global_filter = overrides.use_global_filter;
    config.ransac_homography.inlier_threshold = overrides.ransac_thresh_px;
    config.ransac_homography.max_iters = overrides.ransac_iters;

    // Homography-guided completion options
    config.completion.enable = overrides.completion_enable;
    config.completion.reproj_gate_px = overrides.completion_reproj_gate_px;
    config.completion.min_fit_confidence = overrides.completion_min_fit_confidence;
    if let Some(roi) = overrides.completion_roi_radius_px {
        config.completion.roi_radius_px = roi;
    }
    config.completion.require_perfect_decode = overrides.completion_require_perfect_decode;
    // Center refinement method
    config.circle_refinement = overrides.circle_refinement;
    config.projective_center.enable = config.circle_refinement.uses_projective_center();
    if let Some(shift) = overrides.projective_center_max_shift_px {
        config.projective_center.max_center_shift_px = Some(shift);
    }
    config.projective_center.max_selected_residual = Some(overrides.projective_center_max_residual);
    config.projective_center.min_eig_separation = Some(overrides.projective_center_min_eig_sep);

    // Angular gap and two-ellipse gates
    if let Some(gap) = overrides.max_angular_gap_rad {
        config.outer_fit.max_angular_gap_rad = gap;
        config.inner_fit.max_angular_gap_rad = gap;
    }
    config.inner_fit.require_inner_fit = overrides.require_inner_fit;
    if let Some(n) = overrides.min_outer_edge_points {
        config.outer_fit.min_ransac_points = n;
    }
    if let Some(n) = overrides.min_inner_edge_points {
        config.inner_fit.min_points = n;
    }

    // Outer estimator / decode / scoring tuning options.
    if let Some(v) = overrides.outer_min_theta_consistency {
        if v.is_finite() {
            config.outer_estimation.min_theta_consistency = v.clamp(0.0, 1.0);
        }
    }
    if let Some(v) = overrides.outer_second_peak_min_rel {
        if v.is_finite() {
            config.outer_estimation.second_peak_min_rel = v.clamp(0.0, 1.0);
        }
    }
    if let Some(v) = overrides.decode_min_margin {
        config.decode.min_decode_margin = v;
    }
    if let Some(v) = overrides.decode_max_dist {
        config.decode.max_decode_dist = v;
    }
    if let Some(v) = overrides.decode_min_confidence {
        if v.is_finite() {
            config.decode.min_decode_confidence = v.clamp(0.0, 1.0);
        }
    }
    if let Some(v) = overrides.outer_size_score_weight {
        if v.is_finite() {
            config.outer_fit.size_score_weight = v.clamp(0.0, 1.0);
        }
    }

    // ID correction: CLI --id-correct enables it on top of whatever the config file set.
    if overrides.id_correct_enable {
        config.id_correction.enable = true;
    }

    // Inner-as-outer recovery: --no-inner-as-outer-recovery disables it.
    if !overrides.inner_as_outer_recovery_enable {
        config.inner_as_outer_recovery.enable = false;
    }

    // Self-undistort options
    config.self_undistort.enable = overrides.self_undistort_enable;
    config.self_undistort.lambda_range = overrides.self_undistort_lambda_range;
    config.self_undistort.min_markers = overrides.self_undistort_min_markers;

    config
}

fn validate_correction_compat(overrides: &DetectOverrides) -> CliResult<()> {
    if overrides.camera.is_some() && overrides.self_undistort_enable {
        return Err(
            "camera mapping (--cam-*) and self-undistort (--self-undistort) are mutually exclusive"
                .into(),
        );
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct DetectionJsonOutput<'a> {
    #[serde(flatten)]
    result: &'a ringgrid::DetectionResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    camera: Option<ringgrid::CameraModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposal_frame: Option<ringgrid::DetectionFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposal_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposals: Option<&'a [ringgrid::Proposal]>,
}

fn serialize_detection_output(
    result: &ringgrid::DetectionResult,
    camera: Option<ringgrid::CameraModel>,
    proposals: Option<&[ringgrid::Proposal]>,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&DetectionJsonOutput {
        result,
        camera,
        proposal_frame: proposals.map(|_| ringgrid::DetectionFrame::Image),
        proposal_count: proposals.map(|p| p.len()),
        proposals,
    })
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
        let first = board.marker_by_index(0).expect("marker 0 must exist");
        let last = board
            .marker_by_index(board.n_markers() - 1)
            .expect("last marker must exist");
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

fn load_config_file(args: &CliDetectArgs) -> CliResult<Option<DetectConfigFile>> {
    if let Some(path) = &args.config {
        let text = std::fs::read_to_string(path).map_err(|e| -> CliError {
            format!("Failed to read config file {}: {}", path.display(), e).into()
        })?;
        let cfg: DetectConfigFile = serde_json::from_str(&text).map_err(|e| -> CliError {
            format!("Failed to parse config file {}: {}", path.display(), e).into()
        })?;
        tracing::info!("Loaded detection config from {}", path.display());
        Ok(Some(cfg))
    } else {
        Ok(None)
    }
}

/// Dump the effective default detection configuration as JSON.
///
/// Reflects `--marker-diameter-*` and `--config marker_scale` if provided.
/// The output is a valid `--config` template; remove sections or fields to
/// revert to built-in defaults.
fn run_dump_config(args: &CliDetectArgs) -> CliResult<()> {
    let config_file = load_config_file(args)?;
    let preset = args.to_preset(config_file.as_ref());
    let board = ringgrid::BoardLayout::default();
    let config = ringgrid::DetectConfig::from_target_and_scale_prior(board, preset.marker_scale);
    let dump = DetectConfigDump {
        marker_scale: config.marker_scale,
        inner_fit: config.inner_fit,
        outer_fit: config.outer_fit,
        completion: config.completion,
        projective_center: config.projective_center,
        seed_proposals: config.seed_proposals,
        proposal: config.proposal,
        edge_sample: config.edge_sample,
        decode: config.decode,
        marker_spec: config.marker_spec,
        outer_estimation: config.outer_estimation,
        ransac_homography: config.ransac_homography,
        self_undistort: config.self_undistort,
        id_correction: config.id_correction,
        inner_as_outer_recovery: config.inner_as_outer_recovery,
        dedup_radius: config.dedup_radius,
        max_aspect_ratio: config.max_aspect_ratio,
        use_global_filter: config.use_global_filter,
    };
    println!("{}", serde_json::to_string_pretty(&dump)?);
    Ok(())
}

fn run_detect(args: &CliDetectArgs) -> CliResult<()> {
    // Handle --dump-config before any I/O.
    if args.dump_config {
        return run_dump_config(args);
    }

    if args.marker_diameter.is_some() {
        tracing::warn!(
            "--marker-diameter is legacy fixed-size mode; prefer --marker-diameter-min/--marker-diameter-max"
        );
    }

    // Load config file first: marker scale resolution depends on it.
    let config_file = load_config_file(args)?;
    let preset = args.to_preset(config_file.as_ref());

    let overrides = args.to_overrides()?;
    validate_correction_compat(&overrides)?;

    tracing::info!("Loading image: {}", args.image.display());

    let img = image::open(&args.image).map_err(|e| -> CliError {
        format!("Failed to open image {}: {}", args.image.display(), e).into()
    })?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();

    tracing::info!("Image size: {}x{}", w, h);

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

    let config = build_detect_config(board, preset, config_file.as_ref(), &overrides);

    let detector = ringgrid::Detector::with_config(config);
    let proposals = if args.include_proposals {
        let proposals = detector.propose(&gray);
        tracing::info!(
            "Proposal diagnostics enabled: {} proposals recorded",
            proposals.len()
        );
        Some(proposals)
    } else {
        None
    };
    let result = match overrides.camera.as_ref() {
        Some(camera) => detector.detect_with_mapper(&gray, camera),
        None => detector.detect(&gray),
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
    let json = serialize_detection_output(&result, overrides.camera, proposals.as_deref())?;
    std::fs::write(&args.out, &json)?;
    tracing::info!("Results written to {}", args.out.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_overrides() -> DetectOverrides {
        DetectOverrides {
            use_global_filter: true,
            ransac_thresh_px: 5.0,
            ransac_iters: 2000,
            completion_enable: true,
            completion_reproj_gate_px: 3.0,
            completion_min_fit_confidence: 0.45,
            completion_roi_radius_px: None,
            completion_require_perfect_decode: false,
            camera: None,
            circle_refinement: ringgrid::CircleRefinementMethod::ProjectiveCenter,
            projective_center_max_shift_px: None,
            projective_center_max_residual: 0.25,
            projective_center_min_eig_sep: 1e-6,
            max_angular_gap_rad: None,
            require_inner_fit: false,
            min_outer_edge_points: None,
            min_inner_edge_points: None,
            outer_min_theta_consistency: None,
            outer_second_peak_min_rel: None,
            decode_min_margin: None,
            decode_max_dist: None,
            decode_min_confidence: None,
            outer_size_score_weight: None,
            id_correct_enable: false,
            inner_as_outer_recovery_enable: true,
            self_undistort_enable: false,
            self_undistort_lambda_range: [-8e-7, 8e-7],
            self_undistort_min_markers: 6,
        }
    }

    fn sample_camera() -> ringgrid::CameraModel {
        ringgrid::CameraModel {
            intrinsics: ringgrid::CameraIntrinsics {
                fx: 900.0,
                fy: 900.0,
                cx: 640.0,
                cy: 480.0,
            },
            distortion: ringgrid::RadialTangentialDistortion {
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            },
        }
    }

    #[test]
    fn validate_correction_compat_rejects_camera_plus_self_undistort() {
        let mut overrides = base_overrides();
        overrides.camera = Some(sample_camera());
        overrides.self_undistort_enable = true;
        assert!(validate_correction_compat(&overrides).is_err());
    }

    #[test]
    fn validate_correction_compat_accepts_non_conflicting_modes() {
        let mut with_camera = base_overrides();
        with_camera.camera = Some(sample_camera());
        assert!(validate_correction_compat(&with_camera).is_ok());

        let mut with_self_undistort = base_overrides();
        with_self_undistort.self_undistort_enable = true;
        assert!(validate_correction_compat(&with_self_undistort).is_ok());
    }

    #[test]
    fn serialize_detection_output_includes_camera_when_present() {
        let result = ringgrid::DetectionResult::empty(1280, 960);
        let json =
            serialize_detection_output(&result, Some(sample_camera()), None).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert!(value.get("camera").is_some());
    }

    #[test]
    fn serialize_detection_output_omits_camera_when_absent() {
        let result = ringgrid::DetectionResult::empty(1280, 960);
        let json = serialize_detection_output(&result, None, None).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert!(value.get("camera").is_none());
    }

    #[test]
    fn serialize_detection_output_includes_proposals_when_present() {
        let result = ringgrid::DetectionResult::empty(1280, 960);
        let proposals = vec![ringgrid::Proposal {
            x: 10.0,
            y: 20.0,
            score: 30.0,
        }];
        let json = serialize_detection_output(&result, None, Some(&proposals)).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert_eq!(
            value.get("proposal_frame").and_then(|v| v.as_str()),
            Some("image")
        );
        assert_eq!(
            value.get("proposal_count").and_then(|v| v.as_u64()),
            Some(1)
        );
        assert_eq!(
            value
                .get("proposals")
                .and_then(|v| v.as_array())
                .map(|v| v.len()),
            Some(1)
        );
    }

    #[test]
    fn build_detect_config_applies_tuning_overrides() {
        let mut overrides = base_overrides();
        overrides.outer_min_theta_consistency = Some(0.61);
        overrides.outer_second_peak_min_rel = Some(0.77);
        overrides.decode_min_margin = Some(2);
        overrides.decode_max_dist = Some(1);
        overrides.decode_min_confidence = Some(0.5);
        overrides.outer_size_score_weight = Some(0.33);

        let preset = DetectPreset {
            marker_scale: ringgrid::MarkerScalePrior::new(20.0, 56.0),
        };
        let cfg = build_detect_config(ringgrid::BoardLayout::default(), preset, None, &overrides);

        assert!((cfg.outer_estimation.min_theta_consistency - 0.61).abs() < 1e-6);
        assert!((cfg.outer_estimation.second_peak_min_rel - 0.77).abs() < 1e-6);
        assert_eq!(cfg.decode.min_decode_margin, 2);
        assert_eq!(cfg.decode.max_decode_dist, 1);
        assert!((cfg.decode.min_decode_confidence - 0.5).abs() < 1e-6);
        assert!((cfg.outer_fit.size_score_weight - 0.33).abs() < 1e-6);
    }
}
