//! ringgrid CLI — command-line interface for ring marker detection.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

type CliError = Box<dyn std::error::Error>;
type CliResult<T> = Result<T, CliError>;
const DEFAULT_GEN_TARGET_OUT_DIR: &str = "tools/out/target";
const DEFAULT_GEN_TARGET_BASENAME: &str = "target_print";
const TARGET_SPEC_SCHEMA_V5: &str = "ringgrid.target.v5";

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

#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Detect markers in an image.
    Detect(CliDetectArgs),

    /// Benchmark per-stage detection timing over one or more images.
    ///
    /// Runs single-pass detection with warmup + repeats, reports the median
    /// per-stage wall-clock, and writes a JSON report consumed by
    /// `tools/gen_pages_perf.py` to build the documentation performance page.
    Bench(CliBenchArgs),

    /// Generate a canonical target_spec.json plus printable SVG/PNG target files.
    GenTarget {
        /// Target family to generate.
        #[command(subcommand)]
        command: GenTargetCommands,
    },

    /// Print embedded codebook statistics.
    CodebookInfo,

    /// Print default board specification.
    BoardInfo,

    /// Decode a 16-bit word against the embedded codebook.
    DecodeTest {
        /// Observed 16-bit word (hex, e.g. 0xABCD).
        #[arg(long)]
        word: String,
        /// Embedded codebook profile to use.
        #[arg(long, value_enum, default_value_t = CodebookProfileArg::Base)]
        profile: CodebookProfileArg,
    },
}

#[derive(Debug, Subcommand)]
enum GenTargetCommands {
    /// Hex lattice of 16-sector coded rings (the classic ringgrid target).
    Hex(CliGenHexArgs),
    /// Rectangular lattice of plain (uncoded) rings, optionally with origin dots.
    Rect(CliGenRectArgs),
    /// A built-in target preset.
    Preset(CliGenPresetArgs),
    /// Render from an existing target spec JSON (v5, or legacy v4).
    FromSpec(CliGenFromSpecArgs),
}

#[derive(Debug, Clone, Args)]
struct CliGenOutputArgs {
    /// Output directory for target_spec.json, SVG, PNG, and DXF.
    #[arg(
        long = "out_dir",
        visible_alias = "out-dir",
        default_value = DEFAULT_GEN_TARGET_OUT_DIR
    )]
    out_dir: PathBuf,

    /// Base filename for SVG/PNG outputs.
    #[arg(long, default_value = DEFAULT_GEN_TARGET_BASENAME)]
    basename: String,

    /// PNG raster DPI (also embedded in PNG metadata).
    #[arg(long, default_value_t = 300.0)]
    dpi: f32,

    /// Extra white margin around the board in millimeters.
    #[arg(long = "margin_mm", visible_alias = "margin-mm", default_value_t = 0.0)]
    margin_mm: f32,

    /// Omit the default scale bar from SVG/PNG outputs.
    #[arg(long)]
    no_scale_bar: bool,
}

#[derive(Debug, Clone, Args)]
struct CliGenHexArgs {
    /// Marker center spacing in millimeters.
    #[arg(long = "pitch_mm", visible_alias = "pitch-mm")]
    pitch_mm: f32,

    /// Number of rows in the hex lattice.
    #[arg(long = "rows")]
    rows: usize,

    /// Number of markers in long rows.
    #[arg(long = "long_row_cols", visible_alias = "long-row-cols")]
    long_row_cols: usize,

    /// Outer ring radius in millimeters.
    #[arg(
        long = "marker_outer_radius_mm",
        visible_alias = "marker-outer-radius-mm"
    )]
    marker_outer_radius_mm: f32,

    /// Inner ring radius in millimeters.
    #[arg(
        long = "marker_inner_radius_mm",
        visible_alias = "marker-inner-radius-mm"
    )]
    marker_inner_radius_mm: f32,

    /// Ring width in millimeters.
    #[arg(long = "marker_ring_width_mm", visible_alias = "marker-ring-width-mm")]
    marker_ring_width_mm: f32,

    /// Optional explicit target name. Omitted uses a deterministic geometry-derived name.
    #[arg(long)]
    name: Option<String>,

    #[command(flatten)]
    output: CliGenOutputArgs,
}

impl CliGenHexArgs {
    fn to_target(&self) -> CliResult<ringgrid::TargetLayout> {
        let target = if let Some(name) = &self.name {
            ringgrid::TargetLayout::new(
                name.clone(),
                ringgrid::LatticeGeometry::Hex(ringgrid::HexGeometry {
                    rows: self.rows,
                    long_row_cols: self.long_row_cols,
                    pitch_mm: self.pitch_mm,
                }),
                ringgrid::RingGeometry {
                    outer_radius_mm: self.marker_outer_radius_mm,
                    inner_radius_mm: self.marker_inner_radius_mm,
                },
                ringgrid::MarkerCoding::Coded16(ringgrid::CodedRingSpec {
                    ring_width_mm: self.marker_ring_width_mm,
                    id_assignment: None,
                }),
                None,
            )
        } else {
            // coded_hex generates the deterministic geometry-derived name.
            ringgrid::TargetLayout::coded_hex(
                self.pitch_mm,
                self.rows,
                self.long_row_cols,
                self.marker_outer_radius_mm,
                self.marker_inner_radius_mm,
                self.marker_ring_width_mm,
            )
        };
        target.map_err(|e| -> CliError { format!("invalid target geometry: {e}").into() })
    }
}

/// Origin fiducial dot center in board millimeters.
#[derive(Debug, Clone, Copy)]
struct DotMmArg([f32; 2]);

fn parse_dot_mm(raw: &str) -> Result<DotMmArg, String> {
    let (x, y) = raw
        .split_once(',')
        .ok_or_else(|| format!("expected \"x,y\" millimeters, got '{raw}'"))?;
    let x: f32 = x
        .trim()
        .parse()
        .map_err(|e| format!("invalid x '{x}': {e}"))?;
    let y: f32 = y
        .trim()
        .parse()
        .map_err(|e| format!("invalid y '{y}': {e}"))?;
    Ok(DotMmArg([x, y]))
}

#[derive(Debug, Clone, Args)]
struct CliGenRectArgs {
    /// Marker center spacing in millimeters.
    #[arg(long = "pitch_mm", visible_alias = "pitch-mm")]
    pitch_mm: f32,

    /// Number of rows in the rectangular lattice.
    #[arg(long = "rows")]
    rows: usize,

    /// Number of columns in the rectangular lattice.
    #[arg(long = "cols")]
    cols: usize,

    /// Outer ring radius in millimeters.
    #[arg(
        long = "marker_outer_radius_mm",
        visible_alias = "marker-outer-radius-mm"
    )]
    marker_outer_radius_mm: f32,

    /// Inner ring radius in millimeters.
    #[arg(
        long = "marker_inner_radius_mm",
        visible_alias = "marker-inner-radius-mm"
    )]
    marker_inner_radius_mm: f32,

    /// Origin fiducial dot center in board millimeters, as "x,y".
    /// Repeat for each dot; requires --dot_radius_mm. The dot pattern must
    /// break every rotational symmetry of the lattice.
    #[arg(long = "dot_mm", visible_alias = "dot-mm", value_parser = parse_dot_mm, requires = "dot_radius_mm")]
    dot_mm: Vec<DotMmArg>,

    /// Origin fiducial dot radius in millimeters.
    #[arg(
        long = "dot_radius_mm",
        visible_alias = "dot-radius-mm",
        requires = "dot_mm"
    )]
    dot_radius_mm: Option<f32>,

    /// Optional explicit target name. Omitted uses a deterministic geometry-derived name.
    #[arg(long)]
    name: Option<String>,

    #[command(flatten)]
    output: CliGenOutputArgs,
}

impl CliGenRectArgs {
    fn to_target(&self) -> CliResult<ringgrid::TargetLayout> {
        let name = self.name.clone().unwrap_or_else(|| {
            format!(
                "ringgrid_rect_r{}_c{}_p{:.3}_o{:.3}_i{:.3}",
                self.rows,
                self.cols,
                self.pitch_mm,
                self.marker_outer_radius_mm,
                self.marker_inner_radius_mm
            )
        });
        let fiducials = self
            .dot_radius_mm
            .map(|dot_radius_mm| ringgrid::OriginFiducials {
                dot_radius_mm,
                dots_mm: self.dot_mm.iter().map(|d| d.0).collect(),
            });
        ringgrid::TargetLayout::new(
            name,
            ringgrid::LatticeGeometry::Rect(ringgrid::RectGeometry {
                rows: self.rows,
                cols: self.cols,
                pitch_mm: self.pitch_mm,
            }),
            ringgrid::RingGeometry {
                outer_radius_mm: self.marker_outer_radius_mm,
                inner_radius_mm: self.marker_inner_radius_mm,
            },
            ringgrid::MarkerCoding::Plain,
            fiducials,
        )
        .map_err(|e| -> CliError { format!("invalid target geometry: {e}").into() })
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GenTargetPresetArg {
    /// 24x24 plain rect target: plain rings at 14 mm pitch with
    /// three origin dots (drawing 5256-57-102).
    Rect24x24,
    /// The classic 15-row hex coded ringgrid target (200 mm board).
    DefaultHex,
}

#[derive(Debug, Clone, Args)]
struct CliGenPresetArgs {
    /// Preset to generate.
    #[arg(value_enum)]
    preset: GenTargetPresetArg,

    #[command(flatten)]
    output: CliGenOutputArgs,
}

impl CliGenPresetArgs {
    fn to_target(&self) -> ringgrid::TargetLayout {
        match self.preset {
            GenTargetPresetArg::Rect24x24 => ringgrid::TargetLayout::rect_24x24(),
            GenTargetPresetArg::DefaultHex => ringgrid::TargetLayout::default_hex(),
        }
    }
}

#[derive(Debug, Clone, Args)]
struct CliGenFromSpecArgs {
    /// Path to a target spec JSON file (v5, or legacy v4).
    #[arg(long)]
    spec: PathBuf,

    #[command(flatten)]
    output: CliGenOutputArgs,
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
    /// Unset by default. Effective default prior is 14.0-66.0 when both
    /// min/max are omitted. Can also be set via `--config`
    /// (`marker_scale.diameter_min_px`). CLI takes precedence over config file.
    #[arg(long)]
    marker_diameter_min: Option<f64>,

    /// Maximum marker outer diameter in pixels for scale search.
    ///
    /// Unset by default. Effective default prior is 14.0-66.0 when both
    /// min/max are omitted. Can also be set via `--config`
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

    /// Require a perfect decode (dist=0, margin≥active profile minimum distance)
    /// for completion markers.
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

    /// Proposal-stage image downscale mode.
    ///
    /// `auto` selects a factor from marker diameter (factor = floor(d_min/20),
    /// clamped [1,4]). `off` disables downscaling. A number (1–4) sets an
    /// explicit factor. Default: `off`.
    #[arg(long, default_value = "off")]
    proposal_downscale: ProposalDownscaleArg,

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
    /// Path to a JSON camera calibration file.
    ///
    /// Supported shapes:
    /// - direct `CameraModel` JSON:
    ///   `{ "intrinsics": { ... }, "distortion": { ... } }`
    /// - detector-output wrapper:
    ///   `{ "camera": { "intrinsics": { ... }, "distortion": { ... } } }`
    ///
    /// `distortion` may be omitted and defaults to zero coefficients.
    #[arg(long)]
    calibration: Option<PathBuf>,
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

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
struct CameraModelJson {
    intrinsics: ringgrid::CameraIntrinsics,
    distortion: ringgrid::RadialTangentialDistortion,
}

impl Default for CameraModelJson {
    fn default() -> Self {
        Self {
            intrinsics: ringgrid::CameraIntrinsics {
                fx: 0.0,
                fy: 0.0,
                cx: 0.0,
                cy: 0.0,
            },
            distortion: ringgrid::RadialTangentialDistortion::default(),
        }
    }
}

impl From<CameraModelJson> for ringgrid::CameraModel {
    fn from(value: CameraModelJson) -> Self {
        Self {
            intrinsics: value.intrinsics,
            distortion: value.distortion,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
enum CameraCalibrationFile {
    Direct(CameraModelJson),
    Wrapped { camera: CameraModelJson },
}

fn load_camera_model_from_file(path: &std::path::Path) -> CliResult<ringgrid::CameraModel> {
    let text = std::fs::read_to_string(path).map_err(|e| -> CliError {
        format!("Failed to read calibration file {}: {}", path.display(), e).into()
    })?;
    let parsed: CameraCalibrationFile = serde_json::from_str(&text).map_err(|e| -> CliError {
        format!(
            "Failed to parse calibration file {} as CameraModel JSON: {}",
            path.display(),
            e
        )
        .into()
    })?;
    let model: ringgrid::CameraModel = match parsed {
        CameraCalibrationFile::Direct(model) => model.into(),
        CameraCalibrationFile::Wrapped { camera } => camera.into(),
    };
    if !model.intrinsics.is_valid() {
        return Err(format!(
            "invalid calibration file {}: fx/fy must be finite and non-zero",
            path.display()
        )
        .into());
    }
    Ok(model)
}

impl CliCameraArgs {
    fn to_core(&self) -> CliResult<Option<ringgrid::CameraModel>> {
        let intr = [self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy];
        let any_intr = intr.iter().any(Option::is_some);
        let any_dist = self.cam_k1 != 0.0
            || self.cam_k2 != 0.0
            || self.cam_p1 != 0.0
            || self.cam_p2 != 0.0
            || self.cam_k3 != 0.0;
        if let Some(path) = &self.calibration {
            if any_intr || any_dist {
                return Err(
                    "camera calibration file (--calibration) and inline --cam-* parameters are mutually exclusive"
                        .into(),
                );
            }
            return load_camera_model_from_file(path).map(Some);
        }
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CodebookProfileArg {
    Base,
    Extended,
}

impl CodebookProfileArg {
    fn to_core(self) -> ringgrid::CodebookProfile {
        match self {
            Self::Base => ringgrid::CodebookProfile::Base,
            Self::Extended => ringgrid::CodebookProfile::Extended,
        }
    }
}

/// CLI representation of [`ringgrid::ProposalDownscale`].
///
/// Accepts `auto`, `off`, or an integer factor (1–4).
#[derive(Debug, Clone, Copy)]
struct ProposalDownscaleArg(ringgrid::ProposalDownscale);

impl std::fmt::Display for ProposalDownscaleArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ringgrid::ProposalDownscale::Auto => write!(f, "auto"),
            ringgrid::ProposalDownscale::Off => write!(f, "off"),
            ringgrid::ProposalDownscale::Factor(n) => write!(f, "{n}"),
        }
    }
}

impl std::str::FromStr for ProposalDownscaleArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self(ringgrid::ProposalDownscale::Auto)),
            "off" | "none" => Ok(Self(ringgrid::ProposalDownscale::Off)),
            other => other
                .parse::<u32>()
                .map(|n| Self(ringgrid::ProposalDownscale::Factor(n)))
                .map_err(|_| format!("expected 'auto', 'off', or integer 1-4, got '{s}'")),
        }
    }
}

/// A JSON-loadable detection configuration overlay.
///
/// The file is a (possibly partial) [`ringgrid::DetectConfig`] JSON document.
/// Stage-tuning parameters nest under the `"advanced"` object. Fields omitted
/// from the file keep the value derived from the board geometry and marker
/// scale prior; CLI flags are always applied on top and take precedence.
///
/// The overlay is applied with a recursive JSON merge so that a partially
/// specified section (e.g. `{"advanced": {"completion": {"enable": false}}}`)
/// only overrides the named leaves and leaves sibling fields untouched.
///
/// Use `--dump-config` to print a complete template.
struct DetectConfigFile {
    /// The raw JSON object from the config file, merged later onto the base config.
    overlay: serde_json::Map<String, serde_json::Value>,
}

impl DetectConfigFile {
    /// The `marker_scale` section, if the file sets one.
    ///
    /// Marker scale is resolved before the base config is built (it seeds the
    /// scale-dependent derivation), so it is extracted here rather than merged.
    fn marker_scale(&self) -> Option<ringgrid::MarkerScalePrior> {
        self.overlay
            .get("marker_scale")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
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
    proposal_downscale: ringgrid::ProposalDownscale,
}

impl CliDetectArgs {
    /// Resolve marker scale prior.
    ///
    /// Precedence (highest to lowest):
    /// 1. `--marker-diameter` (fixed single value)
    /// 2. `--marker-diameter-min` / `--marker-diameter-max` (explicitly set)
    /// 3. `marker_scale` section in the JSON config file
    /// 4. Built-in defaults (`MarkerScalePrior::default()`, currently 14-66 px)
    ///
    /// Note: if exactly one bound is provided via CLI, the missing side uses
    /// legacy compatibility fallback (20 or 56).
    fn to_preset(&self, config_file: Option<&DetectConfigFile>) -> DetectPreset {
        let marker_scale = if let Some(d) = self.marker_diameter {
            ringgrid::MarkerScalePrior::from_nominal_diameter_px(d as f32)
        } else if self.marker_diameter_min.is_some() || self.marker_diameter_max.is_some() {
            ringgrid::MarkerScalePrior::new(
                self.marker_diameter_min.unwrap_or(20.0) as f32,
                self.marker_diameter_max.unwrap_or(56.0) as f32,
            )
        } else {
            config_file
                .and_then(|f| f.marker_scale())
                .unwrap_or_default()
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
            proposal_downscale: self.proposal_downscale.0,
        })
    }
}

/// Apply a JSON config-file overlay onto a scale-derived base config.
///
/// Delegates to [`ringgrid::DetectConfig::with_json_overlay`]: recursive merge
/// onto the serialized base, legacy pre-0.8 key normalization, and target
/// re-attachment with geometry/scale re-derivation.
fn apply_config_file_overlay(
    config: ringgrid::DetectConfig,
    config_file: &DetectConfigFile,
) -> CliResult<ringgrid::DetectConfig> {
    config
        .with_json_overlay(serde_json::Value::Object(config_file.overlay.clone()))
        .map_err(|e| -> CliError { format!("invalid config overlay: {e}").into() })
}

fn build_detect_config(
    target: ringgrid::TargetLayout,
    preset: DetectPreset,
    config_file: Option<&DetectConfigFile>,
    overrides: &DetectOverrides,
) -> CliResult<ringgrid::DetectConfig> {
    let mut config =
        ringgrid::DetectConfig::from_target_and_scale_prior(target, preset.marker_scale);

    // Apply the JSON config file as a recursive overlay onto the scale-derived
    // base config. CLI flags applied below always take precedence.
    if let Some(file) = config_file {
        config = apply_config_file_overlay(config, file)?;
    }

    let adv = &mut config.advanced;

    // Global filter options
    adv.use_global_filter = overrides.use_global_filter;
    adv.ransac_homography.inlier_threshold = overrides.ransac_thresh_px;
    adv.ransac_homography.max_iters = overrides.ransac_iters;

    // Homography-guided completion options
    adv.completion.enable = overrides.completion_enable;
    adv.completion.reproj_gate_px = overrides.completion_reproj_gate_px;
    adv.completion.min_fit_confidence = overrides.completion_min_fit_confidence;
    if let Some(roi) = overrides.completion_roi_radius_px {
        adv.completion.roi_radius_px = roi;
    }
    adv.completion.require_perfect_decode = overrides.completion_require_perfect_decode;
    if let Some(shift) = overrides.projective_center_max_shift_px {
        adv.projective_center.max_correction_shift_px = Some(shift);
    }
    adv.projective_center.max_selected_residual = Some(overrides.projective_center_max_residual);
    adv.projective_center.min_eig_separation = Some(overrides.projective_center_min_eig_sep);

    // Angular gap and two-ellipse gates
    if let Some(gap) = overrides.max_angular_gap_rad {
        adv.outer_fit.max_angular_gap_rad = gap;
        adv.inner_fit.max_angular_gap_rad = gap;
    }
    adv.inner_fit.require_inner_fit = overrides.require_inner_fit;
    if let Some(n) = overrides.min_outer_edge_points {
        adv.outer_fit.min_ransac_points = n;
    }
    if let Some(n) = overrides.min_inner_edge_points {
        adv.inner_fit.min_points = n;
    }

    // Outer estimator / decode / scoring tuning options.
    if let Some(v) = overrides.outer_min_theta_consistency
        && v.is_finite()
    {
        adv.outer_estimation.min_theta_consistency = v.clamp(0.0, 1.0);
    }
    if let Some(v) = overrides.outer_second_peak_min_rel
        && v.is_finite()
    {
        adv.outer_estimation.second_peak_min_rel = v.clamp(0.0, 1.0);
    }
    if let Some(v) = overrides.decode_min_margin {
        adv.decode.min_decode_margin = v;
    }
    if let Some(v) = overrides.decode_max_dist {
        adv.decode.max_decode_dist = v;
    }
    if let Some(v) = overrides.decode_min_confidence
        && v.is_finite()
    {
        adv.decode.min_decode_confidence = v.clamp(0.0, 1.0);
    }
    if let Some(v) = overrides.outer_size_score_weight
        && v.is_finite()
    {
        adv.outer_fit.size_score_weight = v.clamp(0.0, 1.0);
    }

    // ID correction: CLI --id-correct enables it on top of whatever the config file set.
    if overrides.id_correct_enable {
        adv.id_correction.enable = true;
    }

    // Inner-as-outer recovery: --no-inner-as-outer-recovery disables it.
    if !overrides.inner_as_outer_recovery_enable {
        adv.inner_as_outer_recovery.enable = false;
    }

    // Proposal downscale
    adv.proposal_downscale = overrides.proposal_downscale;

    // Center refinement method (stable, top-level field).
    config.circle_refinement = overrides.circle_refinement;

    // Self-undistort options (stable, top-level field).
    config.self_undistort.enable = overrides.self_undistort_enable;
    config.self_undistort.lambda_range = overrides.self_undistort_lambda_range;
    config.self_undistort.min_markers = overrides.self_undistort_min_markers;

    Ok(config)
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

/// JSON detection output.
///
/// The slim [`ringgrid::DetectionResult`] fields are flattened at the top
/// level. Per-marker algorithm internals and homography RANSAC statistics —
/// relocated out of the result type in the v0.6 API — are nested under the
/// `diagnostics` object: `diagnostics.markers[i]` describes `detected_markers[i]`
/// (positionally aligned 1:1), and `diagnostics.ransac` holds the RANSAC stats.
#[derive(serde::Serialize)]
struct DetectionJsonOutput<'a> {
    #[serde(flatten)]
    result: &'a ringgrid::DetectionResult,
    diagnostics: &'a ringgrid::diagnostics::DetectionDiagnostics,
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
    diagnostics: &ringgrid::diagnostics::DetectionDiagnostics,
    camera: Option<ringgrid::CameraModel>,
    proposals: Option<&[ringgrid::Proposal]>,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&DetectionJsonOutput {
        result,
        diagnostics,
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
        Commands::Bench(args) => run_bench(&args),
        Commands::GenTarget { command } => run_gen_target(&command),
        Commands::CodebookInfo => run_codebook_info(),

        Commands::BoardInfo => run_board_info(),

        Commands::DecodeTest { word, profile } => run_decode_test(&word, profile),
    }
}

// ── codebook-info ──────────────────────────────────────────────────────

fn run_codebook_info() -> CliResult<()> {
    use ringgrid::CodebookProfile;
    use ringgrid::codebook::{CodebookInfo, codebook_info};

    println!("ringgrid embedded codebook profiles");
    println!("  default profile:      {}", CodebookProfile::Base.as_str());

    for profile in [CodebookProfile::Base, CodebookProfile::Extended] {
        let info: CodebookInfo = codebook_info(profile);
        println!("  {}:", info.profile.as_str());
        println!("    bits per codeword:    {}", info.bits);
        println!("    number of codewords:  {}", info.len);
        println!("    min cyclic Hamming:   {}", info.min_cyclic_dist);
        println!("    generator seed:       {}", info.seed);
        if let Some(first) = info.first_codeword {
            println!("    first codeword:       0x{:04X}", first);
        }
        if let Some(last) = info.last_codeword {
            println!("    last codeword:        0x{:04X}", last);
        }
    }

    Ok(())
}

// ── board-info ────────────────────────────────────────────────────────

fn run_board_info() -> CliResult<()> {
    let target = ringgrid::TargetLayout::default_hex();

    println!("ringgrid default target specification");
    println!("  name:           {}", target.name());
    println!("  schema:         {}", TARGET_SPEC_SCHEMA_V5);
    match target.lattice() {
        ringgrid::LatticeGeometry::Hex(h) => {
            println!("  lattice:        hex");
            println!("  rows:           {}", h.rows);
            println!("  long row cols:  {}", h.long_row_cols);
        }
        ringgrid::LatticeGeometry::Rect(r) => {
            println!("  lattice:        rect");
            println!("  rows:           {}", r.rows);
            println!("  cols:           {}", r.cols);
        }
    }
    println!(
        "  coding:         {}",
        if target.is_coded() {
            "coded16"
        } else {
            "plain"
        }
    );
    println!("  cells:          {}", target.n_cells());
    println!("  pitch:          {} mm", target.pitch_mm());
    if let Some(span) = target.marker_span_mm() {
        println!("  marker span:    {:.3}x{:.3} mm", span[0], span[1]);
    }

    if let (Some(first), Some(last)) = (target.cells().first(), target.cells().last()) {
        println!(
            "  cell 0:         ({:.1}, {:.1}) mm  [u={}, v={}]",
            first.xy_mm[0], first.xy_mm[1], first.coord.u, first.coord.v
        );
        println!(
            "  cell {}:       ({:.1}, {:.1}) mm  [u={}, v={}]",
            target.n_cells() - 1,
            last.xy_mm[0],
            last.xy_mm[1],
            last.coord.u,
            last.coord.v
        );
    }

    Ok(())
}

// ── gen-target ────────────────────────────────────────────────────────

fn run_gen_target(command: &GenTargetCommands) -> CliResult<()> {
    let (target, output) = match command {
        GenTargetCommands::Hex(args) => (args.to_target()?, &args.output),
        GenTargetCommands::Rect(args) => (args.to_target()?, &args.output),
        GenTargetCommands::Preset(args) => (args.to_target(), &args.output),
        GenTargetCommands::FromSpec(args) => (
            ringgrid::TargetLayout::from_json_file(&args.spec).map_err(|e| -> CliError {
                format!("Failed to load target spec {}: {}", args.spec.display(), e).into()
            })?,
            &args.output,
        ),
    };
    write_target_outputs(&target, output)
}

fn write_target_outputs(
    target: &ringgrid::TargetLayout,
    output: &CliGenOutputArgs,
) -> CliResult<()> {
    // Delegate to the shared library helper (single source of truth for target
    // artifact writing, reused by the published `ringgrid` CLI).
    let render = ringgrid::cli::RenderRecipe {
        dpi: output.dpi,
        margin_mm: output.margin_mm,
        scale_bar: !output.no_scale_bar,
        formats: vec![
            ringgrid::cli::Format::Json,
            ringgrid::cli::Format::Svg,
            ringgrid::cli::Format::Png,
            ringgrid::cli::Format::Dxf,
        ],
    };
    let written =
        ringgrid::cli::write_target_artifacts(target, &output.out_dir, &output.basename, &render)
            .map_err(|e| -> CliError { e.to_string().into() })?;
    for path in &written {
        println!("wrote {}", path.display());
    }
    let lattice = match target.lattice() {
        ringgrid::LatticeGeometry::Hex(h) => {
            format!("hex rows={} long_row_cols={}", h.rows, h.long_row_cols)
        }
        ringgrid::LatticeGeometry::Rect(r) => format!("rect rows={} cols={}", r.rows, r.cols),
    };
    let coding = if target.is_coded() {
        "coded16"
    } else {
        "plain"
    };
    println!(
        "Target: {}, schema={}, {}, coding={}, cells={}, pitch={}mm, fiducial dots={}",
        target.name(),
        TARGET_SPEC_SCHEMA_V5,
        lattice,
        coding,
        target.n_cells(),
        target.pitch_mm(),
        target.fiducials().map_or(0, |f| f.dots_mm.len())
    );
    Ok(())
}

// ── decode-test ────────────────────────────────────────────────────────

fn run_decode_test(word_str: &str, profile: CodebookProfileArg) -> CliResult<()> {
    use ringgrid::codebook::{CodewordMatch, decode_word};

    let word_str = word_str
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    let word = u16::from_str_radix(word_str, 16)
        .map_err(|e| -> CliError { format!("invalid hex word: {}", e).into() })?;

    let m: CodewordMatch = decode_word(word, profile.to_core());

    println!("Input word:   0x{:04X} (binary: {:016b})", m.word, m.word);
    println!("Profile:      {}", m.profile.as_str());
    println!("Best match:");
    println!("  id:         {}", m.id);
    println!("  codeword:   0x{:04X}", m.codeword.unwrap_or(0));
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
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| -> CliError {
            format!("Failed to parse config file {}: {}", path.display(), e).into()
        })?;
        let overlay = match value {
            serde_json::Value::Object(obj) => obj,
            _ => {
                return Err(
                    format!("config file {} must contain a JSON object", path.display()).into(),
                );
            }
        };
        tracing::info!("Loaded detection config from {}", path.display());
        Ok(Some(DetectConfigFile { overlay }))
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
    let target = ringgrid::TargetLayout::default_hex();
    let config = ringgrid::DetectConfig::from_target_and_scale_prior(target, preset.marker_scale);
    // `DetectConfig` serializes directly; stage tuning nests under `advanced`.
    // The target layout is `#[serde(skip)]`; loading the dump back through
    // `--config` re-attaches the active target automatically.
    println!("{}", serde_json::to_string_pretty(&config)?);
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

    let target = if let Some(target_path) = &args.target {
        let target =
            ringgrid::TargetLayout::from_json_file(target_path).map_err(|e| -> CliError {
                format!(
                    "Failed to load target spec {}: {}",
                    target_path.display(),
                    e
                )
                .into()
            })?;
        tracing::info!(
            "Loaded target layout '{}' with {} cells",
            target.name(),
            target.n_cells()
        );
        target
    } else {
        ringgrid::TargetLayout::default_hex()
    };

    let config = build_detect_config(target, preset, config_file.as_ref(), &overrides)?;

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
    // Use the diagnostics-returning entry points so the relocated per-marker
    // internals and RANSAC stats can still be emitted in the output JSON.
    let (result, diagnostics) = match overrides.camera.as_ref() {
        Some(camera) => detector.detect_with_mapper_diagnostics(&gray, camera),
        None => detector.detect_with_diagnostics(&gray),
    }
    .map_err(|e| -> CliError { e.to_string().into() })?;

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

    if let Some(ref stats) = diagnostics.ransac {
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
    let json = serialize_detection_output(
        &result,
        &diagnostics,
        overrides.camera,
        proposals.as_deref(),
    )?;
    std::fs::write(&args.out, &json)?;
    tracing::info!("Results written to {}", args.out.display());

    Ok(())
}

// ── bench ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Args)]
struct CliBenchArgs {
    /// Input image to benchmark. Repeat the flag to benchmark several images.
    #[arg(long, required = true)]
    image: Vec<PathBuf>,

    /// Board layout JSON (target specification). Omitted uses the built-in board.
    #[arg(long)]
    target: Option<PathBuf>,

    /// Output path for the timing report JSON.
    #[arg(long)]
    out: PathBuf,

    /// Number of timed repeats per image (median is reported).
    #[arg(long, default_value_t = 9)]
    repeats: usize,

    /// Number of warmup iterations per image (not timed).
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Fixed marker outer diameter in pixels (scale hint). Omitted uses the
    /// default marker-scale prior derived from the board.
    #[arg(long)]
    marker_diameter: Option<f64>,
}

/// Per-image entry of the bench report (the measurement-owned subset of the
/// performance-page `images[]` schema; presentation fields are added by
/// `tools/gen_pages_perf.py`).
#[derive(serde::Serialize)]
struct BenchImageReport {
    file: String,
    width: u32,
    height: u32,
    /// Proposal (raw center) count — maps to the page's `raw_corners`.
    raw_corners: usize,
    /// Total detected markers — maps to the page's `labelled`.
    labelled: usize,
    /// Decoded markers (valid ID) — maps to the page's `markers`.
    markers: usize,
    proposal_ms: f64,
    fit_decode_ms: f64,
    finalize_ms: f64,
    total_ms: f64,
}

#[derive(serde::Serialize)]
struct BenchReport {
    repeats: usize,
    source: String,
    images: Vec<BenchImageReport>,
}

/// Median of a sample set (sorts in place). Returns 0.0 for an empty slice.
fn median(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    }
}

fn run_bench(args: &CliBenchArgs) -> CliResult<()> {
    let target = match &args.target {
        Some(path) => ringgrid::TargetLayout::from_json_file(path).map_err(|e| -> CliError {
            format!("Failed to load target spec {}: {}", path.display(), e).into()
        })?,
        None => ringgrid::TargetLayout::default_hex(),
    };
    let config = match args.marker_diameter {
        Some(d) => ringgrid::DetectConfig::from_target_and_scale_prior(
            target,
            ringgrid::MarkerScalePrior::from_nominal_diameter_px(d as f32),
        ),
        None => ringgrid::DetectConfig::from_target(target),
    };
    let detector = ringgrid::Detector::with_config(config);
    let repeats = args.repeats.max(1);

    let mut images = Vec::with_capacity(args.image.len());
    for image_path in &args.image {
        let img = image::open(image_path).map_err(|e| -> CliError {
            format!("Failed to open image {}: {}", image_path.display(), e).into()
        })?;
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();

        // Proposal count is deterministic; one call matches what detection uses.
        let raw_corners = detector.propose(&gray).len();

        for _ in 0..args.warmup {
            let _ = detector.detect_with_diagnostics(&gray);
        }

        let mut proposal = Vec::with_capacity(repeats);
        let mut fit_decode = Vec::with_capacity(repeats);
        let mut finalize = Vec::with_capacity(repeats);
        let mut total = Vec::with_capacity(repeats);
        let mut labelled = 0usize;
        let mut markers = 0usize;
        for _ in 0..repeats {
            let (result, diagnostics) = detector
                .detect_with_diagnostics(&gray)
                .map_err(|e| -> CliError { e.to_string().into() })?;
            let timings = diagnostics.timings.ok_or_else(|| -> CliError {
                "single-pass detection produced no timings".into()
            })?;
            proposal.push(timings.proposal_ms);
            fit_decode.push(timings.fit_decode_ms);
            finalize.push(timings.finalize_ms);
            total.push(timings.total_ms);
            labelled = result.detected_markers.len();
            markers = result
                .detected_markers
                .iter()
                .filter(|m| m.id.is_some())
                .count();
        }

        let proposal_ms = median(&mut proposal);
        let fit_decode_ms = median(&mut fit_decode);
        let finalize_ms = median(&mut finalize);
        let total_ms = median(&mut total);
        tracing::info!(
            file = %image_path.display(),
            width,
            height,
            raw_corners,
            labelled,
            markers,
            total_p50_ms = total_ms,
            "benchmarked image"
        );

        images.push(BenchImageReport {
            file: image_path.display().to_string(),
            width,
            height,
            raw_corners,
            labelled,
            markers,
            proposal_ms,
            fit_decode_ms,
            finalize_ms,
            total_ms,
        });
    }

    let report = BenchReport {
        repeats,
        source: format!("ringgrid bench --repeats {repeats}"),
        images,
    };
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.out, &json)?;
    tracing::info!("Bench report written to {}", args.out.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::load_from_memory;

    const EXPECTED_TARGET_JSON: &str =
        include_str!("../../ringgrid/tests/fixtures/target_generation/fixture_compact_hex_v5.json");
    const EXPECTED_TARGET_SVG: &str =
        include_str!("../../ringgrid/tests/fixtures/target_generation/fixture_compact_hex.svg");
    const EXPECTED_TARGET_PNG: &[u8] =
        include_bytes!("../../ringgrid/tests/fixtures/target_generation/fixture_compact_hex.png");

    fn normalize_text_newlines(text: &str) -> String {
        text.replace("\r\n", "\n")
    }

    fn temp_output_dir(prefix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ringgrid_cli_{}_{}_{}",
            prefix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ))
    }

    fn png_phys(path: &std::path::Path) -> (u32, u32, u8) {
        let data = std::fs::read(path).expect("read png bytes");
        assert!(data.starts_with(b"\x89PNG\r\n\x1a\n"));
        let mut offset = 8usize;
        while offset + 12 <= data.len() {
            let length =
                u32::from_be_bytes(data[offset..offset + 4].try_into().expect("chunk length"))
                    as usize;
            let chunk_type = &data[offset + 4..offset + 8];
            let chunk_data_start = offset + 8;
            let chunk_data_end = chunk_data_start + length;
            if chunk_type == b"pHYs" {
                let chunk = &data[chunk_data_start..chunk_data_end];
                return (
                    u32::from_be_bytes(chunk[0..4].try_into().expect("xppu")),
                    u32::from_be_bytes(chunk[4..8].try_into().expect("yppu")),
                    chunk[8],
                );
            }
            offset = chunk_data_end + 4;
        }
        panic!("missing pHYs chunk");
    }

    fn fixture_gen_target_args(out_dir: std::path::PathBuf) -> CliGenHexArgs {
        CliGenHexArgs {
            pitch_mm: 8.0,
            rows: 3,
            long_row_cols: 4,
            marker_outer_radius_mm: 4.8,
            marker_inner_radius_mm: 3.2,
            marker_ring_width_mm: 1.152,
            name: Some("fixture_compact_hex".to_string()),
            output: CliGenOutputArgs {
                out_dir,
                basename: "fixture_compact_hex".to_string(),
                dpi: 96.0,
                margin_mm: 0.0,
                no_scale_bar: false,
            },
        }
    }

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
            proposal_downscale: ringgrid::ProposalDownscale::Off,
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

    fn write_temp_json_file(filename: &str, body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "ringgrid_cli_test_{}_{}_{}.json",
            filename,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::write(&path, body).expect("write temp json");
        path
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
        let diagnostics = ringgrid::diagnostics::DetectionDiagnostics::default();
        let json = serialize_detection_output(&result, &diagnostics, Some(sample_camera()), None)
            .expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert!(value.get("camera").is_some());
        assert!(value.get("diagnostics").is_some());
    }

    #[test]
    fn serialize_detection_output_omits_camera_when_absent() {
        let result = ringgrid::DetectionResult::empty(1280, 960);
        let diagnostics = ringgrid::diagnostics::DetectionDiagnostics::default();
        let json =
            serialize_detection_output(&result, &diagnostics, None, None).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert!(value.get("camera").is_none());
    }

    #[test]
    fn serialize_detection_output_includes_proposals_when_present() {
        let result = ringgrid::DetectionResult::empty(1280, 960);
        let diagnostics = ringgrid::diagnostics::DetectionDiagnostics::default();
        let proposals = vec![ringgrid::Proposal {
            x: 10.0,
            y: 20.0,
            score: 30.0,
        }];
        let json = serialize_detection_output(&result, &diagnostics, None, Some(&proposals))
            .expect("serialize");
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
        let cfg = build_detect_config(
            ringgrid::TargetLayout::default_hex(),
            preset,
            None,
            &overrides,
        )
        .expect("build config");

        assert!((cfg.advanced.outer_estimation.min_theta_consistency - 0.61).abs() < 1e-6);
        assert!((cfg.advanced.outer_estimation.second_peak_min_rel - 0.77).abs() < 1e-6);
        assert_eq!(cfg.advanced.decode.min_decode_margin, 2);
        assert_eq!(cfg.advanced.decode.max_decode_dist, 1);
        assert!((cfg.advanced.decode.min_decode_confidence - 0.5).abs() < 1e-6);
        assert!((cfg.advanced.outer_fit.size_score_weight - 0.33).abs() < 1e-6);
    }

    #[test]
    fn build_detect_config_applies_nested_advanced_overlay() {
        // A partial config file: stage tuning nests under "advanced".
        let overlay_obj = serde_json::json!({
            "advanced": {
                "completion": { "enable": false },
                "dedup_radius": 9.5
            }
        });
        let overlay = match overlay_obj {
            serde_json::Value::Object(obj) => obj,
            _ => unreachable!(),
        };
        let config_file = DetectConfigFile { overlay };
        let preset = DetectPreset {
            marker_scale: ringgrid::MarkerScalePrior::new(20.0, 56.0),
        };
        let cfg = build_detect_config(
            ringgrid::TargetLayout::default_hex(),
            preset,
            Some(&config_file),
            &base_overrides(),
        )
        .expect("build config");

        // base_overrides() forces completion.enable = true on top of the file,
        // so the override wins — confirming CLI flags still take precedence.
        assert!(cfg.advanced.completion.enable);
        // dedup_radius has no CLI override, so the overlay value survives.
        assert!((cfg.advanced.dedup_radius - 9.5).abs() < 1e-9);
        // Untouched advanced fields keep their scale-derived/default values.
        assert_eq!(cfg.advanced.inner_fit.min_points, 20);
        // The target is re-attached and re-derives geometry.
        assert!(cfg.target.n_cells() > 0);
    }

    #[test]
    fn dump_config_json_nests_stage_tuning_under_advanced() {
        let config = ringgrid::DetectConfig::from_target_and_scale_prior(
            ringgrid::TargetLayout::default_hex(),
            ringgrid::MarkerScalePrior::new(20.0, 56.0),
        );
        let json = serde_json::to_string(&config).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert!(value.get("advanced").is_some(), "missing advanced object");
        assert!(
            value["advanced"].get("completion").is_some(),
            "completion must nest under advanced"
        );
        assert!(value.get("marker_scale").is_some());
        // board is #[serde(skip)] — not present in the dump.
        assert!(value.get("board").is_none());

        // The dump loads back through DetectConfig deserialization.
        let _: ringgrid::DetectConfig = serde_json::from_value(value).expect("roundtrip");
    }

    #[test]
    fn calibration_file_loads_direct_camera_model_shape() {
        let path = write_temp_json_file(
            "direct_camera_model",
            r#"{
  "intrinsics": { "fx": 900.0, "fy": 920.0, "cx": 640.0, "cy": 480.0 },
  "distortion": { "k1": -0.1, "k2": 0.02, "p1": 0.001, "p2": -0.002, "k3": 0.0 }
}"#,
        );
        let args = CliCameraArgs {
            calibration: Some(path.clone()),
            ..CliCameraArgs::default()
        };
        let model = args
            .to_core()
            .expect("load direct camera model")
            .expect("camera present");
        assert!((model.intrinsics.fx - 900.0).abs() < 1e-12);
        assert!((model.intrinsics.fy - 920.0).abs() < 1e-12);
        assert!((model.distortion.k1 + 0.1).abs() < 1e-12);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn calibration_file_loads_detector_output_wrapper_shape() {
        let path = write_temp_json_file(
            "wrapped_camera_model",
            r#"{
  "camera": {
    "intrinsics": { "fx": 700.0, "fy": 710.0, "cx": 320.0, "cy": 240.0 }
  }
}"#,
        );
        let args = CliCameraArgs {
            calibration: Some(path.clone()),
            ..CliCameraArgs::default()
        };
        let model = args
            .to_core()
            .expect("load wrapped camera model")
            .expect("camera present");
        assert!((model.intrinsics.fx - 700.0).abs() < 1e-12);
        assert_eq!(
            model.distortion,
            ringgrid::RadialTangentialDistortion::default()
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn calibration_file_rejects_mixed_inline_camera_flags() {
        let path = write_temp_json_file(
            "mixed_camera_flags",
            r#"{
  "intrinsics": { "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 480.0 }
}"#,
        );
        let args = CliCameraArgs {
            calibration: Some(path.clone()),
            cam_fx: Some(900.0),
            cam_fy: Some(900.0),
            cam_cx: Some(640.0),
            cam_cy: Some(480.0),
            ..CliCameraArgs::default()
        };
        let err = args.to_core().expect_err("mixed camera inputs must fail");
        assert!(
            err.to_string().contains("mutually exclusive"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn gen_target_subcommand_parses_python_style_flags() {
        let cli = Cli::try_parse_from([
            "ringgrid",
            "gen-target",
            "hex",
            "--pitch_mm",
            "8.0",
            "--rows",
            "3",
            "--long_row_cols",
            "4",
            "--marker_outer_radius_mm",
            "4.8",
            "--marker_inner_radius_mm",
            "3.2",
            "--marker_ring_width_mm",
            "1.152",
            "--out_dir",
            "tools/out/fixture",
        ])
        .expect("parse gen-target cli");
        match cli.command {
            Commands::GenTarget {
                command: GenTargetCommands::Hex(args),
            } => {
                assert_eq!(args.pitch_mm, 8.0);
                assert_eq!(args.rows, 3);
                assert_eq!(args.long_row_cols, 4);
                assert_eq!(
                    args.output.out_dir,
                    std::path::PathBuf::from("tools/out/fixture")
                );
            }
            _ => panic!("unexpected command variant"),
        }
    }

    #[test]
    fn gen_target_rect_parses_dots_and_builds_plain_target() {
        let cli = Cli::try_parse_from([
            "ringgrid",
            "gen-target",
            "rect",
            "--pitch_mm",
            "14.0",
            "--rows",
            "4",
            "--cols",
            "4",
            "--marker_outer_radius_mm",
            "5.6",
            "--marker_inner_radius_mm",
            "2.8",
            "--dot_mm",
            "21,21",
            "--dot_mm",
            "7,21",
            "--dot_radius_mm",
            "1.4",
        ])
        .expect("parse gen-target rect cli");
        let Commands::GenTarget {
            command: GenTargetCommands::Rect(args),
        } = cli.command
        else {
            panic!("unexpected command variant");
        };
        let target = args.to_target().expect("valid rect target");
        assert!(!target.is_coded());
        assert_eq!(target.n_cells(), 16);
        assert_eq!(
            target.fiducials().map(|f| f.dots_mm.len()),
            Some(2),
            "both dots parsed"
        );
    }

    #[test]
    fn gen_target_rect_rejects_dots_without_radius() {
        let result = Cli::try_parse_from([
            "ringgrid",
            "gen-target",
            "rect",
            "--pitch_mm",
            "14.0",
            "--rows",
            "4",
            "--cols",
            "4",
            "--marker_outer_radius_mm",
            "5.6",
            "--marker_inner_radius_mm",
            "2.8",
            "--dot_mm",
            "21,21",
        ]);
        assert!(result.is_err(), "--dot_mm requires --dot_radius_mm");
    }

    #[test]
    fn gen_target_preset_rect_builds_rect_target() {
        let cli = Cli::try_parse_from(["ringgrid", "gen-target", "preset", "rect24x24"])
            .expect("parse gen-target preset cli");
        let Commands::GenTarget {
            command: GenTargetCommands::Preset(args),
        } = cli.command
        else {
            panic!("unexpected command variant");
        };
        let target = args.to_target();
        assert_eq!(target.name(), "rect_24x24");
        assert_eq!(target.n_cells(), 576);
    }

    #[test]
    fn gen_target_writes_committed_fixture_outputs() {
        let out_dir = temp_output_dir("fixture_outputs").join("nested/fixture");
        let args = fixture_gen_target_args(out_dir.clone());
        run_gen_target(&GenTargetCommands::Hex(args)).expect("generate fixture outputs");

        let json_path = out_dir.join("target_spec.json");
        let svg_path = out_dir.join("fixture_compact_hex.svg");
        let png_path = out_dir.join("fixture_compact_hex.png");

        assert_eq!(
            normalize_text_newlines(&std::fs::read_to_string(&json_path).expect("read json")),
            normalize_text_newlines(EXPECTED_TARGET_JSON)
        );
        assert_eq!(
            normalize_text_newlines(&std::fs::read_to_string(&svg_path).expect("read svg")),
            normalize_text_newlines(EXPECTED_TARGET_SVG)
        );

        let expected_ppm = (96.0_f64 * 1000.0 / 25.4).round() as u32;
        assert_eq!(png_phys(&png_path), (expected_ppm, expected_ppm, 1));

        let written_png = load_from_memory(&std::fs::read(&png_path).expect("read png"))
            .expect("decode written png")
            .into_luma8();
        let expected_png = load_from_memory(EXPECTED_TARGET_PNG)
            .expect("decode expected png")
            .into_luma8();
        assert_eq!(written_png.dimensions(), expected_png.dimensions());
        assert_eq!(written_png.as_raw(), expected_png.as_raw());

        let dxf_path = out_dir.join("fixture_compact_hex.dxf");
        let dxf = std::fs::read_to_string(&dxf_path).expect("read dxf");
        assert!(dxf.starts_with("0\nSECTION\n"), "DXF is well-formed");
        assert!(dxf.ends_with("\nEOF\n"), "DXF terminates with EOF");
        assert!(dxf.contains("\nCIRCLE\n"), "DXF carries ring geometry");

        let _ = std::fs::remove_dir_all(out_dir.parent().expect("nested dir"));
    }

    #[test]
    fn gen_target_rejects_geometry_without_code_band_gap() {
        let out_dir = temp_output_dir("invalid_geometry");
        let mut args = fixture_gen_target_args(out_dir);
        args.marker_inner_radius_mm = 4.1;

        let err =
            run_gen_target(&GenTargetCommands::Hex(args)).expect_err("invalid geometry must fail");
        assert!(
            err.to_string().contains("no code band between rings"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gen_target_uses_generated_name_when_name_is_omitted() {
        let out_dir = temp_output_dir("generated_name");
        let mut args = fixture_gen_target_args(out_dir.clone());
        args.name = None;
        args.output.basename = DEFAULT_GEN_TARGET_BASENAME.to_string();
        run_gen_target(&GenTargetCommands::Hex(args)).expect("generate target with auto name");

        let spec: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out_dir.join("target_spec.json")).expect("read spec"),
        )
        .expect("parse spec");
        assert_eq!(
            spec.get("name").and_then(|v| v.as_str()),
            Some("ringgrid_hex_r3_c4_p8.000_o4.800_i3.200_w1.152")
        );

        let _ = std::fs::remove_dir_all(out_dir);
    }

    #[test]
    fn gen_target_rejects_invalid_geometry_and_options() {
        let mut args = fixture_gen_target_args(temp_output_dir("bad_rows"));
        args.rows = 0;
        let err = run_gen_target(&GenTargetCommands::Hex(args)).expect_err("rows=0 must fail");
        assert!(err.to_string().contains("rows"), "unexpected error: {err}");

        let mut args = fixture_gen_target_args(temp_output_dir("bad_margin"));
        args.output.margin_mm = -1.0;
        let err =
            run_gen_target(&GenTargetCommands::Hex(args)).expect_err("negative margin must fail");
        assert!(
            err.to_string().contains("margin"),
            "unexpected error: {err}"
        );

        let mut args = fixture_gen_target_args(temp_output_dir("bad_dpi"));
        args.output.dpi = 0.0;
        let err =
            run_gen_target(&GenTargetCommands::Hex(args)).expect_err("non-positive dpi must fail");
        assert!(err.to_string().contains("dpi"), "unexpected error: {err}");
    }
}
