use crate::board_layout::BoardLayout;
use crate::homography::RansacHomographyConfig;
use crate::marker::{DecodeConfig, MarkerSpec};
use crate::pixelmap::SelfUndistortConfig;
use crate::ring::{EdgeSampleConfig, OuterEstimationConfig};

use super::proposal::ProposalConfig;

/// Seed-injection controls for proposal generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SeedProposalParams {
    /// Radius (pixels) used to merge seed centers with detector proposals.
    pub merge_radius_px: f32,
    /// Score assigned to injected seed proposals.
    pub seed_score: f32,
    /// Maximum number of seeds consumed in one run.
    pub max_seeds: Option<usize>,
}

impl Default for SeedProposalParams {
    fn default() -> Self {
        Self {
            merge_radius_px: 3.0,
            seed_score: 1.0e12,
            max_seeds: Some(512),
        }
    }
}

/// Configuration for homography-guided completion: attempt local fits for
/// missing IDs at H-projected board locations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompletionParams {
    /// Enable completion (runs only when a valid homography is available).
    pub enable: bool,
    /// Radial sampling extent (pixels) used for edge sampling around the prior center.
    pub roi_radius_px: f32,
    /// Maximum allowed reprojection error (pixels) between the fitted center and
    /// the H-projected board center.
    pub reproj_gate_px: f32,
    /// Minimum fit confidence in [0, 1].
    pub min_fit_confidence: f32,
    /// Minimum arc coverage (fraction of rays with both edges found).
    pub min_arc_coverage: f32,
    /// Optional cap on how many completion fits to attempt (in ID order).
    pub max_attempts: Option<usize>,
    /// Skip attempts whose projected center is too close to the image boundary.
    pub image_margin_px: f32,
}

impl Default for CompletionParams {
    fn default() -> Self {
        Self {
            enable: true,
            roi_radius_px: 24.0,
            reproj_gate_px: 3.0,
            min_fit_confidence: 0.45,
            min_arc_coverage: 0.35,
            max_attempts: None,
            image_margin_px: 10.0,
        }
    }
}

/// Projective-only unbiased center recovery from inner/outer conics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProjectiveCenterParams {
    /// Enable projective unbiased center estimation.
    pub enable: bool,
    /// Use `marker_spec.r_inner_expected` as an optional eigenvalue prior.
    pub use_expected_ratio: bool,
    /// Weight of the eigenvalue-vs-ratio penalty term.
    pub ratio_penalty_weight: f64,
    /// Optional maximum allowed shift (pixels) from the pre-correction center.
    ///
    /// When set, large jumps are rejected and the original center is kept.
    pub max_center_shift_px: Option<f64>,
    /// Optional maximum accepted projective-selection residual.
    ///
    /// Higher values are less strict; `None` disables this gate.
    pub max_selected_residual: Option<f64>,
    /// Optional minimum accepted eigenvalue separation used by the selector.
    ///
    /// Low separation indicates unstable conic-pencil eigenpairs.
    pub min_eig_separation: Option<f64>,
}

impl Default for ProjectiveCenterParams {
    fn default() -> Self {
        Self {
            enable: true,
            use_expected_ratio: true,
            ratio_penalty_weight: 1.0,
            max_center_shift_px: None,
            max_selected_residual: Some(0.25),
            min_eig_separation: Some(1e-6),
        }
    }
}

/// Configuration for robust inner ellipse fitting from outer-fit hints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InnerFitConfig {
    /// Minimum number of sampled points required to attempt a fit.
    pub min_points: usize,
    /// Minimum accepted inlier ratio when RANSAC is used.
    pub min_inlier_ratio: f32,
    /// Maximum accepted RMS Sampson residual (px) of the fitted inner ellipse.
    pub max_rms_residual: f64,
    /// Maximum allowed center shift from outer to inner fit center (px).
    pub max_center_shift_px: f64,
    /// Maximum allowed absolute error in recovered scale ratio vs radial hint.
    pub max_ratio_abs_error: f64,
    /// Local half-width (in radius-sample indices) around the radial hint.
    pub local_peak_halfwidth_idx: usize,
    /// RANSAC configuration for robust inner ellipse fitting.
    pub ransac: crate::conic::RansacConfig,
}

impl Default for InnerFitConfig {
    fn default() -> Self {
        Self {
            min_points: 20,
            min_inlier_ratio: 0.5,
            max_rms_residual: 1.0,
            max_center_shift_px: 12.0,
            max_ratio_abs_error: 0.15,
            local_peak_halfwidth_idx: 3,
            ransac: crate::conic::RansacConfig {
                max_iters: 200,
                inlier_threshold: 1.5,
                min_inliers: 8,
                seed: 43,
            },
        }
    }
}

/// Center-correction strategy used after local fits are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CircleRefinementMethod {
    /// Disable center correction.
    None,
    /// Run projective-center recovery from inner/outer conics.
    #[default]
    ProjectiveCenter,
}

impl CircleRefinementMethod {
    /// Returns `true` when this method includes projective-center recovery.
    pub fn uses_projective_center(self) -> bool {
        matches!(self, Self::ProjectiveCenter)
    }
}

/// Scale prior for marker diameter in detector working pixels.
///
/// The detector uses this range to derive proposal radii, outer-edge search,
/// and validation gates. A single known size can be expressed by setting
/// `diameter_min_px == diameter_max_px`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MarkerScalePrior {
    /// Minimum expected marker outer diameter in pixels.
    pub diameter_min_px: f32,
    /// Maximum expected marker outer diameter in pixels.
    pub diameter_max_px: f32,
}

impl MarkerScalePrior {
    const MIN_DIAMETER_FLOOR_PX: f32 = 4.0;

    /// Construct a scale prior from a diameter range in pixels.
    pub fn new(diameter_min_px: f32, diameter_max_px: f32) -> Self {
        let mut out = Self {
            diameter_min_px,
            diameter_max_px,
        };
        out.normalize_in_place();
        out
    }

    /// Construct a fixed-size scale prior from one diameter hint.
    pub fn from_nominal_diameter_px(diameter_px: f32) -> Self {
        Self::new(diameter_px, diameter_px)
    }

    /// Return the normalized diameter range `[min, max]` in pixels.
    pub fn diameter_range_px(self) -> [f32; 2] {
        let n = self.normalized();
        [n.diameter_min_px, n.diameter_max_px]
    }

    /// Return nominal diameter (midpoint of `[min, max]`) in pixels.
    pub fn nominal_diameter_px(self) -> f32 {
        let [d_min, d_max] = self.diameter_range_px();
        0.5 * (d_min + d_max)
    }

    /// Return nominal outer radius in pixels.
    pub fn nominal_outer_radius_px(self) -> f32 {
        self.nominal_diameter_px() * 0.5
    }

    /// Return a normalized copy with finite, ordered, non-degenerate bounds.
    pub fn normalized(self) -> Self {
        let mut out = self;
        out.normalize_in_place();
        out
    }

    fn normalize_in_place(&mut self) {
        let defaults = MarkerScalePrior::default();
        let mut d_min = if self.diameter_min_px.is_finite() {
            self.diameter_min_px
        } else {
            defaults.diameter_min_px
        };
        let mut d_max = if self.diameter_max_px.is_finite() {
            self.diameter_max_px
        } else {
            defaults.diameter_max_px
        };
        if d_min > d_max {
            std::mem::swap(&mut d_min, &mut d_max);
        }
        d_min = d_min.max(Self::MIN_DIAMETER_FLOOR_PX);
        d_max = d_max.max(d_min);
        self.diameter_min_px = d_min;
        self.diameter_max_px = d_max;
    }
}

impl Default for MarkerScalePrior {
    fn default() -> Self {
        Self {
            diameter_min_px: 20.0,
            diameter_max_px: 56.0,
        }
    }
}

/// Top-level detection configuration.
#[derive(Debug, Clone)]
pub struct DetectConfig {
    /// Marker diameter prior (range) in working-frame pixels.
    pub marker_scale: MarkerScalePrior,
    /// Outer edge estimation configuration (anchored on `marker_scale`).
    pub outer_estimation: OuterEstimationConfig,
    /// Proposal generation configuration.
    pub proposal: ProposalConfig,
    /// Seed-injection controls for multi-pass proposal generation.
    pub seed_proposals: SeedProposalParams,
    /// Radial edge sampling configuration.
    pub edge_sample: EdgeSampleConfig,
    /// Marker decode configuration.
    pub decode: DecodeConfig,
    /// Marker geometry specification and estimator controls.
    pub marker_spec: MarkerSpec,
    /// Robust inner ellipse fitting controls shared across pipeline stages.
    pub inner_fit: InnerFitConfig,
    /// Post-fit circle refinement method selector.
    pub circle_refinement: CircleRefinementMethod,
    /// Projective-center recovery controls.
    pub projective_center: ProjectiveCenterParams,
    /// Homography-guided completion controls.
    pub completion: CompletionParams,
    /// Minimum semi-axis for a valid outer ellipse.
    pub min_semi_axis: f64,
    /// Maximum semi-axis for a valid outer ellipse.
    pub max_semi_axis: f64,
    /// Maximum aspect ratio (a/b) for a valid ellipse.
    pub max_aspect_ratio: f64,
    /// NMS dedup radius for final markers (pixels).
    pub dedup_radius: f64,
    /// Enable global homography filtering (requires board spec).
    pub use_global_filter: bool,
    /// RANSAC homography configuration.
    pub ransac_homography: RansacHomographyConfig,
    /// Board layout: marker positions and geometry.
    pub board: BoardLayout,
    /// Self-undistort estimation controls.
    pub self_undistort: SelfUndistortConfig,
}

const EDGE_EXPANSION_FRAC_OUTER: f32 = 0.12;

impl DetectConfig {
    /// Build a configuration with scale-dependent parameters derived from a
    /// marker diameter range and a runtime target layout.
    ///
    /// This is the recommended constructor for library users. After calling it,
    /// individual fields can be overridden as needed.
    pub fn from_target_and_scale_prior(board: BoardLayout, marker_scale: MarkerScalePrior) -> Self {
        let mut cfg = Self {
            marker_scale: marker_scale.normalized(),
            ..Default::default()
        };
        cfg.board = board;
        apply_target_geometry_priors(&mut cfg);
        apply_marker_scale_prior(&mut cfg);
        cfg
    }

    /// Build a configuration from target layout and default marker scale prior.
    pub fn from_target(board: BoardLayout) -> Self {
        Self::from_target_and_scale_prior(board, MarkerScalePrior::default())
    }

    /// Build a configuration from target layout and a fixed marker diameter hint.
    pub fn from_target_and_marker_diameter(board: BoardLayout, diameter_px: f32) -> Self {
        Self::from_target_and_scale_prior(
            board,
            MarkerScalePrior::from_nominal_diameter_px(diameter_px),
        )
    }

    /// Update marker scale prior and re-derive all scale-coupled parameters.
    pub fn set_marker_scale_prior(&mut self, marker_scale: MarkerScalePrior) {
        self.marker_scale = marker_scale.normalized();
        apply_marker_scale_prior(self);
    }

    /// Update marker scale prior from a fixed marker diameter hint.
    pub fn set_marker_diameter_hint_px(&mut self, diameter_px: f32) {
        self.set_marker_scale_prior(MarkerScalePrior::from_nominal_diameter_px(diameter_px));
    }
}

impl Default for DetectConfig {
    fn default() -> Self {
        let mut cfg = Self {
            marker_scale: MarkerScalePrior::default(),
            outer_estimation: OuterEstimationConfig::default(),
            proposal: ProposalConfig::default(),
            seed_proposals: SeedProposalParams::default(),
            edge_sample: EdgeSampleConfig::default(),
            decode: DecodeConfig::default(),
            marker_spec: MarkerSpec::default(),
            inner_fit: InnerFitConfig::default(),
            circle_refinement: CircleRefinementMethod::default(),
            projective_center: ProjectiveCenterParams::default(),
            completion: CompletionParams::default(),
            min_semi_axis: 3.0,
            max_semi_axis: 15.0,
            max_aspect_ratio: 3.0,
            dedup_radius: 6.0,
            use_global_filter: true,
            ransac_homography: RansacHomographyConfig::default(),
            board: BoardLayout::default(),
            self_undistort: SelfUndistortConfig::default(),
        };
        apply_target_geometry_priors(&mut cfg);
        apply_marker_scale_prior(&mut cfg);
        cfg
    }
}

fn apply_marker_scale_prior(config: &mut DetectConfig) {
    config.marker_scale = config.marker_scale.normalized();
    let [d_min, d_max] = config.marker_scale.diameter_range_px();
    let d_nom = config.marker_scale.nominal_diameter_px();
    let r_min = d_min * 0.5;
    let r_max = d_max * 0.5;
    let r_nom = d_nom * 0.5;

    // Proposal search radii
    config.proposal.r_min = (r_min * 0.4).max(2.0);
    config.proposal.r_max = r_max * 1.7;
    config.proposal.nms_radius = (r_min * 0.8).max(2.0);

    // Edge sampling range
    config.edge_sample.r_max = r_max * 2.0;
    config.edge_sample.r_min = 1.5;
    config.outer_estimation.theta_samples = config.edge_sample.n_rays;
    let desired_halfwidth = ((r_max - r_min) * 0.5).max(2.0);
    let base_halfwidth = OuterEstimationConfig::default().search_halfwidth_px;
    config.outer_estimation.search_halfwidth_px = desired_halfwidth.max(base_halfwidth);

    // Ellipse validation bounds
    config.min_semi_axis = (r_min as f64 * 0.3).max(2.0);
    config.max_semi_axis = (r_max as f64 * 2.5).max(config.min_semi_axis);

    // Completion ROI
    config.completion.roi_radius_px = ((d_nom as f64 * 0.75).clamp(24.0, 80.0)) as f32;

    // Projective center max shift
    config.projective_center.max_center_shift_px = Some((2.0 * r_nom) as f64);
}

fn apply_target_geometry_priors(config: &mut DetectConfig) {
    let outer = config.board.marker_outer_radius_mm();
    let inner = config.board.marker_inner_radius_mm();
    if !(outer.is_finite() && inner.is_finite()) || outer <= 0.0 || inner <= 0.0 || inner >= outer {
        return;
    }

    let edge_pad = (outer * EDGE_EXPANSION_FRAC_OUTER).max(0.0);
    let inner_edge = (inner - edge_pad).max(outer * 0.05);
    let outer_edge = outer + edge_pad;
    if inner_edge > 0.0 && inner_edge < outer_edge {
        let r_inner_expected = (inner_edge / outer_edge).clamp(0.1, 0.95);
        config.marker_spec.r_inner_expected = r_inner_expected;
        config.decode.code_band_ratio = (0.5 * (1.0 + r_inner_expected)).clamp(0.2, 0.98);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_fit_config_defaults_are_stable() {
        let core = InnerFitConfig::default();
        assert_eq!(core.min_points, 20);
        assert!((core.min_inlier_ratio - 0.5).abs() < 1e-6);
        assert!((core.max_rms_residual - 1.0).abs() < 1e-9);
        assert!((core.max_center_shift_px - 12.0).abs() < 1e-9);
        assert!((core.max_ratio_abs_error - 0.15).abs() < 1e-9);
        assert_eq!(core.local_peak_halfwidth_idx, 3);
        assert_eq!(core.ransac.max_iters, 200);
        assert!((core.ransac.inlier_threshold - 1.5).abs() < 1e-9);
        assert_eq!(core.ransac.min_inliers, 8);
        assert_eq!(core.ransac.seed, 43);
    }

    #[test]
    fn detect_config_includes_inner_fit_config() {
        let cfg = DetectConfig::default();
        assert_eq!(cfg.inner_fit.min_points, 20);
        assert_eq!(cfg.inner_fit.ransac.min_inliers, 8);
    }
}
