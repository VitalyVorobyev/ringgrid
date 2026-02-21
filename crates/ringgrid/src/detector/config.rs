use crate::board_layout::BoardLayout;
use crate::homography::RansacHomographyConfig;
use crate::marker::{DecodeConfig, MarkerSpec};
use crate::pixelmap::SelfUndistortConfig;
use crate::ring::{EdgeSampleConfig, OuterEstimationConfig};

use super::proposal::ProposalConfig;

/// Seed-injection controls for proposal generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
#[serde(default)]
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
    /// Require a perfect decode (dist=0 and margin ≥ CODEBOOK_MIN_CYCLIC_DIST) for
    /// a completion marker to be accepted.
    ///
    /// When homography prediction accuracy is low (e.g. significant lens distortion
    /// without a calibrated mapper), the H-projected seed can be several pixels off.
    /// Under those conditions, the geometry gates alone are insufficient; requiring a
    /// perfect decode provides an independent quality signal that does not depend on H.
    ///
    /// Default: `false` (backward-compatible). Set to `true` for Scheimpflug / high-
    /// distortion setups where no calibrated camera model is available.
    #[serde(default = "CompletionParams::default_require_perfect_decode")]
    pub require_perfect_decode: bool,
}

impl CompletionParams {
    fn default_require_perfect_decode() -> bool {
        false
    }
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
            require_perfect_decode: false,
        }
    }
}

/// Projective-only unbiased center recovery from inner/outer conics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
#[serde(default)]
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
    /// Confidence multiplier applied when inner ellipse fit fails or is absent.
    ///
    /// Inner fit failure is a reliable signal of poor image quality (heavy blur,
    /// distortion, or edge contamination). Setting this below 1.0 discounts the
    /// decode confidence when the inner ring cannot be fitted, making true markers
    /// in clear regions easier to separate from false detections.
    ///
    /// Default: 0.7 (30 % confidence reduction on inner-fit miss).
    #[serde(default = "InnerFitConfig::default_miss_confidence_factor")]
    pub miss_confidence_factor: f32,
    /// Maximum allowed angular gap (radians) between consecutive inner edge
    /// points. Fits where the largest gap exceeds this are rejected.
    ///
    /// Default: π/2 (90 degrees).
    #[serde(default = "InnerFitConfig::default_max_angular_gap_rad")]
    pub max_angular_gap_rad: f64,
    /// When true, markers are hard-rejected (not just penalized) if the inner
    /// ellipse cannot be fitted. Requires two good ellipses per marker.
    ///
    /// Default: false (backward-compatible).
    #[serde(default = "InnerFitConfig::default_require_inner_fit")]
    pub require_inner_fit: bool,
}

impl InnerFitConfig {
    fn default_miss_confidence_factor() -> f32 {
        0.7
    }
    fn default_max_angular_gap_rad() -> f64 {
        std::f64::consts::FRAC_PI_2
    }
    fn default_require_inner_fit() -> bool {
        false
    }
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
            miss_confidence_factor: 0.7,
            max_angular_gap_rad: Self::default_max_angular_gap_rad(),
            require_inner_fit: false,
        }
    }
}

/// Configuration for robust outer ellipse fitting from sampled edge points.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OuterFitConfig {
    /// Minimum number of sampled points required to attempt direct LS fit.
    pub min_direct_fit_points: usize,
    /// Minimum sampled points required before attempting RANSAC.
    pub min_ransac_points: usize,
    /// RANSAC configuration for robust outer ellipse fitting.
    pub ransac: crate::conic::RansacConfig,
    /// Relative weight of size agreement in outer-hypothesis scoring.
    ///
    /// The score combines decode quality, fit support, size agreement, and
    /// residual quality. This weight controls the size-agreement term and is
    /// normalized with the other terms at runtime.
    ///
    /// Default: `0.15` (preserves legacy behavior).
    #[serde(default = "OuterFitConfig::default_size_score_weight")]
    pub size_score_weight: f32,
    /// Maximum allowed angular gap (radians) between consecutive outer edge
    /// points. Fits where the largest gap exceeds this are rejected.
    ///
    /// Default: π/2 (90 degrees).
    #[serde(default = "OuterFitConfig::default_max_angular_gap_rad")]
    pub max_angular_gap_rad: f64,
}

impl OuterFitConfig {
    fn default_size_score_weight() -> f32 {
        0.15
    }

    fn default_max_angular_gap_rad() -> f64 {
        std::f64::consts::FRAC_PI_2
    }
}

impl Default for OuterFitConfig {
    fn default() -> Self {
        Self {
            min_direct_fit_points: 6,
            min_ransac_points: 8,
            ransac: crate::conic::RansacConfig {
                max_iters: 200,
                inlier_threshold: 1.5,
                min_inliers: 6,
                seed: 42,
            },
            size_score_weight: Self::default_size_score_weight(),
            max_angular_gap_rad: Self::default_max_angular_gap_rad(),
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
/// The detector uses this range to derive proposal radii, outer-edge search
/// windows, ellipse validation bounds, and completion ROI. When the marker
/// scale prior is set via [`DetectConfig::set_marker_scale_prior`] or a
/// constructor, all scale-dependent parameters are auto-derived.
///
/// A single known size can be expressed with
/// [`MarkerScalePrior::from_nominal_diameter_px`]. For scenes where markers
/// vary in apparent size, use [`MarkerScalePrior::new`] with a range.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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

/// Structural ID verification and correction using hex neighborhood consensus.
///
/// Runs after fit-decode and deduplication, before the global RANSAC filter.
/// Uses the board's hex lattice geometry to detect wrong IDs (misidentified by
/// the codebook decoder) and recover missing ones. Each marker's correct ID is
/// implied by its decoded neighbors' positions: neighbors vote on the expected
/// board position of the query marker using a local affine transform (or scale
/// estimate when fewer than 3 neighbors are available).
///
/// Markers that cannot be verified or corrected have their IDs cleared
/// (`id = None`) or are removed entirely depending on `remove_unverified`.
/// This guarantees no wrong IDs reach the global filter or completion stages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct IdCorrectionConfig {
    /// Enable structural ID verification and correction.
    pub enable: bool,
    /// Local-scale search radius multiplier for a fixed single-stage pass.
    ///
    /// The search gate for pair `(i, j)` is:
    /// `dist_px(i,j) <= mul * 0.5 * (outer_radius_px_i + outer_radius_px_j)`.
    ///
    /// When `Some`, this single multiplier is used.
    /// When `None`, `auto_search_radius_outer_muls` is used.
    pub search_radius_outer_mul: Option<f64>,
    /// Local-scale staged search multipliers used when
    /// `search_radius_outer_mul` is `None`.
    pub auto_search_radius_outer_muls: Vec<f64>,
    /// Local-scale neighborhood multiplier for consistency checks.
    pub consistency_outer_mul: f64,
    /// Minimum number of local neighbors required to run consistency checks.
    pub consistency_min_neighbors: usize,
    /// Minimum number of one-hop board-neighbor support edges required for a
    /// non-soft-locked ID to remain assigned.
    pub consistency_min_support_edges: usize,
    /// Maximum allowed contradiction fraction in local consistency checks.
    pub consistency_max_contradiction_frac: f32,
    /// When enabled, exact decodes (`best_dist=0, margin>=2`) are soft-locked:
    /// they are not overridden during normal recovery and only cleared on
    /// strict structural contradiction.
    pub soft_lock_exact_decode: bool,
    /// Maximum image-space radius for spatial neighbor search (pixels).
    ///
    /// Legacy compatibility field.
    ///
    /// New local-scale path does not use this field unless legacy mode is
    /// requested (by setting `auto_search_radius_outer_muls=[]` and
    /// `search_radius_outer_mul=None`).
    pub neighbor_search_radius_px: Option<f64>,
    /// Multipliers used to build the adaptive local-radius schedule when
    /// `neighbor_search_radius_px` is `None`.
    ///
    /// Each multiplier is applied to an estimated `pitch_px` to produce one
    /// local neighbor-search radius. Legacy compatibility field.
    pub auto_neighbor_radius_muls: Vec<f64>,
    /// Optional cap (pixels) applied to every auto-derived neighbor radius.
    /// Legacy compatibility field.
    pub max_auto_neighbor_radius_px: Option<f64>,
    /// Minimum number of independent neighbor votes required to accept a
    /// candidate ID for a marker that already has an id. Default: 2.
    pub min_votes: usize,
    /// Minimum votes to assign an ID to a marker that currently has `id = None`.
    ///
    /// A single high-confidence trusted neighbor is sufficient evidence when
    /// there is no existing wrong ID to protect. Default: 1.
    pub min_votes_recover: usize,
    /// Minimum fraction of total weighted votes the winning candidate must
    /// receive. Default: 0.55 (slight majority).
    pub min_vote_weight_frac: f32,
    /// H-reprojection gate (pixels) used by rough-homography fallback
    /// assignments. Intentionally loose to tolerate significant distortion.
    pub h_reproj_gate_px: f64,
    /// Enable rough-homography fallback for unresolved markers.
    pub homography_fallback_enable: bool,
    /// Minimum trusted markers required before attempting homography fallback.
    pub homography_min_trusted: usize,
    /// Minimum inliers required for fallback homography RANSAC acceptance.
    pub homography_min_inliers: usize,
    /// Maximum RANSAC iterations for fallback homography fitting.
    pub homography_ransac_max_iters: usize,
    /// Allow homography fallback to use recovered (non-anchor) trusted seeds.
    pub homography_use_recovered_seeds: bool,
    /// Number of nearest board IDs to evaluate per unresolved marker during
    /// homography fallback.
    pub homography_candidate_top_k: usize,
    /// Maximum number of iterative correction passes. Default: 5.
    pub max_iters: usize,
    /// When `true`, remove markers that cannot be verified or corrected.
    /// When `false` (default), clear their ID (set to `None`) and keep the
    /// detection so its geometry is available for debugging.
    pub remove_unverified: bool,
    /// Minimum decode confidence for bootstrapping trusted seeds when no
    /// homography is available. Default: 0.7.
    pub seed_min_decode_confidence: f32,
}

impl Default for IdCorrectionConfig {
    fn default() -> Self {
        Self {
            enable: true,
            search_radius_outer_mul: None,
            auto_search_radius_outer_muls: vec![2.4, 2.9, 3.5, 4.2, 5.0],
            consistency_outer_mul: 3.2,
            consistency_min_neighbors: 3,
            consistency_min_support_edges: 2,
            consistency_max_contradiction_frac: 0.5,
            soft_lock_exact_decode: true,
            neighbor_search_radius_px: None,
            auto_neighbor_radius_muls: vec![2.5, 4.0, 6.0, 9.0, 13.5, 18.0],
            max_auto_neighbor_radius_px: Some(120.0),
            min_votes: 2,
            min_votes_recover: 1,
            min_vote_weight_frac: 0.55,
            h_reproj_gate_px: 30.0,
            homography_fallback_enable: true,
            homography_min_trusted: 24,
            homography_min_inliers: 12,
            homography_ransac_max_iters: 1200,
            homography_use_recovered_seeds: false,
            homography_candidate_top_k: 19,
            max_iters: 5,
            remove_unverified: false,
            seed_min_decode_confidence: 0.7,
        }
    }
}

/// Top-level detection configuration.
///
/// Contains all parameters that control the detection pipeline. Use one of the
/// recommended constructors rather than constructing directly:
///
/// - [`DetectConfig::from_target`] — default scale prior
/// - [`DetectConfig::from_target_and_scale_prior`] — explicit scale range
/// - [`DetectConfig::from_target_and_marker_diameter`] — fixed diameter hint
///
/// These constructors auto-derive scale-dependent parameters (proposal radii,
/// edge search windows, validation bounds) from the board geometry and marker
/// scale prior. Individual fields can be tuned after construction.
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
    /// Robust outer ellipse fitting controls shared across pipeline stages.
    pub outer_fit: OuterFitConfig,
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
    /// Structural ID verification and correction using hex neighborhood consensus.
    pub id_correction: IdCorrectionConfig,
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
            outer_fit: OuterFitConfig::default(),
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
            id_correction: IdCorrectionConfig::default(),
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
        assert!((core.miss_confidence_factor - 0.7).abs() < 1e-6);
        assert!(
            (core.max_angular_gap_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "inner max_angular_gap_rad"
        );
        assert!(!core.require_inner_fit);
    }

    #[test]
    fn outer_fit_config_defaults_are_stable() {
        let core = OuterFitConfig::default();
        assert_eq!(core.min_direct_fit_points, 6);
        assert_eq!(core.min_ransac_points, 8);
        assert_eq!(core.ransac.max_iters, 200);
        assert!((core.ransac.inlier_threshold - 1.5).abs() < 1e-9);
        assert_eq!(core.ransac.min_inliers, 6);
        assert_eq!(core.ransac.seed, 42);
        assert!((core.size_score_weight - 0.15).abs() < 1e-6);
        assert!(
            (core.max_angular_gap_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "outer max_angular_gap_rad"
        );
    }

    #[test]
    fn outer_fit_config_deserialize_missing_size_weight_uses_default() {
        let json = r#"{
            "min_direct_fit_points": 6,
            "min_ransac_points": 8,
            "ransac": {
                "max_iters": 200,
                "inlier_threshold": 1.5,
                "min_inliers": 6,
                "seed": 42
            }
        }"#;
        let cfg: OuterFitConfig = serde_json::from_str(json).expect("deserialize outer fit config");
        assert!((cfg.size_score_weight - 0.15).abs() < 1e-6);
    }

    #[test]
    fn detect_config_includes_fit_configs() {
        let cfg = DetectConfig::default();
        assert_eq!(cfg.inner_fit.min_points, 20);
        assert_eq!(cfg.inner_fit.ransac.min_inliers, 8);
        assert_eq!(cfg.outer_fit.min_direct_fit_points, 6);
        assert_eq!(cfg.outer_fit.ransac.min_inliers, 6);
    }

    #[test]
    fn id_correction_config_defaults_are_stable() {
        let cfg = IdCorrectionConfig::default();
        assert!(cfg.enable);
        assert!(cfg.search_radius_outer_mul.is_none());
        assert_eq!(
            cfg.auto_search_radius_outer_muls,
            vec![2.4, 2.9, 3.5, 4.2, 5.0]
        );
        assert!((cfg.consistency_outer_mul - 3.2).abs() < 1e-9);
        assert_eq!(cfg.consistency_min_neighbors, 3);
        assert_eq!(cfg.consistency_min_support_edges, 2);
        assert!((cfg.consistency_max_contradiction_frac - 0.5).abs() < 1e-6);
        assert!(cfg.soft_lock_exact_decode);
        assert!(cfg.neighbor_search_radius_px.is_none());
        assert_eq!(
            cfg.auto_neighbor_radius_muls,
            vec![2.5, 4.0, 6.0, 9.0, 13.5, 18.0]
        );
        assert_eq!(cfg.max_auto_neighbor_radius_px, Some(120.0));
        assert_eq!(cfg.min_votes, 2);
        assert_eq!(cfg.min_votes_recover, 1);
        assert!((cfg.min_vote_weight_frac - 0.55).abs() < 1e-6);
        assert!((cfg.h_reproj_gate_px - 30.0).abs() < 1e-9);
        assert!(cfg.homography_fallback_enable);
        assert_eq!(cfg.homography_min_trusted, 24);
        assert_eq!(cfg.homography_min_inliers, 12);
        assert_eq!(cfg.homography_ransac_max_iters, 1200);
        assert!(!cfg.homography_use_recovered_seeds);
        assert_eq!(cfg.homography_candidate_top_k, 19);
        assert_eq!(cfg.max_iters, 5);
        assert!(!cfg.remove_unverified);
        assert!((cfg.seed_min_decode_confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn id_correction_config_deserialize_legacy_fields_uses_new_defaults() {
        let json = r#"{
            "enable": true,
            "neighbor_search_radius_px": null,
            "min_votes": 2,
            "min_votes_recover": 1,
            "min_vote_weight_frac": 0.55,
            "h_reproj_gate_px": 30.0,
            "max_iters": 5,
            "remove_unverified": false,
            "seed_min_decode_confidence": 0.7
        }"#;
        let cfg: IdCorrectionConfig =
            serde_json::from_str(json).expect("deserialize legacy id correction config");
        assert!(cfg.search_radius_outer_mul.is_none());
        assert_eq!(
            cfg.auto_search_radius_outer_muls,
            vec![2.4, 2.9, 3.5, 4.2, 5.0]
        );
        assert!((cfg.consistency_outer_mul - 3.2).abs() < 1e-9);
        assert_eq!(cfg.consistency_min_neighbors, 3);
        assert_eq!(cfg.consistency_min_support_edges, 2);
        assert!((cfg.consistency_max_contradiction_frac - 0.5).abs() < 1e-6);
        assert!(cfg.soft_lock_exact_decode);
        assert_eq!(
            cfg.auto_neighbor_radius_muls,
            vec![2.5, 4.0, 6.0, 9.0, 13.5, 18.0]
        );
        assert_eq!(cfg.max_auto_neighbor_radius_px, Some(120.0));
        assert!(cfg.homography_fallback_enable);
        assert_eq!(cfg.homography_min_trusted, 24);
        assert_eq!(cfg.homography_min_inliers, 12);
        assert_eq!(cfg.homography_ransac_max_iters, 1200);
        assert!(!cfg.homography_use_recovered_seeds);
        assert_eq!(cfg.homography_candidate_top_k, 19);
    }
}
