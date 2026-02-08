//! Full ring detection pipeline: proposal → edge sampling → fit → decode → global filter.

use image::GrayImage;
use std::collections::HashSet;

use crate::board_layout::BoardLayout;
use crate::camera::PixelMapper;
use crate::debug_dump as dbg;
use crate::homography::{self, RansacHomographyConfig};
use crate::marker_spec::MarkerSpec;
use crate::refine;
use crate::{DetectedMarker, DetectionResult, RansacStats};

use super::decode::DecodeConfig;
use super::edge_sample::EdgeSampleConfig;
use super::outer_estimate::OuterEstimationConfig;
use super::pipeline::dedup::dedup_with_debug as dedup_with_debug_impl;
use super::pipeline::dedup::{
    dedup_by_id as dedup_by_id_impl, dedup_markers as dedup_markers_impl,
};
use super::pipeline::global_filter::global_filter as global_filter_impl;
use super::pipeline::global_filter::global_filter_with_debug as global_filter_with_debug_impl;
use super::proposal::{find_proposals, Proposal, ProposalConfig};
#[path = "detect/completion.rs"]
mod completion;
#[path = "detect/debug_conv.rs"]
mod debug_conv;
#[path = "detect/homography_utils.rs"]
mod homography_utils;
#[path = "detect/inner_fit.rs"]
mod inner_fit;
#[path = "detect/marker_build.rs"]
mod marker_build;
#[path = "detect/outer_fit.rs"]
mod outer_fit;
#[path = "detect/refine_h.rs"]
mod refine_h;
#[path = "detect/stages/mod.rs"]
mod stages;
use completion::{CompletionAttemptRecord, CompletionDebugOptions, CompletionStats};
use outer_fit::{
    compute_center, fit_outer_ellipse_robust_with_reason, marker_outer_radius_expected_px,
    mean_axis_px_from_marker, median_outer_radius_from_neighbors_px, OuterFitCandidate,
};

/// Debug collection options for `detect_rings_with_debug`.
#[derive(Debug, Clone)]
pub struct DebugCollectConfig {
    /// Optional source image path copied into debug dump metadata.
    pub image_path: Option<String>,
    /// Marker diameter used for this run (debug metadata only).
    pub marker_diameter_px: f64,
    /// Maximum number of per-candidate records stored in stage dumps.
    pub max_candidates: usize,
    /// Whether to include sampled edge points in debug output.
    pub store_points: bool,
}

/// Seed-injection controls for proposal generation.
#[derive(Debug, Clone)]
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

/// Two-pass orchestration parameters.
#[derive(Debug, Clone)]
pub struct TwoPassParams {
    /// Seed-injection controls for the second pass.
    pub seed: SeedProposalParams,
    /// Keep pass-1 detections that are not present in pass-2 output.
    pub keep_pass1_markers: bool,
}

impl Default for TwoPassParams {
    fn default() -> Self {
        Self {
            seed: SeedProposalParams::default(),
            keep_pass1_markers: true,
        }
    }
}

/// Configuration for homography-guided completion: attempt local fits for
/// missing IDs at H-projected board locations.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

/// Center-correction strategy used after local fits are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CircleRefinementMethod {
    /// Disable center correction.
    None,
    /// Run projective-center recovery from inner/outer conics.
    #[default]
    ProjectiveCenter,
    /// Run board-plane circle refinement.
    NlBoard,
}

impl CircleRefinementMethod {
    /// Returns `true` when this method includes board-plane circle refinement.
    pub fn uses_nl_refine(self) -> bool {
        matches!(self, Self::NlBoard)
    }

    /// Returns `true` when this method includes projective-center recovery.
    pub fn uses_projective_center(self) -> bool {
        matches!(self, Self::ProjectiveCenter)
    }
}

/// Top-level detection configuration.
#[derive(Debug, Clone)]
pub struct DetectConfig {
    /// Expected marker outer diameter in pixels (for scale-anchored edge extraction).
    pub marker_diameter_px: f32,
    /// Outer edge estimation configuration (anchored on `marker_diameter_px`).
    pub outer_estimation: OuterEstimationConfig,
    /// Proposal generation configuration.
    pub proposal: ProposalConfig,
    /// Radial edge sampling configuration.
    pub edge_sample: EdgeSampleConfig,
    /// Marker decode configuration.
    pub decode: DecodeConfig,
    /// Marker geometry specification and estimator controls.
    pub marker_spec: MarkerSpec,
    /// Optional camera model for distortion-aware processing.
    ///
    /// When set, local fitting/sampling runs in the undistorted pixel frame
    /// and all reported marker geometry/centers use that working frame.
    pub camera: Option<crate::camera::CameraModel>,
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
    /// Enable one-iteration refinement using H.
    pub refine_with_h: bool,
    /// Non-linear per-marker refinement using board-plane circle fits.
    pub nl_refine: refine::RefineParams,
    /// Board layout: marker positions and geometry.
    pub board: BoardLayout,
    /// Self-undistort estimation controls.
    pub self_undistort: crate::self_undistort::SelfUndistortConfig,
}

const EDGE_EXPANSION_FRAC_OUTER: f32 = 0.12;

impl DetectConfig {
    /// Build a configuration with all scale-dependent parameters derived from
    /// the expected marker outer diameter in pixels and a runtime target layout.
    ///
    /// This is the recommended constructor for library users. After calling it,
    /// individual fields can be overridden as needed.
    pub fn from_target_and_marker_diameter(board: BoardLayout, diameter_px: f32) -> Self {
        let r_outer = diameter_px / 2.0;
        let mut cfg = Self {
            marker_diameter_px: diameter_px,
            ..Default::default()
        };
        cfg.board = board;
        apply_target_geometry_priors(&mut cfg);

        // Proposal search radii
        cfg.proposal.r_min = (r_outer * 0.4).max(2.0);
        cfg.proposal.r_max = r_outer * 1.7;
        cfg.proposal.nms_radius = r_outer * 0.8;

        // Edge sampling range
        cfg.edge_sample.r_max = r_outer * 2.0;
        cfg.edge_sample.r_min = 1.5;
        cfg.outer_estimation.theta_samples = cfg.edge_sample.n_rays;

        // Ellipse validation bounds
        cfg.min_semi_axis = (r_outer as f64 * 0.3).max(2.0);
        cfg.max_semi_axis = r_outer as f64 * 2.5;

        // Completion ROI
        cfg.completion.roi_radius_px = ((diameter_px as f64 * 0.75).clamp(24.0, 80.0)) as f32;

        // Projective center max shift
        cfg.projective_center.max_center_shift_px = Some(diameter_px as f64);

        cfg
    }
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            marker_diameter_px: 32.0,
            outer_estimation: OuterEstimationConfig::default(),
            proposal: ProposalConfig::default(),
            edge_sample: EdgeSampleConfig::default(),
            decode: DecodeConfig::default(),
            marker_spec: MarkerSpec::default(),
            camera: None,
            circle_refinement: CircleRefinementMethod::default(),
            projective_center: ProjectiveCenterParams::default(),
            completion: CompletionParams::default(),
            min_semi_axis: 3.0,
            max_semi_axis: 15.0,
            max_aspect_ratio: 3.0,
            dedup_radius: 6.0,
            use_global_filter: true,
            ransac_homography: RansacHomographyConfig::default(),
            refine_with_h: true,
            nl_refine: refine::RefineParams::default(),
            board: BoardLayout::default(),
            self_undistort: crate::self_undistort::SelfUndistortConfig::default(),
        }
    }
}

impl DetectConfig {
    /// Build config from marker diameter using the built-in default board layout.
    ///
    /// Intended for internal CLI compatibility paths.
    #[cfg(feature = "cli-internal")]
    pub fn from_marker_diameter_px(diameter_px: f32) -> Self {
        Self::from_target_and_marker_diameter(BoardLayout::default(), diameter_px)
    }
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
        config.marker_spec.r_inner_expected = (inner_edge / outer_edge).clamp(0.1, 0.95);
    }

    if let (Some(code_outer), Some(code_inner)) = (
        config.board.marker_code_band_outer_radius_mm(),
        config.board.marker_code_band_inner_radius_mm(),
    ) {
        if code_outer.is_finite()
            && code_inner.is_finite()
            && code_outer > code_inner
            && code_outer < outer
        {
            let code_center = 0.5 * (code_outer + code_inner);
            config.decode.code_band_ratio = (code_center / outer_edge).clamp(0.2, 0.98);
        }
    }
}

fn config_mapper(config: &DetectConfig) -> Option<&dyn PixelMapper> {
    config.camera.as_ref().map(|c| c as &dyn PixelMapper)
}

fn capped_len<T>(slice: &[T], max_len: Option<usize>) -> usize {
    max_len.unwrap_or(slice.len()).min(slice.len())
}

pub(super) fn find_proposals_with_seeds(
    gray: &GrayImage,
    proposal_cfg: &ProposalConfig,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> Vec<Proposal> {
    let mut proposals = find_proposals(gray, proposal_cfg);
    if seed_centers_image.is_empty() {
        return proposals;
    }

    let merge_r2 = seed_cfg.merge_radius_px.max(0.0).powi(2);
    for seed in seed_centers_image
        .iter()
        .take(capped_len(seed_centers_image, seed_cfg.max_seeds))
    {
        if !seed[0].is_finite() || !seed[1].is_finite() {
            continue;
        }

        let mut merged = false;
        for p in &mut proposals {
            let dx = p.x - seed[0];
            let dy = p.y - seed[1];
            if dx * dx + dy * dy <= merge_r2 {
                p.score = p.score.max(seed_cfg.seed_score);
                merged = true;
                break;
            }
        }
        if !merged {
            proposals.push(Proposal {
                x: seed[0],
                y: seed[1],
                score: seed_cfg.seed_score,
            });
        }
    }

    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    proposals
}

fn map_marker_image_to_working(
    marker: &DetectedMarker,
    mapper: &dyn PixelMapper,
) -> Option<DetectedMarker> {
    let mut out = marker.clone();
    out.center = mapper.image_to_working_pixel(marker.center)?;

    out.center_projective = marker
        .center_projective
        .and_then(|c| mapper.image_to_working_pixel(c));
    if marker.center_projective.is_some() && out.center_projective.is_none() {
        out.center_projective_residual = None;
    }

    // A non-linear mapper does not preserve ellipse/line structure exactly.
    // Avoid mixed-frame output by dropping these fields for pass-1 fallback markers.
    out.ellipse_outer = None;
    out.ellipse_inner = None;
    out.vanishing_line = None;
    out.edge_points_outer = None;
    out.edge_points_inner = None;
    Some(out)
}

fn merge_two_pass_markers(
    pass1: &[DetectedMarker],
    mut pass2: Vec<DetectedMarker>,
    keep_pass1_markers: bool,
    dedup_radius: f64,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    let mut failed_pass1_reproject = 0usize;
    if keep_pass1_markers {
        let ids_pass2: HashSet<usize> = pass2.iter().filter_map(|m| m.id).collect();
        for m in pass1 {
            if let Some(id) = m.id {
                if ids_pass2.contains(&id) {
                    continue;
                }
            }
            if let Some(mapper) = mapper {
                if let Some(mm) = map_marker_image_to_working(m, mapper) {
                    pass2.push(mm);
                } else {
                    failed_pass1_reproject += 1;
                }
            } else {
                pass2.push(m.clone());
            }
        }
    }

    if failed_pass1_reproject > 0 {
        tracing::warn!(
            failed_pass1_reproject,
            "failed to map some pass-1 markers into mapper working frame; dropping them"
        );
    }

    let mut merged = dedup_markers(pass2, dedup_radius);
    dedup_by_id(&mut merged);
    merged
}

fn collect_seed_centers_image(
    pass1: &DetectionResult,
    seed_cfg: &SeedProposalParams,
) -> Vec<[f32; 2]> {
    pass1
        .detected_markers
        .iter()
        .take(capped_len(&pass1.detected_markers, seed_cfg.max_seeds))
        .filter_map(|m| {
            let x = m.center[0] as f32;
            let y = m.center[1] as f32;
            if x.is_finite() && y.is_finite() {
                Some([x, y])
            } else {
                None
            }
        })
        .collect()
}

fn detect_rings_with_mapper_and_seeds(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> DetectionResult {
    stages::run(gray, config, mapper, seed_centers_image, seed_cfg, None).0
}

/// Run the full ring detection pipeline.
pub fn detect_rings(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    detect_rings_with_mapper_and_seeds(
        gray,
        config,
        config_mapper(config),
        &[],
        &SeedProposalParams::default(),
    )
}

/// Run the full ring detection pipeline with an optional custom pixel mapper.
///
/// This allows users to plug in camera/distortion models via a lightweight trait adapter.
/// When a mapper is provided, detection runs in two passes:
/// raw pass-1, then mapper-aware pass-2 with pass-1 seed injection.
pub fn detect_rings_with_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    if let Some(mapper) = mapper {
        detect_rings_two_pass_with_mapper(gray, config, mapper, &TwoPassParams::default())
    } else {
        detect_rings_with_mapper_and_seeds(gray, config, None, &[], &SeedProposalParams::default())
    }
}

/// Run two-pass detection:
/// - pass-1 in raw image space,
/// - pass-2 with mapper and pass-1 centers injected as proposal seeds.
///
/// Returned detections stay in pass-2 mapper working coordinates. Any retained
/// pass-1 fallback markers are remapped to the same working frame.
pub fn detect_rings_two_pass_with_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
    params: &TwoPassParams,
) -> DetectionResult {
    let pass1 = detect_rings_with_mapper_and_seeds(gray, config, None, &[], &params.seed);
    let seed_centers_image = collect_seed_centers_image(&pass1, &params.seed);

    let mut pass2 = detect_rings_with_mapper_and_seeds(
        gray,
        config,
        Some(mapper),
        &seed_centers_image,
        &params.seed,
    );
    if pass2.detected_markers.is_empty() && !seed_centers_image.is_empty() {
        tracing::info!("seeded pass-2 returned no detections; retrying pass-2 without seeds");
        pass2 = detect_rings_with_mapper_and_seeds(gray, config, Some(mapper), &[], &params.seed);
    }
    pass2.detected_markers = merge_two_pass_markers(
        &pass1.detected_markers,
        pass2.detected_markers,
        params.keep_pass1_markers,
        config.dedup_radius,
        Some(mapper),
    );
    pass2
}

/// Run the full ring detection pipeline and collect a versioned debug dump.
///
/// Debug collection currently uses single-pass execution.
pub fn detect_rings_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
) -> (DetectionResult, dbg::DebugDumpV1) {
    detect_rings_with_debug_and_mapper(gray, config, debug_cfg, config_mapper(config))
}

/// Run the full ring detection pipeline with debug collection and optional custom mapper.
///
/// Debug collection currently uses single-pass execution.
pub fn detect_rings_with_debug_and_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (DetectionResult, dbg::DebugDumpV1) {
    let (result, dump) = stages::run(
        gray,
        config,
        mapper,
        &[],
        &SeedProposalParams::default(),
        Some(debug_cfg),
    );
    (
        result,
        dump.expect("debug dump present when debug_cfg is provided"),
    )
}

pub(super) fn warn_center_correction_without_intrinsics(config: &DetectConfig, has_mapper: bool) {
    if config.circle_refinement == CircleRefinementMethod::None || has_mapper {
        return;
    }

    tracing::warn!(
        "center correction is running without camera intrinsics/undistortion; \
         lens distortion can still bias corrected centers"
    );
}

fn dedup_with_debug(
    markers: Vec<DetectedMarker>,
    cand_idx: Vec<usize>,
    radius: f64,
) -> (Vec<DetectedMarker>, Vec<usize>, dbg::DedupDebugV1) {
    dedup_with_debug_impl(markers, cand_idx, radius)
}

fn global_filter_with_debug(
    markers: &[DetectedMarker],
    cand_idx: &[usize],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
    dbg::RansacDebugV1,
) {
    global_filter_with_debug_impl(markers, cand_idx, config, board)
}

fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> (Vec<DetectedMarker>, dbg::RefineDebugV1) {
    refine_h::refine_with_homography_with_debug(gray, markers, h, config, board, mapper)
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    dedup_by_id_impl(markers);
}

/// Apply global homography RANSAC filter.
///
/// Returns (filtered markers, RANSAC result, stats).
fn global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    global_filter_impl(markers, config, board)
}

/// Refine marker centers using H: project board coords through H as priors,
/// re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    let (refined, _debug) =
        refine_h::refine_with_homography_with_debug(gray, markers, h, config, board, mapper);
    refined
}

/// Try to complete missing IDs using a fitted homography.
///
/// This is intentionally conservative: it only runs when H exists and rejects
/// any fit that deviates from the H-projected center by more than a tight gate.
fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<DetectedMarker>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
    debug: CompletionDebugOptions,
) -> (CompletionStats, Option<Vec<CompletionAttemptRecord>>) {
    completion::complete_with_h(gray, h, markers, config, board, mapper, debug)
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    homography_utils::matrix3_to_array(m)
}

fn mean_reproj_error_px(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    board: &BoardLayout,
) -> f64 {
    homography_utils::mean_reproj_error_px(h, markers, board)
}

fn compute_h_stats(
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    thresh_px: f64,
    board: &BoardLayout,
) -> Option<RansacStats> {
    homography_utils::compute_h_stats(h, markers, thresh_px, board)
}

fn refit_homography_matrix(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> Option<(nalgebra::Matrix3<f64>, RansacStats)> {
    homography_utils::refit_homography_matrix(markers, config, board)
}

/// Remove duplicate detections: keep the highest-confidence marker within dedup_radius.
fn dedup_markers(markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    dedup_markers_impl(markers, radius)
}

pub(super) fn apply_projective_centers(markers: &mut [DetectedMarker], config: &DetectConfig) {
    use crate::conic::Conic2D;
    use crate::projective_center::{
        ring_center_projective_with_debug, RingCenterProjectiveOptions,
    };

    for m in markers.iter_mut() {
        m.center_projective = None;
        m.vanishing_line = None;
        m.center_projective_residual = None;
    }

    if !config.projective_center.enable {
        return;
    }

    let expected_ratio = if config.projective_center.use_expected_ratio {
        Some(config.marker_spec.r_inner_expected as f64)
    } else {
        None
    };
    let opts = RingCenterProjectiveOptions {
        expected_ratio,
        ratio_penalty_weight: config.projective_center.ratio_penalty_weight,
        ..Default::default()
    };

    let mut n_missing_conics = 0usize;
    let mut n_solver_failed = 0usize;
    let mut n_rejected_shift = 0usize;
    let mut n_rejected_residual = 0usize;
    let mut n_rejected_eig_sep = 0usize;
    let mut n_applied = 0usize;

    for m in markers.iter_mut() {
        let (Some(inner), Some(outer)) = (m.ellipse_inner.as_ref(), m.ellipse_outer.as_ref())
        else {
            n_missing_conics += 1;
            continue;
        };

        let center_before = m.center;
        let q_inner = Conic2D::from_ellipse_params(inner).mat;
        let q_outer = Conic2D::from_ellipse_params(outer).mat;
        let Ok(res) = ring_center_projective_with_debug(&q_inner, &q_outer, opts) else {
            n_solver_failed += 1;
            continue;
        };

        if let Some(max_residual) = config.projective_center.max_selected_residual {
            if !res.debug.selected_residual.is_finite()
                || res.debug.selected_residual > max_residual
            {
                n_rejected_residual += 1;
                continue;
            }
        }

        if let Some(min_sep) = config.projective_center.min_eig_separation {
            if !res.debug.selected_eig_separation.is_finite()
                || res.debug.selected_eig_separation < min_sep
            {
                n_rejected_eig_sep += 1;
                continue;
            }
        }

        let center_projective = [res.center.x, res.center.y];
        let dx = center_projective[0] - center_before[0];
        let dy = center_projective[1] - center_before[1];
        let center_shift = (dx * dx + dy * dy).sqrt();
        if let Some(max_shift_px) = config.projective_center.max_center_shift_px {
            if !center_shift.is_finite() || center_shift > max_shift_px {
                n_rejected_shift += 1;
                continue;
            }
        }

        m.center = center_projective;
        m.center_projective = Some(center_projective);
        m.vanishing_line = Some([
            res.vanishing_line[0],
            res.vanishing_line[1],
            res.vanishing_line[2],
        ]);
        m.center_projective_residual = Some(res.debug.selected_residual);
        n_applied += 1;
    }

    tracing::debug!(
        applied = n_applied,
        missing_conics = n_missing_conics,
        solver_failed = n_solver_failed,
        rejected_shift = n_rejected_shift,
        rejected_residual = n_rejected_residual,
        rejected_eig_sep = n_rejected_eig_sep,
        "projective-center application summary"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PixelMapper;
    use crate::conic::ConicCoeffs;
    use crate::FitMetrics;
    use image::GrayImage;
    use image::Luma;
    use nalgebra::Matrix3;
    use nalgebra::Vector3;

    #[test]
    fn debug_dump_does_not_panic_when_stages_skipped() {
        let img = GrayImage::new(64, 64);
        let cfg = DetectConfig {
            use_global_filter: false,
            refine_with_h: false,
            ..DetectConfig::default()
        };

        let dbg_cfg = DebugCollectConfig {
            image_path: Some("dummy.png".to_string()),
            marker_diameter_px: 32.0,
            max_candidates: 10,
            store_points: false,
        };

        let (res, dump) = detect_rings_with_debug(&img, &cfg, &dbg_cfg);
        assert_eq!(res.image_size, [64, 64]);
        assert_eq!(dump.schema_version, crate::debug_dump::DEBUG_SCHEMA_V1);
        assert_eq!(dump.stages.stage0_proposals.n_total, 0);
        assert!(!dump.stages.stage3_ransac.enabled);
    }

    #[test]
    fn seeded_proposals_include_injected_centers() {
        let img = GrayImage::new(32, 32);
        let seeds = vec![[10.0f32, 12.0f32], [20.0f32, 22.0f32]];
        let props = find_proposals_with_seeds(
            &img,
            &ProposalConfig::default(),
            &seeds,
            &SeedProposalParams::default(),
        );
        assert_eq!(props.len(), seeds.len());
    }

    #[test]
    fn map_marker_image_to_working_maps_primary_and_projective_centers() {
        struct ShiftMapper;
        impl PixelMapper for ShiftMapper {
            fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some([image_xy[0] - 5.0, image_xy[1] + 2.0])
            }
            fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some([working_xy[0] + 5.0, working_xy[1] - 2.0])
            }
        }

        let marker = DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center: [1.0, 2.0],
            center_projective: Some([3.0, 4.0]),
            vanishing_line: None,
            center_projective_residual: None,
            ellipse_outer: None,
            ellipse_inner: None,
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        };

        let mapped = map_marker_image_to_working(&marker, &ShiftMapper)
            .expect("center mapping should succeed");
        assert_eq!(mapped.center, [-4.0, 4.0]);
        assert_eq!(mapped.center_projective, Some([-2.0, 6.0]));
        assert!(mapped.ellipse_outer.is_none());
        assert!(mapped.ellipse_inner.is_none());
        assert!(mapped.vanishing_line.is_none());
        assert!(mapped.edge_points_outer.is_none());
        assert!(mapped.edge_points_inner.is_none());
    }

    #[test]
    fn detect_accepts_custom_pixel_mapper_adapter() {
        struct IdentityMapper;
        impl PixelMapper for IdentityMapper {
            fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some(image_xy)
            }
            fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
                Some(working_xy)
            }
        }

        let img = GrayImage::new(64, 64);
        let cfg = DetectConfig {
            use_global_filter: false,
            refine_with_h: false,
            ..DetectConfig::default()
        };
        let dbg_cfg = DebugCollectConfig {
            image_path: None,
            marker_diameter_px: 32.0,
            max_candidates: 10,
            store_points: false,
        };

        let mapper = IdentityMapper;
        let (res, _dump) = detect_rings_with_debug_and_mapper(
            &img,
            &cfg,
            &dbg_cfg,
            Some(&mapper as &dyn PixelMapper),
        );
        assert_eq!(res.image_size, [64, 64]);
    }

    #[test]
    fn completion_adds_marker_at_h_projected_center() {
        let w = 128u32;
        let h = 128u32;

        let cfg = DetectConfig::default();

        // Choose an ID that exists on the default board and project it to the
        // image center with an affine homography.
        let id = 0usize;
        let xy = cfg.board.xy_mm(id).expect("board has id=0");
        let tx = 64.0 - xy[0] as f64;
        let ty = 64.0 - xy[1] as f64;
        let h_matrix = Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0);

        // Render a simple concentric ring at the projected center (no code band).
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 64.0;
                let dy = y as f32 - 64.0;
                let r = (dx * dx + dy * dy).sqrt();

                let bg = 0.85f32;
                let dark = 0.12f32;
                let v = if (12.0..=18.0).contains(&r) { dark } else { bg };
                img.put_pixel(x, y, Luma([(v * 255.0).round() as u8]));
            }
        }

        let mut cfg = DetectConfig {
            refine_with_h: false,
            // Make ellipse validation compatible with our synthetic ring radius.
            min_semi_axis: 6.0,
            max_semi_axis: 30.0,
            ..DetectConfig::default()
        };

        // Completion should attempt only this ID and should not be blocked by decoding.
        cfg.completion.enable = true;
        cfg.completion.max_attempts = Some(1);
        cfg.completion.roi_radius_px = 24.0;
        cfg.completion.reproj_gate_px = 3.0;
        cfg.completion.min_arc_coverage = 0.6;
        cfg.completion.min_fit_confidence = 0.6;
        cfg.decode.min_decode_confidence = 1.0; // force decode rejection (avoid mismatch gate)

        let mut markers: Vec<DetectedMarker> = Vec::new();
        let (stats, _attempts) = complete_with_h(
            &img,
            &h_matrix,
            &mut markers,
            &cfg,
            &cfg.board,
            None,
            CompletionDebugOptions::default(),
        );
        assert_eq!(stats.n_added, 1, "expected one completion addition");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].id, Some(id));
    }

    #[test]
    fn apply_projective_centers_promotes_center_field() {
        fn circle_conic(radius: f64) -> Matrix3<f64> {
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -(radius * radius))
        }

        fn project_conic(q_plane: &Matrix3<f64>, h: &Matrix3<f64>) -> Matrix3<f64> {
            let h_inv = h.try_inverse().expect("invertible homography");
            h_inv.transpose() * q_plane * h_inv
        }

        fn conic_matrix_to_params(q: Matrix3<f64>) -> crate::EllipseParams {
            let q_sym = 0.5 * (q + q.transpose());
            let coeffs = ConicCoeffs([
                q_sym[(0, 0)],
                2.0 * q_sym[(0, 1)],
                q_sym[(1, 1)],
                2.0 * q_sym[(0, 2)],
                2.0 * q_sym[(1, 2)],
                q_sym[(2, 2)],
            ]);
            let e = coeffs.to_ellipse().expect("projected circle is an ellipse");
            crate::EllipseParams::from(e)
        }

        let h = Matrix3::new(1.12, 0.21, 321.0, -0.17, 0.94, 245.0, 8.0e-4, -6.0e-4, 1.0);
        let q_inner = project_conic(&circle_conic(4.0), &h);
        let q_outer = project_conic(&circle_conic(7.0), &h);
        let inner = conic_matrix_to_params(q_inner);
        let outer = conic_matrix_to_params(q_outer);

        let center_before = outer.center_xy;
        let mut markers = vec![DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center: center_before,
            center_projective: None,
            vanishing_line: None,
            center_projective_residual: None,
            ellipse_outer: Some(outer),
            ellipse_inner: Some(inner),
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        }];

        let cfg = DetectConfig::default();
        apply_projective_centers(&mut markers, &cfg);
        let m = &markers[0];

        let gt_h = h * Vector3::new(0.0, 0.0, 1.0);
        let gt_center = [gt_h[0] / gt_h[2], gt_h[1] / gt_h[2]];
        let err =
            ((m.center[0] - gt_center[0]).powi(2) + (m.center[1] - gt_center[1]).powi(2)).sqrt();
        let shift = ((m.center[0] - center_before[0]).powi(2)
            + (m.center[1] - center_before[1]).powi(2))
        .sqrt();

        assert!(
            m.center_projective.is_some(),
            "projective center should be present"
        );
        assert!(
            err < 1e-6,
            "expected near-exact projective center, err={}",
            err
        );
        assert!(
            shift > 1e-3,
            "primary center should be updated from ellipse center, shift={}",
            shift
        );
    }

    #[test]
    fn apply_projective_centers_falls_back_when_shift_gate_rejects() {
        fn circle_conic(radius: f64) -> Matrix3<f64> {
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -(radius * radius))
        }

        fn project_conic(q_plane: &Matrix3<f64>, h: &Matrix3<f64>) -> Matrix3<f64> {
            let h_inv = h.try_inverse().expect("invertible homography");
            h_inv.transpose() * q_plane * h_inv
        }

        fn conic_matrix_to_params(q: Matrix3<f64>) -> crate::EllipseParams {
            let q_sym = 0.5 * (q + q.transpose());
            let coeffs = ConicCoeffs([
                q_sym[(0, 0)],
                2.0 * q_sym[(0, 1)],
                q_sym[(1, 1)],
                2.0 * q_sym[(0, 2)],
                2.0 * q_sym[(1, 2)],
                q_sym[(2, 2)],
            ]);
            let e = coeffs.to_ellipse().expect("projected circle is an ellipse");
            crate::EllipseParams::from(e)
        }

        let h = Matrix3::new(1.12, 0.21, 321.0, -0.17, 0.94, 245.0, 8.0e-4, -6.0e-4, 1.0);
        let q_inner = project_conic(&circle_conic(4.0), &h);
        let q_outer = project_conic(&circle_conic(7.0), &h);
        let inner = conic_matrix_to_params(q_inner);
        let outer = conic_matrix_to_params(q_outer);

        let center_before = outer.center_xy;
        let mut markers = vec![DetectedMarker {
            id: Some(0),
            confidence: 1.0,
            center: center_before,
            center_projective: None,
            vanishing_line: None,
            center_projective_residual: None,
            ellipse_outer: Some(outer),
            ellipse_inner: Some(inner),
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        }];

        let mut cfg = DetectConfig::default();
        cfg.projective_center.max_center_shift_px = Some(1e-6);
        apply_projective_centers(&mut markers, &cfg);
        let m = &markers[0];

        assert_eq!(m.center, center_before);
        assert!(m.center_projective.is_none());
        assert!(m.vanishing_line.is_none());
        assert!(m.center_projective_residual.is_none());
    }
}
