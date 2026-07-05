//! High-level detection API.
//!
//! [`Detector`] is the primary entry point for detecting ring markers.
//! It wraps a [`DetectConfig`] and provides convenience methods for
//! common detection scenarios (config-driven detect and external mapper).

use image::GrayImage;
#[cfg(feature = "std")]
use std::path::Path;

use crate::detector::config::{ScaleTiers, derive_proposal_config};
use crate::detector::{DetectConfig, MarkerScalePrior};
use crate::pipeline;
use crate::pipeline::DetectionDiagnostics;
use crate::pixelmap::PixelMapper;
use crate::proposal::{find_ellipse_centers, find_ellipse_centers_with_heatmap};
use crate::target::TargetLayout;
#[cfg(feature = "std")]
use crate::target::TargetLoadError;
use crate::{DetectionResult, Proposal, ProposalResult};

/// Detection-time failures reported by [`Detector`] methods.
///
/// All built-in lattice × coding combinations detect end-to-end, so detection
/// itself is infallible; the only failure a `Detector` reports is the opt-in
/// [`IncompleteBoard`](Self::IncompleteBoard) gate. This `#[non_exhaustive]`
/// enum reserves room for future failure modes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DetectError {
    /// The configured target layout is not supported by the detection
    /// pipeline in this release.
    #[non_exhaustive]
    UnsupportedTarget {
        /// Name of the configured target layout.
        target_name: String,
        /// Lattice kind of the unsupported combination (`"hex"` or `"rect"`).
        lattice: &'static str,
        /// Coding kind of the unsupported combination (`"coded16"` or `"plain"`).
        coding: &'static str,
    },
    /// The board was not fully detected while
    /// [`DetectConfig::require_complete_board`](crate::DetectConfig::require_complete_board)
    /// was set. Returned only under the strict gate; otherwise incompleteness is
    /// reported non-fatally via
    /// [`DetectionResult::board_complete`](crate::DetectionResult::board_complete).
    IncompleteBoard {
        /// Number of target cells that were labeled.
        found: usize,
        /// Total number of cells the target defines.
        expected: usize,
    },
}

impl std::fmt::Display for DetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTarget {
                target_name,
                lattice,
                coding,
            } => write!(
                f,
                "target '{target_name}' ({lattice} lattice, {coding} coding) is not supported by \
                 the detection pipeline in this release"
            ),
            Self::IncompleteBoard { found, expected } => write!(
                f,
                "incomplete board: {found} of {expected} target cells detected \
                 (require_complete_board is set)"
            ),
        }
    }
}

impl std::error::Error for DetectError {}

/// Primary detection interface.
///
/// Encapsulates target layout and detection configuration.
/// Create once, detect on many images.
/// Target ownership is single-source: it is stored inside `DetectConfig`.
///
/// # Examples
///
/// ```no_run
/// use ringgrid::{Detector, TargetLayout};
/// use image::GrayImage;
/// use std::path::Path;
///
/// let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
/// let detector = Detector::new(target);
/// let image = GrayImage::new(640, 480);
/// let result = detector.detect(&image).unwrap();
/// println!("Found {} markers", result.detected_markers.len());
/// ```
pub struct Detector {
    config: DetectConfig,
}

impl Detector {
    /// Create a detector with a target layout and default
    /// marker-scale search prior.
    pub fn new(target: impl Into<TargetLayout>) -> Self {
        Self {
            config: DetectConfig::from_target(target),
        }
    }

    /// Create a detector with an explicit marker-scale prior.
    pub fn with_marker_scale(
        target: impl Into<TargetLayout>,
        marker_scale: MarkerScalePrior,
    ) -> Self {
        Self {
            config: DetectConfig::from_target_and_scale_prior(target, marker_scale),
        }
    }

    /// Create a detector with a fixed marker-diameter hint.
    pub fn with_marker_diameter_hint(
        target: impl Into<TargetLayout>,
        marker_diameter_px: f32,
    ) -> Self {
        Self::with_marker_scale(
            target,
            MarkerScalePrior::from_nominal_diameter_px(marker_diameter_px),
        )
    }

    /// Load target JSON (`v5` or legacy `v4`) and create a detector in one
    /// step using default marker-scale search prior.
    #[cfg(feature = "std")]
    pub fn from_target_json_file(path: &Path) -> Result<Self, TargetLoadError> {
        Ok(Self::new(TargetLayout::from_json_file(path)?))
    }

    /// Load target JSON and create a detector with explicit marker-scale prior.
    #[cfg(feature = "std")]
    pub fn from_target_json_file_with_scale(
        path: &Path,
        marker_scale: MarkerScalePrior,
    ) -> Result<Self, TargetLoadError> {
        Ok(Self::with_marker_scale(
            TargetLayout::from_json_file(path)?,
            marker_scale,
        ))
    }

    /// Load target JSON and create a detector with fixed marker-diameter hint.
    #[cfg(feature = "std")]
    pub fn from_target_json_file_with_marker_diameter(
        path: &Path,
        marker_diameter_px: f32,
    ) -> Result<Self, TargetLoadError> {
        Ok(Self::with_marker_diameter_hint(
            TargetLayout::from_json_file(path)?,
            marker_diameter_px,
        ))
    }

    /// Create with full config control.
    pub fn with_config(config: DetectConfig) -> Self {
        Self { config }
    }

    /// Access the current configuration.
    pub fn config(&self) -> &DetectConfig {
        &self.config
    }

    /// Mutable access to configuration for post-construction tuning.
    pub fn config_mut(&mut self) -> &mut DetectConfig {
        &mut self.config
    }

    /// Run the single-pass / self-undistort pipeline and return the internal
    /// rich result.
    fn run_detect(&self, image: &GrayImage) -> pipeline::PipelineResult {
        if self.config.self_undistort.enable {
            pipeline::detect_with_self_undistort(image, &self.config)
        } else {
            pipeline::detect_single_pass(image, &self.config)
        }
    }

    /// Convert a finished pipeline result into the public slim result, filling
    /// the board-completeness signal and enforcing the strict gate.
    fn finish(&self, pipeline: pipeline::PipelineResult) -> Result<DetectionResult, DetectError> {
        Ok(self.finish_with_diagnostics(pipeline)?.0)
    }

    /// [`finish`](Self::finish) that also returns the diagnostics half.
    fn finish_with_diagnostics(
        &self,
        pipeline: pipeline::PipelineResult,
    ) -> Result<(DetectionResult, DetectionDiagnostics), DetectError> {
        let (mut result, diagnostics) = pipeline.split();
        self.apply_completeness(&mut result)?;
        Ok((result, diagnostics))
    }

    /// Compute [`DetectionResult::board_complete`] and enforce
    /// [`DetectConfig::require_complete_board`].
    ///
    /// Completeness is defined only when grid assignment ran
    /// (`board_frame.is_some()`); otherwise the signal stays `None`. Under the
    /// strict gate, an incomplete board becomes [`DetectError::IncompleteBoard`]
    /// rather than a low-`board_complete` result.
    fn apply_completeness(&self, result: &mut DetectionResult) -> Result<(), DetectError> {
        if result.board_frame.is_none() {
            return Ok(());
        }
        let expected = self.config.target.n_cells();
        let found = result
            .detected_markers
            .iter()
            .filter(|m| m.grid_coord.is_some())
            .count();
        let complete = found >= expected;
        result.board_complete = Some(complete);
        if self.config.require_complete_board && !complete {
            return Err(DetectError::IncompleteBoard { found, expected });
        }
        Ok(())
    }

    /// Detect markers in a grayscale image.
    ///
    /// When `config.self_undistort.enable` is `false`, runs single-pass
    /// detection in image coordinates.
    ///
    /// When `config.self_undistort.enable` is `true`, runs baseline detection,
    /// estimates a self-undistort model, and optionally runs a second seeded
    /// pass with the estimated mapper.
    ///
    /// Returns the slim [`DetectionResult`]. Use
    /// [`detect_with_diagnostics`](Self::detect_with_diagnostics) to also obtain
    /// per-marker algorithm internals and RANSAC statistics.
    ///
    /// # Errors
    ///
    /// Currently infallible for all built-in targets; the `Result` signature
    /// reserves room for future failure modes (see [`DetectError`]).
    pub fn detect(&self, image: &GrayImage) -> Result<DetectionResult, DetectError> {
        self.finish(self.run_detect(image))
    }

    /// Detect markers and also return opt-in detection diagnostics.
    ///
    /// Behaves exactly like [`detect`](Self::detect) — the returned
    /// [`DetectionResult`] is identical — but additionally yields a
    /// [`DetectionDiagnostics`] carrying per-marker fit/decode metrics, raw edge
    /// sample points, stage provenance, and homography RANSAC statistics.
    ///
    /// `diagnostics.markers` is positionally aligned 1:1 with
    /// `result.detected_markers`.
    pub fn detect_with_diagnostics(
        &self,
        image: &GrayImage,
    ) -> Result<(DetectionResult, DetectionDiagnostics), DetectError> {
        self.finish_with_diagnostics(self.run_detect(image))
    }

    /// Generate pass-1 center proposals in image coordinates.
    ///
    /// This exposes the detector-backed proposal seeds used by single-pass
    /// detection after spacing-aware post-NMS suppression.
    pub fn propose(&self, image: &GrayImage) -> Vec<Proposal> {
        pipeline::proposal_seeds_for_config(image, &self.config)
    }

    /// Generate pass-1 proposals with the vote heatmap in image coordinates.
    ///
    /// The returned heatmap is the post-Gaussian-smoothed vote map used
    /// for thresholding and non-maximum suppression in the proposal stage.
    /// When proposal downscaling is enabled, the heatmap is resampled back into
    /// the original image frame together with the returned proposals.
    pub fn propose_with_heatmap(&self, image: &GrayImage) -> ProposalResult {
        pipeline::proposal_result_for_config(image, &self.config)
    }

    /// Detect markers with robust adaptive scale selection.
    ///
    /// Adaptive mode evaluates several scale candidates and returns the
    /// highest-scoring result. Candidates include:
    /// - probe-selected multi-scale tiers,
    /// - fixed two-tier / four-tier multi-scale presets,
    /// - curated single-pass scale priors.
    ///
    /// Candidate ranking is deterministic and prioritizes:
    /// 1. mapped markers,
    /// 2. homography availability / inlier support,
    /// 3. decoded markers,
    /// 4. geometric quality tie-breakers.
    ///
    /// No manual scale configuration is required. Use
    /// [`detect_adaptive_with_hint`](Self::detect_adaptive_with_hint) when an
    /// approximate marker diameter is known.
    ///
    /// [`detect_multiscale`]: Self::detect_multiscale
    pub fn detect_adaptive(&self, image: &GrayImage) -> Result<DetectionResult, DetectError> {
        self.finish(pipeline::detect_adaptive(image, &self.config))
    }

    /// Return the scale tiers that adaptive detection would use for this image.
    ///
    /// This helper is useful for debugging and reproducibility:
    /// - inspect auto-selected tiers before running detection
    /// - persist exact tiers and replay with [`detect_multiscale`](Self::detect_multiscale)
    ///
    /// When `nominal_diameter_px` is `Some`, returns a hint-centered two-tier
    /// bracket. When `None`, returns probe-selected tiers (or
    /// [`ScaleTiers::four_tier_wide`] fallback if probing fails).
    pub fn adaptive_tiers(
        &self,
        image: &GrayImage,
        nominal_diameter_px: Option<f32>,
    ) -> ScaleTiers {
        pipeline::select_adaptive_tiers(image, nominal_diameter_px)
    }

    /// Adaptive detection with an optional nominal-diameter hint.
    ///
    /// When `nominal_diameter_px` is `Some`, the hint-centered two-tier
    /// bracket is included as the primary candidate while robust fallback
    /// candidates remain enabled. When `None`, behaves identically to
    /// [`detect_adaptive`](Self::detect_adaptive).
    pub fn detect_adaptive_with_hint(
        &self,
        image: &GrayImage,
        nominal_diameter_px: Option<f32>,
    ) -> Result<DetectionResult, DetectError> {
        self.finish(pipeline::detect_adaptive_with_hint(
            image,
            &self.config,
            nominal_diameter_px,
        ))
    }

    /// Detect markers using an explicit set of scale tiers.
    ///
    /// Runs one detection pass per tier (fit/decode + projective centers + ID
    /// correction), merges results with size-consistency-aware dedup, then runs
    /// global filter, completion, and final H refit once on the merged pool.
    ///
    /// Use [`ScaleTiers`] constructors to build the tier set:
    /// - [`ScaleTiers::four_tier_wide`] — 8–220 px
    /// - [`ScaleTiers::two_tier_standard`] — 14–100 px
    /// - [`ScaleTiers::single`] — single-pass, no merge overhead
    pub fn detect_multiscale(
        &self,
        image: &GrayImage,
        tiers: &ScaleTiers,
    ) -> Result<DetectionResult, DetectError> {
        self.finish(pipeline::detect_multiscale(image, &self.config, tiers))
    }

    /// Detect with a custom pixel mapper (two-pass pipeline).
    ///
    /// Pass-1 runs without mapper for seed generation, pass-2 runs with mapper.
    /// Marker centers in the returned result are always image-space.
    /// Mapper-frame centers are exposed via [`DetectedMarker::center_mapped`](crate::DetectedMarker::center_mapped).
    ///
    /// This method always uses the provided mapper and does not run
    /// self-undistort estimation from `config.self_undistort`.
    ///
    /// Returns the slim [`DetectionResult`]. Use
    /// [`detect_with_mapper_diagnostics`](Self::detect_with_mapper_diagnostics)
    /// to also obtain detection diagnostics.
    pub fn detect_with_mapper(
        &self,
        image: &GrayImage,
        mapper: &dyn PixelMapper,
    ) -> Result<DetectionResult, DetectError> {
        self.finish(pipeline::detect_with_mapper(image, &self.config, mapper))
    }

    /// Detect with a custom pixel mapper and also return detection diagnostics.
    ///
    /// Mapper-variant counterpart of
    /// [`detect_with_diagnostics`](Self::detect_with_diagnostics): the returned
    /// [`DetectionResult`] is identical to [`detect_with_mapper`](Self::detect_with_mapper),
    /// and the [`DetectionDiagnostics`] carries per-marker internals and
    /// homography RANSAC statistics.
    ///
    /// `diagnostics.markers` is positionally aligned 1:1 with
    /// `result.detected_markers`.
    pub fn detect_with_mapper_diagnostics(
        &self,
        image: &GrayImage,
        mapper: &dyn PixelMapper,
    ) -> Result<(DetectionResult, DetectionDiagnostics), DetectError> {
        self.finish_with_diagnostics(pipeline::detect_with_mapper(image, &self.config, mapper))
    }
}

/// Generate pass-1 center proposals using target geometry and an explicit
/// marker-scale prior.
pub fn propose_with_marker_scale(
    image: &GrayImage,
    target: &TargetLayout,
    marker_scale: MarkerScalePrior,
) -> Vec<Proposal> {
    let config = derive_proposal_config(target, marker_scale, &crate::ProposalConfig::default());
    find_ellipse_centers(image, &config)
}

/// Generate pass-1 proposals with heatmap using target geometry and an
/// explicit marker-scale prior.
pub fn propose_with_heatmap_and_marker_scale(
    image: &GrayImage,
    target: &TargetLayout,
    marker_scale: MarkerScalePrior,
) -> ProposalResult {
    let config = derive_proposal_config(target, marker_scale, &crate::ProposalConfig::default());
    find_ellipse_centers_with_heatmap(image, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixelmap::PixelMapper;

    struct IdentityMapper;

    impl PixelMapper for IdentityMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some(image_xy)
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some(working_xy)
        }
    }

    #[test]
    fn detector_basic_detect() {
        let det = Detector::with_config(DetectConfig::from_target(TargetLayout::default_hex()));
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img).expect("hex coded target is supported");
        assert!(result.detected_markers.is_empty());
        assert_eq!(result.center_frame, crate::DetectionFrame::Image);
        assert_eq!(result.homography_frame, crate::DetectionFrame::Image);
        assert!(result.self_undistort.is_none());
    }

    #[test]
    fn detector_detect_honors_self_undistort_enable() {
        let mut cfg = DetectConfig::from_target(TargetLayout::default_hex());
        cfg.self_undistort.enable = true;
        cfg.self_undistort.min_markers = 0;
        let det = Detector::with_config(cfg);
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img).expect("hex coded target is supported");
        assert!(result.self_undistort.is_some());
    }

    #[test]
    fn detector_mapper_ignores_self_undistort_config() {
        let mut cfg = DetectConfig::from_target(TargetLayout::default_hex());
        cfg.self_undistort.enable = true;
        cfg.self_undistort.min_markers = 0;
        let det = Detector::with_config(cfg);
        let img = GrayImage::new(200, 200);
        let mapper = IdentityMapper;
        let result = det.detect_with_mapper(&img, &mapper).expect("supported");
        assert_eq!(result.center_frame, crate::DetectionFrame::Image);
        assert_eq!(result.homography_frame, crate::DetectionFrame::Working);
        assert!(result.self_undistort.is_none());
    }

    #[test]
    fn detector_config_mut() {
        let mut det = Detector::with_config(DetectConfig::from_target(TargetLayout::default_hex()));
        det.config_mut().advanced.completion.enable = false;
        assert!(!det.config().advanced.completion.enable);
    }

    #[test]
    fn detector_adaptive_tiers_fallback_matches_four_tier_wide_on_blank_image() {
        let det = Detector::with_config(DetectConfig::from_target(TargetLayout::default_hex()));
        let img = GrayImage::new(200, 200);
        let tiers = det.adaptive_tiers(&img, None);
        let expected = ScaleTiers::four_tier_wide();
        assert_eq!(tiers.tiers().len(), expected.tiers().len());
        for (got, want) in tiers.tiers().iter().zip(expected.tiers().iter()) {
            assert_eq!(
                got.prior.diameter_min_px.to_bits(),
                want.prior.diameter_min_px.to_bits()
            );
            assert_eq!(
                got.prior.diameter_max_px.to_bits(),
                want.prior.diameter_max_px.to_bits()
            );
        }
    }

    #[test]
    fn detector_adaptive_tiers_with_hint_builds_two_tier_bracket() {
        let det = Detector::with_config(DetectConfig::from_target(TargetLayout::default_hex()));
        let img = GrayImage::new(200, 200);
        let tiers = det.adaptive_tiers(&img, Some(32.0));
        assert_eq!(tiers.tiers().len(), 2);
        assert!((tiers.tiers()[0].prior.diameter_min_px - 16.0).abs() < 1e-4);
        assert!((tiers.tiers()[0].prior.diameter_max_px - 33.6).abs() < 1e-4);
        assert!((tiers.tiers()[1].prior.diameter_min_px - 31.92).abs() < 1e-4);
        assert!((tiers.tiers()[1].prior.diameter_max_px - 48.0).abs() < 1e-4);
    }

    fn draw_ring_image(
        w: u32,
        h: u32,
        center: [f32; 2],
        outer_radius: f32,
        inner_radius: f32,
    ) -> GrayImage {
        crate::test_utils::draw_ring_image(w, h, center, outer_radius, inner_radius, 30, 220)
    }

    #[test]
    fn detector_propose_is_deterministic() {
        let cfg = DetectConfig::from_target(TargetLayout::default_hex());
        let det = Detector::with_config(cfg);
        let img = draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0);

        let p1 = det.propose(&img);
        let p2 = det.propose(&img);

        assert!(!p1.is_empty(), "expected at least one proposal");
        assert_eq!(p1.len(), p2.len(), "proposal counts should match");
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.score.to_bits(), b.score.to_bits());
        }
    }

    #[test]
    fn detect_with_diagnostics_result_matches_detect_and_is_aligned() {
        let cfg = DetectConfig::from_target(TargetLayout::default_hex());
        let detector = Detector::with_config(cfg);
        let img = draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0);

        let plain = detector.detect(&img).expect("supported");
        let (rich, diagnostics) = detector.detect_with_diagnostics(&img).expect("supported");

        // detect() must return the exact same slim result as detect_with_diagnostics().
        assert_eq!(plain.detected_markers.len(), rich.detected_markers.len());
        assert_eq!(plain.image_size, rich.image_size);
        assert_eq!(plain.center_frame, rich.center_frame);
        assert_eq!(plain.homography_frame, rich.homography_frame);
        for (a, b) in plain
            .detected_markers
            .iter()
            .zip(rich.detected_markers.iter())
        {
            assert_eq!(a.id, b.id);
            assert_eq!(a.center, b.center);
            assert_eq!(a.confidence.to_bits(), b.confidence.to_bits());
        }

        // Diagnostics markers are positionally aligned 1:1 with detected markers.
        assert_eq!(diagnostics.markers.len(), rich.detected_markers.len());
    }

    #[test]
    fn detect_with_diagnostics_populates_single_pass_timings() {
        let cfg = DetectConfig::from_target(TargetLayout::default_hex());
        let detector = Detector::with_config(cfg);
        let img = draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0);

        let (_result, diagnostics) = detector.detect_with_diagnostics(&img).expect("supported");
        let timings = diagnostics
            .timings
            .expect("single-pass detection surfaces stage timings");

        // Each stage is non-negative and the end-to-end total is at least as
        // large as any single stage it contains.
        assert!(timings.proposal_ms >= 0.0);
        assert!(timings.fit_decode_ms >= 0.0);
        assert!(timings.finalize_ms >= 0.0);
        assert!(timings.total_ms >= timings.proposal_ms);
        assert!(timings.total_ms >= timings.fit_decode_ms);
        assert!(timings.total_ms >= timings.finalize_ms);
    }

    #[test]
    fn detector_proposal_methods_consistent() {
        let cfg = DetectConfig::from_target(TargetLayout::default_hex());
        let detector = Detector::with_config(cfg);
        let img = draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0);

        let proposals = detector.propose(&img);
        let result = detector.propose_with_heatmap(&img);

        assert_eq!(proposals, result.proposals);
        assert_eq!(
            result.heatmap.len(),
            img.width() as usize * img.height() as usize
        );
    }

    #[test]
    fn size_aware_free_proposal_apis_match_detector_with_marker_hint() {
        let board = TargetLayout::default_hex();
        let detector = Detector::with_marker_diameter_hint(board.clone(), 32.0);
        let img = draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0);

        let free = propose_with_marker_scale(
            &img,
            &board,
            MarkerScalePrior::from_nominal_diameter_px(32.0),
        );
        let detector_out = detector.propose(&img);
        assert_eq!(free, detector_out);

        let free_diag = propose_with_heatmap_and_marker_scale(
            &img,
            &board,
            MarkerScalePrior::from_nominal_diameter_px(32.0),
        );
        let detector_diag = detector.propose_with_heatmap(&img);
        assert_eq!(free_diag.proposals, detector_diag.proposals);
        assert_eq!(free_diag.heatmap, detector_diag.heatmap);
    }

    #[test]
    fn plain_rect_target_detects_without_error() {
        // The transitional UnsupportedTarget gate is gone: every built-in
        // lattice × coding combination runs the pipeline end-to-end. An empty
        // image simply yields an empty result.
        let det = Detector::new(TargetLayout::rect_24x24());
        let img = GrayImage::new(64, 64);
        let result = det.detect(&img).expect("plain rect target must detect");
        assert!(result.detected_markers.is_empty());
        assert!(result.board_frame.is_none(), "nothing labeled ⇒ no frame");
        assert!(
            det.detect_multiscale(&img, &ScaleTiers::single(MarkerScalePrior::default()))
                .is_ok()
        );
    }

    #[test]
    fn detector_proposal_apis_honor_proposal_downscale() {
        let mut cfg = DetectConfig::from_target(TargetLayout::default_hex());
        cfg.advanced.proposal_downscale = crate::ProposalDownscale::Factor(4);
        let detector = Detector::with_config(cfg.clone());
        let img = draw_ring_image(101, 98, [50.0, 49.0], 20.0, 10.0);

        let proposals = detector.propose(&img);
        let result = detector.propose_with_heatmap(&img);
        let expected = pipeline::proposal_result_for_config(&img, &cfg);

        assert_eq!(proposals, expected.proposals);
        assert_eq!(result, expected);
        assert_eq!(result.image_size, [101, 98]);
        assert_eq!(result.heatmap.len(), 101 * 98);
    }
}
