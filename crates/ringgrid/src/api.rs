//! High-level detection API.
//!
//! [`Detector`] is the primary entry point for detecting ring markers.
//! It wraps a [`DetectConfig`] and provides convenience methods for
//! common detection scenarios (config-driven detect and external mapper).

use image::GrayImage;
use std::path::Path;

use crate::board_layout::{BoardLayout, BoardLayoutLoadError};
use crate::detector::config::{derive_proposal_config, ScaleTiers};
use crate::detector::{DetectConfig, MarkerScalePrior};
use crate::pipeline;
use crate::pixelmap::PixelMapper;
use crate::proposal::{find_ellipse_centers, find_ellipse_centers_with_heatmap};
use crate::{DetectionResult, Proposal, ProposalResult};

/// Primary detection interface.
///
/// Encapsulates board layout and detection configuration.
/// Create once, detect on many images.
/// Board ownership is single-source: it is stored inside `DetectConfig`.
///
/// # Examples
///
/// ```no_run
/// use ringgrid::{BoardLayout, Detector};
/// use image::GrayImage;
/// use std::path::Path;
///
/// let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
/// let detector = Detector::new(board);
/// let image = GrayImage::new(640, 480);
/// let result = detector.detect(&image);
/// println!("Found {} markers", result.detected_markers.len());
/// ```
pub struct Detector {
    config: DetectConfig,
}

impl Detector {
    /// Create a detector with a board layout and default
    /// marker-scale search prior.
    pub fn new(board: BoardLayout) -> Self {
        Self {
            config: DetectConfig::from_target(board),
        }
    }

    /// Create a detector with an explicit marker-scale prior.
    pub fn with_marker_scale(board: BoardLayout, marker_scale: MarkerScalePrior) -> Self {
        Self {
            config: DetectConfig::from_target_and_scale_prior(board, marker_scale),
        }
    }

    /// Create a detector with a fixed marker-diameter hint.
    pub fn with_marker_diameter_hint(board: BoardLayout, marker_diameter_px: f32) -> Self {
        Self::with_marker_scale(
            board,
            MarkerScalePrior::from_nominal_diameter_px(marker_diameter_px),
        )
    }

    /// Load target JSON and create a detector in one step using default
    /// marker-scale search prior.
    pub fn from_target_json_file(path: &Path) -> Result<Self, BoardLayoutLoadError> {
        Ok(Self::new(BoardLayout::from_json_file(path)?))
    }

    /// Load target JSON and create a detector with explicit marker-scale prior.
    pub fn from_target_json_file_with_scale(
        path: &Path,
        marker_scale: MarkerScalePrior,
    ) -> Result<Self, BoardLayoutLoadError> {
        Ok(Self::with_marker_scale(
            BoardLayout::from_json_file(path)?,
            marker_scale,
        ))
    }

    /// Load target JSON and create a detector with fixed marker-diameter hint.
    pub fn from_target_json_file_with_marker_diameter(
        path: &Path,
        marker_diameter_px: f32,
    ) -> Result<Self, BoardLayoutLoadError> {
        Ok(Self::with_marker_diameter_hint(
            BoardLayout::from_json_file(path)?,
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

    /// Detect markers in a grayscale image.
    ///
    /// When `config.self_undistort.enable` is `false`, runs single-pass
    /// detection in image coordinates.
    ///
    /// When `config.self_undistort.enable` is `true`, runs baseline detection,
    /// estimates a self-undistort model, and optionally runs a second seeded
    /// pass with the estimated mapper.
    pub fn detect(&self, image: &GrayImage) -> DetectionResult {
        if self.config.self_undistort.enable {
            pipeline::detect_with_self_undistort(image, &self.config)
        } else {
            pipeline::detect_single_pass(image, &self.config)
        }
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
    pub fn detect_adaptive(&self, image: &GrayImage) -> DetectionResult {
        pipeline::detect_adaptive(image, &self.config)
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
    ) -> DetectionResult {
        pipeline::detect_adaptive_with_hint(image, &self.config, nominal_diameter_px)
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
    pub fn detect_multiscale(&self, image: &GrayImage, tiers: &ScaleTiers) -> DetectionResult {
        pipeline::detect_multiscale(image, &self.config, tiers)
    }

    /// Detect with a custom pixel mapper (two-pass pipeline).
    ///
    /// Pass-1 runs without mapper for seed generation, pass-2 runs with mapper.
    /// Marker centers in the returned result are always image-space.
    /// Mapper-frame centers are exposed via `DetectedMarker.center_mapped`.
    ///
    /// This method always uses the provided mapper and does not run
    /// self-undistort estimation from `config.self_undistort`.
    pub fn detect_with_mapper(
        &self,
        image: &GrayImage,
        mapper: &dyn PixelMapper,
    ) -> DetectionResult {
        pipeline::detect_with_mapper(image, &self.config, mapper)
    }
}

/// Generate pass-1 center proposals using board geometry and an explicit
/// marker-scale prior.
pub fn propose_with_marker_scale(
    image: &GrayImage,
    board: &BoardLayout,
    marker_scale: MarkerScalePrior,
) -> Vec<Proposal> {
    let config = derive_proposal_config(board, marker_scale, &crate::ProposalConfig::default());
    find_ellipse_centers(image, &config)
}

/// Generate pass-1 proposals with heatmap using board geometry and an explicit
/// marker-scale prior.
pub fn propose_with_heatmap_and_marker_scale(
    image: &GrayImage,
    board: &BoardLayout,
    marker_scale: MarkerScalePrior,
) -> ProposalResult {
    let config = derive_proposal_config(board, marker_scale, &crate::ProposalConfig::default());
    find_ellipse_centers_with_heatmap(image, &config)
}

/// Generate pass-1 center proposals using board geometry and a fixed marker
/// diameter hint.
pub fn propose_with_marker_diameter(
    image: &GrayImage,
    board: &BoardLayout,
    marker_diameter_px: f32,
) -> Vec<Proposal> {
    propose_with_marker_scale(
        image,
        board,
        MarkerScalePrior::from_nominal_diameter_px(marker_diameter_px),
    )
}

/// Generate pass-1 proposals with heatmap using board geometry and a fixed
/// marker diameter hint.
pub fn propose_with_heatmap_and_marker_diameter(
    image: &GrayImage,
    board: &BoardLayout,
    marker_diameter_px: f32,
) -> ProposalResult {
    propose_with_heatmap_and_marker_scale(
        image,
        board,
        MarkerScalePrior::from_nominal_diameter_px(marker_diameter_px),
    )
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
        let det = Detector::with_config(DetectConfig::from_target(BoardLayout::default()));
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img);
        assert!(result.detected_markers.is_empty());
        assert_eq!(result.center_frame, crate::DetectionFrame::Image);
        assert_eq!(result.homography_frame, crate::DetectionFrame::Image);
        assert!(result.self_undistort.is_none());
    }

    #[test]
    fn detector_detect_honors_self_undistort_enable() {
        let mut cfg = DetectConfig::from_target(BoardLayout::default());
        cfg.self_undistort.enable = true;
        cfg.self_undistort.min_markers = 0;
        let det = Detector::with_config(cfg);
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img);
        assert!(result.self_undistort.is_some());
    }

    #[test]
    fn detector_mapper_ignores_self_undistort_config() {
        let mut cfg = DetectConfig::from_target(BoardLayout::default());
        cfg.self_undistort.enable = true;
        cfg.self_undistort.min_markers = 0;
        let det = Detector::with_config(cfg);
        let img = GrayImage::new(200, 200);
        let mapper = IdentityMapper;
        let result = det.detect_with_mapper(&img, &mapper);
        assert_eq!(result.center_frame, crate::DetectionFrame::Image);
        assert_eq!(result.homography_frame, crate::DetectionFrame::Working);
        assert!(result.self_undistort.is_none());
    }

    #[test]
    fn detector_config_mut() {
        let mut det = Detector::with_config(DetectConfig::from_target(BoardLayout::default()));
        det.config_mut().completion.enable = false;
        assert!(!det.config().completion.enable);
    }

    #[test]
    fn detector_adaptive_tiers_fallback_matches_four_tier_wide_on_blank_image() {
        let det = Detector::with_config(DetectConfig::from_target(BoardLayout::default()));
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
        let det = Detector::with_config(DetectConfig::from_target(BoardLayout::default()));
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
        let cfg = DetectConfig::from_target(BoardLayout::default());
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
    fn detector_proposal_methods_consistent() {
        let cfg = DetectConfig::from_target(BoardLayout::default());
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
        let board = BoardLayout::default();
        let detector = Detector::with_marker_diameter_hint(board.clone(), 32.0);
        let img = draw_ring_image(128, 128, [64.0, 64.0], 24.0, 12.0);

        let free = propose_with_marker_diameter(&img, &board, 32.0);
        let detector_out = detector.propose(&img);
        assert_eq!(free, detector_out);

        let free_diag = propose_with_heatmap_and_marker_diameter(&img, &board, 32.0);
        let detector_diag = detector.propose_with_heatmap(&img);
        assert_eq!(free_diag.proposals, detector_diag.proposals);
        assert_eq!(free_diag.heatmap, detector_diag.heatmap);
    }

    #[test]
    fn detector_proposal_apis_honor_proposal_downscale() {
        let mut cfg = DetectConfig::from_target(BoardLayout::default());
        cfg.proposal_downscale = crate::ProposalDownscale::Factor(4);
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
