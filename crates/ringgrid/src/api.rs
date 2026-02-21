//! High-level detection API.
//!
//! [`Detector`] is the primary entry point for detecting ring markers.
//! It wraps a [`DetectConfig`] and provides convenience methods for
//! common detection scenarios (config-driven detect and external mapper).

use image::GrayImage;
use std::path::Path;

use crate::board_layout::{BoardLayout, BoardLayoutLoadError};
use crate::detector::proposal::find_proposals;
use crate::detector::{DetectConfig, MarkerScalePrior};
use crate::pipeline;
use crate::pixelmap::PixelMapper;
use crate::{DetectionResult, Proposal};

/// Primary detection interface.
///
/// Encapsulates board layout and detection configuration.
/// Create once, detect on many images.
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
    /// This exposes the same proposal stage used by single-pass detection.
    pub fn propose(&self, image: &GrayImage) -> Vec<Proposal> {
        find_proposals(image, &self.config.proposal)
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

    fn draw_ring_image(
        w: u32,
        h: u32,
        center: [f32; 2],
        outer_radius: f32,
        inner_radius: f32,
    ) -> GrayImage {
        use image::Luma;

        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let d = (dx * dx + dy * dy).sqrt();
                let pix = if d >= inner_radius && d <= outer_radius {
                    30u8
                } else {
                    220u8
                };
                img.put_pixel(x, y, Luma([pix]));
            }
        }
        img
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
}
