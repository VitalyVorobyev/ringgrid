//! High-level detection API.
//!
//! [`Detector`] is the primary entry point for detecting ring markers.
//! It wraps a [`DetectConfig`] and provides convenience methods for
//! common detection scenarios (with/without camera, debug, mapper).

use image::GrayImage;
use std::path::Path;

use crate::board_layout::BoardLayout;
#[cfg(feature = "cli-internal")]
use crate::debug_dump::DebugDump;
#[cfg(feature = "cli-internal")]
use crate::detector::DebugCollectConfig;
use crate::detector::{DetectConfig, MarkerScalePrior};
use crate::pipeline;
use crate::pixelmap::PixelMapper;
use crate::DetectionResult;

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
/// let board = BoardLayout::from_json_file(Path::new("crates/ringgrid/examples/target.json")).unwrap();
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
    pub fn from_target_json_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(BoardLayout::from_json_file(path)?))
    }

    /// Load target JSON and create a detector with explicit marker-scale prior.
    pub fn from_target_json_file_with_scale(
        path: &Path,
        marker_scale: MarkerScalePrior,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::with_marker_scale(
            BoardLayout::from_json_file(path)?,
            marker_scale,
        ))
    }

    /// Load target JSON and create a detector with fixed marker-diameter hint.
    pub fn from_target_json_file_with_marker_diameter(
        path: &Path,
        marker_diameter_px: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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

    /// Detect markers in a grayscale image (single-pass, no distortion mapping).
    pub fn detect(&self, image: &GrayImage) -> DetectionResult {
        pipeline::detect_rings(image, &self.config, None)
    }

    /// Detect with a custom pixel mapper (two-pass pipeline).
    ///
    /// Pass-1 runs without mapper for seed generation, pass-2 runs with mapper.
    /// Results are in mapper working frame.
    pub fn detect_with_mapper(
        &self,
        image: &GrayImage,
        mapper: &dyn PixelMapper,
    ) -> DetectionResult {
        pipeline::detect_rings_with_mapper(image, &self.config, Some(mapper))
    }

    /// Detect with automatic self-undistortion estimation.
    ///
    /// Runs a baseline detection, then estimates a division-model distortion
    /// correction from ellipse edge points. If the model improves reprojection,
    /// re-runs detection with the estimated mapper.
    ///
    /// Requires `config.self_undistort.enable = true` (set via config_mut).
    pub fn detect_with_self_undistort(&self, image: &GrayImage) -> DetectionResult {
        pipeline::detect_rings_with_self_undistort(image, &self.config)
    }

    /// Detect with debug dump collection (single-pass).
    #[cfg(feature = "cli-internal")]
    pub fn detect_with_debug(
        &self,
        image: &GrayImage,
        debug_cfg: &DebugCollectConfig,
        mapper: Option<&dyn PixelMapper>,
    ) -> (DetectionResult, DebugDump) {
        pipeline::detect_rings_with_debug(image, &self.config, debug_cfg, mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_basic_detect() {
        let det = Detector::with_config(DetectConfig::from_target(BoardLayout::default()));
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img);
        assert!(result.detected_markers.is_empty());
    }

    #[test]
    fn detector_config_mut() {
        let mut det = Detector::with_config(DetectConfig::from_target(BoardLayout::default()));
        det.config_mut().completion.enable = false;
        assert!(!det.config().completion.enable);
    }
}
