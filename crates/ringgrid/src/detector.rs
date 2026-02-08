//! High-level detection API.
//!
//! [`Detector`] is the primary entry point for detecting ring markers.
//! It wraps a [`DetectConfig`] and provides convenience methods for
//! common detection scenarios (with/without camera, debug, mapper).

use image::GrayImage;
use std::path::Path;

use crate::board_layout::BoardLayout;
use crate::camera::{CameraModel, PixelMapper};
use crate::debug_dump::DebugDumpV1;
use crate::ring::detect::{DebugCollectConfig, DetectConfig};
use crate::DetectionResult;

/// Target specification describing the board to detect.
///
/// Wraps a [`BoardLayout`] with marker positions and geometry.
/// This struct exists as an extension point for future additions
/// (codebook selection, marker geometry variants) without changing
/// the [`Detector`] constructor signature.
#[derive(Debug, Clone)]
pub struct TargetSpec {
    /// Board layout: marker positions, geometry, and metadata.
    pub board: BoardLayout,
}

impl TargetSpec {
    /// Create from a board layout.
    pub fn new(board: BoardLayout) -> Self {
        Self { board }
    }

    /// Create from the embedded default board.
    pub fn default_board() -> Self {
        Self {
            board: BoardLayout::default(),
        }
    }

    /// Load from a board JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            board: BoardLayout::from_json_file(path)?,
        })
    }
}

impl Default for TargetSpec {
    fn default() -> Self {
        Self::default_board()
    }
}

/// Primary detection interface.
///
/// Encapsulates target specification and detection configuration.
/// Create once, detect on many images.
///
/// # Examples
///
/// ```no_run
/// use ringgrid_core::detector::Detector;
/// use image::GrayImage;
///
/// let detector = Detector::new(32.0);
/// let image = GrayImage::new(640, 480);
/// let result = detector.detect(&image);
/// println!("Found {} markers", result.detected_markers.len());
/// ```
pub struct Detector {
    config: DetectConfig,
}

impl Detector {
    /// Create a detector for the default board with the given marker diameter.
    pub fn new(marker_diameter_px: f32) -> Self {
        Self {
            config: DetectConfig::from_marker_diameter_px(marker_diameter_px),
        }
    }

    /// Create a detector with a custom target and default-scaled config.
    pub fn with_target(target: TargetSpec, marker_diameter_px: f32) -> Self {
        let mut config = DetectConfig::from_marker_diameter_px(marker_diameter_px);
        config.board = target.board;
        Self { config }
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
    pub fn detect(&self, image: &GrayImage) -> DetectionResult {
        crate::ring::detect::detect_rings(image, &self.config)
    }

    /// Detect with a camera model (sets camera in config, runs standard pipeline).
    pub fn detect_with_camera(&self, image: &GrayImage, camera: &CameraModel) -> DetectionResult {
        let mut cfg = self.config.clone();
        cfg.camera = Some(*camera);
        crate::ring::detect::detect_rings(image, &cfg)
    }

    /// Detect with a custom pixel mapper (two-pass pipeline).
    pub fn detect_with_mapper(
        &self,
        image: &GrayImage,
        mapper: &dyn PixelMapper,
    ) -> DetectionResult {
        crate::ring::detect::detect_rings_with_mapper(image, &self.config, Some(mapper))
    }

    /// Detect with debug dump collection.
    pub fn detect_with_debug(
        &self,
        image: &GrayImage,
        debug_cfg: &DebugCollectConfig,
    ) -> (DetectionResult, DebugDumpV1) {
        crate::ring::detect::detect_rings_with_debug(image, &self.config, debug_cfg)
    }

    /// Detect markers, then estimate and optionally apply a self-undistort model.
    ///
    /// Runs the standard pipeline first. If the self-undistort config (in
    /// `DetectConfig`) is enabled and enough markers with inner+outer edge points
    /// are found, estimates a 1-parameter division-model distortion. If the
    /// improvement exceeds the threshold, re-runs detection with the estimated
    /// model as a `PixelMapper` (two-pass pipeline).
    pub fn detect_with_self_undistort(&self, image: &GrayImage) -> DetectionResult {
        use crate::ring::detect::{detect_rings, detect_rings_two_pass_with_mapper, TwoPassParams};
        use crate::self_undistort::estimate_self_undistort;

        let mut result = detect_rings(image, &self.config);
        let su_cfg = &self.config.self_undistort;
        if !su_cfg.enable {
            return result;
        }

        let image_size = result.image_size;
        let su_result = match estimate_self_undistort(
            &result.detected_markers,
            image_size,
            su_cfg,
            Some(&self.config.board),
        ) {
            Some(r) => r,
            None => return result,
        };

        if su_result.applied {
            let model = su_result.model;
            let pass2 = detect_rings_two_pass_with_mapper(
                image,
                &self.config,
                &model,
                &TwoPassParams::default(),
            );
            result = pass2;
        }

        result.self_undistort = Some(su_result);
        result
    }

    /// Detect with known camera intrinsics for precision mode.
    ///
    /// Thin wrapper around [`detect_with_camera`](Self::detect_with_camera).
    pub fn detect_precision(&self, image: &GrayImage, camera: &CameraModel) -> DetectionResult {
        self.detect_with_camera(image, camera)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_basic_detect() {
        let det = Detector::new(32.0);
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img);
        assert!(result.detected_markers.is_empty());
    }

    #[test]
    fn detector_with_target() {
        let target = TargetSpec::default();
        assert_eq!(target.board.n_markers(), 203);
        let det = Detector::with_target(target, 32.0);
        assert_eq!(det.config().board.n_markers(), 203);
    }

    #[test]
    fn detector_config_mut() {
        let mut det = Detector::new(32.0);
        det.config_mut().completion.enable = false;
        assert!(!det.config().completion.enable);
    }
}
