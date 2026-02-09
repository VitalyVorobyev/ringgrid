//! High-level detection API.
//!
//! [`Detector`] is the primary entry point for detecting ring markers.
//! It wraps a [`DetectConfig`] and provides convenience methods for
//! common detection scenarios (with/without camera, debug, mapper).

use image::GrayImage;
use std::path::Path;

use crate::board_layout::BoardLayout;
use crate::camera::{CameraModel, PixelMapper};
#[cfg(feature = "cli-internal")]
use crate::debug_dump::DebugDump;
#[cfg(feature = "cli-internal")]
use crate::ring::detect::DebugCollectConfig;
use crate::ring::detect::DetectConfig;
use crate::DetectionResult;

/// Target specification describing the board to detect.
///
/// Public API v1 requires runtime target loading from JSON.
#[derive(Debug, Clone)]
pub struct TargetSpec {
    board: BoardLayout,
}

impl TargetSpec {
    /// Load from a board JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            board: BoardLayout::from_json_file(path)?,
        })
    }

    /// Access board layout metadata.
    pub fn board(&self) -> &BoardLayout {
        &self.board
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
/// use ringgrid::{Detector, TargetSpec};
/// use image::GrayImage;
/// use std::path::Path;
///
/// let target = TargetSpec::from_json_file(Path::new("crates/ringgrid/examples/target.json")).unwrap();
/// let detector = Detector::new(target, 32.0);
/// let image = GrayImage::new(640, 480);
/// let result = detector.detect(&image);
/// println!("Found {} markers", result.detected_markers.len());
/// ```
pub struct Detector {
    config: DetectConfig,
}

impl Detector {
    /// Create a detector with a runtime target specification.
    pub fn new(target: TargetSpec, marker_diameter_px: f32) -> Self {
        Self {
            config: DetectConfig::from_target_and_marker_diameter(target.board, marker_diameter_px),
        }
    }

    /// Load target JSON and create a detector in one step.
    pub fn from_target_json_file(
        path: &Path,
        marker_diameter_px: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(
            TargetSpec::from_json_file(path)?,
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
    #[cfg(feature = "cli-internal")]
    pub fn detect_with_debug(
        &self,
        image: &GrayImage,
        debug_cfg: &DebugCollectConfig,
    ) -> (DetectionResult, DebugDump) {
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
        use crate::ring::detect::{detect_rings, detect_rings_pass2_with_seeds, TwoPassParams};
        use crate::self_undistort::estimate_self_undistort;

        let mut result = detect_rings(image, &self.config);
        let su_cfg = &self.config.self_undistort;
        if !su_cfg.enable {
            return result;
        }

        let su_result = match estimate_self_undistort(
            &result.detected_markers,
            result.image_size,
            su_cfg,
            Some(&self.config.board),
        ) {
            Some(r) => r,
            None => return result,
        };

        if su_result.applied {
            let model = su_result.model;
            // Reuse pass-1 detections as seeds for pass-2 (saves one full pipeline run).
            result = detect_rings_pass2_with_seeds(
                image,
                &self.config,
                &model,
                &result,
                &TwoPassParams::default(),
            );
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
        let det = Detector::with_config(DetectConfig::from_target_and_marker_diameter(
            BoardLayout::default(),
            32.0,
        ));
        let img = GrayImage::new(200, 200);
        let result = det.detect(&img);
        assert!(result.detected_markers.is_empty());
    }

    #[test]
    fn detector_config_mut() {
        let mut det = Detector::with_config(DetectConfig::from_target_and_marker_diameter(
            BoardLayout::default(),
            32.0,
        ));
        det.config_mut().completion.enable = false;
        assert!(!det.config().completion.enable);
    }
}
