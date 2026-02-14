//! High-level detection pipeline orchestration.
//!
//! This module defines call order and stage boundaries. Algorithmic primitives
//! are implemented in `crate::detector` and imported here.

use image::GrayImage;

use crate::debug_dump as dbg;
use crate::detector;
use crate::detector::proposal::find_proposals;
use crate::detector::DetectedMarker;
use crate::pixelmap::PixelMapper;

/// Full detection result for a single image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionResult {
    /// Detected markers in detector working pixel coordinates.
    pub detected_markers: Vec<DetectedMarker>,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
    /// Fitted board-to-working-frame homography (3x3, row-major), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homography: Option<[[f64; 3]; 3]>,
    /// RANSAC statistics, if homography was fitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac: Option<crate::homography::RansacStats>,
    /// Estimated self-undistort division model, if self-undistort was run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_undistort: Option<crate::pixelmap::SelfUndistortResult>,
}

impl DetectionResult {
    /// Construct an empty result for an image with the provided dimensions.
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            detected_markers: Vec::new(),
            image_size: [width, height],
            homography: None,
            ransac: None,
            self_undistort: None,
        }
    }
}

pub(crate) use crate::detector::inner_fit;
pub(crate) use crate::detector::marker_build;
pub(crate) use crate::detector::outer_fit;
pub(crate) use crate::detector::{
    CompletionAttemptRecord, CompletionDebugOptions, CompletionStats,
};
pub(crate) use crate::detector::{DebugCollectConfig, DetectConfig};

pub(crate) use crate::homography::{
    compute_h_stats, matrix3_to_array, mean_reproj_error_px, refit_homography_matrix,
};
pub(crate) use detector::{
    apply_projective_centers, complete_with_h, dedup_by_id, dedup_markers, dedup_with_debug,
    global_filter, global_filter_with_debug, reapply_projective_centers, refine_with_homography,
    refine_with_homography_with_debug, warn_center_correction_without_intrinsics,
};

mod finalize;
mod fit_decode;
mod run;
mod two_pass;

// ---------------------------------------------------------------------------
// Single-pass entry points
// ---------------------------------------------------------------------------

/// Single-pass detection. Mapper (if provided) is used for distortion-aware
/// sampling within the single pass — it does NOT trigger two-pass detection.
pub(crate) fn detect_single_pass(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    let proposals = find_proposals(gray, &config.proposal);
    run::run(gray, config, mapper, proposals, None).0
}

/// Single-pass detection with debug dump collection.
pub fn detect_single_pass_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (DetectionResult, dbg::DebugDump) {
    let proposals = find_proposals(gray, &config.proposal);
    let (result, dump) = run::run(gray, config, mapper, proposals, Some(debug_cfg));
    (
        result,
        dump.expect("debug dump present when debug_cfg is provided"),
    )
}

// ---------------------------------------------------------------------------
// Two-pass entry points (delegated to two_pass module)
// ---------------------------------------------------------------------------

/// Two-pass detection (pass-1 raw, pass-2 with mapper + seeds).
pub(crate) fn detect_two_pass(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
) -> DetectionResult {
    two_pass::detect_two_pass(gray, config, mapper)
}

/// Single-pass + optional self-undistort second pass.
pub(crate) fn detect_with_self_undistort(
    gray: &GrayImage,
    config: &DetectConfig,
) -> DetectionResult {
    two_pass::detect_with_self_undistort(gray, config)
}
