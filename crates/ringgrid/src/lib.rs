//! # ringgrid
//!
//! Pure-Rust detector for dense coded ring calibration targets on a hex lattice.
//!
//! ringgrid detects ring markers in grayscale images, decodes their 16-sector
//! binary IDs from the shipped baseline 893-codeword profile (with an opt-in
//! extended profile available for advanced use), fits subpixel ellipses via
//! Fitzgibbon's direct method with RANSAC, and estimates a board-to-image
//! homography. No OpenCV dependency — all image processing is in Rust.
//!
//! ## Detection Modes
//!
//! - **Simple** — [`Detector::detect`]: single-pass detection in image coordinates.
//!   Use when the camera has negligible distortion.
//! - **External mapper** — [`Detector::detect_with_mapper`]: two-pass pipeline
//!   with a [`PixelMapper`] (e.g. [`CameraModel`]) for distortion-aware detection.
//! - **Self-undistort** — [`Detector::detect`] with
//!   [`SelfUndistortConfig::enable`] set to `true`: estimates a 1-parameter
//!   division-model distortion from detected markers and optionally re-runs
//!   detection with the estimated correction.
//!
//! ## Quick Start
//!
//! ```no_run
//! use ringgrid::{BoardLayout, Detector};
//! use std::path::Path;
//!
//! let board = BoardLayout::from_json_file(Path::new("target.json")).unwrap();
//! let image = image::open("photo.png").unwrap().to_luma8();
//!
//! let detector = Detector::new(board);
//! let result = detector.detect(&image);
//!
//! for marker in &result.detected_markers {
//!     if let Some(id) = marker.id {
//!         println!("Marker {id} at ({:.1}, {:.1})", marker.center[0], marker.center[1]);
//!     }
//! }
//! ```
//!
//! ## Coordinate Frames
//!
//! Marker centers ([`DetectedMarker::center`]) are always in image-pixel
//! coordinates, regardless of mapper usage. When a [`PixelMapper`] is active,
//! [`DetectedMarker::center_mapped`] provides the working-frame (undistorted)
//! coordinates, and the homography maps board coordinates to the working frame.
//! [`DetectedMarker::board_xy_mm`] provides board-space marker coordinates in
//! millimeters when a valid decoded ID is available on the active board.
//!
//! See [`DetectionResult::center_frame`] and [`DetectionResult::homography_frame`]
//! for the frame metadata on each result.

mod api;
mod board_layout;
mod conic;
mod detector;
mod homography;
mod marker;
mod pipeline;
mod pixelmap;
pub mod proposal;
mod ring;
mod target_generation;
#[cfg(test)]
pub(crate) mod test_utils;

// ── Public API ──────────────────────────────────────────────────────────

// High-level detector facade and proposal-only convenience helpers
pub use api::{
    propose_with_heatmap_and_marker_diameter, propose_with_heatmap_and_marker_scale,
    propose_with_marker_diameter, propose_with_marker_scale, Detector,
};

// Proposal module (standalone ellipse center detection)
pub use proposal::{find_ellipse_centers, find_ellipse_centers_with_heatmap};
pub use proposal::{Proposal, ProposalConfig, ProposalResult};

// Result types
pub use detector::{DetectedMarker, DetectionSource, FitMetrics, InnerFitReason, InnerFitStatus};
pub use homography::RansacStats;
pub use marker::DecodeMetrics;
pub use pipeline::{DetectionFrame, DetectionResult};

// Configuration
pub use detector::{
    CircleRefinementMethod, CompletionParams, DetectConfig, IdCorrectionConfig,
    InnerAsOuterRecoveryConfig, InnerFitConfig, MarkerScalePrior, OuterFitConfig,
    ProjectiveCenterParams, ProposalDownscale, ScaleTier, ScaleTiers, SeedProposalParams,
};
pub use homography::RansacHomographyConfig;

// Sub-configs
pub use marker::{CodebookProfile, DecodeConfig};
pub use ring::{EdgeSampleConfig, OuterEstimationConfig};

// Geometry
pub use board_layout::{
    BoardLayout, BoardLayoutLoadError, BoardLayoutValidationError, BoardMarker,
};
pub use conic::Ellipse;
pub use marker::MarkerSpec;
pub use target_generation::{PngTargetOptions, SvgTargetOptions, TargetGenerationError};

// Camera / distortion
// These raw codebook/codec modules are re-exported for the ringgrid-cli diagnostic
// commands (codebook-info, decode-test). They are not part of the stable library
// API — external code should use the high-level Detector interface.
#[doc(hidden)]
pub use marker::codebook;
#[doc(hidden)]
pub use marker::codec;
pub use pixelmap::{
    CameraIntrinsics, CameraModel, DivisionModel, PixelMapper, RadialTangentialDistortion,
    SelfUndistortConfig, SelfUndistortResult,
};
