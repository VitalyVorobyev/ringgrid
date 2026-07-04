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
//! use ringgrid::{Detector, TargetLayout};
//! use std::path::Path;
//!
//! let target = TargetLayout::from_json_file(Path::new("target.json")).unwrap();
//! let image = image::open("photo.png").unwrap().to_luma8();
//!
//! let detector = Detector::new(target);
//! let result = detector.detect(&image).unwrap();
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

// The public API is fully documented; this lint keeps it that way.
#![warn(missing_docs)]

mod api;
mod conic;
mod detector;
mod homography;
mod marker;
mod pipeline;
mod pixelmap;
mod proposal;
mod ring;
mod target;
mod target_generation;
#[cfg(test)]
pub(crate) mod test_utils;

// ── Public API ──────────────────────────────────────────────────────────

// High-level detector facade and proposal-only convenience helpers
pub use api::{
    DetectError, Detector, propose_with_heatmap_and_marker_scale, propose_with_marker_scale,
};

// Proposal module (standalone ellipse center detection)
pub use proposal::{Proposal, ProposalConfig, ProposalResult};
pub use proposal::{find_ellipse_centers, find_ellipse_centers_with_heatmap};

// Result types — slim, stable primary output of `Detector::detect`.
pub use pipeline::{BoardFrame, DetectedMarker, DetectionFrame, DetectionResult};

/// Opt-in diagnostics channel returned by
/// [`Detector::detect_with_diagnostics`]: per-marker fit and decode metrics,
/// homography RANSAC statistics, and pipeline stage timings.
///
/// These types are deliberately not at the crate root: the stable primary
/// output is [`DetectionResult`]; the diagnostics surface may evolve faster
/// between releases.
pub mod diagnostics {
    pub use crate::detector::{DetectionSource, FitMetrics, InnerFitReason, InnerFitStatus};
    pub use crate::homography::RansacStats;
    pub use crate::marker::DecodeMetrics;
    pub use crate::pipeline::{DetectionDiagnostics, MarkerDiagnostics, StageTimings};
}

// Configuration
pub use conic::RansacConfig;
pub use detector::{
    AdvancedDetectConfig, CircleRefinementMethod, CompletionConfig, DetectConfig,
    IdCorrectionConfig, InnerAsOuterRecoveryConfig, InnerFitConfig, MarkerScalePrior,
    OuterFitConfig, ProjectiveCenterConfig, ProposalDownscale, ScaleTier, ScaleTiers,
    SeedProposalConfig,
};

// Sub-configs
pub use marker::{AngularAggregator, CodebookProfile, DecodeConfig, GradPolarity};
pub use ring::{EdgeSampleConfig, OuterEstimationConfig};

/// Inspection helpers for the embedded 16-sector codebook profiles.
pub mod codebook {
    pub use crate::marker::{CodebookInfo, CodewordMatch, codebook_info, decode_word};
}

// Geometry — compositional target model. Legacy v4 `board_spec.json` files load
// via `TargetLayout::from_json_*` (schema auto-migration); the deprecated
// `BoardLayout`/`BoardMarker` Rust types were removed in 0.9.
pub use target::{
    CodedRingSpec, HexGeometry, LatticeGeometry, MarkerCoding, OriginFiducials, RectGeometry,
    RingGeometry, TargetCell, TargetLayout, TargetLoadError, TargetValidationError,
};

pub use conic::Ellipse;
pub use marker::MarkerSpecConfig;
pub use target_generation::{PngTargetOptions, SvgTargetOptions, TargetGenerationError};

// Camera / distortion
pub use pixelmap::{
    CameraIntrinsics, CameraModel, DivisionModel, PixelMapper, RadialTangentialDistortion,
    SelfUndistortConfig, SelfUndistortResult, UndistortConfig,
};
