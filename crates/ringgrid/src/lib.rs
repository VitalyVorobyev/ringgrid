//! # ringgrid
//!
//! Pure-Rust detector for dense coded ring calibration targets on a hex lattice.
//!
//! ringgrid detects ring markers in grayscale images, decodes their 16-sector
//! binary IDs from a 893-codeword codebook, fits subpixel ellipses via
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
mod ring;

// ── Public API ──────────────────────────────────────────────────────────

// High-level detector facade
pub use api::Detector;

// Result types
pub use detector::{DetectedMarker, FitMetrics};
pub use homography::RansacStats;
pub use marker::DecodeMetrics;
pub use pipeline::{DetectionFrame, DetectionResult};

// Configuration
pub use detector::{
    CircleRefinementMethod, CompletionParams, DetectConfig, InnerFitConfig, MarkerScalePrior,
    ProjectiveCenterParams, SeedProposalParams,
};
pub use homography::RansacHomographyConfig;

// Geometry
pub use board_layout::{BoardLayout, BoardMarker};
pub use conic::Ellipse;
pub use marker::MarkerSpec;

// Camera / distortion
pub use marker::codebook;
pub use marker::codec;
pub use pixelmap::{
    CameraIntrinsics, CameraModel, DivisionModel, PixelMapper, RadialTangentialDistortion,
    SelfUndistortConfig, SelfUndistortResult,
};
