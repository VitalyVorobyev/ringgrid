//! ringgrid — pure-Rust detector for coded ring calibration targets.
//!
//! The high-level stage order is owned by `pipeline` (internal):
//! proposal -> local fit/decode -> dedup -> global filter -> refinement -> completion.
//! Low-level fitting/sampling primitives are implemented in `detector` and `ring`.

mod api;
mod board_layout;
mod conic;
mod debug_dump;
mod detector;
mod homography;
mod marker;
mod pipeline;
mod pixelmap;
mod ring;

// ── Public API ──────────────────────────────────────────────────────────

// High-level detector facade
pub use api::{Detector, TargetSpec};

// Result types
pub use detector::{DetectedMarker, FitMetrics};
pub use homography::RansacStats;
pub use marker::DecodeMetrics;
pub use pipeline::DetectionResult;

// Configuration
pub use detector::{
    CircleRefinementMethod, CompletionParams, DetectConfig, MarkerScalePrior,
    ProjectiveCenterParams,
};
pub use homography::RansacHomographyConfig;

// Geometry
pub use board_layout::{BoardLayout, BoardMarker};
pub use conic::Ellipse;
pub use marker::MarkerSpec;

// Camera / distortion
pub use pixelmap::{
    CameraIntrinsics, CameraModel, DivisionModel, PixelMapper, RadialTangentialDistortion,
    SelfUndistortConfig, SelfUndistortResult,
};

// ── Feature-gated (CLI-internal) ────────────────────────────────────────

#[cfg(feature = "cli-internal")]
pub use debug_dump::DebugDump;
#[cfg(feature = "cli-internal")]
pub use detector::DebugCollectConfig;
#[cfg(feature = "cli-internal")]
pub use marker::codebook;
#[cfg(feature = "cli-internal")]
pub use marker::codec;
#[cfg(feature = "cli-internal")]
pub use pipeline::detect_rings_with_debug;
