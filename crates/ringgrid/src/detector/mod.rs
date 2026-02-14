//! Detection primitives (fit/decode/filter/correction) independent of orchestration.
//!
//! The `pipeline` module owns the high-level call order. This module provides
//! reusable algorithmic building blocks and shared configuration types.

pub(crate) mod center_correction;
pub(crate) mod completion;
pub(crate) mod dedup;
pub(crate) mod global_filter;
pub(crate) mod inner_fit;
pub(crate) mod marker_build;
pub(crate) mod outer_fit;
pub(crate) mod proposal;
pub(crate) mod refine_h;

pub(crate) mod config;

pub(crate) use center_correction::{
    apply_projective_centers, reapply_projective_centers, warn_center_correction_without_intrinsics,
};
pub(crate) use completion::{
    complete_with_h, CompletionAttemptRecord, CompletionDebugOptions, CompletionStats,
};
pub use config::{
    CircleRefinementMethod, CompletionParams, DebugCollectConfig, DetectConfig, MarkerScalePrior,
    ProjectiveCenterParams, SeedProposalParams,
};
pub(crate) use dedup::dedup_with_debug;
pub use dedup::{dedup_by_id, dedup_markers};
pub use global_filter::{global_filter, global_filter_with_debug};
pub use marker_build::{DetectedMarker, FitMetrics};
pub(crate) use outer_fit::{
    compute_center, fit_outer_ellipse_robust_with_reason, marker_outer_radius_expected_px,
    mean_axis_px_from_marker, median_outer_radius_from_neighbors_px, OuterFitCandidate,
};
pub(crate) use refine_h::{refine_with_homography, refine_with_homography_with_debug};
