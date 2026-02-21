//! Detection primitives (fit/decode/filter/correction) independent of orchestration.
//!
//! The `pipeline` module owns the high-level call order. This module provides
//! reusable algorithmic building blocks and shared configuration types.

pub(crate) mod center_correction;
pub(crate) mod completion;
pub(crate) mod dedup;
pub(crate) mod global_filter;
pub(crate) mod id_correction;
pub(crate) mod inner_fit;
pub(crate) mod marker_build;
pub(crate) mod outer_fit;
pub(crate) mod proposal;

pub(crate) mod config;

pub(crate) use center_correction::{
    apply_projective_centers, warn_center_correction_without_intrinsics,
};
pub(crate) use completion::{complete_with_h, CompletionStats};
pub use config::{
    CircleRefinementMethod, CompletionParams, DetectConfig, IdCorrectionConfig, InnerFitConfig,
    MarkerScalePrior, OuterFitConfig, ProjectiveCenterParams, SeedProposalParams,
};
pub use dedup::{dedup_by_id, dedup_markers};
pub use global_filter::global_filter;
pub(crate) use id_correction::verify_and_correct_ids;
pub use marker_build::{DetectedMarker, FitMetrics};
pub(crate) use outer_fit::{
    fit_outer_candidate_from_prior_for_completion, median_outer_radius_from_neighbors_px,
    OuterFitCandidate,
};
pub use proposal::{Proposal, ProposalConfig};
