//! High-level detection pipeline orchestration.
//!
//! This module defines call order and stage boundaries. Algorithmic primitives
//! are implemented in `crate::detector` and imported here.

use image::GrayImage;

use crate::board_layout::BoardLayout;
use crate::debug_dump as dbg;
use crate::detector;
use crate::pixelmap::PixelMapper;
use crate::{DetectedMarker, DetectionResult};

pub(crate) use crate::detector::config_mapper;
pub(crate) use crate::detector::inner_fit;
pub(crate) use crate::detector::marker_build;
pub(crate) use crate::detector::outer_fit;
pub(crate) use crate::detector::{
    CompletionAttemptRecord, CompletionDebugOptions, CompletionStats,
};
pub(crate) use crate::detector::{
    DebugCollectConfig, DetectConfig, SeedProposalParams, TwoPassParams,
};

pub(crate) use detector::{
    apply_projective_centers, complete_with_h, compute_h_stats, dedup_by_id, dedup_markers,
    dedup_with_debug, global_filter, global_filter_with_debug, matrix3_to_array,
    mean_reproj_error_px, refine_with_homography_with_debug, refit_homography_matrix,
    warn_center_correction_without_intrinsics,
};

mod stages;
mod two_pass;

pub use two_pass::{
    detect_rings, detect_rings_two_pass_with_mapper, detect_rings_with_mapper,
    detect_rings_with_self_undistort,
};

pub(super) fn find_proposals_with_seeds(
    gray: &GrayImage,
    proposal_cfg: &crate::detector::proposal::ProposalConfig,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> Vec<crate::detector::proposal::Proposal> {
    two_pass::find_proposals_with_seeds(gray, proposal_cfg, seed_centers_image, seed_cfg)
}

/// Run the full ring detection pipeline and collect a versioned debug dump.
///
/// Debug collection currently uses single-pass execution.
pub fn detect_rings_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
) -> (DetectionResult, dbg::DebugDump) {
    detect_rings_with_debug_and_mapper(gray, config, debug_cfg, config_mapper(config))
}

/// Run the full ring detection pipeline with debug collection and optional custom mapper.
///
/// Debug collection currently uses single-pass execution.
pub fn detect_rings_with_debug_and_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (DetectionResult, dbg::DebugDump) {
    let (result, dump) = stages::run(
        gray,
        config,
        mapper,
        &[],
        &SeedProposalParams::default(),
        Some(debug_cfg),
    );
    (
        result,
        dump.expect("debug dump present when debug_cfg is provided"),
    )
}

/// Refine marker centers using H: project board coords through H as priors,
/// then re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
    board: &BoardLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    let (refined, _debug) =
        refine_with_homography_with_debug(gray, markers, h, config, board, mapper);
    refined
}
