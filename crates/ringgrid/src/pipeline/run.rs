//! Top-level pipeline orchestrator: fit_decode → finalize.

use super::*;
use crate::debug_dump::DebugDump;
use crate::detector::proposal::find_proposals;
use crate::detector::proposal::Proposal;
use crate::pixelmap::{estimate_self_undistort, PixelMapper};

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    proposals: Vec<Proposal>,
    debug_cfg: Option<&DebugCollectConfig>,
) -> (DetectionResult, Option<DebugDump>) {
    let (w, h) = gray.dimensions();
    let fit_out = super::fit_decode::run(gray, config, mapper, proposals, debug_cfg);
    super::finalize::run(gray, fit_out, [w, h], config, mapper, debug_cfg)
}

// ---------------------------------------------------------------------------
// Single-pass helper (used internally by two-pass for pass-1)
// ---------------------------------------------------------------------------
pub fn detect_single_pass(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let proposals = find_proposals(gray, &config.proposal);
    run(gray, config, None, proposals, None).0
}

/// Single-pass detection with debug dump collection.
pub fn detect_single_pass_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (DetectionResult, DebugDump) {
    let proposals = find_proposals(gray, &config.proposal);
    let (result, dump) = run::run(gray, config, mapper, proposals, Some(debug_cfg));
    (
        result,
        dump.expect("debug dump present when debug_cfg is provided"),
    )
}

// ---------------------------------------------------------------------------
// Pass-2 orchestration (shared by two-pass and self-undistort)
// ---------------------------------------------------------------------------
fn run_pass2(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
    pass1: &DetectionResult,
) -> DetectionResult {
    let seed_params = &config.seed_proposals;
    let proposals = pass1.seed_proposals(seed_params.max_seeds);
    run(gray, config, Some(mapper), proposals, None).0
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Two-pass detection: pass-1 without mapper, pass-2 with mapper + seeds.
///
/// Returned detections are in mapper working-frame coordinates. Any retained
/// pass-1 fallback markers are remapped to the same working frame.
pub(crate) fn detect_with_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
) -> DetectionResult {
    let pass1 = detect_single_pass(gray, config);
    run_pass2(gray, config, mapper, &pass1)
}

/// Detect markers, then estimate and optionally apply a self-undistort model.
///
/// Runs a baseline pass first. If `config.self_undistort.enable` is true and
/// enough markers with edge points are available, estimates a division-model
/// mapper and re-runs pass-2 with seeded proposals.
pub(crate) fn detect_with_self_undistort(
    gray: &GrayImage,
    config: &DetectConfig,
) -> DetectionResult {
    let mut result = detect_single_pass(gray, config);
    let su_cfg = &config.self_undistort;
    if !su_cfg.enable {
        return result;
    }

    let su_result = match estimate_self_undistort(
        &result.detected_markers,
        result.image_size,
        su_cfg,
        Some(&config.board),
    ) {
        Some(r) => r,
        None => return result,
    };

    if su_result.applied {
        let model = su_result.model;
        result = run_pass2(gray, config, &model, &result);
    }

    result.self_undistort = Some(su_result);
    result
}
