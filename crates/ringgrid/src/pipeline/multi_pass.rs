//! Multi-pass detection orchestration.
//!
//! Two-pass detection: pass-1 runs without mapper for seed generation,
//! pass-2 runs with mapper and pass-1 centers injected as proposal seeds.

use image::GrayImage;

use crate::detector::proposal::find_proposals;
use crate::pixelmap::{estimate_self_undistort, PixelMapper};

use super::run as stages;
use super::{DetectConfig, DetectionResult};

// ---------------------------------------------------------------------------
// Single-pass helper (used internally by two-pass for pass-1)
// ---------------------------------------------------------------------------

fn run_single_pass(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let proposals = find_proposals(gray, &config.proposal);
    stages::run(gray, config, None, proposals, None).0
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
    // TODO: use pass1 markers as proposals without find_proposals_with_seeds
    let proposals = pass1.seed_proposals(seed_params.max_seeds);
    let pass2 = stages::run(gray, config, Some(mapper), proposals, None).0;
    pass2
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
    let pass1 = run_single_pass(gray, config);
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
    let mut result = run_single_pass(gray, config);
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
