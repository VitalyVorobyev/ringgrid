//! Two-pass detection and self-undistort orchestration.
//!
//! Two-pass detection: pass-1 runs without mapper for seed generation,
//! pass-2 runs with mapper and pass-1 centers injected as proposal seeds.

use image::GrayImage;
use std::collections::HashSet;

use crate::detector::config::SeedProposalParams;
use crate::detector::proposal::{find_proposals, Proposal, ProposalConfig};
use crate::detector::DetectedMarker;
use crate::pixelmap::{estimate_self_undistort, PixelMapper};

use super::run as stages;
use super::{dedup_by_id, dedup_markers, DetectConfig, DetectionResult};

// ---------------------------------------------------------------------------
// Seed-related helpers (private, use hardcoded defaults)
// ---------------------------------------------------------------------------

fn seed_params() -> SeedProposalParams {
    SeedProposalParams::default()
}

fn collect_seed_centers(pass1: &DetectionResult) -> Vec<[f32; 2]> {
    let params = seed_params();
    let max = params.max_seeds.unwrap_or(pass1.detected_markers.len());
    pass1
        .detected_markers
        .iter()
        .take(max.min(pass1.detected_markers.len()))
        .filter_map(|m| {
            let x = m.center[0] as f32;
            let y = m.center[1] as f32;
            if x.is_finite() && y.is_finite() {
                Some([x, y])
            } else {
                None
            }
        })
        .collect()
}

fn find_proposals_with_seeds(
    gray: &GrayImage,
    proposal_cfg: &ProposalConfig,
    seed_centers_image: &[[f32; 2]],
) -> Vec<Proposal> {
    let mut proposals = find_proposals(gray, proposal_cfg);
    if seed_centers_image.is_empty() {
        return proposals;
    }

    let params = seed_params();
    let merge_r2 = params.merge_radius_px.max(0.0).powi(2);
    let max_seeds = params
        .max_seeds
        .unwrap_or(seed_centers_image.len())
        .min(seed_centers_image.len());

    for seed in seed_centers_image.iter().take(max_seeds) {
        if !seed[0].is_finite() || !seed[1].is_finite() {
            continue;
        }

        let mut merged = false;
        for p in &mut proposals {
            let dx = p.x - seed[0];
            let dy = p.y - seed[1];
            if dx * dx + dy * dy <= merge_r2 {
                p.score = p.score.max(params.seed_score);
                merged = true;
                break;
            }
        }
        if !merged {
            proposals.push(Proposal {
                x: seed[0],
                y: seed[1],
                score: params.seed_score,
            });
        }
    }

    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    proposals
}

// ---------------------------------------------------------------------------
// Marker remapping and merging
// ---------------------------------------------------------------------------

fn map_marker_image_to_working(
    marker: &DetectedMarker,
    mapper: &dyn PixelMapper,
) -> Option<DetectedMarker> {
    let mut out = marker.clone();
    out.center = mapper.image_to_working_pixel(marker.center)?;

    out.center_projective = marker
        .center_projective
        .and_then(|c| mapper.image_to_working_pixel(c));
    if marker.center_projective.is_some() && out.center_projective.is_none() {
        out.center_projective_residual = None;
    }

    // A non-linear mapper does not preserve ellipse structure exactly.
    // Avoid mixed-frame output by dropping these fields for pass-1 fallback markers.
    out.ellipse_outer = None;
    out.ellipse_inner = None;
    out.edge_points_outer = None;
    out.edge_points_inner = None;
    Some(out)
}

fn merge_two_pass_markers(
    pass1: &[DetectedMarker],
    mut pass2: Vec<DetectedMarker>,
    dedup_radius: f64,
    mapper: &dyn PixelMapper,
) -> Vec<DetectedMarker> {
    // Always keep pass-1 markers not present in pass-2 (remapped to working frame).
    let ids_pass2: HashSet<usize> = pass2.iter().filter_map(|m| m.id).collect();
    let mut failed_pass1_reproject = 0usize;

    for m in pass1 {
        if let Some(id) = m.id {
            if ids_pass2.contains(&id) {
                continue;
            }
        }
        if let Some(mm) = map_marker_image_to_working(m, mapper) {
            pass2.push(mm);
        } else {
            failed_pass1_reproject += 1;
        }
    }

    if failed_pass1_reproject > 0 {
        tracing::warn!(
            failed_pass1_reproject,
            "failed to map some pass-1 markers into mapper working frame; dropping them"
        );
    }

    let mut merged = dedup_markers(pass2, dedup_radius);
    dedup_by_id(&mut merged);
    merged
}

// ---------------------------------------------------------------------------
// Single-pass helper (used internally by two-pass for pass-1)
// ---------------------------------------------------------------------------

fn run_single_pass(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    let proposals = find_proposals(gray, &config.proposal);
    stages::run(gray, config, mapper, proposals, None).0
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
    let seeds = collect_seed_centers(pass1);
    let proposals = find_proposals_with_seeds(gray, &config.proposal, &seeds);
    let mut pass2 = stages::run(gray, config, Some(mapper), proposals, None).0;

    if pass2.detected_markers.is_empty() && !seeds.is_empty() {
        tracing::info!("seeded pass-2 returned no detections; retrying pass-2 without seeds");
        let proposals = find_proposals(gray, &config.proposal);
        pass2 = stages::run(gray, config, Some(mapper), proposals, None).0;
    }

    pass2.detected_markers = merge_two_pass_markers(
        &pass1.detected_markers,
        pass2.detected_markers,
        config.dedup_radius,
        mapper,
    );
    pass2
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Two-pass detection: pass-1 without mapper, pass-2 with mapper + seeds.
///
/// Returned detections are in mapper working-frame coordinates. Any retained
/// pass-1 fallback markers are remapped to the same working frame.
pub(super) fn detect_two_pass(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
) -> DetectionResult {
    let pass1 = run_single_pass(gray, config, None);
    run_pass2(gray, config, mapper, &pass1)
}

/// Detect markers, then estimate and optionally apply a self-undistort model.
///
/// Runs a baseline pass first. If `config.self_undistort.enable` is true and
/// enough markers with edge points are available, estimates a division-model
/// mapper and re-runs pass-2 with seeded proposals.
pub(super) fn detect_with_self_undistort(
    gray: &GrayImage,
    config: &DetectConfig,
) -> DetectionResult {
    let mut result = run_single_pass(gray, config, None);
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
