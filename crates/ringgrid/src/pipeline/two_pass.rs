use image::GrayImage;
use std::collections::HashSet;

use crate::pixelmap::PixelMapper;
use crate::{DetectedMarker, DetectionResult};

use super::config_mapper;
use super::stages;
use super::{dedup_by_id, dedup_markers, DetectConfig, SeedProposalParams, TwoPassParams};
use crate::detector::proposal::{find_proposals, Proposal, ProposalConfig};

fn capped_len<T>(slice: &[T], max_len: Option<usize>) -> usize {
    max_len.unwrap_or(slice.len()).min(slice.len())
}

pub(super) fn find_proposals_with_seeds(
    gray: &GrayImage,
    proposal_cfg: &ProposalConfig,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> Vec<Proposal> {
    let mut proposals = find_proposals(gray, proposal_cfg);
    if seed_centers_image.is_empty() {
        return proposals;
    }

    let merge_r2 = seed_cfg.merge_radius_px.max(0.0).powi(2);
    for seed in seed_centers_image
        .iter()
        .take(capped_len(seed_centers_image, seed_cfg.max_seeds))
    {
        if !seed[0].is_finite() || !seed[1].is_finite() {
            continue;
        }

        let mut merged = false;
        for p in &mut proposals {
            let dx = p.x - seed[0];
            let dy = p.y - seed[1];
            if dx * dx + dy * dy <= merge_r2 {
                p.score = p.score.max(seed_cfg.seed_score);
                merged = true;
                break;
            }
        }
        if !merged {
            proposals.push(Proposal {
                x: seed[0],
                y: seed[1],
                score: seed_cfg.seed_score,
            });
        }
    }

    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    proposals
}

pub(super) fn map_marker_image_to_working(
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
    keep_pass1_markers: bool,
    dedup_radius: f64,
    mapper: Option<&dyn PixelMapper>,
) -> Vec<DetectedMarker> {
    let mut failed_pass1_reproject = 0usize;
    if keep_pass1_markers {
        let ids_pass2: HashSet<usize> = pass2.iter().filter_map(|m| m.id).collect();
        for m in pass1 {
            if let Some(id) = m.id {
                if ids_pass2.contains(&id) {
                    continue;
                }
            }
            if let Some(mapper) = mapper {
                if let Some(mm) = map_marker_image_to_working(m, mapper) {
                    pass2.push(mm);
                } else {
                    failed_pass1_reproject += 1;
                }
            } else {
                pass2.push(m.clone());
            }
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

fn collect_seed_centers_image(
    pass1: &DetectionResult,
    seed_cfg: &SeedProposalParams,
) -> Vec<[f32; 2]> {
    pass1
        .detected_markers
        .iter()
        .take(capped_len(&pass1.detected_markers, seed_cfg.max_seeds))
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

fn detect_rings_with_mapper_and_seeds(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> DetectionResult {
    stages::run(gray, config, mapper, seed_centers_image, seed_cfg, None).0
}

/// Run the full ring detection pipeline.
pub fn detect_rings(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    detect_rings_with_mapper_and_seeds(
        gray,
        config,
        config_mapper(config),
        &[],
        &SeedProposalParams::default(),
    )
}

/// Run the full ring detection pipeline with an optional custom pixel mapper.
///
/// This allows users to plug in camera/distortion models via a lightweight trait adapter.
/// When a mapper is provided, detection runs in two passes:
/// raw pass-1, then mapper-aware pass-2 with pass-1 seed injection.
pub fn detect_rings_with_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    if let Some(mapper) = mapper {
        detect_rings_two_pass_with_mapper(gray, config, mapper, &TwoPassParams::default())
    } else {
        detect_rings_with_mapper_and_seeds(gray, config, None, &[], &SeedProposalParams::default())
    }
}

/// Run pass-2 of two-pass detection using existing pass-1 results as seeds.
///
/// This avoids re-running pass-1 when the caller already has pass-1 detections
/// (e.g. from a prior `detect_rings` call).
fn detect_rings_pass2_with_seeds(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
    pass1: &DetectionResult,
    params: &TwoPassParams,
) -> DetectionResult {
    let seed_centers_image = collect_seed_centers_image(pass1, &params.seed);

    let mut pass2 = detect_rings_with_mapper_and_seeds(
        gray,
        config,
        Some(mapper),
        &seed_centers_image,
        &params.seed,
    );
    if pass2.detected_markers.is_empty() && !seed_centers_image.is_empty() {
        tracing::info!("seeded pass-2 returned no detections; retrying pass-2 without seeds");
        pass2 = detect_rings_with_mapper_and_seeds(gray, config, Some(mapper), &[], &params.seed);
    }
    pass2.detected_markers = merge_two_pass_markers(
        &pass1.detected_markers,
        pass2.detected_markers,
        params.keep_pass1_markers,
        config.dedup_radius,
        Some(mapper),
    );
    pass2
}

/// Run two-pass detection:
/// - pass-1 in raw image space,
/// - pass-2 with mapper and pass-1 centers injected as proposal seeds.
///
/// Returned detections stay in pass-2 mapper working coordinates. Any retained
/// pass-1 fallback markers are remapped to the same working frame.
pub fn detect_rings_two_pass_with_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
    params: &TwoPassParams,
) -> DetectionResult {
    let pass1 = detect_rings_with_mapper_and_seeds(gray, config, None, &[], &params.seed);
    detect_rings_pass2_with_seeds(gray, config, mapper, &pass1, params)
}

/// Detect markers, then estimate and optionally apply a self-undistort model.
///
/// Runs a baseline pass first. If `config.self_undistort.enable` is true and
/// enough markers with edge points are available, estimates a division-model
/// mapper and re-runs pass-2 with seeded proposals.
pub fn detect_rings_with_self_undistort(
    gray: &GrayImage,
    config: &DetectConfig,
) -> DetectionResult {
    use crate::pixelmap::estimate_self_undistort;

    let mut result = detect_rings(gray, config);
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
        result =
            detect_rings_pass2_with_seeds(gray, config, &model, &result, &TwoPassParams::default());
    }

    result.self_undistort = Some(su_result);
    result
}
