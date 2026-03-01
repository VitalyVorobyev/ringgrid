//! Top-level pipeline orchestrator: fit_decode → finalize.

use super::*;
use crate::detector::config::{ScaleTier, ScaleTiers};
use crate::detector::dedup::merge_multiscale_markers;
use crate::detector::marker_build::DetectionSource;
use crate::detector::proposal::find_proposals;
use crate::detector::proposal::Proposal;
use crate::pixelmap::{estimate_self_undistort, PixelMapper};

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    proposals: Vec<Proposal>,
    source: DetectionSource,
) -> DetectionResult {
    let fit_markers = super::fit_decode::run(gray, config, mapper, proposals, source);
    super::finalize::run(gray, fit_markers, config, mapper)
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
    run(
        gray,
        config,
        Some(mapper),
        proposals,
        DetectionSource::SeededPass,
    )
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
pub fn detect_single_pass(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let proposals = find_proposals(gray, &config.proposal);
    run(gray, config, None, proposals, DetectionSource::FitDecoded)
}

/// Two-pass detection: pass-1 without mapper, pass-2 with mapper + seeds.
///
/// Returned marker centers are always image-space (`DetectedMarker.center`).
/// When a mapper is active, mapper-frame centers are preserved in
/// `DetectedMarker.center_mapped` and homography frame metadata is set to
/// `DetectionFrame::Working`.
pub fn detect_with_mapper(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: &dyn PixelMapper,
) -> DetectionResult {
    let pass1 = detect_single_pass(gray, config);
    run_pass2(gray, config, mapper, &pass1)
}

// ---------------------------------------------------------------------------
// Multi-scale / adaptive entry points
// ---------------------------------------------------------------------------

/// Run a single scale tier through fit/decode and per-tier finalization
/// (projective centers + ID correction).
///
/// Returns a raw marker list **without** global filter, completion, or final H
/// refit. Call [`detect_multiscale`] to merge and finalize across tiers.
pub(crate) fn detect_premerge(
    gray: &GrayImage,
    config: &DetectConfig,
) -> Vec<crate::DetectedMarker> {
    let proposals = find_proposals(gray, &config.proposal);
    let fit_markers =
        super::fit_decode::run(gray, config, None, proposals, DetectionSource::FitDecoded);
    super::finalize::finalize_premerge(fit_markers, config)
}

/// Detect markers across multiple scale tiers.
///
/// For each tier:
/// 1. Clones `config` and applies the tier's scale prior.
/// 2. Runs fit/decode + projective centers + ID correction (`detect_premerge`).
///
/// After all tiers:
/// 3. Merges results with size-consistency-aware dedup
///    ([`merge_multiscale_markers`]).
/// 4. Runs global filter + completion + final H refit once on the merged pool.
///
/// The merge dedup radius is `min_tier_diameter × 0.6`, clamped to at least
/// `config.dedup_radius`.
pub fn detect_multiscale(
    gray: &GrayImage,
    config: &DetectConfig,
    tiers: &ScaleTiers,
) -> DetectionResult {
    warn_center_correction_without_intrinsics(config, false);

    let mut all_markers: Vec<crate::DetectedMarker> = Vec::new();

    for tier in tiers.tiers() {
        let mut tier_config = config.clone();
        tier_config.set_marker_scale_prior(tier.prior);

        tracing::debug!(
            d_min = tier.prior.diameter_min_px,
            d_max = tier.prior.diameter_max_px,
            "running scale tier"
        );

        let tier_markers = detect_premerge(gray, &tier_config);
        tracing::debug!(n = tier_markers.len(), "tier produced markers before merge");
        all_markers.extend(tier_markers);
    }

    // Merge dedup radius: small enough not to suppress neighboring markers,
    // large enough to collapse same-marker duplicates from overlapping tiers.
    let min_tier_d = tiers
        .tiers()
        .iter()
        .map(|t| t.prior.diameter_min_px as f64)
        .fold(f64::INFINITY, f64::min);
    let dedup_radius = (min_tier_d * 0.6).max(config.dedup_radius);

    let merged = merge_multiscale_markers(all_markers, dedup_radius, 6);

    tracing::info!(
        n_merged = merged.len(),
        n_tiers = tiers.tiers().len(),
        dedup_radius,
        "merged markers from all tiers"
    );

    super::finalize::finalize_postmerge(gray, merged, config, None)
}

/// Select adaptive scale tiers for one image.
///
/// When `nominal_diameter_px` is `Some`, skips the scale probe and builds a
/// two-tier bracket around `[0.5×, 1.5×]` the hint with a small overlap at the
/// split point.
///
/// When `nominal_diameter_px` is `None`, runs scale probing and derives tiers
/// from dominant radii. Falls back to [`ScaleTiers::four_tier_wide`] when the
/// probe returns no usable signal.
pub fn select_adaptive_tiers(gray: &GrayImage, nominal_diameter_px: Option<f32>) -> ScaleTiers {
    match nominal_diameter_px {
        Some(d) => {
            let d_lo = (d * 0.5).max(4.0);
            let d_hi = d * 1.5;
            // Split at nominal with 5 % overlap to avoid a tier boundary gap.
            let d_split = d * 1.05;
            ScaleTiers(vec![
                ScaleTier::new(d_lo, d_split),
                ScaleTier::new(d_split * 0.95, d_hi),
            ])
        }
        None => {
            let probe_radii = super::scale_probe::scale_probe(gray, 64, 16);
            if probe_radii.is_empty() {
                tracing::debug!(
                    "scale probe found no dominant radii; using four_tier_wide fallback"
                );
                ScaleTiers::four_tier_wide()
            } else {
                tracing::debug!(n = probe_radii.len(), "scale probe succeeded");
                ScaleTiers::from_detected_radii(&probe_radii)
            }
        }
    }
}

/// Adaptive detection using automatically selected scale tiers.
///
/// Equivalent to [`detect_adaptive_with_hint`] with `nominal_diameter_px=None`.
pub fn detect_adaptive(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let tiers = select_adaptive_tiers(gray, None);
    detect_multiscale(gray, config, &tiers)
}

/// Adaptive detection with an optional nominal-diameter hint.
///
/// When `nominal_diameter_px` is `Some`, skips the scale probe and builds a
/// two-tier bracket around `[0.5×, 1.5×]` the hint. When `None`, behaves
/// identically to [`detect_adaptive`].
pub fn detect_adaptive_with_hint(
    gray: &GrayImage,
    config: &DetectConfig,
    nominal_diameter_px: Option<f32>,
) -> DetectionResult {
    let tiers = select_adaptive_tiers(gray, nominal_diameter_px);
    detect_multiscale(gray, config, &tiers)
}

/// Detect markers, then estimate and optionally apply a self-undistort model.
///
/// Runs a baseline pass first. If `config.self_undistort.enable` is true and
/// enough markers with edge points are available, estimates a division-model
/// mapper and re-runs pass-2 with seeded proposals.
pub fn detect_with_self_undistort(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
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
