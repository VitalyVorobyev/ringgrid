//! Top-level pipeline orchestrator: fit_decode → finalize.

use super::*;
use crate::detector::config::{MarkerScalePrior, ScaleTier, ScaleTiers};
use crate::detector::dedup::merge_multiscale_markers;
use crate::detector::marker_build::DetectionSource;
use crate::pixelmap::{estimate_self_undistort, PixelMapper};
use crate::proposal::{
    find_ellipse_centers_with_heatmap, Proposal, ProposalConfig, ProposalResult,
};
use image::{ImageBuffer, Luma};
use std::collections::HashSet;
use std::time::Instant;

#[inline]
fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    proposals: Vec<Proposal>,
    source: DetectionSource,
) -> DetectionResult {
    let total_start = Instant::now();

    let fit_decode_start = Instant::now();
    let fit_markers = super::fit_decode::run(gray, config, mapper, proposals, source);
    let fit_decode_elapsed = fit_decode_start.elapsed();
    let fit_marker_count = fit_markers.len();

    let finalize_start = Instant::now();
    let result = super::finalize::run(gray, fit_markers, config, mapper);
    let finalize_elapsed = finalize_start.elapsed();

    tracing::info!(
        markers_after_fit_decode = fit_marker_count,
        markers_final = result.detected_markers.len(),
        fit_decode_ms = duration_ms(fit_decode_elapsed),
        finalize_ms = duration_ms(finalize_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "pipeline timing summary"
    );

    result
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
// Downscaled proposal generation
// ---------------------------------------------------------------------------

/// Run proposal generation, optionally on a downscaled image copy.
///
/// When `factor > 1`, the image is resized to `(w/factor, h/factor)` using
/// triangle (bilinear) interpolation, proposals are found on the smaller image,
/// and coordinates are scaled back to original image space.
fn rescale_proposals_to_image_space(
    proposals: Vec<Proposal>,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Proposal> {
    proposals
        .into_iter()
        .map(|p| Proposal {
            x: p.x * scale_x,
            y: p.y * scale_y,
            score: p.score,
        })
        .collect()
}

fn resize_heatmap_to_image_space(
    heatmap: Vec<f32>,
    small_w: u32,
    small_h: u32,
    w: u32,
    h: u32,
) -> Vec<f32> {
    let heatmap_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(small_w, small_h, heatmap)
        .expect("proposal heatmap dimensions match");
    image::imageops::resize(&heatmap_img, w, h, image::imageops::FilterType::Triangle).into_raw()
}

fn proposal_result_with_downscale(
    gray: &GrayImage,
    proposal_config: &ProposalConfig,
    factor: u32,
) -> ProposalResult {
    if factor <= 1 {
        return find_ellipse_centers_with_heatmap(gray, proposal_config);
    }

    let (w, h) = gray.dimensions();
    let small_w = (w / factor).max(4);
    let small_h = (h / factor).max(4);
    let scale_x = w as f32 / small_w as f32;
    let scale_y = h as f32 / small_h as f32;
    let distance_scale = 0.5 * (scale_x + scale_y);

    let small = image::imageops::resize(
        gray,
        small_w,
        small_h,
        image::imageops::FilterType::Triangle,
    );

    let mut scaled_config = proposal_config.clone();
    scaled_config.r_min /= distance_scale;
    scaled_config.r_max /= distance_scale;
    scaled_config.min_distance /= distance_scale;

    let small_result = find_ellipse_centers_with_heatmap(&small, &scaled_config);
    let proposals = rescale_proposals_to_image_space(small_result.proposals, scale_x, scale_y);
    let heatmap = resize_heatmap_to_image_space(small_result.heatmap, small_w, small_h, w, h);

    ProposalResult {
        image_size: [w, h],
        proposals,
        heatmap,
    }
}

fn find_proposals_with_downscale(
    gray: &GrayImage,
    proposal_config: &ProposalConfig,
    factor: u32,
) -> Vec<Proposal> {
    proposal_result_with_downscale(gray, proposal_config, factor).proposals
}

pub(crate) fn proposal_seeds_for_config(gray: &GrayImage, config: &DetectConfig) -> Vec<Proposal> {
    let factor = config.proposal_downscale.resolve(config.marker_scale);
    find_proposals_with_downscale(gray, &config.proposal, factor)
}

pub(crate) fn proposal_result_for_config(
    gray: &GrayImage,
    config: &DetectConfig,
) -> ProposalResult {
    let factor = config.proposal_downscale.resolve(config.marker_scale);
    proposal_result_with_downscale(gray, &config.proposal, factor)
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
pub fn detect_single_pass(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let total_start = Instant::now();
    let (image_width, image_height) = gray.dimensions();

    let factor = config.proposal_downscale.resolve(config.marker_scale);

    let proposal_start = Instant::now();
    let proposals = proposal_seeds_for_config(gray, config);
    let proposal_elapsed = proposal_start.elapsed();
    let proposal_count = proposals.len();

    let downstream_start = Instant::now();
    let result = run(gray, config, None, proposals, DetectionSource::FitDecoded);
    let downstream_elapsed = downstream_start.elapsed();

    tracing::info!(
        image_width,
        image_height,
        proposals = proposal_count,
        proposal_downscale_factor = factor,
        markers = result.detected_markers.len(),
        proposal_ms = duration_ms(proposal_elapsed),
        downstream_ms = duration_ms(downstream_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "detect_single_pass timing summary"
    );

    result
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
    let total_start = Instant::now();

    let pass1_start = Instant::now();
    let pass1 = detect_single_pass(gray, config);
    let pass1_elapsed = pass1_start.elapsed();

    let pass2_start = Instant::now();
    let result = run_pass2(gray, config, mapper, &pass1);
    let pass2_elapsed = pass2_start.elapsed();

    tracing::info!(
        pass1_markers = pass1.detected_markers.len(),
        markers_final = result.detected_markers.len(),
        pass1_ms = duration_ms(pass1_elapsed),
        pass2_ms = duration_ms(pass2_elapsed),
        total_ms = duration_ms(total_start.elapsed()),
        "detect_with_mapper timing summary"
    );

    result
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
    let proposals = proposal_seeds_for_config(gray, config);
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

#[derive(Debug, Clone, Copy)]
struct AdaptiveCandidateScore {
    mapped: usize,
    has_ransac: usize,
    ransac_inliers: usize,
    decoded: usize,
    ransac_neg_err_px: f64,
    mean_confidence: f64,
    total: usize,
}

impl AdaptiveCandidateScore {
    fn from_result(result: &DetectionResult) -> Self {
        let mut mapped = 0usize;
        let mut decoded = 0usize;
        let mut mapped_conf_sum = 0.0f64;

        for marker in &result.detected_markers {
            if marker.id.is_some() {
                decoded += 1;
                if marker.board_xy_mm.is_some() {
                    mapped += 1;
                    mapped_conf_sum += f64::from(marker.confidence);
                }
            }
        }

        let mean_confidence = if mapped > 0 {
            mapped_conf_sum / mapped as f64
        } else {
            0.0
        };

        let (has_ransac, ransac_inliers, ransac_neg_err_px) = result
            .ransac
            .as_ref()
            .map(|stats| (1usize, stats.n_inliers, -stats.mean_err_px))
            .unwrap_or((0usize, 0usize, f64::NEG_INFINITY));

        Self {
            mapped,
            has_ransac,
            ransac_inliers,
            decoded,
            ransac_neg_err_px,
            mean_confidence,
            total: result.detected_markers.len(),
        }
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self.mapped > other.mapped
            || (self.mapped == other.mapped && self.has_ransac > other.has_ransac)
            || (self.mapped == other.mapped
                && self.has_ransac == other.has_ransac
                && self.ransac_inliers > other.ransac_inliers)
            || (self.mapped == other.mapped
                && self.has_ransac == other.has_ransac
                && self.ransac_inliers == other.ransac_inliers
                && self.decoded > other.decoded)
            || (self.mapped == other.mapped
                && self.has_ransac == other.has_ransac
                && self.ransac_inliers == other.ransac_inliers
                && self.decoded == other.decoded
                && self.ransac_neg_err_px > other.ransac_neg_err_px)
            || (self.mapped == other.mapped
                && self.has_ransac == other.has_ransac
                && self.ransac_inliers == other.ransac_inliers
                && self.decoded == other.decoded
                && self.ransac_neg_err_px == other.ransac_neg_err_px
                && self.mean_confidence > other.mean_confidence)
            || (self.mapped == other.mapped
                && self.has_ransac == other.has_ransac
                && self.ransac_inliers == other.ransac_inliers
                && self.decoded == other.decoded
                && self.ransac_neg_err_px == other.ransac_neg_err_px
                && self.mean_confidence == other.mean_confidence
                && self.total > other.total)
    }
}

#[derive(Debug, Clone)]
struct AdaptiveCandidate {
    label: &'static str,
    tiers: ScaleTiers,
}

fn tiers_signature(tiers: &ScaleTiers) -> Vec<(u32, u32)> {
    tiers
        .tiers()
        .iter()
        .map(|tier| {
            (
                tier.prior.diameter_min_px.to_bits(),
                tier.prior.diameter_max_px.to_bits(),
            )
        })
        .collect()
}

fn candidate_single_prior(
    label: &'static str,
    diameter_min_px: f32,
    diameter_max_px: f32,
) -> AdaptiveCandidate {
    AdaptiveCandidate {
        label,
        tiers: ScaleTiers::single(MarkerScalePrior::new(diameter_min_px, diameter_max_px)),
    }
}

fn build_adaptive_candidates(
    gray: &GrayImage,
    config: &DetectConfig,
    nominal_diameter_px: Option<f32>,
) -> Vec<AdaptiveCandidate> {
    let mut candidates: Vec<AdaptiveCandidate> = Vec::new();
    let mut seen: HashSet<Vec<(u32, u32)>> = HashSet::new();

    let mut push_candidate = |candidate: AdaptiveCandidate| {
        let signature = tiers_signature(&candidate.tiers);
        if seen.insert(signature) {
            candidates.push(candidate);
        }
    };

    push_candidate(AdaptiveCandidate {
        label: "probe",
        tiers: select_adaptive_tiers(gray, nominal_diameter_px),
    });
    push_candidate(AdaptiveCandidate {
        label: "two_tier_standard",
        tiers: ScaleTiers::two_tier_standard(),
    });
    push_candidate(AdaptiveCandidate {
        label: "four_tier_wide",
        tiers: ScaleTiers::four_tier_wide(),
    });
    push_candidate(AdaptiveCandidate {
        label: "single_config",
        tiers: ScaleTiers::single(config.marker_scale),
    });

    // Curated fallback priors mirror historical RTV3D tuning presets.
    push_candidate(candidate_single_prior("single_14_80", 14.0, 80.0));
    push_candidate(candidate_single_prior("single_18_100", 18.0, 100.0));
    push_candidate(candidate_single_prior("single_16_120", 16.0, 120.0));
    push_candidate(candidate_single_prior("single_10_120", 10.0, 120.0));
    push_candidate(candidate_single_prior("single_14_140", 14.0, 140.0));
    push_candidate(candidate_single_prior("single_20_140", 20.0, 140.0));
    push_candidate(candidate_single_prior("single_8_220", 8.0, 220.0));

    candidates
}

fn detect_adaptive_candidates(
    gray: &GrayImage,
    config: &DetectConfig,
    nominal_diameter_px: Option<f32>,
) -> DetectionResult {
    let candidates = build_adaptive_candidates(gray, config, nominal_diameter_px);
    let mut best_result: Option<DetectionResult> = None;
    let mut best_score: Option<AdaptiveCandidateScore> = None;
    let mut best_label = "<none>";

    for candidate in candidates {
        let result = detect_multiscale(gray, config, &candidate.tiers);
        let score = AdaptiveCandidateScore::from_result(&result);

        let should_replace = best_score
            .as_ref()
            .is_none_or(|best| score.is_better_than(best));

        if should_replace {
            best_label = candidate.label;
            best_score = Some(score);
            best_result = Some(result);
        }
    }

    tracing::debug!(
        selected = best_label,
        mapped = best_score.as_ref().map(|s| s.mapped),
        decoded = best_score.as_ref().map(|s| s.decoded),
        has_ransac = best_score.as_ref().map(|s| s.has_ransac),
        inliers = best_score.as_ref().map(|s| s.ransac_inliers),
        "adaptive candidate selected"
    );

    best_result.expect("adaptive candidate list must not be empty")
}

/// Adaptive detection using automatically selected scale tiers.
///
/// Equivalent to [`detect_adaptive_with_hint`] with `nominal_diameter_px=None`.
pub fn detect_adaptive(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    detect_adaptive_candidates(gray, config, None)
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
    detect_adaptive_candidates(gray, config, nominal_diameter_px)
}

/// Detect markers, then estimate and optionally apply a self-undistort model.
///
/// Runs a baseline pass first. If `config.self_undistort.enable` is true and
/// enough markers with edge points are available, estimates a division-model
/// mapper and re-runs pass-2 with seeded proposals.
pub fn detect_with_self_undistort(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let total_start = Instant::now();

    let pass1_start = Instant::now();
    let mut result = detect_single_pass(gray, config);
    let pass1_elapsed = pass1_start.elapsed();
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
        let pass2_start = Instant::now();
        result = run_pass2(gray, config, &model, &result);
        tracing::info!(
            pass1_ms = duration_ms(pass1_elapsed),
            pass2_ms = duration_ms(pass2_start.elapsed()),
            total_ms = duration_ms(total_start.elapsed()),
            markers_final = result.detected_markers.len(),
            "detect_with_self_undistort timing summary"
        );
    }

    result.self_undistort = Some(su_result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score_from(
        mapped: usize,
        decoded: usize,
        has_ransac: bool,
        inliers: usize,
        mean_err_px: f64,
    ) -> AdaptiveCandidateScore {
        let mut result = DetectionResult {
            detected_markers: (0..decoded)
                .map(|i| {
                    let mut marker = crate::DetectedMarker {
                        id: Some(i),
                        confidence: 0.8,
                        ..crate::DetectedMarker::default()
                    };
                    if i < mapped {
                        marker.board_xy_mm = Some([0.0, 0.0]);
                    }
                    marker
                })
                .collect(),
            ..DetectionResult::default()
        };

        if has_ransac {
            result.ransac = Some(crate::homography::RansacStats {
                n_candidates: decoded,
                n_inliers: inliers,
                threshold_px: 4.0,
                mean_err_px,
                p95_err_px: mean_err_px,
            });
        }

        AdaptiveCandidateScore::from_result(&result)
    }

    #[test]
    fn adaptive_score_prefers_mapped_over_all_other_axes() {
        let a = score_from(12, 12, false, 0, 0.0);
        let b = score_from(10, 20, true, 20, 0.1);
        assert!(a.is_better_than(&b));
    }

    #[test]
    fn adaptive_score_prefers_ransac_when_mapped_ties() {
        let a = score_from(12, 12, true, 10, 0.6);
        let b = score_from(12, 12, false, 0, 0.0);
        assert!(a.is_better_than(&b));
    }

    #[test]
    fn proposal_rescaling_uses_actual_resize_ratios() {
        let proposals = vec![Proposal {
            x: 24.0,
            y: 23.0,
            score: 7.0,
        }];

        let scaled = rescale_proposals_to_image_space(proposals, 101.0 / 25.0, 98.0 / 24.0);
        let best = scaled[0];

        assert!((best.x - 96.96).abs() < 1.0e-5);
        assert!((best.y - (23.0 * 98.0 / 24.0)).abs() < 1.0e-5);
        assert_ne!(best.x.to_bits(), (24.0f32 * 4.0f32).to_bits());
        assert_ne!(best.y.to_bits(), (23.0f32 * 4.0f32).to_bits());
    }
}
