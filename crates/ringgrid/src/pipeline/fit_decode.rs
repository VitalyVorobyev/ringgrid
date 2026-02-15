use image::GrayImage;
use std::cmp::Ordering;
use std::collections::HashMap;

use super::inner_fit;
use super::marker_build::{
    decode_metrics_from_result, fit_metrics_with_inner, inner_ellipse_params, marker_with_defaults,
};
use super::outer_fit::{
    compute_center, fit_outer_ellipse_robust_with_reason, marker_outer_radius_expected_px,
    OuterFitCandidate,
};
use super::{dedup_by_id, dedup_markers, DetectConfig};
use crate::detector::proposal::Proposal;
use crate::detector::DetectedMarker;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;

struct CandidateProcessContext<'a> {
    gray: &'a GrayImage,
    config: &'a DetectConfig,
    mapper: Option<&'a dyn PixelMapper>,
    sampler: DistortionAwareSampler<'a>,
}

fn fallback_fit_confidence(
    edge: &crate::ring::edge_sample::EdgeSampleResult,
    outer_ransac: Option<&crate::conic::RansacResult>,
) -> f32 {
    let arc_cov = edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32;
    let inlier_ratio = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0);
    (arc_cov * inlier_ratio).clamp(0.0, 1.0)
}

fn select_proposals_for_fit(
    mut proposals: Vec<Proposal>,
    max_candidates: Option<usize>,
) -> Vec<Proposal> {
    let Some(max_candidates) = max_candidates else {
        return proposals;
    };
    if proposals.len() <= max_candidates {
        return proposals;
    }

    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    proposals.truncate(max_candidates);
    proposals
}

fn process_candidate(
    proposal: Proposal,
    ctx: &CandidateProcessContext<'_>,
) -> Result<DetectedMarker, String> {
    let Some(center_prior) = ctx.sampler.image_to_working_xy([proposal.x, proposal.y]) else {
        return Err("proposal_unmappable".to_string());
    };

    let fit = match fit_outer_ellipse_robust_with_reason(
        ctx.gray,
        center_prior,
        marker_outer_radius_expected_px(ctx.config),
        ctx.config,
        ctx.mapper,
        false,
        false
    ) {
        Ok(v) => v,
        Err(reason) => return Err(format!("outer_fit:{reason}")),
    };

    let OuterFitCandidate {
        edge,
        outer,
        outer_ransac,
        decode_result,
        ..
    } = fit;

    let center = compute_center(&outer);
    let inner_fit = inner_fit::fit_inner_ellipse_from_outer_hint(
        ctx.gray,
        &outer,
        &ctx.config.marker_spec,
        ctx.mapper,
        &ctx.config.inner_fit,
        false,
    );
    if inner_fit.status != inner_fit::InnerFitStatus::Ok {
        tracing::trace!(
            "inner fit rejected/failed at proposal ({:.1},{:.1}): status={:?}, reason={:?}",
            proposal.x,
            proposal.y,
            inner_fit.status,
            inner_fit.reason
        );
    }
    let inner_params = inner_ellipse_params(&inner_fit);

    let fit_metrics = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner_fit);
    let confidence = decode_result
        .as_ref()
        .map(|d| d.confidence)
        .unwrap_or_else(|| fallback_fit_confidence(&edge, outer_ransac.as_ref()));
    let decode_metrics = decode_metrics_from_result(decode_result.as_ref());
    let marker_id = decode_result.as_ref().map(|d| d.id);
    let outer_points = edge.outer_points;
    let inner_points = inner_fit.points_inner;

    Ok(marker_with_defaults(
        marker_id,
        confidence,
        center,
        Some(outer),
        inner_params,
        Some(outer_points),
        Some(inner_points),
        fit_metrics,
        decode_metrics,
    ))
}

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    proposals: Vec<Proposal>,
) -> Vec<DetectedMarker> {
    let input_count = proposals.len();
    tracing::info!("{} proposals found", input_count);
    let proposals = select_proposals_for_fit(proposals, config.proposal.max_candidates);
    if proposals.len() != input_count {
        tracing::info!(
            "proposal cap active: evaluating {} / {} proposals",
            proposals.len(),
            input_count
        );
    }

    let sampler = DistortionAwareSampler::new(gray, mapper);

    let ctx = CandidateProcessContext {
        gray,
        config,
        mapper,
        sampler,
    };

    let mut markers: Vec<DetectedMarker> = Vec::new();
    let mut reject_reasons: HashMap<String, usize> = HashMap::new();
    for proposal in proposals {
        match process_candidate(proposal, &ctx) {
            Ok(marker) => markers.push(marker),
            Err(reason) => *reject_reasons.entry(reason).or_insert(0) += 1,
        }
    }
    if !reject_reasons.is_empty() {
        let mut summary: Vec<(String, usize)> = reject_reasons.into_iter().collect();
        summary.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let rejected_total: usize = summary.iter().map(|(_, n)| *n).sum();
        let top = summary
            .iter()
            .take(4)
            .map(|(reason, n)| format!("{reason}={n}"))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::debug!(
            "fit/decode rejected {} proposals (top reasons: {})",
            rejected_total,
            top
        );
    }

    markers = dedup_markers(markers, config.dedup_radius);
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());

    markers
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn draw_ring_image(
        w: u32,
        h: u32,
        center: [f32; 2],
        outer_radius: f32,
        inner_radius: f32,
    ) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let d = (dx * dx + dy * dy).sqrt();
                let pix = if d >= inner_radius && d <= outer_radius {
                    24u8
                } else {
                    230u8
                };
                img.put_pixel(x, y, Luma([pix]));
            }
        }
        img
    }

    fn nearest_marker(markers: &[DetectedMarker], center: [f64; 2]) -> Option<&DetectedMarker> {
        markers.iter().min_by(|a, b| {
            let da = (a.center[0] - center[0]) * (a.center[0] - center[0])
                + (a.center[1] - center[1]) * (a.center[1] - center[1]);
            let db = (b.center[0] - center[0]) * (b.center[0] - center[0])
                + (b.center[1] - center[1]) * (b.center[1] - center[1]);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        })
    }

    #[test]
    fn fit_decode_honors_inner_fit_config() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 11.75f32; // ratio ≈ 0.49, aligned with default target geometry prior
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);
        let proposals = vec![
            Proposal {
                x: center[0],
                y: center[1],
                score: 10.0,
            },
            Proposal {
                x: center[0] + 1.0,
                y: center[1],
                score: 9.0,
            },
            Proposal {
                x: center[0],
                y: center[1] + 1.0,
                score: 8.0,
            },
        ];

        let mut relaxed = DetectConfig::default();
        relaxed.set_marker_diameter_hint_px(outer_radius * 2.0);
        relaxed.inner_fit.min_points = 1;
        relaxed.inner_fit.min_inlier_ratio = 0.0;
        relaxed.inner_fit.max_rms_residual = f64::INFINITY;
        relaxed.inner_fit.max_center_shift_px = f64::INFINITY;
        relaxed.inner_fit.max_ratio_abs_error = f64::INFINITY;

        let relaxed_out = run(&img, &relaxed, None, proposals.clone());
        assert!(
            !relaxed_out.is_empty(),
            "expected at least one marker with relaxed inner-fit params"
        );
        let relaxed_marker = nearest_marker(&relaxed_out, [center[0] as f64, center[1] as f64])
            .expect("nearest marker");
        assert!(
            relaxed_marker.ellipse_inner.is_some(),
            "expected inner ellipse with relaxed inner-fit params"
        );

        let mut strict = relaxed.clone();
        strict.inner_fit.min_points = usize::MAX;

        let strict_out = run(&img, &strict, None, proposals);
        assert!(
            !strict_out.is_empty(),
            "expected marker to remain present when inner-fit is disabled by strict gate"
        );
        let strict_marker = nearest_marker(&strict_out, [center[0] as f64, center[1] as f64])
            .expect("nearest marker");
        assert!(
            strict_marker.ellipse_inner.is_none(),
            "expected no inner ellipse when min_points gate is impossible"
        );
    }

    #[test]
    fn fit_decode_respects_proposal_cap() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 11.75f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);
        let proposals = vec![Proposal {
            x: center[0],
            y: center[1],
            score: 10.0,
        }];

        let mut cfg = DetectConfig::default();
        cfg.set_marker_diameter_hint_px(outer_radius * 2.0);
        cfg.proposal.max_candidates = Some(0);

        let out = run(&img, &cfg, None, proposals);
        assert!(
            out.is_empty(),
            "expected no markers when proposal cap is zero"
        );
    }
}
