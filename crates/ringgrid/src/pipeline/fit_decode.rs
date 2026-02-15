use image::GrayImage;

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

pub(super) struct FitDecodeCoreOutput {
    pub(super) markers: Vec<DetectedMarker>,
}

struct CandidateProcessContext<'a> {
    gray: &'a GrayImage,
    config: &'a DetectConfig,
    mapper: Option<&'a dyn PixelMapper>,
    sampler: DistortionAwareSampler<'a>,
    inner_fit_cfg: &'a inner_fit::InnerFitConfig,
}

fn process_candidate(
    proposal: Proposal,
    ctx: &CandidateProcessContext<'_>,
) -> Option<DetectedMarker> {
    let center_prior = ctx.sampler.image_to_working_xy([proposal.x, proposal.y])?;

    let fit = match fit_outer_ellipse_robust_with_reason(
        ctx.gray,
        center_prior,
        marker_outer_radius_expected_px(ctx.config),
        ctx.config,
        ctx.mapper,
        &ctx.config.edge_sample,
        false,
    ) {
        Ok(v) => v,
        Err(_) => return None,
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
        ctx.inner_fit_cfg,
        false,
    );
    if inner_fit.status != inner_fit::InnerFitStatus::Ok {
        tracing::trace!(
            "inner fit rejected/failed: status={:?}, reason={:?}",
            inner_fit.status,
            inner_fit.reason
        );
    }
    let inner_params = inner_ellipse_params(&inner_fit);

    let fit_metrics = fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner_fit);
    let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
    let decode_metrics = decode_metrics_from_result(decode_result.as_ref());

    Some(marker_with_defaults(
        decode_result.as_ref().map(|d| d.id),
        confidence,
        center,
        Some(outer),
        inner_params,
        Some(edge.outer_points.clone()),
        Some(inner_fit.points_inner.clone()),
        fit_metrics,
        decode_metrics,
    ))
}

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    proposals: Vec<Proposal>,
) -> FitDecodeCoreOutput {
    tracing::info!("{} proposals found", proposals.len());

    let inner_fit_cfg = inner_fit::InnerFitConfig::default();
    let sampler = DistortionAwareSampler::new(gray, mapper);

    let ctx = CandidateProcessContext {
        gray,
        config,
        mapper,
        sampler,
        inner_fit_cfg: &inner_fit_cfg,
    };

    let mut markers: Vec<DetectedMarker> = proposals
        .iter()
        .copied()
        .filter_map(|proposal| process_candidate(proposal, &ctx))
        .collect();

    markers = dedup_markers(markers, config.dedup_radius);
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());

    FitDecodeCoreOutput { markers }
}
