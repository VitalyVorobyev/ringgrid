use super::super::marker_build::{
    decode_metrics_from_result, fit_metrics_from_outer, marker_with_defaults,
};
use super::super::*;

pub(super) fn run(gray: &GrayImage, config: &DetectConfig) -> Vec<DetectedMarker> {
    // Stage 1: Find candidate centers
    let proposals = find_proposals(gray, &config.proposal);
    tracing::info!("{} proposals found", proposals.len());
    let use_projective_center =
        config.circle_refinement.uses_projective_center() && config.projective_center.enable;
    let inner_fit_cfg = inner_fit::InnerFitConfig::default();
    let sampler =
        crate::ring::edge_sample::DistortionAwareSampler::new(gray, config.camera.as_ref());

    // Stages 2-5: For each proposal, sample edges → fit → decode
    let mut markers: Vec<DetectedMarker> = Vec::new();

    for proposal in &proposals {
        let center_prior = match sampler.image_to_working_xy([proposal.x, proposal.y]) {
            Some(v) => v,
            None => continue,
        };
        // Stage 2-4: Robust outer edge extraction → outer fit → decode
        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            center_prior,
            marker_outer_radius_expected_px(config),
            config,
            &config.edge_sample,
            false,
        ) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            decode_result,
            ..
        } = fit;

        // Compute center
        let center = compute_center(&outer);

        // Stage 4b: Robust inner fit for every accepted outer fit.
        let inner_fit = inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            config.camera.as_ref(),
            &inner_fit_cfg,
            false,
        );
        if inner_fit.status != inner_fit::InnerFitStatus::Ok {
            tracing::trace!(
                "inner fit rejected/failed: status={:?}, reason={:?}",
                inner_fit.status,
                inner_fit.reason
            );
        }
        let inner_params = inner_fit.ellipse_inner.as_ref().map(ellipse_to_params);

        // Compute fit metrics
        let fit = fit_metrics_from_outer(
            &edge,
            &outer,
            outer_ransac.as_ref(),
            inner_fit.points_inner.len(),
            inner_fit.ransac_inlier_ratio_inner,
            inner_fit.rms_residual_inner,
        );

        // Build marker
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);

        let decode_metrics = decode_metrics_from_result(decode_result.as_ref());

        let marker = marker_with_defaults(
            decode_result.as_ref().map(|d| d.id),
            confidence,
            center,
            Some(ellipse_to_params(&outer)),
            inner_params,
            fit,
            decode_metrics,
        );

        markers.push(marker);
    }

    // Stage 5: Promote projective centers so downstream stages use unbiased centers.
    if use_projective_center {
        apply_projective_centers(&mut markers, config);
    }

    // Stage 6: Dedup by center proximity
    markers = dedup_markers(markers, config.dedup_radius);

    // Stage 6b: Dedup by ID — keep best confidence per decoded ID
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());
    markers
}
