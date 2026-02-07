use super::super::marker_build::{
    decode_metrics_from_result, fit_metrics_from_outer, inner_params_from_estimate,
    marker_with_defaults,
};
use super::super::*;

pub(super) fn run(gray: &GrayImage, config: &DetectConfig) -> Vec<DetectedMarker> {
    // Stage 1: Find candidate centers
    let proposals = find_proposals(gray, &config.proposal);
    tracing::info!("{} proposals found", proposals.len());
    let use_projective_center =
        config.circle_refinement.uses_projective_center() && config.projective_center.enable;

    // Stages 2-5: For each proposal, sample edges → fit → decode
    let mut markers: Vec<DetectedMarker> = Vec::new();

    for proposal in &proposals {
        // Stage 2-4: Robust outer edge extraction → outer fit → decode
        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            [proposal.x, proposal.y],
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

        // Compute fit metrics
        let fit = fit_metrics_from_outer(&edge, &outer, outer_ransac.as_ref());

        // Stage 4b: Inner edge estimation for every accepted outer fit.
        let est = estimate_inner_scale_from_outer(gray, &outer, &config.marker_spec, false);
        let inner_params = inner_params_from_estimate(
            &outer,
            est.status,
            est.r_inner_found,
            config.marker_spec.r_inner_expected,
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
