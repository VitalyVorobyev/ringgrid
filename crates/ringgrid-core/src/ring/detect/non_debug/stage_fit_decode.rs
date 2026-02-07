use super::super::*;

pub(super) fn run(gray: &GrayImage, config: &DetectConfig) -> Vec<DetectedMarker> {
    // Stage 1: Find candidate centers
    let proposals = find_proposals(gray, &config.proposal);
    tracing::info!("{} proposals found", proposals.len());

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
        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: 0,
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: None,
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: None,
        };

        // Stage 4b: Inner edge estimation for every accepted outer fit.
        let est = estimate_inner_scale_from_outer(gray, &outer, &config.marker_spec, false);
        let inner_params = if est.status == InnerStatus::Ok {
            let s = est
                .r_inner_found
                .unwrap_or(config.marker_spec.r_inner_expected) as f64;
            Some(EllipseParams {
                center_xy: [outer.cx, outer.cy],
                semi_axes: [outer.a * s, outer.b * s],
                angle: outer.angle,
            })
        } else {
            None
        };

        // Build marker
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);

        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let marker = DetectedMarker {
            id: decode_result.as_ref().map(|d| d.id),
            confidence,
            center,
            center_projective: None,
            vanishing_line: None,
            center_projective_residual: None,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params,
            fit,
            decode: decode_metrics,
        };

        markers.push(marker);
    }

    // Stage 5: Dedup by center proximity
    markers = dedup_markers(markers, config.dedup_radius);

    // Stage 5b: Dedup by ID — keep best confidence per decoded ID
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());
    markers
}
