use super::super::debug_conv;
use super::super::marker_build::{
    decode_metrics_from_result, fit_metrics_with_inner, inner_ellipse_params, marker_with_defaults,
};
use super::super::*;

use crate::debug_dump as dbg;

pub(super) struct FitDecodeCoreOutput {
    pub(super) markers: Vec<DetectedMarker>,
    pub(super) marker_cand_idx: Vec<usize>,
    pub(super) stage0: Option<dbg::StageDebugV1>,
    pub(super) stage1: Option<dbg::StageDebugV1>,
    pub(super) stage2: Option<dbg::DedupDebugV1>,
}

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
    debug_cfg: Option<&DebugCollectConfig>,
) -> FitDecodeCoreOutput {
    let proposals = find_proposals_with_seeds(gray, &config.proposal, seed_centers_image, seed_cfg);
    tracing::info!("{} proposals found", proposals.len());

    let use_projective_center =
        config.circle_refinement.uses_projective_center() && config.projective_center.enable;
    let inner_fit_cfg = inner_fit::InnerFitConfig::default();
    let sampler = crate::ring::edge_sample::DistortionAwareSampler::new(gray, mapper);

    let n_rec = debug_cfg
        .map(|cfg| proposals.len().min(cfg.max_candidates))
        .unwrap_or(0);
    let mut stage0 = debug_cfg.map(|_| dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    });
    if let Some(stage0) = stage0.as_mut() {
        for (i, p) in proposals.iter().take(n_rec).enumerate() {
            stage0.candidates.push(dbg::CandidateDebugV1 {
                cand_idx: i,
                proposal: dbg::ProposalDebugV1 {
                    center_xy: [p.x, p.y],
                    score: p.score,
                },
                ring_fit: None,
                decode: None,
                decision: dbg::DecisionDebugV1 {
                    status: dbg::DecisionStatusV1::Accepted,
                    reason: "proposal".to_string(),
                },
                derived: dbg::DerivedDebugV1 {
                    id: None,
                    confidence: None,
                    center_xy: None,
                },
            });
        }
    }

    let mut stage1 = debug_cfg.map(|_| dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    });

    let mut markers: Vec<DetectedMarker> = Vec::new();
    let mut marker_cand_idx: Vec<usize> = Vec::new();

    for (i, proposal) in proposals.iter().enumerate() {
        let center_prior = match sampler.image_to_working_xy([proposal.x, proposal.y]) {
            Some(v) => v,
            None => continue,
        };

        let mut cand_debug = if debug_cfg.is_some() && i < n_rec {
            Some(dbg::CandidateDebugV1 {
                cand_idx: i,
                proposal: dbg::ProposalDebugV1 {
                    center_xy: [proposal.x, proposal.y],
                    score: proposal.score,
                },
                ring_fit: None,
                decode: None,
                decision: dbg::DecisionDebugV1 {
                    status: dbg::DecisionStatusV1::Rejected,
                    reason: "unprocessed".to_string(),
                },
                derived: dbg::DerivedDebugV1 {
                    id: None,
                    confidence: None,
                    center_xy: None,
                },
            })
        } else {
            None
        };

        let store_points = debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false);
        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            center_prior,
            marker_outer_radius_expected_px(config),
            config,
            mapper,
            &config.edge_sample,
            store_points,
        ) {
            Ok(v) => v,
            Err(reason) => {
                if let Some(cd) = cand_debug.as_mut() {
                    cd.decision = dbg::DecisionDebugV1 {
                        status: dbg::DecisionStatusV1::Rejected,
                        reason,
                    };
                }
                if let (Some(stage1), Some(cd)) = (stage1.as_mut(), cand_debug) {
                    stage1.candidates.push(cd);
                }
                continue;
            }
        };

        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            outer_estimate,
            chosen_hypothesis,
            decode_result,
            decode_diag,
            ..
        } = fit;

        let center = compute_center(&outer);
        let inner_fit = inner_fit::fit_inner_ellipse_from_outer_hint(
            gray,
            &outer,
            &config.marker_spec,
            mapper,
            &inner_fit_cfg,
            store_points,
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
        let marker = marker_with_defaults(
            decode_result.as_ref().map(|d| d.id),
            confidence,
            center,
            Some(crate::EllipseParams::from(&outer)),
            inner_params.clone(),
            fit_metrics.clone(),
            decode_metrics,
        );

        markers.push(marker);
        marker_cand_idx.push(i);

        if let Some(cd) = cand_debug.as_mut() {
            let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
            cd.ring_fit = Some(dbg::RingFitDebugV1 {
                center_xy_fit: [center[0] as f32, center[1] as f32],
                edges: dbg::RingEdgesDebugV1 {
                    n_angles_total: edge.n_total_rays,
                    n_angles_with_both: edge.n_good_rays,
                    inner_peak_r: if edge.inner_radii.is_empty() {
                        None
                    } else {
                        Some(edge.inner_radii.clone())
                    },
                    outer_peak_r: Some(edge.outer_radii.clone()),
                },
                outer_estimation: Some(debug_conv::outer_estimation_debug(
                    &outer_estimate,
                    chosen_hypothesis,
                )),
                ellipse_outer: Some(debug_conv::ellipse_from_conic(&outer)),
                ellipse_inner: inner_params.as_ref().map(debug_conv::ellipse_from_params),
                inner_estimation: Some(debug_conv::inner_estimation_debug(
                    &inner_fit.estimate,
                )),
                metrics: debug_conv::ring_fit_metrics(
                    &fit_metrics,
                    arc_cov,
                    inner_params.is_some(),
                    true,
                ),
                points_outer: if store_points {
                    Some(
                        edge.outer_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                points_inner: if store_points {
                    Some(
                        inner_fit
                            .points_inner
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
            });

            cd.decode = Some(dbg::DecodeDebugV1 {
                sector_means: decode_diag.sector_intensities,
                threshold: decode_diag.threshold,
                observed_word_hex: format!("0x{:04X}", decode_diag.used_word),
                inverted_used: decode_diag.inverted_used,
                r#match: dbg::DecodeMatchDebugV1 {
                    best_id: decode_diag.best_id,
                    best_rotation: decode_diag.best_rotation,
                    best_dist: decode_diag.best_dist,
                    margin: decode_diag.margin,
                    decode_confidence: decode_diag.decode_confidence,
                },
                accepted: Some(decode_result.is_some()),
                reject_reason: decode_diag.reject_reason.clone(),
            });

            let derived_id = decode_result.as_ref().map(|d| d.id);
            cd.decision = dbg::DecisionDebugV1 {
                status: dbg::DecisionStatusV1::Accepted,
                reason: if let Some(r) = decode_diag.reject_reason {
                    format!("ok_with_decode_reject:{}", r)
                } else {
                    "ok".to_string()
                },
            };
            cd.derived = dbg::DerivedDebugV1 {
                id: derived_id,
                confidence: Some(confidence),
                center_xy: Some([center[0] as f32, center[1] as f32]),
            };
        }

        if let (Some(stage1), Some(cd)) = (stage1.as_mut(), cand_debug) {
            stage1.candidates.push(cd);
        }
    }

    if use_projective_center {
        apply_projective_centers(&mut markers, config);
        if debug_cfg.is_some() {
            for (marker, &cand_idx) in markers.iter().zip(marker_cand_idx.iter()) {
                if cand_idx < n_rec {
                    if let Some(stage1) = stage1.as_mut() {
                        if let Some(cd) = stage1.candidates.get_mut(cand_idx) {
                            cd.derived.center_xy =
                                Some([marker.center[0] as f32, marker.center[1] as f32]);
                        }
                    }
                }
            }
        }
    }

    let (markers, marker_cand_idx, stage2) = if debug_cfg.is_some() {
        let (m, idx, d) = dedup_with_debug(markers, marker_cand_idx, config.dedup_radius);
        (m, idx, Some(d))
    } else {
        let mut m = dedup_markers(markers, config.dedup_radius);
        dedup_by_id(&mut m);
        (m, Vec::new(), None)
    };

    tracing::info!("{} markers detected after dedup", markers.len());

    FitDecodeCoreOutput {
        markers,
        marker_cand_idx,
        stage0,
        stage1,
        stage2,
    }
}
