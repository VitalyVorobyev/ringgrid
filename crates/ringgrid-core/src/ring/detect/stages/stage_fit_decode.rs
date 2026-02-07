#[cfg(feature = "debug-trace")]
use super::super::debug_conv;
use super::super::marker_build::{
    decode_metrics_from_result, fit_metrics_from_outer, marker_with_defaults,
};
use super::super::*;

#[cfg(feature = "debug-trace")]
use crate::debug_dump as dbg;
#[cfg(feature = "debug-trace")]
use crate::ring::inner_estimate::{InnerStatus, Polarity};
#[cfg(feature = "debug-trace")]
use crate::ring::outer_estimate::OuterStatus;

struct FitDecodeCoreOutput {
    markers: Vec<DetectedMarker>,
    #[cfg(feature = "debug-trace")]
    marker_cand_idx: Vec<usize>,
    #[cfg(feature = "debug-trace")]
    stage0: Option<dbg::StageDebugV1>,
    #[cfg(feature = "debug-trace")]
    stage1: Option<dbg::StageDebugV1>,
    #[cfg(feature = "debug-trace")]
    stage2: Option<dbg::DedupDebugV1>,
}

#[cfg(feature = "debug-trace")]
pub(super) struct FitDecodeDebugOutput {
    pub(super) markers: Vec<DetectedMarker>,
    pub(super) marker_cand_idx: Vec<usize>,
    pub(super) stage0: dbg::StageDebugV1,
    pub(super) stage1: dbg::StageDebugV1,
    pub(super) stage2: dbg::DedupDebugV1,
}

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> Vec<DetectedMarker> {
    #[cfg(feature = "debug-trace")]
    let out = run_core(gray, config, mapper, seed_centers_image, seed_cfg, None);
    #[cfg(not(feature = "debug-trace"))]
    let out = run_core(gray, config, mapper, seed_centers_image, seed_cfg);
    out.markers
}

#[cfg(feature = "debug-trace")]
pub(super) fn run_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
    debug_cfg: &DebugCollectConfig,
) -> FitDecodeDebugOutput {
    let out = run_core(
        gray,
        config,
        mapper,
        seed_centers_image,
        seed_cfg,
        Some(debug_cfg),
    );
    FitDecodeDebugOutput {
        markers: out.markers,
        marker_cand_idx: out.marker_cand_idx,
        stage0: out.stage0.expect("stage0 debug should be present"),
        stage1: out.stage1.expect("stage1 debug should be present"),
        stage2: out.stage2.expect("stage2 debug should be present"),
    }
}

fn run_core(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
    #[cfg(feature = "debug-trace")] debug_cfg: Option<&DebugCollectConfig>,
) -> FitDecodeCoreOutput {
    let proposals = find_proposals_with_seeds(gray, &config.proposal, seed_centers_image, seed_cfg);
    tracing::info!("{} proposals found", proposals.len());

    let use_projective_center =
        config.circle_refinement.uses_projective_center() && config.projective_center.enable;
    let inner_fit_cfg = inner_fit::InnerFitConfig::default();
    let sampler = crate::ring::edge_sample::DistortionAwareSampler::new(gray, mapper);

    #[cfg(feature = "debug-trace")]
    let n_rec = debug_cfg
        .map(|cfg| proposals.len().min(cfg.max_candidates))
        .unwrap_or(0);
    #[cfg(feature = "debug-trace")]
    let mut stage0 = debug_cfg.map(|_| dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    });
    #[cfg(feature = "debug-trace")]
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

    #[cfg(feature = "debug-trace")]
    let mut stage1 = debug_cfg.map(|_| dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    });

    let mut markers: Vec<DetectedMarker> = Vec::new();
    #[cfg(feature = "debug-trace")]
    let mut marker_cand_idx: Vec<usize> = Vec::new();

    for (i, proposal) in proposals.iter().enumerate() {
        #[cfg(not(feature = "debug-trace"))]
        let _ = i;
        let center_prior = match sampler.image_to_working_xy([proposal.x, proposal.y]) {
            Some(v) => v,
            None => continue,
        };

        #[cfg(feature = "debug-trace")]
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

        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            center_prior,
            marker_outer_radius_expected_px(config),
            config,
            mapper,
            &config.edge_sample,
            #[cfg(feature = "debug-trace")]
            debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false),
            #[cfg(not(feature = "debug-trace"))]
            false,
        ) {
            Ok(v) => v,
            Err(_reason) => {
                #[cfg(feature = "debug-trace")]
                if let Some(cd) = cand_debug.as_mut() {
                    cd.decision = dbg::DecisionDebugV1 {
                        status: dbg::DecisionStatusV1::Rejected,
                        reason: _reason,
                    };
                }
                #[cfg(feature = "debug-trace")]
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
            #[cfg(feature = "debug-trace")]
            outer_estimate,
            #[cfg(feature = "debug-trace")]
            chosen_hypothesis,
            decode_result,
            #[cfg(feature = "debug-trace")]
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
            #[cfg(feature = "debug-trace")]
            debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false),
            #[cfg(not(feature = "debug-trace"))]
            false,
        );
        if inner_fit.status != inner_fit::InnerFitStatus::Ok {
            tracing::trace!(
                "inner fit rejected/failed: status={:?}, reason={:?}",
                inner_fit.status,
                inner_fit.reason
            );
        }
        let inner_params = inner_fit.ellipse_inner.as_ref().map(crate::EllipseParams::from);

        let fit_metrics = fit_metrics_from_outer(
            &edge,
            &outer,
            outer_ransac.as_ref(),
            inner_fit.points_inner.len(),
            inner_fit.ransac_inlier_ratio_inner,
            inner_fit.rms_residual_inner,
        );

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
        #[cfg(feature = "debug-trace")]
        marker_cand_idx.push(i);

        #[cfg(feature = "debug-trace")]
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
                outer_estimation: Some({
                    let chosen = outer_estimate.hypotheses.get(chosen_hypothesis);
                    dbg::OuterEstimationDebugV1 {
                        r_outer_expected_px: outer_estimate.r_outer_expected_px,
                        search_window_px: outer_estimate.search_window_px,
                        r_outer_found_px: chosen.map(|h| h.r_outer_px),
                        polarity: outer_estimate.polarity.map(|p| match p {
                            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                        }),
                        peak_strength: chosen.map(|h| h.peak_strength),
                        theta_consistency: chosen.map(|h| h.theta_consistency),
                        status: match outer_estimate.status {
                            OuterStatus::Ok => dbg::OuterEstimationStatusDebugV1::Ok,
                            OuterStatus::Rejected => dbg::OuterEstimationStatusDebugV1::Rejected,
                            OuterStatus::Failed => dbg::OuterEstimationStatusDebugV1::Failed,
                        },
                        reason: outer_estimate.reason.clone(),
                        hypotheses: outer_estimate
                            .hypotheses
                            .iter()
                            .map(|h| dbg::OuterHypothesisDebugV1 {
                                r_outer_px: h.r_outer_px,
                                peak_strength: h.peak_strength,
                                theta_consistency: h.theta_consistency,
                            })
                            .collect(),
                        chosen_hypothesis: Some(chosen_hypothesis),
                        radial_response_agg: outer_estimate.radial_response_agg.clone(),
                        r_samples: outer_estimate.r_samples.clone(),
                    }
                }),
                ellipse_outer: Some(debug_conv::ellipse_from_conic(&outer)),
                ellipse_inner: inner_params.as_ref().map(debug_conv::ellipse_from_params),
                inner_estimation: Some(dbg::InnerEstimationDebugV1 {
                    r_inner_expected: inner_fit.estimate.r_inner_expected,
                    search_window: inner_fit.estimate.search_window,
                    r_inner_found: inner_fit.estimate.r_inner_found,
                    polarity: inner_fit.estimate.polarity.map(|p| match p {
                        Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                        Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                    }),
                    peak_strength: inner_fit.estimate.peak_strength,
                    theta_consistency: inner_fit.estimate.theta_consistency,
                    status: match inner_fit.estimate.status {
                        InnerStatus::Ok => dbg::InnerEstimationStatusDebugV1::Ok,
                        InnerStatus::Rejected => dbg::InnerEstimationStatusDebugV1::Rejected,
                        InnerStatus::Failed => dbg::InnerEstimationStatusDebugV1::Failed,
                    },
                    reason: inner_fit.estimate.reason.clone(),
                    radial_response_agg: inner_fit.estimate.radial_response_agg.clone(),
                    r_samples: inner_fit.estimate.r_samples.clone(),
                }),
                metrics: debug_conv::ring_fit_metrics(
                    &fit_metrics,
                    arc_cov,
                    inner_params.is_some(),
                    true,
                ),
                points_outer: if debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false) {
                    Some(
                        edge.outer_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                points_inner: if debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false) {
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

        #[cfg(feature = "debug-trace")]
        if let (Some(stage1), Some(cd)) = (stage1.as_mut(), cand_debug) {
            stage1.candidates.push(cd);
        }
    }

    if use_projective_center {
        apply_projective_centers(&mut markers, config);
        #[cfg(feature = "debug-trace")]
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

    #[cfg(feature = "debug-trace")]
    let (markers, marker_cand_idx, stage2) = if debug_cfg.is_some() {
        let (m, idx, d) = dedup_with_debug(markers, marker_cand_idx, config.dedup_radius);
        (m, idx, Some(d))
    } else {
        let mut m = dedup_markers(markers, config.dedup_radius);
        dedup_by_id(&mut m);
        (m, Vec::new(), None)
    };
    #[cfg(not(feature = "debug-trace"))]
    let markers = {
        let mut m = dedup_markers(markers, config.dedup_radius);
        dedup_by_id(&mut m);
        m
    };

    tracing::info!("{} markers detected after dedup", markers.len());

    FitDecodeCoreOutput {
        markers,
        #[cfg(feature = "debug-trace")]
        marker_cand_idx,
        #[cfg(feature = "debug-trace")]
        stage0,
        #[cfg(feature = "debug-trace")]
        stage1,
        #[cfg(feature = "debug-trace")]
        stage2,
    }
}
