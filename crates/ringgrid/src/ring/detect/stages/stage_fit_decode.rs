use super::super::marker_build::{
    decode_metrics_from_result, fit_metrics_with_inner, inner_ellipse_params, marker_with_defaults,
};
use super::super::*;

use crate::debug_dump as dbg;
use crate::ring::edge_sample::EdgeSampleResult;

pub(super) struct FitDecodeCoreOutput {
    pub(super) markers: Vec<DetectedMarker>,
    pub(super) marker_cand_idx: Vec<usize>,
    pub(super) stage0: Option<dbg::StageDebug>,
    pub(super) stage1: Option<dbg::StageDebug>,
    pub(super) stage2: Option<dbg::DedupDebug>,
}

fn edge_for_debug(edge: &EdgeSampleResult, store_points: bool) -> EdgeSampleResult {
    if store_points {
        return edge.clone();
    }
    let mut out = edge.clone();
    out.outer_points.clear();
    out.inner_points.clear();
    out
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
    let mut stage0 = debug_cfg.map(|_| dbg::StageDebug {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    });
    if let Some(stage0) = stage0.as_mut() {
        for (i, p) in proposals.iter().take(n_rec).enumerate() {
            stage0.candidates.push(dbg::CandidateDebug {
                cand_idx: i,
                proposal: *p,
                ring_fit: None,
                decode: None,
                decision: dbg::DecisionDebug {
                    status: dbg::DecisionStatus::Accepted,
                    reason: "proposal".to_string(),
                },
                derived: dbg::DerivedDebug {
                    id: None,
                    confidence: None,
                    center_xy: None,
                },
            });
        }
    }

    let mut stage1 = debug_cfg.map(|_| dbg::StageDebug {
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
            Some(dbg::CandidateDebug {
                cand_idx: i,
                proposal: *proposal,
                ring_fit: None,
                decode: None,
                decision: dbg::DecisionDebug {
                    status: dbg::DecisionStatus::Rejected,
                    reason: "unprocessed".to_string(),
                },
                derived: dbg::DerivedDebug {
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
                    cd.decision = dbg::DecisionDebug {
                        status: dbg::DecisionStatus::Rejected,
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
            Some(crate::EllipseParams::from(outer)),
            inner_params.clone(),
            Some(edge.outer_points.clone()),
            Some(inner_fit.points_inner.clone()),
            fit_metrics.clone(),
            decode_metrics.clone(),
        );

        markers.push(marker);
        marker_cand_idx.push(i);

        if let Some(cd) = cand_debug.as_mut() {
            cd.ring_fit = Some(dbg::RingFitDebug {
                center_xy_fit: center,
                edge: edge_for_debug(&edge, store_points),
                outer_estimation: Some(outer_estimate.clone()),
                chosen_outer_hypothesis: Some(chosen_hypothesis),
                ellipse_outer: Some(crate::EllipseParams::from(outer)),
                ellipse_inner: inner_params.clone(),
                inner_estimation: Some(inner_fit.estimate.clone()),
                fit: fit_metrics.clone(),
                inner_points_fit: if store_points {
                    Some(inner_fit.points_inner.clone())
                } else {
                    None
                },
            });

            cd.decode = Some(dbg::DecodeDebug {
                diagnostics: decode_diag.clone(),
                result: decode_result.clone(),
                decode_metrics: decode_metrics.clone(),
            });

            let derived_id = decode_result.as_ref().map(|d| d.id);
            cd.decision = dbg::DecisionDebug {
                status: dbg::DecisionStatus::Accepted,
                reason: if let Some(r) = decode_diag.reject_reason.as_ref() {
                    format!("ok_with_decode_reject:{}", r)
                } else {
                    "ok".to_string()
                },
            };
            cd.derived = dbg::DerivedDebug {
                id: derived_id,
                confidence: Some(confidence),
                center_xy: Some(center),
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
                            cd.derived.center_xy = Some(marker.center);
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
