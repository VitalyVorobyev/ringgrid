use super::super::marker_build::{
    decode_metrics_from_result, fit_metrics_with_inner, inner_ellipse_params, marker_with_defaults,
};
use super::super::*;

use crate::debug_dump as dbg;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleResult};
use crate::ring::proposal::Proposal;

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

fn stage0_candidate(cand_idx: usize, proposal: Proposal) -> dbg::CandidateDebug {
    dbg::CandidateDebug {
        cand_idx,
        proposal,
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
    }
}

fn stage1_candidate_stub(cand_idx: usize, proposal: Proposal) -> dbg::CandidateDebug {
    dbg::CandidateDebug {
        cand_idx,
        proposal,
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
    }
}

struct DebugState {
    collect_debug: bool,
    store_points: bool,
    n_rec: usize,
    stage0: Option<dbg::StageDebug>,
    stage1: Option<dbg::StageDebug>,
    stage1_slot_by_cand_idx: Option<Vec<Option<usize>>>,
}

impl DebugState {
    fn new(proposals: &[Proposal], debug_cfg: Option<&DebugCollectConfig>) -> Self {
        let collect_debug = debug_cfg.is_some();
        let store_points = debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false);
        let n_rec = debug_cfg
            .map(|cfg| proposals.len().min(cfg.max_candidates))
            .unwrap_or(0);

        let stage0 = if collect_debug {
            let mut stage = dbg::StageDebug {
                n_total: proposals.len(),
                n_recorded: n_rec,
                candidates: Vec::with_capacity(n_rec),
                notes: Vec::new(),
            };
            for (i, p) in proposals.iter().take(n_rec).enumerate() {
                stage.candidates.push(stage0_candidate(i, *p));
            }
            Some(stage)
        } else {
            None
        };

        let stage1 = if collect_debug {
            Some(dbg::StageDebug {
                n_total: proposals.len(),
                n_recorded: n_rec,
                candidates: Vec::with_capacity(n_rec),
                notes: Vec::new(),
            })
        } else {
            None
        };

        let stage1_slot_by_cand_idx = if collect_debug {
            Some(vec![None; n_rec])
        } else {
            None
        };

        Self {
            collect_debug,
            store_points,
            n_rec,
            stage0,
            stage1,
            stage1_slot_by_cand_idx,
        }
    }

    fn should_record_candidate(&self, cand_idx: usize) -> bool {
        self.collect_debug && cand_idx < self.n_rec
    }

    fn record_stage1_candidate(&mut self, candidate: Option<dbg::CandidateDebug>) {
        let Some(candidate) = candidate else {
            return;
        };
        let Some(stage1) = self.stage1.as_mut() else {
            return;
        };

        let cand_idx = candidate.cand_idx;
        let slot = stage1.candidates.len();
        stage1.candidates.push(candidate);

        if let Some(map) = self.stage1_slot_by_cand_idx.as_mut() {
            if cand_idx < map.len() {
                map[cand_idx] = Some(slot);
            }
        }
    }

    fn sync_projective_centers(&mut self, markers: &[DetectedMarker], marker_cand_idx: &[usize]) {
        let Some(stage1) = self.stage1.as_mut() else {
            return;
        };
        let Some(map) = self.stage1_slot_by_cand_idx.as_ref() else {
            return;
        };

        for (marker, &cand_idx) in markers.iter().zip(marker_cand_idx.iter()) {
            let Some(Some(slot)) = map.get(cand_idx).copied() else {
                continue;
            };
            if let Some(cd) = stage1.candidates.get_mut(slot) {
                cd.derived.center_xy = Some(marker.center);
            }
        }
    }

    fn into_stages(self) -> (Option<dbg::StageDebug>, Option<dbg::StageDebug>) {
        (self.stage0, self.stage1)
    }
}

struct CandidateProcessContext<'a> {
    gray: &'a GrayImage,
    config: &'a DetectConfig,
    mapper: Option<&'a dyn crate::camera::PixelMapper>,
    sampler: DistortionAwareSampler<'a>,
    inner_fit_cfg: &'a inner_fit::InnerFitConfig,
    store_points: bool,
}

struct CandidateProcessOutput {
    marker: Option<DetectedMarker>,
    marker_cand_idx: Option<usize>,
    debug_candidate: Option<dbg::CandidateDebug>,
}

fn process_candidate(
    cand_idx: usize,
    proposal: Proposal,
    ctx: &CandidateProcessContext<'_>,
    record_debug: bool,
) -> CandidateProcessOutput {
    let center_prior = match ctx.sampler.image_to_working_xy([proposal.x, proposal.y]) {
        Some(v) => v,
        None => {
            return CandidateProcessOutput {
                marker: None,
                marker_cand_idx: None,
                debug_candidate: None,
            };
        }
    };

    let mut cand_debug = record_debug.then(|| stage1_candidate_stub(cand_idx, proposal));

    let fit = match fit_outer_ellipse_robust_with_reason(
        ctx.gray,
        center_prior,
        marker_outer_radius_expected_px(ctx.config),
        ctx.config,
        ctx.mapper,
        &ctx.config.edge_sample,
        ctx.store_points,
    ) {
        Ok(v) => v,
        Err(reason) => {
            if let Some(cd) = cand_debug.as_mut() {
                cd.decision = dbg::DecisionDebug {
                    status: dbg::DecisionStatus::Rejected,
                    reason,
                };
            }
            return CandidateProcessOutput {
                marker: None,
                marker_cand_idx: None,
                debug_candidate: cand_debug,
            };
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
        ctx.gray,
        &outer,
        &ctx.config.marker_spec,
        ctx.mapper,
        ctx.inner_fit_cfg,
        ctx.store_points,
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

    if let Some(cd) = cand_debug.as_mut() {
        cd.ring_fit = Some(dbg::RingFitDebug {
            center_xy_fit: center,
            edge: edge_for_debug(&edge, ctx.store_points),
            outer_estimation: Some(outer_estimate),
            chosen_outer_hypothesis: Some(chosen_hypothesis),
            ellipse_outer: Some(crate::EllipseParams::from(outer)),
            ellipse_inner: inner_params.clone(),
            inner_estimation: Some(inner_fit.estimate.clone()),
            fit: fit_metrics,
            inner_points_fit: if ctx.store_points {
                Some(inner_fit.points_inner)
            } else {
                None
            },
        });

        cd.decode = Some(dbg::DecodeDebug {
            diagnostics: decode_diag.clone(),
            result: decode_result.clone(),
            decode_metrics,
        });

        cd.decision = dbg::DecisionDebug {
            status: dbg::DecisionStatus::Accepted,
            reason: if let Some(r) = decode_diag.reject_reason.as_ref() {
                format!("ok_with_decode_reject:{}", r)
            } else {
                "ok".to_string()
            },
        };
        cd.derived = dbg::DerivedDebug {
            id: decode_result.as_ref().map(|d| d.id),
            confidence: Some(confidence),
            center_xy: Some(center),
        };
    }

    CandidateProcessOutput {
        marker: Some(marker),
        marker_cand_idx: Some(cand_idx),
        debug_candidate: cand_debug,
    }
}

fn run_dedup_phase(
    markers: Vec<DetectedMarker>,
    marker_cand_idx: Vec<usize>,
    dedup_radius: f64,
    collect_debug: bool,
) -> (Vec<DetectedMarker>, Vec<usize>, Option<dbg::DedupDebug>) {
    if collect_debug {
        let (m, idx, d) = dedup_with_debug(markers, marker_cand_idx, dedup_radius);
        (m, idx, Some(d))
    } else {
        let mut m = dedup_markers(markers, dedup_radius);
        dedup_by_id(&mut m);
        (m, Vec::new(), None)
    }
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
    let sampler = DistortionAwareSampler::new(gray, mapper);

    let mut debug_state = DebugState::new(&proposals, debug_cfg);

    let ctx = CandidateProcessContext {
        gray,
        config,
        mapper,
        sampler,
        inner_fit_cfg: &inner_fit_cfg,
        store_points: debug_state.store_points,
    };

    let mut markers: Vec<DetectedMarker> = Vec::new();
    let mut marker_cand_idx: Vec<usize> = Vec::new();

    for (i, proposal) in proposals.iter().copied().enumerate() {
        let out = process_candidate(i, proposal, &ctx, debug_state.should_record_candidate(i));

        if let Some(marker) = out.marker {
            markers.push(marker);
        }
        if let Some(ci) = out.marker_cand_idx {
            marker_cand_idx.push(ci);
        }
        debug_state.record_stage1_candidate(out.debug_candidate);
    }

    if use_projective_center {
        apply_projective_centers(&mut markers, config);
        debug_state.sync_projective_centers(&markers, &marker_cand_idx);
    }

    let (markers, marker_cand_idx, stage2) = run_dedup_phase(
        markers,
        marker_cand_idx,
        config.dedup_radius,
        debug_state.collect_debug,
    );

    tracing::info!("{} markers detected after dedup", markers.len());

    let (stage0, stage1) = debug_state.into_stages();
    FitDecodeCoreOutput {
        markers,
        marker_cand_idx,
        stage0,
        stage1,
        stage2,
    }
}
