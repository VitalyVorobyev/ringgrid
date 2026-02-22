use image::GrayImage;

use crate::conic::Ellipse;
use crate::marker::decode::decode_marker_with_diagnostics_and_mapper;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleConfig, EdgeSampleResult};
use crate::ring::inner_estimate::Polarity;
use crate::ring::outer_estimate::{
    estimate_outer_from_prior_with_mapper, OuterEstimateFailure, OuterHypothesis, OuterStatus,
};

use super::DetectConfig;

mod sampling;
mod scoring;
mod solver;

pub(crate) use sampling::{max_angular_gap, median_outer_radius_from_neighbors_px};

/// Stable reject code for outer-fit candidate creation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OuterFitRejectReason {
    OuterEstimateInvalidSearchWindow,
    OuterEstimateInsufficientThetaCoverage,
    OuterEstimateNoPolarityCandidates,
    OuterEstimateMissingPolarity,
    NoValidHypothesis,
    AngularGapTooLarge,
}

impl OuterFitRejectReason {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::OuterEstimateInvalidSearchWindow => "outer_estimate_invalid_search_window",
            Self::OuterEstimateInsufficientThetaCoverage => {
                "outer_estimate_insufficient_theta_coverage"
            }
            Self::OuterEstimateNoPolarityCandidates => "outer_estimate_no_polarity_candidates",
            Self::OuterEstimateMissingPolarity => "outer_estimate_missing_polarity",
            Self::NoValidHypothesis => "no_valid_hypothesis",
            Self::AngularGapTooLarge => "angular_gap_too_large",
        }
    }
}

impl std::fmt::Display for OuterFitRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// Structured context attached to an outer-fit reject.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum OuterFitRejectContext {
    ThetaCoverage {
        observed_coverage: f32,
        min_required_coverage: f32,
    },
    NoValidHypothesis {
        hypothesis_count: usize,
    },
    AngularGapTooLarge {
        observed_gap_rad: f64,
        max_allowed_gap_rad: f64,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct OuterFitReject {
    pub(crate) reason: OuterFitRejectReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) context: Option<OuterFitRejectContext>,
}

impl OuterFitReject {
    fn new(reason: OuterFitRejectReason, context: Option<OuterFitRejectContext>) -> Self {
        Self { reason, context }
    }
}

fn map_outer_estimate_reject(failure: Option<OuterEstimateFailure>) -> OuterFitReject {
    match failure {
        Some(OuterEstimateFailure::InvalidSearchWindow) => {
            OuterFitReject::new(OuterFitRejectReason::OuterEstimateInvalidSearchWindow, None)
        }
        Some(OuterEstimateFailure::NoPolarityCandidates) => OuterFitReject::new(
            OuterFitRejectReason::OuterEstimateNoPolarityCandidates,
            None,
        ),
        Some(OuterEstimateFailure::InsufficientThetaCoverage {
            observed,
            min_required,
        }) => OuterFitReject::new(
            OuterFitRejectReason::OuterEstimateInsufficientThetaCoverage,
            Some(OuterFitRejectContext::ThetaCoverage {
                observed_coverage: observed,
                min_required_coverage: min_required,
            }),
        ),
        None => OuterFitReject::new(
            OuterFitRejectReason::OuterEstimateNoPolarityCandidates,
            None,
        ),
    }
}

pub(crate) struct OuterFitCandidate {
    pub(crate) edge: EdgeSampleResult,
    pub(crate) outer: Ellipse,
    pub(crate) outer_ransac: Option<crate::conic::RansacResult>,
    pub(crate) decode_result: Option<crate::marker::decode::DecodeResult>,
    pub(crate) score: f32,
}

fn baseline_edge_cfg(config: &DetectConfig) -> &EdgeSampleConfig {
    &config.edge_sample
}

fn completion_edge_cfg(config: &DetectConfig) -> EdgeSampleConfig {
    let mut edge_cfg = config.edge_sample.clone();
    let params = &config.completion;
    edge_cfg.r_max = params.roi_radius_px.clamp(8.0, 200.0);

    let n_rays = edge_cfg.n_rays.max(1);
    let min_rays = ((n_rays as f32) * params.min_arc_coverage).ceil().max(6.0) as usize;
    edge_cfg.min_rays_with_ring = min_rays.min(n_rays);

    edge_cfg
}

struct OuterFitEvalContext<'a> {
    gray: &'a GrayImage,
    center_prior: [f32; 2],
    r_expected: f32,
    pol: Polarity,
    config: &'a DetectConfig,
    edge_cfg: &'a EdgeSampleConfig,
    sampler: DistortionAwareSampler<'a>,
    mapper: Option<&'a dyn PixelMapper>,
}

fn evaluate_hypothesis(
    ctx: &OuterFitEvalContext<'_>,
    hyp: &OuterHypothesis,
) -> Option<OuterFitCandidate> {
    let (outer_points, outer_radii) = sampling::collect_outer_edge_points_near_radius(
        ctx.sampler,
        ctx.center_prior,
        hyp.r_outer_px,
        ctx.pol,
        ctx.edge_cfg,
        ctx.config.outer_estimation.refine_halfwidth_px,
    );

    if outer_points.len() < ctx.edge_cfg.min_rays_with_ring {
        return None;
    }

    let mut edge = EdgeSampleResult {
        outer_points,
        inner_points: Vec::new(),
        outer_radii,
        inner_radii: Vec::new(),
        n_good_rays: 0,
        n_total_rays: ctx.edge_cfg.n_rays.max(8),
    };
    // The outer edge sampler only records rays where an outer edge is found.
    edge.n_good_rays = edge.outer_points.len();

    let (outer, outer_ransac) = solver::fit_outer_ellipse_with_reason(&edge, ctx.config).ok()?;

    let gap = sampling::max_angular_gap(outer.center(), &edge.outer_points);
    if gap > ctx.config.outer_fit.max_angular_gap_rad {
        return None;
    }

    let (decode_result, diagnostics) =
        decode_marker_with_diagnostics_and_mapper(ctx.gray, &outer, &ctx.config.decode, ctx.mapper);

    let score = scoring::score_outer_candidate(
        &edge,
        &outer,
        outer_ransac.as_ref(),
        diagnostics.decode_confidence,
        ctx.r_expected,
        ctx.config.outer_fit.size_score_weight,
    );

    Some(OuterFitCandidate {
        edge,
        outer,
        outer_ransac,
        decode_result,
        score,
    })
}

pub(crate) fn fit_outer_candidate_from_prior(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> Result<OuterFitCandidate, OuterFitReject> {
    fit_outer_candidate_from_prior_with_edge_cfg(
        gray,
        center_prior,
        r_outer_expected_px,
        config,
        mapper,
        baseline_edge_cfg(config),
    )
}

pub(crate) fn fit_outer_candidate_from_prior_for_completion(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> Result<OuterFitCandidate, OuterFitReject> {
    let edge_cfg = completion_edge_cfg(config);
    fit_outer_candidate_from_prior_with_edge_cfg(
        gray,
        center_prior,
        r_outer_expected_px,
        config,
        mapper,
        &edge_cfg,
    )
}

fn fit_outer_candidate_from_prior_with_edge_cfg(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    edge_cfg: &EdgeSampleConfig,
) -> Result<OuterFitCandidate, OuterFitReject> {
    let r_expected = r_outer_expected_px.max(2.0);
    let sampler = DistortionAwareSampler::new(gray, mapper);

    let outer_estimate = estimate_outer_from_prior_with_mapper(
        gray,
        center_prior,
        r_expected,
        &config.outer_estimation,
        edge_cfg.n_rays.max(8),
        mapper,
        false,
    );
    if outer_estimate.status != OuterStatus::Ok || outer_estimate.hypotheses.is_empty() {
        return Err(map_outer_estimate_reject(outer_estimate.failure));
    }

    let pol = outer_estimate.polarity.ok_or_else(|| {
        OuterFitReject::new(OuterFitRejectReason::OuterEstimateMissingPolarity, None)
    })?;

    let ctx = OuterFitEvalContext {
        gray,
        center_prior,
        r_expected,
        pol,
        config,
        edge_cfg,
        sampler,
        mapper,
    };

    let mut best: Option<OuterFitCandidate> = None;

    for hyp in &outer_estimate.hypotheses {
        let Some(cand) = evaluate_hypothesis(&ctx, hyp) else {
            continue;
        };
        match &best {
            Some(b) if b.score >= cand.score => {}
            _ => best = Some(cand),
        }
    }

    best.ok_or_else(|| {
        OuterFitReject::new(
            OuterFitRejectReason::NoValidHypothesis,
            Some(OuterFitRejectContext::NoValidHypothesis {
                hypothesis_count: outer_estimate.hypotheses.len(),
            }),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn draw_ring_image(
        w: u32,
        h: u32,
        center: [f32; 2],
        outer_radius: f32,
        inner_radius: f32,
    ) -> GrayImage {
        crate::test_utils::draw_ring_image(w, h, center, outer_radius, inner_radius, 24, 230)
    }

    #[test]
    fn completion_edge_cfg_derivation_is_bounded() {
        let mut cfg = DetectConfig::default();
        cfg.edge_sample.n_rays = 20;
        cfg.edge_sample.r_max = 99.0;
        cfg.completion.roi_radius_px = 4.0;
        cfg.completion.min_arc_coverage = 0.42;

        let edge_cfg = completion_edge_cfg(&cfg);
        assert!((edge_cfg.r_max - 8.0).abs() < 1e-6);
        assert_eq!(edge_cfg.min_rays_with_ring, 9);
        assert_eq!(edge_cfg.r_min, cfg.edge_sample.r_min);
        assert!((edge_cfg.r_step - cfg.edge_sample.r_step).abs() < 1e-6);
        assert!((edge_cfg.min_ring_depth - cfg.edge_sample.min_ring_depth).abs() < 1e-6);
    }

    #[test]
    fn baseline_edge_cfg_returns_config_field() {
        let cfg = DetectConfig::default();
        assert!(std::ptr::eq(baseline_edge_cfg(&cfg), &cfg.edge_sample));
    }

    #[test]
    fn outer_fit_reject_reason_serialization_is_stable() {
        let reason = OuterFitRejectReason::NoValidHypothesis;
        assert_eq!(reason.to_string(), "no_valid_hypothesis");
        let json = serde_json::to_string(&reason).expect("serialize outer fit reason");
        assert_eq!(json, "\"no_valid_hypothesis\"");
    }

    #[test]
    fn outer_estimate_failure_mapping_extracts_theta_coverage_context() {
        let reject =
            map_outer_estimate_reject(Some(OuterEstimateFailure::InsufficientThetaCoverage {
                observed: 0.31,
                min_required: 0.60,
            }));
        assert_eq!(
            reject.reason,
            OuterFitRejectReason::OuterEstimateInsufficientThetaCoverage
        );
        match reject.context {
            Some(OuterFitRejectContext::ThetaCoverage {
                observed_coverage,
                min_required_coverage,
            }) => {
                assert!((observed_coverage - 0.31).abs() < 1e-6);
                assert!((min_required_coverage - 0.60).abs() < 1e-6);
            }
            other => panic!("unexpected outer-estimate context: {other:?}"),
        }
    }

    #[test]
    fn baseline_entry_point_finds_candidate_on_synthetic_ring() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);
        let mut cfg = DetectConfig::default();
        cfg.edge_sample.r_min = 1.5;
        cfg.edge_sample.r_max = 48.0;

        let out = fit_outer_candidate_from_prior(&img, center, outer_radius, &cfg, None)
            .expect("baseline outer fit should produce a candidate");
        assert!(out.edge.outer_points.len() >= cfg.edge_sample.min_rays_with_ring);
    }

    #[test]
    fn completion_entry_point_uses_completion_edge_policy() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);

        let mut cfg = DetectConfig::default();
        cfg.edge_sample.r_max = 12.0;
        cfg.completion.roi_radius_px = 40.0;

        let baseline = fit_outer_candidate_from_prior(&img, center, outer_radius, &cfg, None);
        assert!(baseline.is_err());

        let completion =
            fit_outer_candidate_from_prior_for_completion(&img, center, outer_radius, &cfg, None)
                .expect("completion outer fit should use completion-derived edge config");
        let completion_cfg = completion_edge_cfg(&cfg);
        assert!(completion.edge.outer_points.len() >= completion_cfg.min_rays_with_ring);
    }

    #[test]
    fn outer_fit_solver_thresholds_are_configurable() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);

        let mut cfg = DetectConfig::default();
        cfg.edge_sample.r_min = 1.5;
        cfg.edge_sample.r_max = 48.0;
        cfg.outer_fit.min_ransac_points = usize::MAX;

        let out = fit_outer_candidate_from_prior(&img, center, outer_radius, &cfg, None)
            .expect("outer fit should still succeed via direct fit path");
        assert!(out.outer_ransac.is_none());
    }
}
