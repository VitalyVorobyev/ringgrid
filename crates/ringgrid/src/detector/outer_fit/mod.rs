use image::GrayImage;

use crate::conic::Ellipse;
use crate::marker::decode::{TargetCodes, decode_marker_with_diagnostics_and_mapper};
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleConfig, EdgeSampleResult};
use crate::ring::inner_estimate::Polarity;
use crate::ring::outer_estimate::{
    OuterEstimateFailure, OuterEstimationConfig, OuterHypothesis, OuterStatus,
    estimate_outer_from_prior_with_mapper,
};

use super::DetectConfig;

mod sampling;
mod scoring;
mod solver;

pub(crate) use sampling::{max_angular_gap, max_outer_edge_radius_px};

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
    &config.advanced.edge_sample
}

/// Edge sampling for a completion refit of a ring expected at `r_expected`
/// (the neighbour-median outer radius at its cell): the detector's sampler,
/// reaching exactly as far as the per-ray expected-radius gate accepts —
/// whatever the scale prior's range — with the completion arc-coverage
/// requirement.
fn completion_edge_cfg(config: &DetectConfig, r_expected: f32) -> EdgeSampleConfig {
    let mut edge_cfg = config.advanced.edge_sample.clone();
    let params = &config.advanced.completion;
    edge_cfg.r_max = max_outer_edge_radius_px(r_expected);

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
    refine_halfwidth_px: f32,
    sampler: DistortionAwareSampler<'a>,
    mapper: Option<&'a dyn PixelMapper>,
}

/// Ellipse-exact resampling rounds after the hypothesis-model pass.
const MAX_ELLIPSE_RESAMPLE_ROUNDS: usize = 2;

fn edge_result(
    ctx: &OuterFitEvalContext<'_>,
    outer_points: Vec<[f64; 2]>,
    outer_radii: Vec<f32>,
) -> EdgeSampleResult {
    // The outer edge sampler only records rays where an outer edge is found.
    EdgeSampleResult {
        n_good_rays: outer_points.len(),
        outer_points,
        outer_radii,
        n_total_rays: ctx.edge_cfg.n_rays.max(8),
    }
}

/// Refine half-window for the hypothesis-model (seed) pass: the configured
/// window, widened to the second-harmonic model's own error bound on an
/// eccentric ring, so the seed fit gets enough rays.
///
/// A second-harmonic fit `c0 + A cos 2θ` to an ellipse's polar radius is off
/// by the fourth harmonic, at most about `A² / c0` (axis ratio 0.56: bound
/// 3.5 px, true maximum 3.3 px; 0.70: 1.5 vs 1.3; 0.81: 0.6 vs 0.5). The
/// widening is capped at half the dark band's width along the minor axis, so
/// the search cannot reach the ring's inner edge or the paper beyond the
/// neighbouring ring. A circle (`A = 0`) keeps the configured window.
fn seed_refine_halfwidth_px(ctx: &OuterFitEvalContext<'_>, hyp: &OuterHypothesis) -> f32 {
    let Some(model) = hyp.radius_model.as_ref().filter(|m| m.c0 > 0.0) else {
        return ctx.refine_halfwidth_px;
    };
    let amplitude = model.amplitude();
    let model_error = amplitude * amplitude / model.c0;
    let inner_ratio = ctx
        .config
        .advanced
        .marker_spec
        .r_inner_expected
        .clamp(0.05, 0.95);
    let band_cap = 0.5 * (model.c0 - amplitude).max(0.0) * (1.0 - inner_ratio);
    ctx.refine_halfwidth_px.max(model_error.min(band_cap))
}

/// Fit the outer ellipse to the hypothesis-model edge points, then re-sample
/// the edge at the fitted ellipse on every ray and refit, for as long as that
/// finds more edge points.
///
/// The hypothesis tracks the edge with a second-harmonic radius model, exact
/// only to first order in the eccentricity: at an axis ratio of 0.56 it is
/// off by up to 3.3 px against a ±1 px refine window, so a sharp edge is lost
/// on most rays (a blurred one, wider than the error, survives). Any ellipse
/// through the rays that did hit is a far better model; the seed fit
/// therefore needs only the solver's own minimum of points, while the
/// accepted fit still needs `min_rays_with_ring`.
fn fit_with_ellipse_resampling(
    ctx: &OuterFitEvalContext<'_>,
    outer_points: Vec<[f64; 2]>,
    outer_radii: Vec<f32>,
) -> Option<(
    EdgeSampleResult,
    Ellipse,
    Option<crate::conic::RansacResult>,
)> {
    let min_rays = ctx.edge_cfg.min_rays_with_ring;
    let mut edge = edge_result(ctx, outer_points, outer_radii);
    let mut best = None;
    for round in 0..=MAX_ELLIPSE_RESAMPLE_ROUNDS {
        let Ok((outer, ransac)) = solver::fit_outer_ellipse_with_reason(&edge, ctx.config) else {
            break;
        };
        let n_points = edge.outer_points.len();
        let seed = outer;
        if n_points >= min_rays {
            best = Some((edge, outer, ransac));
        }
        if round == MAX_ELLIPSE_RESAMPLE_ROUNDS {
            break;
        }
        let (points, radii) = sampling::collect_outer_edge_points_near_ellipse(
            ctx.sampler,
            ctx.center_prior,
            &seed,
            ctx.r_expected,
            ctx.pol,
            ctx.edge_cfg,
            ctx.refine_halfwidth_px,
        );
        if points.len() <= n_points {
            break;
        }
        edge = edge_result(ctx, points, radii);
    }
    best
}

fn evaluate_hypothesis(
    ctx: &OuterFitEvalContext<'_>,
    hyp: &OuterHypothesis,
) -> Option<OuterFitCandidate> {
    let (outer_points, outer_radii) = sampling::collect_outer_edge_points_near_radius(
        ctx.sampler,
        ctx.center_prior,
        hyp,
        ctx.r_expected,
        ctx.pol,
        ctx.edge_cfg,
        seed_refine_halfwidth_px(ctx, hyp),
    );

    let (edge, outer, outer_ransac) = fit_with_ellipse_resampling(ctx, outer_points, outer_radii)?;

    let gap = sampling::max_angular_gap(outer.center(), &edge.outer_points);
    if gap > ctx.config.advanced.outer_fit.max_angular_gap_rad {
        return None;
    }

    // Evidence for the decode-confidence scoring slot: the sampled code band
    // for coded targets, photometric ring evidence for plain ones (which have
    // no code band to sample).
    let (decode_result, evidence_confidence) =
        if let Some(codes) = TargetCodes::of(&ctx.config.target) {
            let (decode_result, diagnostics) = decode_marker_with_diagnostics_and_mapper(
                ctx.gray,
                &outer,
                &ctx.config.advanced.decode,
                &codes,
                ctx.mapper,
            );
            (decode_result, diagnostics.decode_confidence)
        } else {
            let evidence = scoring::plain_ring_evidence(
                ctx.sampler,
                &outer,
                ctx.config.advanced.marker_spec.r_inner_expected,
            );
            (None, evidence)
        };

    let score = scoring::score_outer_candidate(
        &edge,
        &outer,
        outer_ransac.as_ref(),
        evidence_confidence,
        ctx.r_expected,
        ctx.config.advanced.outer_fit.size_score_weight,
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
    fit_outer_candidate_with_sampling(
        gray,
        center_prior,
        r_outer_expected_px,
        config,
        mapper,
        &config.advanced.outer_estimation,
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
    let edge_cfg = completion_edge_cfg(config, r_outer_expected_px);
    fit_outer_candidate_with_sampling(
        gray,
        center_prior,
        r_outer_expected_px,
        config,
        mapper,
        &config.advanced.outer_estimation,
        &edge_cfg,
    )
}

/// Outer fit with explicit sampling parameters: `estimation` drives the
/// radius-hypothesis search and the per-hypothesis edge refinement window,
/// `edge_cfg` the edge sampler. All other fit / decode settings come from
/// `config`. The `fit_outer_candidate_from_prior*` entry points pass the
/// detector's own sampling; inner-as-outer recovery passes its overrides.
pub(crate) fn fit_outer_candidate_with_sampling(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    estimation: &OuterEstimationConfig,
    edge_cfg: &EdgeSampleConfig,
) -> Result<OuterFitCandidate, OuterFitReject> {
    let r_expected = r_outer_expected_px.max(2.0);
    let sampler = DistortionAwareSampler::new(gray, mapper);

    let outer_estimate = estimate_outer_from_prior_with_mapper(
        gray,
        center_prior,
        r_expected,
        estimation,
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
        refine_halfwidth_px: estimation.refine_halfwidth_px,
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
    fn completion_edge_cfg_reach_follows_expected_radius() {
        let mut cfg = DetectConfig::default();
        cfg.advanced.edge_sample.n_rays = 20;
        cfg.advanced.edge_sample.r_max = 30.0;
        cfg.advanced.completion.min_arc_coverage = 0.42;

        // A ring larger than the scale prior's reach: completion reaches as
        // far as the expected-radius gate accepts (1.4 x 50 px).
        let edge_cfg = completion_edge_cfg(&cfg, 50.0);
        assert!((edge_cfg.r_max - 70.0).abs() < 1e-4);
        assert!((completion_edge_cfg(&cfg, 10.0).r_max - 14.0).abs() < 1e-4);
        assert_eq!(edge_cfg.min_rays_with_ring, 9);
        assert_eq!(edge_cfg.r_min, cfg.advanced.edge_sample.r_min);
        assert!((edge_cfg.r_step - cfg.advanced.edge_sample.r_step).abs() < 1e-6);
        assert!((edge_cfg.min_ring_depth - cfg.advanced.edge_sample.min_ring_depth).abs() < 1e-6);
    }

    #[test]
    fn baseline_edge_cfg_returns_config_field() {
        let cfg = DetectConfig::default();
        assert!(std::ptr::eq(
            baseline_edge_cfg(&cfg),
            &cfg.advanced.edge_sample
        ));
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
        cfg.advanced.edge_sample.r_min = 1.5;
        cfg.advanced.edge_sample.r_max = 48.0;

        let out = fit_outer_candidate_from_prior(&img, center, outer_radius, &cfg, None)
            .expect("baseline outer fit should produce a candidate");
        assert!(out.edge.outer_points.len() >= cfg.advanced.edge_sample.min_rays_with_ring);
    }

    /// A sharp, 2×-supersampled plain ring (filled annulus) whose outer
    /// boundary is the ellipse `(a, b, angle)` and inner boundary the same
    /// ellipse scaled by one half.
    fn draw_eccentric_ring(w: u32, h: u32, e: Ellipse) -> GrayImage {
        let (s, c) = e.angle.sin_cos();
        let inside = |x: f64, y: f64, k: f64| {
            let (dx, dy) = (x - e.cx, y - e.cy);
            let (u, v) = (c * dx + s * dy, -s * dx + c * dy);
            (u / (k * e.a)).powi(2) + (v / (k * e.b)).powi(2) <= 1.0
        };
        GrayImage::from_fn(w, h, |x, y| {
            let mut ink = 0u32;
            for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let (px, py) = (f64::from(x) + ox - 0.5, f64::from(y) + oy - 0.5);
                ink += u32::from(inside(px, py, 1.0) && !inside(px, py, 0.5));
            }
            image::Luma([(230 - ink * 50) as u8])
        })
    }

    /// The second-harmonic hypothesis model is off by up to ~3 px at an axis
    /// ratio of 0.55: on a sharp edge the ±1 px refine window loses most
    /// rays. The seed window and the ellipse-exact resampling recover them.
    #[test]
    fn sharp_eccentric_ring_is_fitted() {
        let truth = Ellipse {
            cx: 100.3,
            cy: 90.6,
            a: 57.0,
            b: 31.0,
            angle: 0.5,
        };
        let img = draw_eccentric_ring(200, 180, truth);
        let mut cfg = DetectConfig::from_target(crate::TargetLayout::rect_24x24());
        cfg.set_marker_scale_prior(crate::MarkerScalePrior::new(40.0, 140.0));
        let center = [truth.cx as f32, truth.cy as f32];
        let out = fit_outer_candidate_from_prior(
            &img,
            center,
            cfg.marker_scale.nominal_outer_radius_px(),
            &cfg,
            None,
        )
        .expect("eccentric sharp ring must fit");
        assert!((out.outer.a - truth.a).abs() < 1.0, "a {}", out.outer.a);
        assert!((out.outer.b - truth.b).abs() < 1.0, "b {}", out.outer.b);
        let err = (out.outer.cx - truth.cx).hypot(out.outer.cy - truth.cy);
        assert!(err < 0.3, "centre error {err}");
    }

    #[test]
    fn ray_ellipse_radius_matches_the_polar_form() {
        let e = Ellipse {
            cx: 10.0,
            cy: -4.0,
            a: 30.0,
            b: 12.0,
            angle: 0.3,
        };
        for k in 0..16 {
            let t = f64::from(k) * std::f64::consts::TAU / 16.0;
            let r = sampling::ray_ellipse_radius(
                &e,
                [e.cx as f32, e.cy as f32],
                [t.cos() as f32, t.sin() as f32],
            )
            .expect("ray from the centre hits the ellipse");
            let phi = t - e.angle;
            let polar = e.a * e.b / ((e.b * phi.cos()).powi(2) + (e.a * phi.sin()).powi(2)).sqrt();
            assert!(
                (f64::from(r) - polar).abs() < 1e-3,
                "theta {t}: {r} vs {polar}"
            );
        }
    }

    #[test]
    fn completion_entry_point_uses_completion_edge_policy() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);

        // The scale prior's reach (12 px) is short of the 24 px ring; the
        // completion refit reaches from the ring's expected radius instead.
        let mut cfg = DetectConfig::default();
        cfg.advanced.edge_sample.r_max = 12.0;

        let baseline = fit_outer_candidate_from_prior(&img, center, outer_radius, &cfg, None);
        assert!(baseline.is_err());

        let completion =
            fit_outer_candidate_from_prior_for_completion(&img, center, outer_radius, &cfg, None)
                .expect("completion outer fit should use completion-derived edge config");
        let completion_cfg = completion_edge_cfg(&cfg, outer_radius);
        assert!(completion.edge.outer_points.len() >= completion_cfg.min_rays_with_ring);
    }

    #[test]
    fn outer_fit_solver_thresholds_are_configurable() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);

        let mut cfg = DetectConfig::default();
        cfg.advanced.edge_sample.r_min = 1.5;
        cfg.advanced.edge_sample.r_max = 48.0;
        cfg.advanced.outer_fit.min_ransac_points = usize::MAX;

        let out = fit_outer_candidate_from_prior(&img, center, outer_radius, &cfg, None)
            .expect("outer fit should still succeed via direct fit path");
        assert!(out.outer_ransac.is_none());
    }
}
