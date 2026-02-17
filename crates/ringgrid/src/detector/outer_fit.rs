use image::GrayImage;

use crate::conic::{
    fit_ellipse_direct, rms_sampson_distance, try_fit_ellipse_ransac, Ellipse, RansacConfig,
};
use crate::marker::decode::decode_marker_with_diagnostics_and_mapper;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleConfig, EdgeSampleResult};
use crate::ring::inner_estimate::Polarity;
use crate::ring::outer_estimate::{
    estimate_outer_from_prior_with_mapper, OuterHypothesis, OuterStatus,
};
use crate::DetectedMarker;

use super::DetectConfig;

/// Stable reject code for outer-fit candidate creation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OuterFitRejectReason {
    OuterEstimateInvalidSearchWindow,
    OuterEstimateInsufficientThetaCoverage,
    OuterEstimateNoPolarityCandidates,
    OuterEstimateUnknownFailure,
    OuterEstimateMissingPolarity,
    NoValidHypothesis,
}

impl OuterFitRejectReason {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::OuterEstimateInvalidSearchWindow => "outer_estimate_invalid_search_window",
            Self::OuterEstimateInsufficientThetaCoverage => {
                "outer_estimate_insufficient_theta_coverage"
            }
            Self::OuterEstimateNoPolarityCandidates => "outer_estimate_no_polarity_candidates",
            Self::OuterEstimateUnknownFailure => "outer_estimate_unknown_failure",
            Self::OuterEstimateMissingPolarity => "outer_estimate_missing_polarity",
            Self::NoValidHypothesis => "no_valid_hypothesis",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OuterEllipseFitReject {
    TooFewPoints,
    DirectFitFailed,
    InvalidEllipse,
}

fn fit_outer_ellipse_with_reason(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Result<(Ellipse, Option<crate::conic::RansacResult>), OuterEllipseFitReject> {
    // Fit outer ellipse
    let ransac_config = RansacConfig {
        max_iters: 200,
        inlier_threshold: 1.5,
        min_inliers: 6,
        seed: 42,
    };

    let (outer, outer_ransac) = if edge.outer_points.len() >= 8 {
        match try_fit_ellipse_ransac(&edge.outer_points, &ransac_config) {
            Ok(r) => (r.ellipse, Some(r)),
            Err(_) => {
                // Fall back to direct fit
                match fit_ellipse_direct(&edge.outer_points) {
                    Some(e) => (e, None),
                    None => return Err(OuterEllipseFitReject::DirectFitFailed),
                }
            }
        }
    } else if edge.outer_points.len() >= 6 {
        match fit_ellipse_direct(&edge.outer_points) {
            Some(e) => (e, None),
            None => return Err(OuterEllipseFitReject::DirectFitFailed),
        }
    } else {
        return Err(OuterEllipseFitReject::TooFewPoints);
    };

    // Validate outer ellipse
    if outer.a < config.min_semi_axis
        || outer.a > config.max_semi_axis
        || outer.b < config.min_semi_axis
        || outer.b > config.max_semi_axis
        || outer.aspect_ratio() > config.max_aspect_ratio
        || !outer.is_valid()
    {
        return Err(OuterEllipseFitReject::InvalidEllipse);
    }

    Ok((outer, outer_ransac))
}

pub(crate) fn median_outer_radius_from_neighbors_px(
    projected_center: [f64; 2],
    markers: &[DetectedMarker],
    k: usize,
) -> Option<f32> {
    let mut candidates: Vec<(f64, f32)> = Vec::new();
    for m in markers {
        let r = match m.ellipse_outer {
            Some(v) => v.mean_axis(),
            None => continue,
        };
        let dx = m.center[0] - projected_center[0];
        let dy = m.center[1] - projected_center[1];
        let d2 = dx * dx + dy * dy;
        if d2.is_finite() {
            candidates.push((d2, r as f32));
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let radii: Vec<f32> = candidates
        .iter()
        .take(k.max(1).min(candidates.len()))
        .map(|(_, r)| *r)
        .collect();
    Some(median_f32(&radii))
}

fn sample_signed_radial_derivative(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r: f32,
) -> Option<f32> {
    let h = 0.25f32;
    if r <= h {
        return None;
    }

    let x1 = ray_origin[0] + ray_dir[0] * (r + h);
    let y1 = ray_origin[1] + ray_dir[1] * (r + h);
    let x0 = ray_origin[0] + ray_dir[0] * (r - h);
    let y0 = ray_origin[1] + ray_dir[1] * (r - h);
    let i1 = sampler.sample_checked(x1, y1)?;
    let i0 = sampler.sample_checked(x0, y0)?;
    Some((i1 - i0) / (2.0 * h))
}

fn passes_ring_depth_gate(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r: f32,
    pol: Polarity,
    min_ring_depth: f32,
) -> bool {
    let band = 2.0f32;
    let x_in = ray_origin[0] + ray_dir[0] * (r - band);
    let y_in = ray_origin[1] + ray_dir[1] * (r - band);
    let x_out = ray_origin[0] + ray_dir[0] * (r + band);
    let y_out = ray_origin[1] + ray_dir[1] * (r + band);
    let Some(i_in) = sampler.sample_checked(x_in, y_in) else {
        return false;
    };
    let Some(i_out) = sampler.sample_checked(x_out, y_out) else {
        return false;
    };
    let signed_depth = match pol {
        Polarity::Pos => i_out - i_in,
        Polarity::Neg => i_in - i_out,
    };
    signed_depth >= min_ring_depth
}

fn pick_best_radius_on_ray(
    sampler: DistortionAwareSampler<'_>,
    ray_origin: [f32; 2],
    ray_dir: [f32; 2],
    r0: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> Option<f32> {
    let refine_hw = refine_halfwidth_px.clamp(0.0, 4.0);
    let refine_step = edge_cfg.r_step.clamp(0.25, 1.0);
    let n_ref = ((refine_hw / refine_step).ceil() as i32).max(1);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_r = None::<f32>;

    for k in -n_ref..=n_ref {
        let r = r0 + k as f32 * refine_step;
        if r < edge_cfg.r_min || r > edge_cfg.r_max {
            continue;
        }

        let Some(d) = sample_signed_radial_derivative(sampler, ray_origin, ray_dir, r) else {
            continue;
        };
        let score = match pol {
            Polarity::Pos => d,
            Polarity::Neg => -d,
        };

        if score > best_score {
            best_score = score;
            best_r = Some(r);
        }
    }

    match best_r {
        Some(r) if best_score.is_finite() && best_score > 0.0 => Some(r),
        _ => None,
    }
}

fn collect_outer_edge_points_near_radius(
    sampler: DistortionAwareSampler<'_>,
    center_prior: [f32; 2],
    r0: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let n_t = edge_cfg.n_rays.max(8);

    let mut outer_points = Vec::with_capacity(n_t);
    let mut outer_radii = Vec::with_capacity(n_t);

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let ray_dir = [theta.cos(), theta.sin()];

        let Some(r) = pick_best_radius_on_ray(
            sampler,
            center_prior,
            ray_dir,
            r0,
            pol,
            edge_cfg,
            refine_halfwidth_px,
        ) else {
            continue;
        };

        if !passes_ring_depth_gate(
            sampler,
            center_prior,
            ray_dir,
            r,
            pol,
            edge_cfg.min_ring_depth,
        ) {
            continue;
        }

        let x = center_prior[0] + ray_dir[0] * r;
        let y = center_prior[1] + ray_dir[1] * r;
        outer_points.push([x as f64, y as f64]);
        outer_radii.push(r);
    }

    (outer_points, outer_radii)
}

fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn parse_theta_coverage_reason(reason: &str) -> Option<(f32, f32)> {
    const PREFIX: &str = "insufficient_theta_coverage(";
    let payload = reason.strip_prefix(PREFIX)?.strip_suffix(')')?;
    let (observed, min_required) = payload.split_once('<')?;
    let observed = observed.parse::<f32>().ok()?;
    let min_required = min_required.parse::<f32>().ok()?;
    Some((observed, min_required))
}

fn map_outer_estimate_reject(reason: Option<&str>) -> OuterFitReject {
    match reason {
        Some("invalid_search_window") => {
            OuterFitReject::new(OuterFitRejectReason::OuterEstimateInvalidSearchWindow, None)
        }
        Some("no_polarity_candidates") => OuterFitReject::new(
            OuterFitRejectReason::OuterEstimateNoPolarityCandidates,
            None,
        ),
        Some(raw) => {
            if let Some((observed, min_required)) = parse_theta_coverage_reason(raw) {
                OuterFitReject::new(
                    OuterFitRejectReason::OuterEstimateInsufficientThetaCoverage,
                    Some(OuterFitRejectContext::ThetaCoverage {
                        observed_coverage: observed,
                        min_required_coverage: min_required,
                    }),
                )
            } else {
                OuterFitReject::new(OuterFitRejectReason::OuterEstimateUnknownFailure, None)
            }
        }
        None => OuterFitReject::new(OuterFitRejectReason::OuterEstimateUnknownFailure, None),
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

fn build_outer_estimation_cfg(
    config: &DetectConfig,
    edge_cfg: &EdgeSampleConfig,
) -> crate::ring::OuterEstimationConfig {
    let mut outer_cfg = config.outer_estimation.clone();
    outer_cfg.theta_samples = edge_cfg.n_rays.max(8);
    outer_cfg
}

fn score_outer_candidate(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    decode_confidence: f32,
    r_expected: f32,
) -> f32 {
    let decode_score = decode_confidence.clamp(0.0, 1.0);
    let arc_cov = (edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32).clamp(0.0, 1.0);
    let inlier_ratio = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let fit_support = (arc_cov * inlier_ratio).clamp(0.0, 1.0);

    let mean_axis = ((outer.a + outer.b) * 0.5) as f32;
    let size_score = 1.0 - ((mean_axis - r_expected).abs() / r_expected.max(1.0)).min(1.0);

    let residual = rms_sampson_distance(outer, &edge.outer_points) as f32;
    let residual = if residual.is_finite() {
        residual.max(0.0)
    } else {
        f32::INFINITY
    };
    let residual_score = 1.0 / (1.0 + residual);

    0.55 * decode_score + 0.25 * fit_support + 0.15 * size_score + 0.05 * residual_score
}

struct OuterFitEvalContext<'a> {
    gray: &'a GrayImage,
    center_prior: [f32; 2],
    r_expected: f32,
    pol: Polarity,
    config: &'a DetectConfig,
    edge_cfg: &'a EdgeSampleConfig,
    outer_cfg: &'a crate::ring::OuterEstimationConfig,
    sampler: DistortionAwareSampler<'a>,
    mapper: Option<&'a dyn PixelMapper>,
}

fn evaluate_hypothesis(
    ctx: &OuterFitEvalContext<'_>,
    hyp: &OuterHypothesis,
) -> Option<OuterFitCandidate> {
    let (outer_points, outer_radii) = collect_outer_edge_points_near_radius(
        ctx.sampler,
        ctx.center_prior,
        hyp.r_outer_px,
        ctx.pol,
        ctx.edge_cfg,
        ctx.outer_cfg.refine_halfwidth_px,
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

    let (outer, outer_ransac) = fit_outer_ellipse_with_reason(&edge, ctx.config).ok()?;

    let (decode_result, diagnostics) =
        decode_marker_with_diagnostics_and_mapper(ctx.gray, &outer, &ctx.config.decode, ctx.mapper);

    let score = score_outer_candidate(
        &edge,
        &outer,
        outer_ransac.as_ref(),
        diagnostics.decode_confidence,
        ctx.r_expected,
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
    let outer_cfg = build_outer_estimation_cfg(config, edge_cfg);
    let sampler = DistortionAwareSampler::new(gray, mapper);

    let outer_estimate = estimate_outer_from_prior_with_mapper(
        gray,
        center_prior,
        r_expected,
        &outer_cfg,
        mapper,
        false,
    );
    if outer_estimate.status != OuterStatus::Ok || outer_estimate.hypotheses.is_empty() {
        return Err(map_outer_estimate_reject(outer_estimate.reason.as_deref()));
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
        outer_cfg: &outer_cfg,
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
    use image::Luma;

    fn draw_ring_image(
        w: u32,
        h: u32,
        center: [f32; 2],
        outer_radius: f32,
        inner_radius: f32,
    ) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let d = (dx * dx + dy * dy).sqrt();
                let pix = if d >= inner_radius && d <= outer_radius {
                    24u8
                } else {
                    230u8
                };
                img.put_pixel(x, y, Luma([pix]));
            }
        }
        img
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
    fn outer_estimate_reason_mapping_extracts_theta_coverage_context() {
        let reject = map_outer_estimate_reject(Some("insufficient_theta_coverage(0.31<0.60)"));
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
    fn pick_best_radius_on_ray_hits_synthetic_outer_edge() {
        let center = [64.0f32, 64.0f32];
        let outer_radius = 24.0f32;
        let inner_radius = 12.0f32;
        let img = draw_ring_image(128, 128, center, outer_radius, inner_radius);
        let sampler = DistortionAwareSampler::new(&img, None);

        let edge_cfg = EdgeSampleConfig {
            r_min: 1.5,
            r_max: 40.0,
            r_step: 0.5,
            ..EdgeSampleConfig::default()
        };

        let r = pick_best_radius_on_ray(
            sampler,
            center,
            [1.0, 0.0],
            outer_radius,
            Polarity::Pos,
            &edge_cfg,
            2.0,
        )
        .expect("expected radius pick on synthetic ring");

        assert!((r - outer_radius).abs() <= 1.5);
        assert!(passes_ring_depth_gate(
            sampler,
            center,
            [1.0, 0.0],
            r,
            Polarity::Pos,
            edge_cfg.min_ring_depth,
        ));
    }

    #[test]
    fn ring_depth_gate_rejects_low_depth_ray() {
        let img = GrayImage::from_pixel(64, 64, Luma([128u8]));
        let sampler = DistortionAwareSampler::new(&img, None);
        assert!(!passes_ring_depth_gate(
            sampler,
            [32.0, 32.0],
            [1.0, 0.0],
            16.0,
            Polarity::Pos,
            0.01,
        ));
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
}
