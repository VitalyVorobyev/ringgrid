use image::GrayImage;

use crate::camera::PixelMapper;
use crate::conic::{
    fit_ellipse_direct, rms_sampson_distance, try_fit_ellipse_ransac, Ellipse, RansacConfig,
};
use crate::ring::decode::decode_marker_with_diagnostics_and_mapper;
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleConfig, EdgeSampleResult};
use crate::ring::inner_estimate::Polarity;
use crate::ring::outer_estimate::{
    estimate_outer_from_prior_with_mapper, OuterEstimate, OuterStatus,
};
use crate::{DetectedMarker, EllipseParams};

use super::DetectConfig;

pub(super) fn fit_outer_ellipse_with_reason(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Result<(Ellipse, Option<crate::conic::RansacResult>), String> {
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
                    None => return Err("fit_outer:direct_failed".to_string()),
                }
            }
        }
    } else if edge.outer_points.len() >= 6 {
        match fit_ellipse_direct(&edge.outer_points) {
            Some(e) => (e, None),
            None => return Err("fit_outer:direct_failed".to_string()),
        }
    } else {
        return Err("fit_outer:too_few_points".to_string());
    };

    // Validate outer ellipse
    if outer.a < config.min_semi_axis
        || outer.a > config.max_semi_axis
        || outer.b < config.min_semi_axis
        || outer.b > config.max_semi_axis
        || outer.aspect_ratio() > config.max_aspect_ratio
        || !outer.is_valid()
    {
        return Err("fit_outer:invalid_ellipse".to_string());
    }

    Ok((outer, outer_ransac))
}

/// Marker center used by the detector.
///
/// We use the outer ellipse center as the base estimate. Inner edge estimation
/// is constrained to be concentric with the outer ellipse and is not allowed
/// to bias the center when unreliable.
pub(super) fn compute_center(outer: &Ellipse) -> [f64; 2] {
    [outer.cx, outer.cy]
}

pub(super) fn marker_outer_radius_expected_px(config: &DetectConfig) -> f32 {
    (config.marker_diameter_px * 0.5).max(2.0)
}

fn mean_axis_px_from_params(params: &EllipseParams) -> f32 {
    ((params.semi_axes[0] + params.semi_axes[1]) * 0.5) as f32
}

pub(super) fn mean_axis_px_from_marker(marker: &DetectedMarker) -> Option<f32> {
    marker.ellipse_outer.as_ref().map(mean_axis_px_from_params)
}

pub(super) fn median_outer_radius_from_neighbors_px(
    projected_center: [f64; 2],
    markers: &[DetectedMarker],
    k: usize,
) -> Option<f32> {
    let mut candidates: Vec<(f64, f32)> = Vec::new();
    for m in markers {
        let r = match mean_axis_px_from_marker(m) {
            Some(v) if v.is_finite() && v > 1.0 => v,
            _ => continue,
        };
        let dx = m.center[0] - projected_center[0];
        let dy = m.center[1] - projected_center[1];
        let d2 = dx * dx + dy * dy;
        if d2.is_finite() {
            candidates.push((d2, r));
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

fn sample_outer_edge_points(
    sampler: DistortionAwareSampler<'_>,
    center_prior: [f32; 2],
    r0: f32,
    pol: Polarity,
    edge_cfg: &EdgeSampleConfig,
    refine_halfwidth_px: f32,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let n_t = edge_cfg.n_rays.max(8);
    let cx = center_prior[0];
    let cy = center_prior[1];

    let refine_hw = refine_halfwidth_px.clamp(0.0, 4.0);
    let refine_step = edge_cfg.r_step.clamp(0.25, 1.0);
    let n_ref = ((refine_hw / refine_step).ceil() as i32).max(1);

    let mut outer_points = Vec::with_capacity(n_t);
    let mut outer_radii = Vec::with_capacity(n_t);

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let dx = theta.cos();
        let dy = theta.sin();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_r = None::<f32>;

        for k in -n_ref..=n_ref {
            let r = r0 + k as f32 * refine_step;
            if r < edge_cfg.r_min || r > edge_cfg.r_max {
                continue;
            }

            // dI/dr at r via small central difference
            let h = 0.25f32;
            if r <= h {
                continue;
            }

            let x1 = cx + dx * (r + h);
            let y1 = cy + dy * (r + h);
            let x0 = cx + dx * (r - h);
            let y0 = cy + dy * (r - h);
            let i1 = match sampler.sample_checked(x1, y1) {
                Some(v) => v,
                None => continue,
            };
            let i0 = match sampler.sample_checked(x0, y0) {
                Some(v) => v,
                None => continue,
            };
            let d = (i1 - i0) / (2.0 * h);

            let score = match pol {
                Polarity::Pos => d,
                Polarity::Neg => -d,
            };

            if score > best_score {
                best_score = score;
                best_r = Some(r);
            }
        }

        let r = match best_r {
            Some(r) if best_score.is_finite() && best_score > 0.0 => r,
            _ => continue,
        };

        // Ring depth check across the chosen edge.
        let band = 2.0f32;
        let x_in = cx + dx * (r - band);
        let y_in = cy + dy * (r - band);
        let x_out = cx + dx * (r + band);
        let y_out = cy + dy * (r + band);
        let i_in = match sampler.sample_checked(x_in, y_in) {
            Some(v) => v,
            None => continue,
        };
        let i_out = match sampler.sample_checked(x_out, y_out) {
            Some(v) => v,
            None => continue,
        };
        let signed_depth = match pol {
            Polarity::Pos => i_out - i_in,
            Polarity::Neg => i_in - i_out,
        };
        if signed_depth < edge_cfg.min_ring_depth {
            continue;
        }

        let x = cx + dx * r;
        let y = cy + dy * r;
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

pub(super) struct OuterFitCandidate {
    pub(super) edge: EdgeSampleResult,
    pub(super) outer: Ellipse,
    pub(super) outer_ransac: Option<crate::conic::RansacResult>,
    pub(super) outer_estimate: OuterEstimate,
    pub(super) chosen_hypothesis: usize,
    pub(super) decode_result: Option<crate::ring::decode::DecodeResult>,
    pub(super) decode_diag: crate::ring::decode::DecodeDiagnostics,
    pub(super) score: f32,
}

pub(super) fn fit_outer_ellipse_robust_with_reason(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    edge_cfg: &EdgeSampleConfig,
    store_response: bool,
) -> Result<OuterFitCandidate, String> {
    let r_expected = r_outer_expected_px.max(2.0);

    let mut outer_cfg = config.outer_estimation.clone();
    outer_cfg.theta_samples = edge_cfg.n_rays.max(8);
    let sampler = DistortionAwareSampler::new(gray, mapper);

    let outer_estimate = estimate_outer_from_prior_with_mapper(
        gray,
        center_prior,
        r_expected,
        &outer_cfg,
        mapper,
        store_response,
    );
    if outer_estimate.status != OuterStatus::Ok || outer_estimate.hypotheses.is_empty() {
        return Err(format!(
            "outer_estimate:{}",
            outer_estimate
                .reason
                .as_deref()
                .unwrap_or("unknown_failure")
        ));
    }

    let pol = outer_estimate
        .polarity
        .ok_or_else(|| "outer_estimate:no_polarity".to_string())?;

    let mut best: Option<OuterFitCandidate> = None;

    for (hi, hyp) in outer_estimate.hypotheses.iter().enumerate() {
        let (outer_points, outer_radii) = sample_outer_edge_points(
            sampler,
            center_prior,
            hyp.r_outer_px,
            pol,
            edge_cfg,
            outer_cfg.refine_halfwidth_px,
        );

        if outer_points.len() < edge_cfg.min_rays_with_ring {
            continue;
        }

        let outer_radius = median_f32(&outer_radii);
        let edge = EdgeSampleResult {
            center: center_prior,
            outer_points,
            inner_points: Vec::new(),
            outer_radius,
            inner_radius: 0.0,
            outer_radii,
            inner_radii: Vec::new(),
            n_good_rays: 0,
            n_total_rays: edge_cfg.n_rays.max(8),
        };
        // The outer edge sampler only records rays where an outer edge is found.
        let mut edge = edge;
        edge.n_good_rays = edge.outer_points.len();

        let (outer, outer_ransac) = match fit_outer_ellipse_with_reason(&edge, config) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let (decode_result, decode_diag) =
            decode_marker_with_diagnostics_and_mapper(gray, &outer, &config.decode, mapper);

        let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
        let inlier_ratio = outer_ransac
            .as_ref()
            .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
            .unwrap_or(1.0);
        let residual = rms_sampson_distance(&outer, &edge.outer_points) as f32;

        let mean_axis = ((outer.a + outer.b) * 0.5) as f32;
        let size_err = ((mean_axis - r_expected).abs() / r_expected.max(1.0)).min(2.0);
        let size_score = (1.0 - size_err).clamp(0.0, 1.0);

        let decode_score = decode_diag.decode_confidence;

        let score = 2.0 * decode_score + 0.7 * (arc_cov * inlier_ratio) + 0.3 * size_score
            - 0.05 * residual;

        let cand = OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            outer_estimate: outer_estimate.clone(),
            chosen_hypothesis: hi,
            decode_result,
            decode_diag,
            score,
        };

        match &best {
            Some(b) if b.score >= cand.score => {}
            _ => {
                best = Some(cand);
            }
        }
    }

    best.ok_or_else(|| "outer_fit:no_valid_hypothesis".to_string())
}
