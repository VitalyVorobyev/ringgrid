use image::GrayImage;

use crate::camera::CameraModel;
use crate::conic::{self, Ellipse, RansacConfig};
use crate::marker_spec::MarkerSpec;
use crate::ring::edge_sample::DistortionAwareSampler;
use crate::ring::inner_estimate::{
    estimate_inner_scale_from_outer_with_camera, InnerEstimate, InnerStatus, Polarity,
};

/// Outcome category for robust inner ellipse fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InnerFitStatus {
    /// Inner ellipse fit is valid and accepted by quality gates.
    Ok,
    /// A candidate was fitted but rejected by quality gates.
    Rejected,
    /// Fitting could not be completed (insufficient/invalid data).
    Failed,
}

/// Tuning parameters for robust inner ellipse fitting.
///
/// Outer ellipse is used only as a geometric/search prior.
#[derive(Debug, Clone)]
pub(super) struct InnerFitConfig {
    /// Minimum number of sampled points required to attempt a fit.
    pub min_points: usize,
    /// Minimum accepted inlier ratio when RANSAC is used.
    pub min_inlier_ratio: f32,
    /// Maximum accepted RMS Sampson residual (px) of the fitted inner ellipse.
    pub max_rms_residual: f64,
    /// Maximum allowed center shift from outer to inner fit center (px).
    pub max_center_shift_px: f64,
    /// Maximum allowed absolute error in recovered scale ratio vs radial hint.
    pub max_ratio_abs_error: f64,
    /// Local half-width (in radius-sample indices) around the radial hint.
    pub local_peak_halfwidth_idx: usize,
    /// RANSAC config for robust inner ellipse fitting.
    pub ransac: RansacConfig,
}

impl Default for InnerFitConfig {
    fn default() -> Self {
        Self {
            min_points: 20,
            min_inlier_ratio: 0.5,
            max_rms_residual: 1.0,
            max_center_shift_px: 12.0,
            max_ratio_abs_error: 0.15,
            local_peak_halfwidth_idx: 3,
            ransac: RansacConfig {
                max_iters: 200,
                inlier_threshold: 1.5,
                min_inliers: 8,
                seed: 43,
            },
        }
    }
}

/// Robust inner fit result with shared inner-estimate diagnostics.
#[derive(Debug, Clone)]
pub(super) struct InnerFitResult {
    /// Shared radial-hint estimation output.
    pub estimate: InnerEstimate,
    /// Final inner-ellipse fit status.
    pub status: InnerFitStatus,
    /// Optional reject/failure reason from fit stage.
    pub reason: Option<String>,
    /// Fitted inner ellipse (present only when status is `Ok`).
    pub ellipse_inner: Option<Ellipse>,
    /// Candidate inner-edge points used for fitting.
    pub points_inner: Vec<[f64; 2]>,
    /// RANSAC inlier ratio for the fitted inner ellipse (if RANSAC path was used).
    pub ransac_inlier_ratio_inner: Option<f32>,
    /// RMS Sampson residual of the fitted inner ellipse.
    pub rms_residual_inner: Option<f64>,
}

fn outer_scaled_point(outer: &Ellipse, theta: f32, r_norm: f32) -> [f32; 2] {
    let ca = (outer.angle as f32).cos();
    let sa = (outer.angle as f32).sin();
    let ct = theta.cos();
    let st = theta.sin();
    let vx = (outer.a as f32) * r_norm * ct;
    let vy = (outer.b as f32) * r_norm * st;
    let x = (outer.cx as f32) + ca * vx - sa * vy;
    let y = (outer.cy as f32) + sa * vx + ca * vy;
    [x, y]
}

fn signed_peak_value(v: f32, pol: Polarity) -> f32 {
    match pol {
        Polarity::Pos => v,
        Polarity::Neg => -v,
    }
}

fn refine_peak_idx_subpixel(curve: &[f32], idx: usize, pol: Polarity) -> f32 {
    if idx == 0 || idx + 1 >= curve.len() {
        return idx as f32;
    }
    let y0 = signed_peak_value(curve[idx - 1], pol);
    let y1 = signed_peak_value(curve[idx], pol);
    let y2 = signed_peak_value(curve[idx + 1], pol);
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-6 {
        return idx as f32;
    }
    let offs = 0.5 * (y0 - y2) / denom;
    (idx as f32 + offs.clamp(-1.0, 1.0)).clamp(0.0, (curve.len() - 1) as f32)
}

fn idx_to_r(r_samples: &[f32], idx_f: f32) -> f32 {
    if r_samples.is_empty() {
        return 0.0;
    }
    if r_samples.len() == 1 {
        return r_samples[0];
    }
    let i0 = idx_f.floor().clamp(0.0, (r_samples.len() - 1) as f32) as usize;
    let i1 = (i0 + 1).min(r_samples.len() - 1);
    let t = (idx_f - i0 as f32).clamp(0.0, 1.0);
    r_samples[i0] * (1.0 - t) + r_samples[i1] * t
}

fn sample_inner_points_from_hint(
    gray: &GrayImage,
    outer: &Ellipse,
    estimate: &InnerEstimate,
    cfg: &InnerFitConfig,
    camera: Option<&CameraModel>,
) -> Vec<[f64; 2]> {
    let (Some(r_hint), Some(pol)) = (estimate.r_inner_found, estimate.polarity) else {
        return Vec::new();
    };

    let window = estimate.search_window;
    let n_r = 64usize.max(((window[1] - window[0]).abs() / 0.0025).round() as usize);
    let n_t = 96usize;
    if n_r < 8 || window[1] <= window[0] {
        return Vec::new();
    }
    let r_step = (window[1] - window[0]) / (n_r as f32 - 1.0);
    let r_samples: Vec<f32> = (0..n_r).map(|i| window[0] + i as f32 * r_step).collect();

    let hint_idx = r_samples
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - r_hint)
                .abs()
                .partial_cmp(&(b.1 - r_hint).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let sampler = DistortionAwareSampler::new(gray, camera);

    let mut points = Vec::<[f64; 2]>::with_capacity(n_t);
    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let mut profile = Vec::with_capacity(n_r);
        let mut ok = true;
        for &r in &r_samples {
            let p = outer_scaled_point(outer, theta, r);
            let Some(v) = sampler.sample_checked(p[0], p[1]) else {
                ok = false;
                break;
            };
            profile.push(v);
        }
        if !ok {
            continue;
        }

        let mut d = vec![0.0f32; n_r];
        for ri in 0..n_r {
            if ri == 0 {
                d[ri] = (profile[1] - profile[0]) / r_step;
            } else if ri + 1 == n_r {
                d[ri] = (profile[n_r - 1] - profile[n_r - 2]) / r_step;
            } else {
                d[ri] = (profile[ri + 1] - profile[ri - 1]) / (2.0 * r_step);
            }
        }

        let lo = hint_idx.saturating_sub(cfg.local_peak_halfwidth_idx);
        let hi = (hint_idx + cfg.local_peak_halfwidth_idx).min(n_r - 1);
        let local_i = (lo..=hi)
            .max_by(|&i, &j| {
                signed_peak_value(d[i], pol)
                    .partial_cmp(&signed_peak_value(d[j], pol))
                    .unwrap()
            })
            .unwrap_or(hint_idx);
        let idx_f = refine_peak_idx_subpixel(&d, local_i, pol);
        let r = idx_to_r(&r_samples, idx_f);
        let p = outer_scaled_point(outer, theta, r);
        points.push([p[0] as f64, p[1] as f64]);
    }

    points
}

/// Robustly fit inner ellipse points using outer ellipse only as search prior.
///
/// The radial hint/polarity stage is delegated to `estimate_inner_scale_from_outer`
/// so there is a single source of truth for inner-edge signal extraction.
pub(super) fn fit_inner_ellipse_from_outer_hint(
    gray: &GrayImage,
    outer: &Ellipse,
    spec: &MarkerSpec,
    camera: Option<&CameraModel>,
    cfg: &InnerFitConfig,
    store_response: bool,
) -> InnerFitResult {
    let estimate =
        estimate_inner_scale_from_outer_with_camera(gray, outer, spec, camera, store_response);

    if estimate.status != InnerStatus::Ok {
        return InnerFitResult {
            estimate,
            status: InnerFitStatus::Failed,
            reason: Some("inner_estimate_not_ok".to_string()),
            ellipse_inner: None,
            points_inner: Vec::new(),
            ransac_inlier_ratio_inner: None,
            rms_residual_inner: None,
        };
    }
    if estimate.r_inner_found.is_none() || estimate.polarity.is_none() {
        return InnerFitResult {
            estimate,
            status: InnerFitStatus::Failed,
            reason: Some("inner_estimate_missing_hint".to_string()),
            ellipse_inner: None,
            points_inner: Vec::new(),
            ransac_inlier_ratio_inner: None,
            rms_residual_inner: None,
        };
    }

    let points_inner = sample_inner_points_from_hint(gray, outer, &estimate, cfg, camera);
    if points_inner.len() < cfg.min_points {
        return InnerFitResult {
            estimate,
            status: InnerFitStatus::Failed,
            reason: Some(format!(
                "insufficient_inner_points({}<{})",
                points_inner.len(),
                cfg.min_points
            )),
            ellipse_inner: None,
            points_inner,
            ransac_inlier_ratio_inner: None,
            rms_residual_inner: None,
        };
    }

    let (ellipse_inner, inlier_ratio): (Ellipse, Option<f32>) = if points_inner.len() >= 8 {
        match conic::try_fit_ellipse_ransac(&points_inner, &cfg.ransac) {
            Ok(r) => (
                r.ellipse,
                Some(r.num_inliers as f32 / points_inner.len().max(1) as f32),
            ),
            Err(_) => match conic::fit_ellipse_direct(&points_inner) {
                Some((_, e)) => (e, None),
                None => {
                    return InnerFitResult {
                        estimate,
                        status: InnerFitStatus::Failed,
                        reason: Some("inner_fit_failed".to_string()),
                        ellipse_inner: None,
                        points_inner,
                        ransac_inlier_ratio_inner: None,
                        rms_residual_inner: None,
                    };
                }
            },
        }
    } else {
        match conic::fit_ellipse_direct(&points_inner) {
            Some((_, e)) => (e, None),
            None => {
                return InnerFitResult {
                    estimate,
                    status: InnerFitStatus::Failed,
                    reason: Some("inner_fit_failed".to_string()),
                    ellipse_inner: None,
                    points_inner,
                    ransac_inlier_ratio_inner: None,
                    rms_residual_inner: None,
                };
            }
        }
    };

    let mut reject_reason: Option<String> = None;
    if let Some(ir) = inlier_ratio {
        if ir < cfg.min_inlier_ratio {
            reject_reason = Some(format!(
                "inlier_ratio_low({:.3}<{:.3})",
                ir, cfg.min_inlier_ratio
            ));
        }
    }

    let dx = ellipse_inner.cx - outer.cx;
    let dy = ellipse_inner.cy - outer.cy;
    let center_shift = (dx * dx + dy * dy).sqrt();
    if reject_reason.is_none() && center_shift > cfg.max_center_shift_px {
        reject_reason = Some(format!(
            "center_shift_too_large({:.3}>{:.3})",
            center_shift, cfg.max_center_shift_px
        ));
    }

    let mean_outer = ((outer.a + outer.b) * 0.5).max(1e-9);
    let mean_inner = (ellipse_inner.a + ellipse_inner.b) * 0.5;
    let ratio_meas = mean_inner / mean_outer;
    if reject_reason.is_none() {
        if let Some(r_hint) = estimate.r_inner_found {
            if (ratio_meas - r_hint as f64).abs() > cfg.max_ratio_abs_error {
                reject_reason = Some(format!(
                    "ratio_mismatch({:.3} vs {:.3})",
                    ratio_meas, r_hint
                ));
            }
        }
    }

    let rms_residual_inner = conic::rms_sampson_distance(&ellipse_inner, &points_inner);
    if reject_reason.is_none()
        && (!rms_residual_inner.is_finite() || rms_residual_inner > cfg.max_rms_residual)
    {
        reject_reason = Some(format!(
            "inner_rms_residual_high({:.3}>{:.3})",
            rms_residual_inner, cfg.max_rms_residual
        ));
    }

    if let Some(reason) = reject_reason {
        return InnerFitResult {
            estimate,
            status: InnerFitStatus::Rejected,
            reason: Some(reason),
            ellipse_inner: None,
            points_inner,
            ransac_inlier_ratio_inner: inlier_ratio,
            rms_residual_inner: Some(rms_residual_inner),
        };
    }

    InnerFitResult {
        estimate,
        status: InnerFitStatus::Ok,
        reason: None,
        ellipse_inner: Some(ellipse_inner),
        points_inner,
        ransac_inlier_ratio_inner: inlier_ratio,
        rms_residual_inner: Some(rms_residual_inner),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marker_spec::InnerGradPolarity;
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
                    26u8
                } else {
                    230u8
                };
                img.put_pixel(x, y, Luma([pix]));
            }
        }
        img
    }

    #[test]
    fn inner_fit_recovers_circle_inner_from_outer_hint() {
        let center = [64.0f32, 64.0f32];
        let r_out = 26.0f32;
        let r_in = 16.5f32;
        let img = draw_ring_image(128, 128, center, r_out, r_in);

        let outer = Ellipse {
            cx: center[0] as f64,
            cy: center[1] as f64,
            a: r_out as f64,
            b: r_out as f64,
            angle: 0.0,
        };
        let spec = MarkerSpec {
            r_inner_expected: r_in / r_out,
            inner_search_halfwidth: 0.12,
            inner_grad_polarity: InnerGradPolarity::LightToDark,
            ..MarkerSpec::default()
        };

        let res = fit_inner_ellipse_from_outer_hint(
            &img,
            &outer,
            &spec,
            None,
            &InnerFitConfig::default(),
            false,
        );
        assert_eq!(res.status, InnerFitStatus::Ok, "reason={:?}", res.reason);
        assert_eq!(res.estimate.status, InnerStatus::Ok);
        let e = res.ellipse_inner.expect("inner ellipse");
        assert!((e.cx - outer.cx).abs() < 1.2);
        assert!((e.cy - outer.cy).abs() < 1.2);
        let mean = (e.a + e.b) * 0.5;
        assert!((mean - r_in as f64).abs() < 1.5, "mean={:.3}", mean);
    }
}
