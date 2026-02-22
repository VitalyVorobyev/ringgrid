use image::GrayImage;

use super::config::InnerFitConfig;
use crate::conic::{self, Ellipse};
use crate::marker::MarkerSpec;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;
use crate::ring::inner_estimate::{
    estimate_inner_scale_from_outer_with_mapper, InnerEstimate, InnerStatus, Polarity,
};
use crate::ring::radial_profile;

/// Outcome category for robust inner ellipse fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InnerFitStatus {
    /// Inner ellipse fit is valid and accepted by quality gates.
    Ok,
    /// A candidate was fitted but rejected by quality gates.
    Rejected,
    /// Fitting could not be completed (insufficient/invalid data).
    Failed,
}

/// Stable reject/failure code for inner fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InnerFitReason {
    EstimateNotOk,
    EstimateMissingHint,
    InsufficientPoints,
    FitFailed,
    InlierRatioLow,
    CenterShiftTooLarge,
    RatioMismatch,
    RmsResidualHigh,
    AngularGapTooLarge,
}

impl InnerFitReason {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::EstimateNotOk => "estimate_not_ok",
            Self::EstimateMissingHint => "estimate_missing_hint",
            Self::InsufficientPoints => "insufficient_points",
            Self::FitFailed => "fit_failed",
            Self::InlierRatioLow => "inlier_ratio_low",
            Self::CenterShiftTooLarge => "center_shift_too_large",
            Self::RatioMismatch => "ratio_mismatch",
            Self::RmsResidualHigh => "rms_residual_high",
            Self::AngularGapTooLarge => "angular_gap_too_large",
        }
    }
}

impl std::fmt::Display for InnerFitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// Structured diagnostics for inner-fit reject/failure reasons.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum InnerFitReasonContext {
    EstimateStatus {
        status: InnerStatus,
    },
    InsufficientPoints {
        observed_points: usize,
        min_required_points: usize,
    },
    InlierRatioLow {
        observed_inlier_ratio: f32,
        min_required_inlier_ratio: f32,
    },
    CenterShiftTooLarge {
        observed_shift_px: f64,
        max_allowed_shift_px: f64,
    },
    RatioMismatch {
        measured_ratio: f64,
        hint_ratio: f32,
        max_allowed_abs_error: f64,
    },
    RmsResidualHigh {
        observed_rms_residual: f64,
        max_allowed_rms_residual: f64,
    },
    AngularGapTooLarge {
        observed_gap_rad: f64,
        max_allowed_gap_rad: f64,
    },
}

/// Robust inner fit result.
#[derive(Debug, Clone)]
pub(crate) struct InnerFitResult {
    /// Final inner-ellipse fit status.
    pub status: InnerFitStatus,
    /// Optional reject/failure reason from fit stage.
    pub reason: Option<InnerFitReason>,
    /// Optional structured reject/failure context.
    pub reason_context: Option<InnerFitReasonContext>,
    /// Fitted inner ellipse (present only when status is `Ok`).
    pub ellipse_inner: Option<Ellipse>,
    /// Candidate inner-edge points used for fitting.
    pub points_inner: Vec<[f64; 2]>,
    /// RANSAC inlier ratio for the fitted inner ellipse (if RANSAC path was used).
    pub ransac_inlier_ratio_inner: Option<f32>,
    /// RMS Sampson residual of the fitted inner ellipse.
    pub rms_residual_inner: Option<f64>,
    /// Maximum angular gap (radians) between consecutive inner edge points.
    pub max_angular_gap: Option<f64>,
    /// Theta consistency score from the inner estimate stage (fraction of theta
    /// samples that agree on the inner edge location). Present when estimation ran.
    pub theta_consistency: Option<f32>,
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

fn sample_inner_points_from_hint(
    gray: &GrayImage,
    outer: &Ellipse,
    estimate: &InnerEstimate,
    cfg: &InnerFitConfig,
    mapper: Option<&dyn PixelMapper>,
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
    let hint_idx = ((r_hint - window[0]) / r_step)
        .round()
        .clamp(0.0, (n_r - 1) as f32) as usize;
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let cx = outer.cx as f32;
    let cy = outer.cy as f32;
    let a = outer.a as f32;
    let b = outer.b as f32;
    let ca = (outer.angle as f32).cos();
    let sa = (outer.angle as f32).sin();

    let mut points = Vec::<[f64; 2]>::with_capacity(n_t);
    let mut profile = vec![0.0f32; n_r];
    let mut d = vec![0.0f32; n_r];
    let d_theta = 2.0 * std::f32::consts::PI / n_t as f32;
    let c_step = d_theta.cos();
    let s_step = d_theta.sin();
    let mut ct = 1.0f32;
    let mut st = 0.0f32;
    for _ in 0..n_t {
        let ray_x = ca * (a * ct) - sa * (b * st);
        let ray_y = sa * (a * ct) + ca * (b * st);

        let mut ok = true;
        for (ri, sample) in profile.iter_mut().enumerate().take(n_r) {
            let r = window[0] + ri as f32 * r_step;
            let x = cx + ray_x * r;
            let y = cy + ray_y * r;
            let Some(v) = sampler.sample_checked(x, y) else {
                ok = false;
                break;
            };
            *sample = v;
        }
        if ok {
            radial_profile::radial_derivative_into(&profile, r_step, &mut d);
            radial_profile::smooth_3point(&mut d);

            let lo = hint_idx.saturating_sub(cfg.local_peak_halfwidth_idx);
            let hi = (hint_idx + cfg.local_peak_halfwidth_idx).min(n_r - 1);
            let local_i = match pol {
                Polarity::Pos => (lo..=hi)
                    .max_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap())
                    .unwrap_or(hint_idx),
                Polarity::Neg => (lo..=hi)
                    .min_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap())
                    .unwrap_or(hint_idx),
            };
            let idx_f = refine_peak_idx_subpixel(&d, local_i, pol);
            let r = (window[0] + idx_f * r_step).clamp(window[0], window[1]);
            points.push([(cx + ray_x * r) as f64, (cy + ray_y * r) as f64]);
        }

        let next_ct = ct * c_step - st * s_step;
        let next_st = st * c_step + ct * s_step;
        ct = next_ct;
        st = next_st;
    }

    points
}

/// Robustly fit inner ellipse points using outer ellipse only as search prior.
///
/// The radial hint/polarity stage is delegated to `estimate_inner_scale_from_outer`
/// so there is a single source of truth for inner-edge signal extraction.
pub(crate) fn fit_inner_ellipse_from_outer_hint(
    gray: &GrayImage,
    outer: &Ellipse,
    spec: &MarkerSpec,
    mapper: Option<&dyn PixelMapper>,
    cfg: &InnerFitConfig,
    store_response: bool,
) -> InnerFitResult {
    let estimate =
        estimate_inner_scale_from_outer_with_mapper(gray, outer, spec, mapper, store_response);

    if estimate.status != InnerStatus::Ok {
        return InnerFitResult {
            status: InnerFitStatus::Failed,
            reason: Some(InnerFitReason::EstimateNotOk),
            reason_context: Some(InnerFitReasonContext::EstimateStatus {
                status: estimate.status,
            }),
            ellipse_inner: None,
            points_inner: Vec::new(),
            ransac_inlier_ratio_inner: None,
            rms_residual_inner: None,
            max_angular_gap: None,
            theta_consistency: estimate.theta_consistency,
        };
    }
    if estimate.r_inner_found.is_none() || estimate.polarity.is_none() {
        return InnerFitResult {
            status: InnerFitStatus::Failed,
            reason: Some(InnerFitReason::EstimateMissingHint),
            reason_context: None,
            ellipse_inner: None,
            points_inner: Vec::new(),
            ransac_inlier_ratio_inner: None,
            rms_residual_inner: None,
            max_angular_gap: None,
            theta_consistency: estimate.theta_consistency,
        };
    }

    let points_inner = sample_inner_points_from_hint(gray, outer, &estimate, cfg, mapper);
    if points_inner.len() < cfg.min_points {
        return InnerFitResult {
            status: InnerFitStatus::Failed,
            reason: Some(InnerFitReason::InsufficientPoints),
            reason_context: Some(InnerFitReasonContext::InsufficientPoints {
                observed_points: points_inner.len(),
                min_required_points: cfg.min_points,
            }),
            ellipse_inner: None,
            points_inner,
            ransac_inlier_ratio_inner: None,
            rms_residual_inner: None,
            max_angular_gap: None,
            theta_consistency: estimate.theta_consistency,
        };
    }

    let (ellipse_inner, inlier_ratio): (Ellipse, Option<f32>) = if points_inner.len() >= 8 {
        match conic::try_fit_ellipse_ransac(&points_inner, &cfg.ransac) {
            Ok(r) => (
                r.ellipse,
                Some(r.num_inliers as f32 / points_inner.len().max(1) as f32),
            ),
            Err(_) => match conic::fit_ellipse_direct(&points_inner) {
                Some(e) => (e, None),
                None => {
                    return InnerFitResult {
                        status: InnerFitStatus::Failed,
                        reason: Some(InnerFitReason::FitFailed),
                        reason_context: None,
                        ellipse_inner: None,
                        points_inner,
                        ransac_inlier_ratio_inner: None,
                        rms_residual_inner: None,
                        max_angular_gap: None,
                        theta_consistency: estimate.theta_consistency,
                    };
                }
            },
        }
    } else {
        match conic::fit_ellipse_direct(&points_inner) {
            Some(e) => (e, None),
            None => {
                return InnerFitResult {
                    status: InnerFitStatus::Failed,
                    reason: Some(InnerFitReason::FitFailed),
                    reason_context: None,
                    ellipse_inner: None,
                    points_inner,
                    ransac_inlier_ratio_inner: None,
                    rms_residual_inner: None,
                    max_angular_gap: None,
                    theta_consistency: estimate.theta_consistency,
                };
            }
        }
    };

    let mut reject_reason: Option<InnerFitReason> = None;
    let mut reject_context: Option<InnerFitReasonContext> = None;
    if let Some(ir) = inlier_ratio {
        if ir < cfg.min_inlier_ratio {
            reject_reason = Some(InnerFitReason::InlierRatioLow);
            reject_context = Some(InnerFitReasonContext::InlierRatioLow {
                observed_inlier_ratio: ir,
                min_required_inlier_ratio: cfg.min_inlier_ratio,
            });
        }
    }

    let dx = ellipse_inner.cx - outer.cx;
    let dy = ellipse_inner.cy - outer.cy;
    let center_shift = (dx * dx + dy * dy).sqrt();
    if reject_reason.is_none() && center_shift > cfg.max_center_shift_px {
        reject_reason = Some(InnerFitReason::CenterShiftTooLarge);
        reject_context = Some(InnerFitReasonContext::CenterShiftTooLarge {
            observed_shift_px: center_shift,
            max_allowed_shift_px: cfg.max_center_shift_px,
        });
    }

    let mean_outer = ((outer.a + outer.b) * 0.5).max(1e-9);
    let mean_inner = (ellipse_inner.a + ellipse_inner.b) * 0.5;
    let ratio_meas = mean_inner / mean_outer;
    if reject_reason.is_none() {
        if let Some(r_hint) = estimate.r_inner_found {
            if (ratio_meas - r_hint as f64).abs() > cfg.max_ratio_abs_error {
                reject_reason = Some(InnerFitReason::RatioMismatch);
                reject_context = Some(InnerFitReasonContext::RatioMismatch {
                    measured_ratio: ratio_meas,
                    hint_ratio: r_hint,
                    max_allowed_abs_error: cfg.max_ratio_abs_error,
                });
            }
        }
    }

    let rms_residual_inner = conic::rms_sampson_distance(&ellipse_inner, &points_inner);
    let inner_angular_gap = crate::detector::outer_fit::max_angular_gap(
        [ellipse_inner.cx, ellipse_inner.cy],
        &points_inner,
    );

    if reject_reason.is_none()
        && (!rms_residual_inner.is_finite() || rms_residual_inner > cfg.max_rms_residual)
    {
        reject_reason = Some(InnerFitReason::RmsResidualHigh);
        reject_context = Some(InnerFitReasonContext::RmsResidualHigh {
            observed_rms_residual: rms_residual_inner,
            max_allowed_rms_residual: cfg.max_rms_residual,
        });
    }

    if reject_reason.is_none() && inner_angular_gap > cfg.max_angular_gap_rad {
        reject_reason = Some(InnerFitReason::AngularGapTooLarge);
        reject_context = Some(InnerFitReasonContext::AngularGapTooLarge {
            observed_gap_rad: inner_angular_gap,
            max_allowed_gap_rad: cfg.max_angular_gap_rad,
        });
    }

    if let Some(reason) = reject_reason {
        return InnerFitResult {
            status: InnerFitStatus::Rejected,
            reason: Some(reason),
            reason_context: reject_context,
            ellipse_inner: None,
            points_inner,
            ransac_inlier_ratio_inner: inlier_ratio,
            rms_residual_inner: Some(rms_residual_inner),
            max_angular_gap: Some(inner_angular_gap),
            theta_consistency: estimate.theta_consistency,
        };
    }

    InnerFitResult {
        status: InnerFitStatus::Ok,
        reason: None,
        reason_context: None,
        ellipse_inner: Some(ellipse_inner),
        points_inner,
        ransac_inlier_ratio_inner: inlier_ratio,
        rms_residual_inner: Some(rms_residual_inner),
        max_angular_gap: Some(inner_angular_gap),
        theta_consistency: estimate.theta_consistency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marker::GradPolarity;
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
            inner_grad_polarity: GradPolarity::LightToDark,
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
        assert_eq!(
            res.status,
            InnerFitStatus::Ok,
            "reason={:?} context={:?}",
            res.reason,
            res.reason_context
        );
        let e = res.ellipse_inner.expect("inner ellipse");
        assert!((e.cx - outer.cx).abs() < 1.2);
        assert!((e.cy - outer.cy).abs() < 1.2);
        let mean = (e.a + e.b) * 0.5;
        assert!((mean - r_in as f64).abs() < 1.5, "mean={:.3}", mean);
    }

    #[test]
    fn inner_fit_reason_serialization_is_stable() {
        let reason = InnerFitReason::RmsResidualHigh;
        assert_eq!(reason.to_string(), "rms_residual_high");
        let json = serde_json::to_string(&reason).expect("serialize inner fit reason");
        assert_eq!(json, "\"rms_residual_high\"");
    }
}
