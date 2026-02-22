//! Inner edge estimation anchored on the fitted outer ellipse.
//!
//! The inner edge is estimated in the outer-ellipse-normalized coordinate
//! system by aggregating radial edge responses over theta. This avoids fitting
//! an unconstrained inner ellipse from potentially-confusing code-band edges.

use image::GrayImage;

use crate::conic::Ellipse;
use crate::marker::{GradPolarity, MarkerSpec};
use crate::pixelmap::PixelMapper;

use super::edge_sample::DistortionAwareSampler;
use super::radial_estimator::{scan_radial_derivatives, RadialSampleGrid};
use super::radial_profile;
pub use super::radial_profile::Polarity;

/// Outcome category for inner-edge estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InnerStatus {
    /// Inner-edge estimate is valid and passed quality gates.
    Ok,
    /// Inner-edge estimate was computed but rejected by quality gates.
    Rejected,
    /// Estimation failed (invalid inputs or insufficient data).
    Failed,
}

/// Typed failure/rejection reason for inner-edge estimation.
///
/// Replaces the old `reason: Option<String>` field so callers receive
/// structured data rather than a formatted string.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InnerEstimateFailure {
    /// The outer ellipse is invalid or too small to sample.
    InvalidOuterEllipse,
    /// The normalized inner search window is degenerate.
    InvalidSearchWindow,
    /// Fewer than `min_theta_coverage` fraction of rays produced a valid sample.
    InsufficientThetaCoverage { observed: f32, min_required: f32 },
    /// No polarity produced a valid estimate.
    NoPolarityCandidates,
    /// The selected peak is out of the valid [0.2, 0.9] normalized range.
    ScaleOutOfBounds { r_star: f32 },
    /// The selected peak sits at the edge of the search window.
    PeakAtSearchWindowEdge,
    /// Theta consistency is below the minimum threshold.
    ThetaInconsistent { observed: f32, min_required: f32 },
}

/// Inner-edge estimation result anchored on the fitted outer ellipse.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InnerEstimate {
    /// Expected normalized inner radius (`Rin/Rout`).
    pub r_inner_expected: f32,
    /// Search window in normalized radius units.
    pub search_window: [f32; 2],
    /// Recovered normalized inner radius when available.
    pub r_inner_found: Option<f32>,
    /// Selected radial derivative polarity.
    pub polarity: Option<Polarity>,
    /// Absolute aggregated peak magnitude at the selected radius.
    pub peak_strength: Option<f32>,
    /// Fraction of theta samples consistent with the selected radius.
    pub theta_consistency: Option<f32>,
    /// Final estimator status.
    pub status: InnerStatus,
    /// Typed failure/rejection reason (present when `status` is not `Ok`).
    pub failure: Option<InnerEstimateFailure>,
    /// Optional aggregated radial response profile (for debug/analysis).
    pub radial_response_agg: Option<Vec<f32>>,
    /// Optional sampled normalized radii corresponding to `radial_response_agg`.
    pub r_samples: Option<Vec<f32>>,
}

/// Estimate inner ring scale from a fitted outer ellipse.
///
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub fn estimate_inner_scale_from_outer_with_mapper(
    gray: &GrayImage,
    outer: &Ellipse,
    spec: &MarkerSpec,
    mapper: Option<&dyn PixelMapper>,
    store_response: bool,
) -> InnerEstimate {
    if !outer.is_valid() || outer.a < 2.0 || outer.b < 2.0 {
        return InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: spec.search_window(),
            r_inner_found: None,
            polarity: None,
            peak_strength: None,
            theta_consistency: None,
            status: InnerStatus::Failed,
            failure: Some(InnerEstimateFailure::InvalidOuterEllipse),
            radial_response_agg: None,
            r_samples: None,
        };
    }

    let mut window = spec.search_window();
    // Clamp to plausible normalized bounds. (These are intentionally wide.)
    window[0] = window[0].clamp(0.2, 0.9);
    window[1] = window[1].clamp(0.2, 0.9);
    if window[1] <= window[0] + 1e-6 {
        return InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: window,
            r_inner_found: None,
            polarity: None,
            peak_strength: None,
            theta_consistency: None,
            status: InnerStatus::Failed,
            failure: Some(InnerEstimateFailure::InvalidSearchWindow),
            radial_response_agg: None,
            r_samples: None,
        };
    }

    let n_r = spec.radial_samples.max(5);
    let n_t = spec.theta_samples.max(8);
    let polarity_candidates: Vec<Polarity> = match spec.inner_grad_polarity {
        GradPolarity::DarkToLight => vec![Polarity::Pos],
        GradPolarity::LightToDark => vec![Polarity::Neg],
        GradPolarity::Auto => vec![Polarity::Neg, Polarity::Pos],
    };
    let track_pos = polarity_candidates.contains(&Polarity::Pos);
    let track_neg = polarity_candidates.contains(&Polarity::Neg);
    let Some(grid) = RadialSampleGrid::from_window(window, n_r) else {
        return InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: window,
            r_inner_found: None,
            polarity: None,
            peak_strength: None,
            theta_consistency: None,
            status: InnerStatus::Failed,
            failure: Some(InnerEstimateFailure::InvalidSearchWindow),
            radial_response_agg: None,
            r_samples: None,
        };
    };

    // Precompute rotation for ellipse sampling
    let cx = outer.cx as f32;
    let cy = outer.cy as f32;
    let a = outer.a as f32;
    let b = outer.b as f32;
    let ca = (outer.angle as f32).cos();
    let sa = (outer.angle as f32).sin();
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let scan = scan_radial_derivatives(
        grid,
        n_t,
        track_pos,
        track_neg,
        |ct, st, r_samples, i_vals| {
            for (ri, &r) in r_samples.iter().enumerate() {
                // v = [a*r*cosθ, b*r*sinθ] then rotate by ellipse angle
                let vx = a * r * ct;
                let vy = b * r * st;
                let x = cx + ca * vx - sa * vy;
                let y = cy + sa * vx + ca * vy;
                let Some(sample) = sampler.sample_checked(x, y) else {
                    return false;
                };
                i_vals[ri] = sample;
            }
            true
        },
    );

    let coverage = scan.coverage();
    if coverage < spec.min_theta_coverage {
        return InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: window,
            r_inner_found: None,
            polarity: None,
            peak_strength: None,
            theta_consistency: Some(coverage),
            status: InnerStatus::Failed,
            failure: Some(InnerEstimateFailure::InsufficientThetaCoverage {
                observed: coverage,
                min_required: spec.min_theta_coverage,
            }),
            radial_response_agg: None,
            r_samples: if store_response {
                Some(scan.grid.r_samples.clone())
            } else {
                None
            },
        };
    }

    fn find_peak_idx(agg: &[f32], pol: Polarity) -> (usize, f32) {
        let idx = radial_profile::peak_idx(agg, pol);
        (idx, agg[idx])
    }

    let agg_resp = scan.aggregate_response(&spec.aggregator);

    let mut best: Option<(InnerEstimate, f32)> = None;

    for pol in polarity_candidates {
        let (peak_idx, peak_val) = find_peak_idx(&agg_resp, pol);
        let r_star = scan.grid.r_samples[peak_idx];

        // Consistency: how many per-theta peaks agree with r_star
        let per_theta = scan.per_theta_peaks(pol);
        let theta_consistency =
            radial_profile::theta_consistency(per_theta, r_star, scan.grid.r_step, 0.02);

        let peak_strength = peak_val.abs();

        let mut status = InnerStatus::Ok;
        let mut failure = None;
        if !(0.2..=0.9).contains(&r_star) {
            status = InnerStatus::Rejected;
            failure = Some(InnerEstimateFailure::ScaleOutOfBounds { r_star });
        } else if peak_idx == 0 || peak_idx + 1 == n_r {
            status = InnerStatus::Rejected;
            failure = Some(InnerEstimateFailure::PeakAtSearchWindowEdge);
        } else if theta_consistency < spec.min_theta_consistency {
            status = InnerStatus::Rejected;
            failure = Some(InnerEstimateFailure::ThetaInconsistent {
                observed: theta_consistency,
                min_required: spec.min_theta_consistency,
            });
        }

        let est = InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: window,
            r_inner_found: Some(r_star),
            polarity: Some(pol),
            peak_strength: Some(peak_strength),
            theta_consistency: Some(theta_consistency),
            status,
            failure,
            radial_response_agg: if store_response {
                Some(agg_resp.clone())
            } else {
                None
            },
            r_samples: if store_response {
                Some(scan.grid.r_samples.clone())
            } else {
                None
            },
        };

        // Heuristic score for auto polarity selection
        let score = peak_strength * theta_consistency;
        match &best {
            Some((_, best_score)) if *best_score >= score => {}
            _ => {
                best = Some((est, score));
            }
        }
    }

    best.map(|(e, _)| e).unwrap_or(InnerEstimate {
        r_inner_expected: spec.r_inner_expected,
        search_window: window,
        r_inner_found: None,
        polarity: None,
        peak_strength: None,
        theta_consistency: None,
        status: InnerStatus::Failed,
        failure: Some(InnerEstimateFailure::NoPolarityCandidates),
        radial_response_agg: None,
        r_samples: if store_response {
            Some(scan.grid.r_samples.clone())
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn blur_gray(img: &GrayImage, sigma: f32) -> GrayImage {
        crate::test_utils::blur_gray(img, sigma)
    }

    #[test]
    fn inner_scale_estimator_ignores_codeband_edge() {
        // Synthetic circular marker:
        // - Outer edge at r_outer
        // - Inner edge at r_inner
        // - Code band starts slightly outside r_inner, creating a strong
        //   (dark->light) edge that must NOT be selected when expecting
        //   the inner (light->dark) edge.
        let w = 128u32;
        let h = 128u32;
        let cx = 64.0f32;
        let cy = 64.0f32;
        let r_outer = 30.0f32;
        let r_inner = 20.0f32;
        let code_inner = 21.0f32;
        let code_outer = 29.0f32;

        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let mut val = 0.85f32;

                // Dark annulus between inner and outer edges.
                if r >= r_inner && r <= r_outer {
                    val = 0.10;
                }

                // Code band inside the annulus (overwrites interior but leaves edges dark).
                if r >= code_inner && r <= code_outer {
                    // Simple 16-sector pattern.
                    let ang = dy.atan2(dx);
                    let sector = (((ang / (2.0 * std::f32::consts::PI) + 0.5) * 16.0) as i32)
                        .rem_euclid(16) as usize;
                    let bit = (sector % 2) as u8;
                    val = if bit == 1 { 1.0 } else { 0.15 };
                }

                img.put_pixel(x, y, Luma([(val * 255.0).round() as u8]));
            }
        }

        let img = blur_gray(&img, 1.2);

        let outer_ellipse = Ellipse {
            cx: cx as f64,
            cy: cy as f64,
            a: r_outer as f64,
            b: r_outer as f64,
            angle: 0.0,
        };

        let spec = MarkerSpec {
            r_inner_expected: r_inner / r_outer,
            inner_grad_polarity: GradPolarity::LightToDark,
            inner_search_halfwidth: 0.08,
            theta_samples: 64,
            radial_samples: 64,
            min_theta_coverage: 0.5,
            ..MarkerSpec::default()
        };

        let est =
            estimate_inner_scale_from_outer_with_mapper(&img, &outer_ellipse, &spec, None, true);
        assert_eq!(
            est.status,
            InnerStatus::Ok,
            "inner estimate should succeed: {:?}",
            est.failure
        );
        let r_found = est.r_inner_found.unwrap();
        assert!(
            (r_found - spec.r_inner_expected).abs() < 0.06,
            "r_found {:.3} should be near expected {:.3}",
            r_found,
            spec.r_inner_expected
        );
        assert_eq!(est.polarity, Some(Polarity::Neg));
        assert!(est.theta_consistency.unwrap_or(0.0) >= spec.min_theta_coverage);
    }
}
