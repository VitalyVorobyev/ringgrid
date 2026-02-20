//! Outer edge estimation from a center prior and expected marker scale.
//!
//! The outer edge can be confused with stronger inner-ring or code-band edges
//! under blur/high contrast. This estimator anchors the search to the expected
//! outer radius (derived from marker scale prior) and aggregates radial edge
//! responses over theta, similarly to the inner estimator.

use image::GrayImage;

use crate::marker::{AngularAggregator, GradPolarity};
use crate::pixelmap::PixelMapper;

use super::edge_sample::DistortionAwareSampler;
use super::radial_estimator::{scan_radial_derivatives, RadialSampleGrid};
use super::radial_profile;
use super::radial_profile::Polarity;

/// Outcome category for outer-edge estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OuterStatus {
    /// Outer-radius estimate is valid and passed quality gates.
    Ok,
    /// Estimation failed (invalid inputs or insufficient data).
    Failed,
}

/// Candidate outer-radius hypothesis from aggregated radial response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OuterHypothesis {
    /// Outer radius in pixels.
    pub r_outer_px: f32,
    /// Absolute aggregated peak magnitude.
    pub peak_strength: f32,
    /// Fraction of per-theta peaks close to `r_outer_px`.
    pub theta_consistency: f32,
}

/// Outer-radius estimation result with optional debug traces.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OuterEstimate {
    /// Expected outer radius (pixels) from current scale prior.
    pub r_outer_expected_px: f32,
    /// Radial search window `[min, max]` in pixels.
    pub search_window_px: [f32; 2],
    /// Selected response polarity.
    pub polarity: Option<Polarity>,
    /// Candidate hypotheses sorted by quality (best first).
    pub hypotheses: Vec<OuterHypothesis>,
    /// Final estimator status.
    pub status: OuterStatus,
    /// Optional human-readable reject/failure reason.
    pub reason: Option<String>,
    /// Optional aggregated radial response profile (for debug/analysis).
    pub radial_response_agg: Option<Vec<f32>>,
    /// Optional sampled radii corresponding to `radial_response_agg`.
    pub r_samples: Option<Vec<f32>>,
}

/// Configuration for outer-radius estimation around a center prior.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OuterEstimationConfig {
    /// Search half-width around the expected outer radius, in pixels.
    pub search_halfwidth_px: f32,
    /// Number of radial samples used to build the aggregated response.
    pub radial_samples: usize,
    /// Number of theta samples (rays).
    pub theta_samples: usize,
    /// Aggregation method across theta.
    pub aggregator: AngularAggregator,
    /// Expected polarity of `dI/dr` at the outer edge.
    pub grad_polarity: GradPolarity,
    /// Minimum fraction of theta samples required for an estimate.
    pub min_theta_coverage: f32,
    /// Minimum fraction of theta samples that must agree with the selected peak.
    pub min_theta_consistency: f32,
    /// If set, emit up to two hypotheses (best + runner-up) when runner-up is comparable.
    pub allow_two_hypotheses: bool,
    /// Runner-up must be at least this fraction of the best peak strength.
    pub second_peak_min_rel: f32,
    /// Per-theta local refinement half-width around the chosen radius.
    pub refine_halfwidth_px: f32,
}

impl Default for OuterEstimationConfig {
    fn default() -> Self {
        Self {
            search_halfwidth_px: 4.0,
            radial_samples: 64,
            theta_samples: 48,
            aggregator: AngularAggregator::Median,
            grad_polarity: GradPolarity::DarkToLight,
            min_theta_coverage: 0.6,
            min_theta_consistency: 0.35,
            allow_two_hypotheses: true,
            second_peak_min_rel: 0.85,
            refine_halfwidth_px: 1.0,
        }
    }
}

fn find_local_peaks(score: &[f32]) -> Vec<usize> {
    if score.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 1..(score.len() - 1) {
        if score[i].is_finite() && score[i] >= score[i - 1] && score[i] >= score[i + 1] {
            out.push(i);
        }
    }
    out
}

/// Estimate outer radius around a center prior using radial derivatives.
///
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub fn estimate_outer_from_prior_with_mapper(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    cfg: &OuterEstimationConfig,
    mapper: Option<&dyn PixelMapper>,
    store_response: bool,
) -> OuterEstimate {
    let r_expected = r_outer_expected_px.max(1.0);
    let hw = cfg.search_halfwidth_px.max(0.5);
    let mut window = [r_expected - hw, r_expected + hw];
    window[0] = window[0].max(1.0);
    if window[1] <= window[0] + 1e-3 {
        return OuterEstimate {
            r_outer_expected_px: r_expected,
            search_window_px: window,
            polarity: None,
            hypotheses: Vec::new(),
            status: OuterStatus::Failed,
            reason: Some("invalid_search_window".to_string()),
            radial_response_agg: None,
            r_samples: None,
        };
    }

    let n_r = cfg.radial_samples.max(7);
    let n_t = cfg.theta_samples.max(8);
    let polarity_candidates: Vec<Polarity> = match cfg.grad_polarity {
        GradPolarity::DarkToLight => vec![Polarity::Pos],
        GradPolarity::LightToDark => vec![Polarity::Neg],
        GradPolarity::Auto => vec![Polarity::Pos, Polarity::Neg],
    };
    let track_pos = polarity_candidates.contains(&Polarity::Pos);
    let track_neg = polarity_candidates.contains(&Polarity::Neg);
    let Some(grid) = RadialSampleGrid::from_window(window, n_r) else {
        return OuterEstimate {
            r_outer_expected_px: r_expected,
            search_window_px: window,
            polarity: None,
            hypotheses: Vec::new(),
            status: OuterStatus::Failed,
            reason: Some("invalid_search_window".to_string()),
            radial_response_agg: None,
            r_samples: None,
        };
    };
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let cx = center_prior[0];
    let cy = center_prior[1];

    let scan = scan_radial_derivatives(
        grid,
        n_t,
        track_pos,
        track_neg,
        |ct, st, r_samples, i_vals| {
            for (ri, &r) in r_samples.iter().enumerate() {
                let x = cx + ct * r;
                let y = cy + st * r;
                let Some(sample) = sampler.sample_checked(x, y) else {
                    return false;
                };
                i_vals[ri] = sample;
            }
            true
        },
    );
    let n_r = scan.grid.r_samples.len();

    let coverage = scan.coverage();
    if scan.n_valid_theta == 0 || coverage < cfg.min_theta_coverage {
        return OuterEstimate {
            r_outer_expected_px: r_expected,
            search_window_px: window,
            polarity: None,
            hypotheses: Vec::new(),
            status: OuterStatus::Failed,
            reason: Some(format!(
                "insufficient_theta_coverage({:.2}<{:.2})",
                coverage, cfg.min_theta_coverage
            )),
            radial_response_agg: None,
            r_samples: if store_response {
                Some(scan.grid.r_samples.clone())
            } else {
                None
            },
        };
    }

    let agg_resp = scan.aggregate_response(&cfg.aggregator);

    let mut best: Option<(OuterEstimate, f32)> = None;

    for pol in polarity_candidates {
        // Convert to a score where "larger is better" regardless of polarity.
        let score_vec: Vec<f32> = match pol {
            Polarity::Pos => agg_resp.clone(),
            Polarity::Neg => agg_resp.iter().map(|v| -v).collect(),
        };

        // Candidate peaks: local maxima in score_vec.
        let mut peaks = find_local_peaks(&score_vec);
        if peaks.is_empty() {
            // Fallback: global max if no local maxima found.
            if let Some((idx, _)) = score_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                peaks.push(idx);
            }
        }

        // Sort peaks by score desc.
        peaks.sort_by(|&a, &b| score_vec[b].partial_cmp(&score_vec[a]).unwrap());

        let per_theta = scan.per_theta_peaks(pol);

        let mut hypotheses: Vec<OuterHypothesis> = Vec::new();
        let mut best_strength = None::<f32>;

        for &pi in &peaks {
            if pi == 0 || pi + 1 == n_r {
                continue;
            }
            let r_star = scan.grid.r_samples[pi];
            let peak_strength = agg_resp[pi].abs();
            if best_strength.is_none() {
                best_strength = Some(peak_strength);
            } else if !cfg.allow_two_hypotheses {
                break;
            } else if let Some(bs) = best_strength {
                if peak_strength < (cfg.second_peak_min_rel.clamp(0.0, 1.0) * bs) {
                    break;
                }
            }

            let theta_consistency =
                radial_profile::theta_consistency(per_theta, r_star, scan.grid.r_step, 0.75);

            if theta_consistency < cfg.min_theta_consistency {
                continue;
            }

            hypotheses.push(OuterHypothesis {
                r_outer_px: r_star,
                peak_strength,
                theta_consistency,
            });

            if hypotheses.len() >= 2 {
                break;
            }
        }

        if hypotheses.is_empty() {
            continue;
        }

        let primary = &hypotheses[0];
        let score = primary.peak_strength * primary.theta_consistency;

        let est = OuterEstimate {
            r_outer_expected_px: r_expected,
            search_window_px: window,
            polarity: Some(pol),
            hypotheses,
            status: OuterStatus::Ok,
            reason: None,
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

        match &best {
            Some((_, best_score)) if *best_score >= score => {}
            _ => {
                best = Some((est, score));
            }
        }
    }

    best.map(|(e, _)| e).unwrap_or(OuterEstimate {
        r_outer_expected_px: r_expected,
        search_window_px: window,
        polarity: None,
        hypotheses: Vec::new(),
        status: OuterStatus::Failed,
        reason: Some("no_polarity_candidates".to_string()),
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
        let (w, h) = img.dimensions();
        let mut f = image::ImageBuffer::<Luma<f32>, Vec<f32>>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                f.put_pixel(x, y, Luma([img.get_pixel(x, y)[0] as f32 / 255.0]));
            }
        }
        let blurred = imageproc::filter::gaussian_blur_f32(&f, sigma);
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = blurred.get_pixel(x, y)[0].clamp(0.0, 1.0);
                out.put_pixel(x, y, Luma([(v * 255.0).round() as u8]));
            }
        }
        out
    }

    #[test]
    fn outer_estimator_prefers_expected_outer_over_strong_inner_edge() {
        // Synthetic circular marker:
        // - Outer edge at r_outer (weak contrast)
        // - Strong edge at r_strong (< r_outer), simulating inner/code edges.
        let w = 128u32;
        let h = 128u32;
        let cx = 64.0f32;
        let cy = 64.0f32;
        let r_outer = 30.0f32;
        let r_strong = 22.0f32;

        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let mut val = 0.85f32;

                // Outer dark band (weak)
                if (r - r_outer).abs() <= 1.5 {
                    val = 0.40;
                }

                // Strong inner dark band
                if (r - r_strong).abs() <= 1.5 {
                    val = 0.05;
                }

                img.put_pixel(x, y, Luma([(val * 255.0).round() as u8]));
            }
        }
        let img = blur_gray(&img, 1.0);

        let cfg = OuterEstimationConfig {
            search_halfwidth_px: 4.0,
            radial_samples: 64,
            theta_samples: 64,
            aggregator: AngularAggregator::Median,
            grad_polarity: GradPolarity::DarkToLight,
            min_theta_coverage: 0.5,
            min_theta_consistency: 0.35,
            allow_two_hypotheses: true,
            second_peak_min_rel: 0.7,
            refine_halfwidth_px: 1.0,
        };

        // If we search around r_outer, we should land near r_outer, not the stronger inner edge.
        let est = estimate_outer_from_prior_with_mapper(&img, [cx, cy], r_outer, &cfg, None, true);
        assert_eq!(
            est.status,
            OuterStatus::Ok,
            "outer estimate failed: {:?}",
            est.reason
        );
        assert_eq!(est.polarity, Some(Polarity::Pos));
        let r_found = est.hypotheses[0].r_outer_px;
        assert!(
            (r_found - r_outer).abs() < 2.0,
            "r_found {:.2} should be near expected {:.2}",
            r_found,
            r_outer
        );
    }
}
