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
use super::radial_estimator::{RadialSampleGrid, RadialScanResult, scan_radial_derivatives};
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

/// Typed failure reason for outer-edge estimation.
///
/// Replaces the old `reason: Option<String>` field so callers receive
/// structured data instead of a formatted string to be parsed back.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OuterEstimateFailure {
    /// The radial search window is degenerate (window width ≤ 0).
    InvalidSearchWindow,
    /// Fewer than `min_theta_coverage` fraction of rays produced a valid sample.
    InsufficientThetaCoverage { observed: f32, min_required: f32 },
    /// No polarity produced a hypothesis that passed quality gates.
    NoPolarityCandidates,
}

/// Second-harmonic radius model `r(θ) = c0 + c1·cos 2θ + c2·sin 2θ`.
///
/// This is the first-order radial signature of an eccentric (elliptical)
/// ring seen from its center: the edge radius oscillates at twice the ray
/// angle. A circle has `c1 = c2 = 0`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RadialHarmonic {
    /// Mean radius (pixels).
    pub c0: f32,
    /// cos(2θ) coefficient (pixels).
    pub c1: f32,
    /// sin(2θ) coefficient (pixels).
    pub c2: f32,
}

impl RadialHarmonic {
    /// Model radius at ray angle `theta` (radians).
    #[inline]
    pub fn radius_at(&self, theta: f32) -> f32 {
        let (s, c) = (2.0 * theta).sin_cos();
        self.c0 + self.c1 * c + self.c2 * s
    }

    /// Peak radial deviation from the mean radius (pixels).
    #[inline]
    pub fn amplitude(&self) -> f32 {
        (self.c1 * self.c1 + self.c2 * self.c2).sqrt()
    }
}

/// Candidate outer-radius hypothesis from aggregated radial response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OuterHypothesis {
    /// Outer radius in pixels.
    pub r_outer_px: f32,
    /// Absolute aggregated peak magnitude.
    pub peak_strength: f32,
    /// Fraction of per-theta peaks close to `r_outer_px` (or to the
    /// eccentricity model when one is attached).
    pub theta_consistency: f32,
    /// Eccentricity-aware radius model fitted to the per-theta peaks.
    ///
    /// Present when the per-theta peak field is well explained by a
    /// second-harmonic (elliptical) radius oscillation around this
    /// hypothesis. Downstream edge sampling recenters each ray's local
    /// search on `r_outer_px + (model.radius_at(θ) − model.c0)`.
    #[serde(default)]
    pub radius_model: Option<RadialHarmonic>,
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
    /// Typed failure reason (present when `status` is `Failed`).
    pub failure: Option<OuterEstimateFailure>,
    /// Optional aggregated radial response profile (for debug/analysis).
    pub radial_response_agg: Option<Vec<f32>>,
    /// Optional sampled radii corresponding to `radial_response_agg`.
    pub r_samples: Option<Vec<f32>>,
}

/// Configuration for outer-radius estimation around a center prior.
///
/// The number of theta samples (rays) is **not** stored here; it is passed
/// explicitly to `estimate_outer_from_prior_with_mapper` so the caller can
/// synchronise it with the edge-sampling resolution without a silent override.
/// Set [`crate::EdgeSampleConfig::n_rays`] to control angular density for both stages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct OuterEstimationConfig {
    /// Search half-width around the expected outer radius, in pixels.
    pub search_halfwidth_px: f32,
    /// Number of radial samples used to build the aggregated response.
    ///
    /// Same convention as [`crate::MarkerSpecConfig::radial_samples`], calibrated
    /// independently for the outer estimation stage.
    pub radial_samples: usize,
    /// Aggregation method across theta.
    ///
    /// Same convention as [`crate::MarkerSpecConfig::aggregator`], applied to the outer
    /// radial profile.
    pub aggregator: AngularAggregator,
    /// Expected polarity of `dI/dr` at the outer edge.
    pub grad_polarity: GradPolarity,
    /// Minimum fraction of theta samples required for an estimate.
    ///
    /// Same convention as [`crate::MarkerSpecConfig::min_theta_coverage`], calibrated
    /// independently for the outer estimation stage.
    pub min_theta_coverage: f32,
    /// Minimum fraction of theta samples that must agree with the selected peak.
    ///
    /// Same convention as [`crate::MarkerSpecConfig::min_theta_consistency`]; the outer
    /// estimator uses a stricter default (0.35) than the inner estimator (0.25)
    /// because the outer edge is anchored to a scale prior.
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

fn failed_outer_estimate(
    r_outer_expected_px: f32,
    search_window_px: [f32; 2],
    failure: OuterEstimateFailure,
    r_samples: Option<Vec<f32>>,
) -> OuterEstimate {
    OuterEstimate {
        r_outer_expected_px,
        search_window_px,
        polarity: None,
        hypotheses: Vec::new(),
        status: OuterStatus::Failed,
        failure: Some(failure),
        radial_response_agg: None,
        r_samples,
    }
}

/// Least-squares fit of [`RadialHarmonic`] to `(angle, radius)` pairs.
///
/// Returns `None` when there are too few points or the normal equations are
/// singular (e.g. degenerate angular coverage).
fn fit_radial_harmonic_lsq(angles: &[f32], radii: &[f32]) -> Option<RadialHarmonic> {
    const MIN_POINTS: usize = 8;
    debug_assert_eq!(angles.len(), radii.len());
    if angles.len() < MIN_POINTS {
        return None;
    }

    // Normal equations for the basis [1, cos 2θ, sin 2θ] in f64.
    let (mut sc, mut ss, mut scc, mut sss, mut scs) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    let (mut sr, mut src, mut srs) = (0.0f64, 0.0, 0.0);
    for (&theta, &r) in angles.iter().zip(radii) {
        let (s, c) = (2.0 * f64::from(theta)).sin_cos();
        let r = f64::from(r);
        sc += c;
        ss += s;
        scc += c * c;
        sss += s * s;
        scs += c * s;
        sr += r;
        src += r * c;
        srs += r * s;
    }
    let n = angles.len() as f64;
    let a = nalgebra::Matrix3::new(n, sc, ss, sc, scc, scs, ss, scs, sss);
    let b = nalgebra::Vector3::new(sr, src, srs);
    let x = a.lu().solve(&b)?;
    let model = RadialHarmonic {
        c0: x[0] as f32,
        c1: x[1] as f32,
        c2: x[2] as f32,
    };
    (model.c0.is_finite() && model.c1.is_finite() && model.c2.is_finite()).then_some(model)
}

/// An eccentricity-model fit together with its own quality measure.
struct HarmonicFit {
    model: RadialHarmonic,
    /// RMS of the inlier residuals of the final fit (pixels). A genuine
    /// elliptical signature is explained well by the model (RMS near the
    /// radial quantization); a fit that merely chases noise carries an
    /// amplitude comparable to its own residuals.
    inlier_rms: f32,
}

/// Fit an eccentricity model to the per-theta peak field, with one
/// outlier-rejection refit round (rays locked onto the wrong edge bias a
/// plain least-squares fit).
fn fit_radial_harmonic(angles: &[f32], radii: &[f32], r_step: f32) -> Option<HarmonicFit> {
    let first = fit_radial_harmonic_lsq(angles, radii)?;

    let mut abs_residuals: Vec<f32> = angles
        .iter()
        .zip(radii)
        .map(|(&theta, &r)| (r - first.radius_at(theta)).abs())
        .collect();
    let mid = abs_residuals.len() / 2;
    let (_, mad, _) = abs_residuals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let inlier_tol = (3.0 * *mad).max(r_step);

    let mut in_angles = Vec::with_capacity(angles.len());
    let mut in_radii = Vec::with_capacity(radii.len());
    for (&theta, &r) in angles.iter().zip(radii) {
        if (r - first.radius_at(theta)).abs() <= inlier_tol {
            in_angles.push(theta);
            in_radii.push(r);
        }
    }
    let model = if in_angles.len() == angles.len() {
        first
    } else {
        fit_radial_harmonic_lsq(&in_angles, &in_radii).unwrap_or(first)
    };

    let mut rms_sq = 0.0f32;
    for (&theta, &r) in in_angles.iter().zip(&in_radii) {
        let res = r - model.radius_at(theta);
        rms_sq += res * res;
    }
    let inlier_rms = (rms_sq / in_angles.len().max(1) as f32).sqrt();

    Some(HarmonicFit { model, inlier_rms })
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

/// Build the search window and construct the `RadialSampleGrid` for outer estimation.
///
/// Returns `(window, polarity_candidates, track_pos, track_neg, grid)` on
/// success, or an early-return `OuterEstimate` on failure.
#[allow(clippy::type_complexity)]
fn setup_outer_search_window(
    r_outer_expected_px: f32,
    cfg: &OuterEstimationConfig,
) -> Result<([f32; 2], Vec<Polarity>, bool, bool, RadialSampleGrid), OuterEstimate> {
    let r_expected = r_outer_expected_px.max(1.0);
    let hw = cfg.search_halfwidth_px.max(0.5);
    let mut window = [r_expected - hw, r_expected + hw];
    window[0] = window[0].max(1.0);
    if window[1] <= window[0] + 1e-3 {
        return Err(failed_outer_estimate(
            r_expected,
            window,
            OuterEstimateFailure::InvalidSearchWindow,
            None,
        ));
    }

    let n_r = cfg.radial_samples.max(7);
    let polarity_candidates: Vec<Polarity> = match cfg.grad_polarity {
        GradPolarity::DarkToLight => vec![Polarity::Pos],
        GradPolarity::LightToDark => vec![Polarity::Neg],
        GradPolarity::Auto => vec![Polarity::Pos, Polarity::Neg],
    };
    let track_pos = polarity_candidates.contains(&Polarity::Pos);
    let track_neg = polarity_candidates.contains(&Polarity::Neg);
    let Some(grid) = RadialSampleGrid::from_window(window, n_r) else {
        return Err(failed_outer_estimate(
            r_expected,
            window,
            OuterEstimateFailure::InvalidSearchWindow,
            None,
        ));
    };

    Ok((window, polarity_candidates, track_pos, track_neg, grid))
}

/// Run the radial derivative scan and check theta coverage.
///
/// Returns the `RadialScanResult` on success, or an early-return
/// `OuterEstimate` on failure.
///
/// `fail_ctx` holds `(r_expected, window, store_response)` for the error path.
fn execute_outer_radial_scan(
    gray: &GrayImage,
    center_prior: [f32; 2],
    grid: RadialSampleGrid,
    polarity: (usize, bool, bool),
    mapper: Option<&dyn PixelMapper>,
    cfg: &OuterEstimationConfig,
    fail_ctx: (f32, [f32; 2], bool),
) -> Result<RadialScanResult, OuterEstimate> {
    let (n_t, track_pos, track_neg) = polarity;
    let (r_expected, window, store_response) = fail_ctx;
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

    let coverage = scan.coverage();
    if scan.n_valid_theta == 0 || coverage < cfg.min_theta_coverage {
        return Err(failed_outer_estimate(
            r_expected,
            window,
            OuterEstimateFailure::InsufficientThetaCoverage {
                observed: coverage,
                min_required: cfg.min_theta_coverage,
            },
            store_response.then(|| scan.grid.r_samples.clone()),
        ));
    }

    Ok(scan)
}

/// Fraction of per-theta peaks within tolerance of the eccentricity model.
///
/// Mirrors [`radial_profile::theta_consistency`], but measures each peak
/// against the model radius at its own ray angle instead of one constant
/// radius — a strongly eccentric ring has peaks spread over ±amplitude that
/// are all consistent with a single elliptical edge.
fn theta_consistency_with_model(
    per_theta_peaks: &[f32],
    angles: &[f32],
    model: &RadialHarmonic,
    r_step: f32,
    min_delta: f32,
) -> f32 {
    let delta = (4.0 * r_step).max(min_delta);
    let n_close = per_theta_peaks
        .iter()
        .zip(angles)
        .filter(|&(&r, &theta)| (r - model.radius_at(theta)).abs() <= delta)
        .count();
    n_close as f32 / per_theta_peaks.len().max(1) as f32
}

/// Attach-gates for the eccentricity model on one hypothesis.
///
/// The model must describe the same edge as the aggregated peak (`c0` close
/// to `r_star`), stay a plausible ellipse (bounded relative amplitude),
/// oscillate more than the constant-radius consistency tolerance already
/// absorbs (below that it cannot help), and carry an amplitude that clearly
/// exceeds its own residual noise — a fit chasing a noisy (e.g. heavily
/// blurred) peak field fails the SNR gate, keeping the constant-radius path
/// in charge instead of dragging per-ray refine centers off the true edge.
fn model_applies_to_peak(
    fit: &HarmonicFit,
    r_star: f32,
    window: [f32; 2],
    consistency_delta: f32,
) -> bool {
    const MAX_RELATIVE_AMPLITUDE: f32 = 0.35;
    const MIN_AMPLITUDE_TO_NOISE: f32 = 2.0;
    let model = &fit.model;
    let half_window = 0.5 * (window[1] - window[0]);
    model.c0.is_finite()
        && (model.c0 - r_star).abs() <= 0.5 * half_window
        && model.amplitude() >= consistency_delta
        && model.amplitude() <= MAX_RELATIVE_AMPLITUDE * model.c0
        && model.amplitude() >= MIN_AMPLITUDE_TO_NOISE * fit.inlier_rms
}

/// Find peaks in the aggregated response for a single polarity and assemble
/// up to two `OuterHypothesis` entries that pass theta-consistency gating.
fn build_hypotheses_for_polarity(
    agg_resp: &[f32],
    pol: Polarity,
    scan: &RadialScanResult,
    n_r: usize,
    cfg: &OuterEstimationConfig,
) -> Vec<OuterHypothesis> {
    // Convert to a score where "larger is better" regardless of polarity.
    let score_vec: Vec<f32> = match pol {
        Polarity::Pos => agg_resp.to_vec(),
        Polarity::Neg => agg_resp.iter().map(|v| -v).collect(),
    };

    // Candidate peaks: local maxima in score_vec.
    let mut peaks = find_local_peaks(&score_vec);
    if peaks.is_empty() {
        // Fallback: global max if no local maxima found.
        if let Some((idx, _)) = score_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
        {
            peaks.push(idx);
        }
    }

    // Sort peaks by score desc.
    peaks.sort_by(|&a, &b| score_vec[b].total_cmp(&score_vec[a]));

    let per_theta = scan.per_theta_peaks(pol);
    let window = [
        scan.grid.r_samples[0],
        scan.grid.r_samples[scan.grid.r_samples.len() - 1],
    ];

    // One eccentricity model per polarity: the per-theta peak field is a
    // property of the scan, not of an individual aggregated peak. The delta
    // matches the theta-consistency tolerance below.
    let consistency_delta = (4.0 * scan.grid.r_step).max(0.75);
    let harmonic = fit_radial_harmonic(scan.valid_theta_angles(), per_theta, scan.grid.r_step);

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
        } else if let Some(bs) = best_strength
            && peak_strength < (cfg.second_peak_min_rel.clamp(0.0, 1.0) * bs)
        {
            break;
        }

        let constant_consistency =
            radial_profile::theta_consistency(per_theta, r_star, scan.grid.r_step, 0.75);

        // The model earns its place only by explaining the peak field strictly
        // better than the constant radius; ties keep the legacy path.
        let (radius_model, theta_consistency) = match harmonic
            .as_ref()
            .filter(|fit| model_applies_to_peak(fit, r_star, window, consistency_delta))
        {
            Some(fit) => {
                let model_consistency = theta_consistency_with_model(
                    per_theta,
                    scan.valid_theta_angles(),
                    &fit.model,
                    scan.grid.r_step,
                    0.75,
                );
                if model_consistency > constant_consistency {
                    (Some(fit.model), model_consistency)
                } else {
                    (None, constant_consistency)
                }
            }
            None => (None, constant_consistency),
        };

        if theta_consistency < cfg.min_theta_consistency {
            continue;
        }

        hypotheses.push(OuterHypothesis {
            r_outer_px: r_star,
            peak_strength,
            theta_consistency,
            radius_model,
        });

        if hypotheses.len() >= 2 {
            break;
        }
    }

    hypotheses
}

/// Estimate outer radius around a center prior using radial derivatives.
///
/// `theta_samples` sets the number of angular rays. Pass
/// `edge_sample.n_rays` to keep the outer-estimation and edge-sampling
/// resolutions in sync without a silent config override.
///
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub fn estimate_outer_from_prior_with_mapper(
    gray: &GrayImage,
    center_prior: [f32; 2],
    r_outer_expected_px: f32,
    cfg: &OuterEstimationConfig,
    theta_samples: usize,
    mapper: Option<&dyn PixelMapper>,
    store_response: bool,
) -> OuterEstimate {
    let r_expected = r_outer_expected_px.max(1.0);
    let n_t = theta_samples.max(8);

    let (window, polarity_candidates, track_pos, track_neg, grid) =
        match setup_outer_search_window(r_outer_expected_px, cfg) {
            Ok(v) => v,
            Err(est) => return est,
        };

    let scan = match execute_outer_radial_scan(
        gray,
        center_prior,
        grid,
        (n_t, track_pos, track_neg),
        mapper,
        cfg,
        (r_expected, window, store_response),
    ) {
        Ok(s) => s,
        Err(est) => return est,
    };
    let n_r = scan.grid.r_samples.len();

    let agg_resp = scan.aggregate_response(&cfg.aggregator);

    let mut best: Option<(OuterEstimate, f32)> = None;

    for pol in polarity_candidates {
        let hypotheses = build_hypotheses_for_polarity(&agg_resp, pol, &scan, n_r, cfg);

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
            failure: None,
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

    best.map(|(e, _)| e).unwrap_or_else(|| {
        failed_outer_estimate(
            r_expected,
            window,
            OuterEstimateFailure::NoPolarityCandidates,
            store_response.then(|| scan.grid.r_samples.clone()),
        )
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
    fn radial_harmonic_radius_and_amplitude_match_closed_form() {
        let m = RadialHarmonic {
            c0: 20.0,
            c1: 3.0,
            c2: -4.0,
        };
        // At theta = 0: 2*theta = 0 → cos = 1, sin = 0 → c0 + c1.
        assert!((m.radius_at(0.0) - 23.0).abs() < 1e-5);
        // At theta = PI/4: 2*theta = PI/2 → cos = 0, sin = 1 → c0 + c2.
        assert!((m.radius_at(std::f32::consts::FRAC_PI_4) - 16.0).abs() < 1e-4);
        // Amplitude is the L2 norm of (c1, c2) = sqrt(9 + 16) = 5.
        assert!((m.amplitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn find_local_peaks_reports_interior_maxima() {
        // Two clear interior maxima at indices 1 and 3.
        assert_eq!(find_local_peaks(&[0.0, 1.0, 0.0, 2.0, 0.0]), vec![1, 3]);
        // A plateau: both shoulders qualify (>= comparison on both sides).
        assert_eq!(find_local_peaks(&[0.0, 1.0, 1.0, 0.0]), vec![1, 2]);
        // Monotone curves have no interior peak.
        assert!(find_local_peaks(&[3.0, 2.0, 1.0]).is_empty());
        // Endpoints are never reported and short curves yield nothing.
        assert!(find_local_peaks(&[1.0, 2.0]).is_empty());
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
            aggregator: AngularAggregator::Median,
            grad_polarity: GradPolarity::DarkToLight,
            min_theta_coverage: 0.5,
            min_theta_consistency: 0.35,
            allow_two_hypotheses: true,
            second_peak_min_rel: 0.7,
            refine_halfwidth_px: 1.0,
        };

        // If we search around r_outer, we should land near r_outer, not the stronger inner edge.
        let est =
            estimate_outer_from_prior_with_mapper(&img, [cx, cy], r_outer, &cfg, 64, None, true);
        assert_eq!(
            est.status,
            OuterStatus::Ok,
            "outer estimate failed: {:?}",
            est.failure
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

    #[test]
    fn fit_radial_harmonic_recovers_elliptical_signature() {
        // Synthesize per-theta peaks from a known model plus quantization noise.
        let truth = RadialHarmonic {
            c0: 27.0,
            c1: 2.4,
            c2: -1.1,
        };
        let n = 48;
        let angles: Vec<f32> = (0..n)
            .map(|i| i as f32 * 2.0 * std::f32::consts::PI / n as f32)
            .collect();
        let radii: Vec<f32> = angles
            .iter()
            .enumerate()
            .map(|(i, &t)| truth.radius_at(t) + if i % 2 == 0 { 0.15 } else { -0.15 })
            .collect();

        let fit = fit_radial_harmonic(&angles, &radii, 0.4).expect("fit succeeds");
        let m = fit.model;
        assert!((m.c0 - truth.c0).abs() < 0.1, "c0 {:.3}", m.c0);
        assert!((m.c1 - truth.c1).abs() < 0.1, "c1 {:.3}", m.c1);
        assert!((m.c2 - truth.c2).abs() < 0.1, "c2 {:.3}", m.c2);
        assert!(
            fit.inlier_rms <= 0.2,
            "clean signal should have small residual RMS, got {:.3}",
            fit.inlier_rms
        );
    }

    #[test]
    fn fit_radial_harmonic_rejects_contaminated_rays() {
        // A sixth of the rays lock onto a much closer (inner) edge — a
        // realistic partial-occlusion level. The outlier-rejection round must
        // keep the fit on the outer signature. (At contamination approaching
        // the MAD breakdown the fit stays biased; the `model_applies_to_peak`
        // gate then rejects it and the constant-radius path takes over.)
        let truth = RadialHarmonic {
            c0: 27.0,
            c1: 2.0,
            c2: 0.0,
        };
        let n = 48;
        let angles: Vec<f32> = (0..n)
            .map(|i| i as f32 * 2.0 * std::f32::consts::PI / n as f32)
            .collect();
        let radii: Vec<f32> = angles
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if i % 6 == 0 {
                    20.0 // inner-edge contamination
                } else {
                    truth.radius_at(t)
                }
            })
            .collect();

        let fit = fit_radial_harmonic(&angles, &radii, 0.4).expect("fit succeeds");
        let m = fit.model;
        assert!(
            (m.c0 - truth.c0).abs() < 0.5,
            "contaminated c0 {:.3} should stay near 27",
            m.c0
        );
        assert!((m.c1 - truth.c1).abs() < 0.5, "c1 {:.3}", m.c1);
    }

    #[test]
    fn outer_estimator_attaches_model_on_eccentric_ring() {
        // Elliptic dark ring with amplitude (a-b)/2 = 3 px: per-theta peaks
        // spread over ±3 px, well past the constant-radius consistency
        // tolerance, so the estimator needs the eccentricity model.
        let w = 160u32;
        let h = 160u32;
        let cx = 80.0f32;
        let cy = 80.0f32;
        let a = 30.0f32;
        let b = 24.0f32;
        let angle = 0.4f32;

        let ca = angle.cos();
        let sa = angle.sin();
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let xr = ca * dx + sa * dy;
                let yr = -sa * dx + ca * dy;
                let rho = ((xr / a).powi(2) + (yr / b).powi(2)).sqrt();
                let val: f32 = if (0.55..=1.0).contains(&rho) {
                    0.15
                } else {
                    0.9
                };
                img.put_pixel(x, y, Luma([(val * 255.0).round() as u8]));
            }
        }
        let img = blur_gray(&img, 1.0);

        let r_mean = 0.5 * (a + b);
        let cfg = OuterEstimationConfig {
            search_halfwidth_px: 6.0,
            ..OuterEstimationConfig::default()
        };

        let est =
            estimate_outer_from_prior_with_mapper(&img, [cx, cy], r_mean, &cfg, 64, None, false);
        assert_eq!(
            est.status,
            OuterStatus::Ok,
            "eccentric outer estimate failed: {:?}",
            est.failure
        );
        let hyp = &est.hypotheses[0];
        let model = hyp
            .radius_model
            .as_ref()
            .expect("eccentric ring should attach a radius model");
        // Amplitude should reflect (a - b) / 2 = 3 px.
        assert!(
            (model.amplitude() - 3.0).abs() < 1.0,
            "model amplitude {:.2} should be near 3.0",
            model.amplitude()
        );
        assert!(
            hyp.theta_consistency >= cfg.min_theta_consistency,
            "model-based consistency {:.2} should pass the gate",
            hyp.theta_consistency
        );
    }
}
