//! Inner edge estimation anchored on the fitted outer ellipse.
//!
//! The inner edge is estimated in the outer-ellipse-normalized coordinate
//! system by aggregating radial edge responses over theta. This avoids fitting
//! an unconstrained inner ellipse from potentially-confusing code-band edges.

use image::GrayImage;

use crate::conic::Ellipse;
use crate::marker::{GradPolarity, MarkerSpecConfig};
use crate::pixelmap::PixelMapper;

use super::edge_sample::DistortionAwareSampler;
use super::radial_estimator::{RadialSampleGrid, RadialScanResult, scan_radial_derivatives};
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

/// The nominal search window ([`MarkerSpecConfig::search_window`]) with its
/// inner bound moved `inward_reach` towards the centre (see [`blur_reach`]:
/// the hole inside the inner edge holds no other edge, so only that bound
/// moves).
fn search_window(spec: &MarkerSpecConfig, inward_reach: f32) -> [f32; 2] {
    let [lo, hi] = spec.search_window();
    [lo - inward_reach.max(0.0), hi]
}

/// Build the search window, validate it, and construct the `RadialSampleGrid`.
///
/// Returns `(window, polarity_candidates, grid)` on
/// success, or an early-return `InnerEstimate` on failure.
#[allow(clippy::type_complexity)]
fn setup_inner_search_window(
    spec: &MarkerSpecConfig,
    inward_reach: f32,
    n_r: usize,
) -> Result<([f32; 2], Vec<Polarity>, RadialSampleGrid), InnerEstimate> {
    let mut window = search_window(spec, inward_reach);
    window[0] = window[0].clamp(0.2, 0.9);
    window[1] = window[1].clamp(0.2, 0.9);
    if window[1] <= window[0] + 1e-6 {
        return Err(InnerEstimate {
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
        });
    }

    let polarity_candidates: Vec<Polarity> = match spec.inner_grad_polarity {
        GradPolarity::DarkToLight => vec![Polarity::Pos],
        GradPolarity::LightToDark => vec![Polarity::Neg],
        GradPolarity::Auto => vec![Polarity::Neg, Polarity::Pos],
    };
    let Some(grid) = RadialSampleGrid::from_window(window, n_r) else {
        return Err(InnerEstimate {
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
        });
    };

    Ok((window, polarity_candidates, grid))
}

/// Iterate over polarity candidates and select the best inner-radius hypothesis.
///
/// Returns the best `InnerEstimate` found, or `None` if no polarity produced a
/// valid candidate.
fn select_best_polarity_hypothesis(
    polarity_candidates: &[Polarity],
    agg_resp: &[f32],
    scan: &RadialScanResult,
    spec: &MarkerSpecConfig,
    window: [f32; 2],
    n_r: usize,
    store_response: bool,
) -> Option<InnerEstimate> {
    fn find_peak_idx(agg: &[f32], pol: Polarity) -> (usize, f32) {
        let idx = radial_profile::peak_idx(agg, pol);
        (idx, agg[idx])
    }

    let mut best: Option<(InnerEstimate, f32)> = None;

    for &pol in polarity_candidates {
        let (peak_idx, peak_val) = find_peak_idx(agg_resp, pol);
        let r_star = scan.grid.r_samples[peak_idx];

        // Rays vote within the nominal window's width: the window itself, or
        // that width centred on the selected radius when blur widened it (a
        // ray's strongest response anywhere in the wider window is often the
        // blank hole's noise, not the edge).
        let band = if window[1] - window[0] > 2.0 * spec.inner_search_halfwidth + 1e-6 {
            let hw = spec.inner_search_halfwidth;
            [r_star - hw, r_star + hw]
        } else {
            window
        };
        let per_theta = scan.per_theta_peaks_in(pol, band);
        let theta_consistency =
            radial_profile::theta_consistency(&per_theta, r_star, scan.grid.r_step, 0.02);

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
                Some(agg_resp.to_vec())
            } else {
                None
            },
            r_samples: if store_response {
                Some(scan.grid.r_samples.clone())
            } else {
                None
            },
        };

        let score = peak_strength * theta_consistency;
        match &best {
            Some((_, best_score)) if *best_score >= score => {}
            _ => {
                best = Some((est, score));
            }
        }
    }

    best.map(|(e, _)| e)
}

/// How far (outer-normalized radius) blur can pull the inner-edge peak below
/// its sharp position, for an outer edge whose blur ramp is
/// `outer_edge_width_px` wide on an outer ellipse of mean semi-axis
/// `outer_radius_px`.
///
/// Blur merges each ring stroke's two edges, and the far edge's
/// opposite-signed derivative pushes the near edge's peak away from the stroke
/// by up to the blur sigma: the inner edge's peak moves towards the centre and
/// the outer edge's peak outwards, so the measured ratio falls by up to
/// `(1 + r_inner_expected) · σ / R`. The width of a Gaussian-blurred step is
/// `σ·√(2π)` ([`crate::ring::edge_quality::outer_edge_width_px`]).
fn blur_reach(
    spec: &MarkerSpecConfig,
    outer_edge_width_px: Option<f32>,
    outer_radius_px: f32,
) -> f32 {
    let Some(width) = outer_edge_width_px.filter(|w| w.is_finite() && *w > 0.0) else {
        return 0.0;
    };
    let sigma = width / (2.0 * std::f32::consts::PI).sqrt();
    (1.0 + spec.r_inner_expected) * sigma / outer_radius_px.max(1.0)
}

/// Estimate inner ring scale from a fitted outer ellipse.
///
/// Searches the nominal window ([`MarkerSpecConfig::search_window`]) first. When its peak sits on the window's inner bound — the inner
/// edge lies further in, where blur moves it — it searches once more with the
/// inner bound moved in by the blur reach of the outer edge's measured width
/// `outer_edge_width_px` ([`crate::ring::edge_quality::outer_edge_width_px`]).
/// Rings whose nominal search succeeds, or fails otherwise, are unaffected.
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub fn estimate_inner_scale_from_outer_with_mapper(
    gray: &GrayImage,
    outer: &Ellipse,
    spec: &MarkerSpecConfig,
    outer_edge_width_px: Option<f32>,
    mapper: Option<&dyn PixelMapper>,
    store_response: bool,
) -> InnerEstimate {
    let nominal = estimate_in_window(gray, outer, spec, 0.0, mapper, store_response);
    let peak_at_inner_bound = nominal.failure == Some(InnerEstimateFailure::PeakAtSearchWindowEdge)
        && nominal
            .r_inner_found
            .is_some_and(|r| r <= nominal.search_window[0]);
    let reach = blur_reach(spec, outer_edge_width_px, outer.mean_axis() as f32);
    if peak_at_inner_bound && reach > 0.0 {
        estimate_in_window(gray, outer, spec, reach, mapper, store_response)
    } else {
        nominal
    }
}

/// One inner-edge search over the nominal window with its inner bound moved in
/// by `inward_reach` (see [`search_window`]).
fn estimate_in_window(
    gray: &GrayImage,
    outer: &Ellipse,
    spec: &MarkerSpecConfig,
    inward_reach: f32,
    mapper: Option<&dyn PixelMapper>,
    store_response: bool,
) -> InnerEstimate {
    if !outer.is_valid() || outer.a < 2.0 || outer.b < 2.0 {
        return InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: search_window(spec, inward_reach),
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

    let n_r = spec.radial_samples.max(5);
    let n_t = spec.theta_samples.max(8);

    let (window, polarity_candidates, grid) =
        match setup_inner_search_window(spec, inward_reach, n_r) {
            Ok(v) => v,
            Err(est) => return est,
        };

    // Precompute rotation for ellipse sampling
    let cx = outer.cx as f32;
    let cy = outer.cy as f32;
    let a = outer.a as f32;
    let b = outer.b as f32;
    let ca = (outer.angle as f32).cos();
    let sa = (outer.angle as f32).sin();
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let scan = scan_radial_derivatives(grid, n_t, false, false, |ct, st, r_samples, i_vals| {
        for (ri, &r) in r_samples.iter().enumerate() {
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
    });

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

    let agg_resp = scan.aggregate_response(&spec.aggregator);

    select_best_polarity_hypothesis(
        &polarity_candidates,
        &agg_resp,
        &scan,
        spec,
        window,
        n_r,
        store_response,
    )
    .unwrap_or(InnerEstimate {
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

        let spec = MarkerSpecConfig {
            r_inner_expected: r_inner / r_outer,
            inner_grad_polarity: GradPolarity::LightToDark,
            inner_search_halfwidth: 0.08,
            theta_samples: 64,
            radial_samples: 64,
            min_theta_coverage: 0.5,
            ..MarkerSpecConfig::default()
        };

        let est = estimate_inner_scale_from_outer_with_mapper(
            &img,
            &outer_ellipse,
            &spec,
            None,
            None,
            true,
        );
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

    /// A small coded ring of the default target's geometry (strokes centred
    /// on 3.2 / 4.8 mm, 1.153 mm wide; 16 code sectors between them),
    /// blurred to σ = `sigma` px.
    fn blurred_coded_ring(outer_px: f32, sigma: f32) -> (GrayImage, Ellipse, MarkerSpecConfig) {
        let (w, c) = (96u32, 48.0f32);
        let mm = outer_px / 4.8;
        let half_stroke = 0.5 * 1.153 * mm;
        let (r_in, r_out) = (3.2 * mm, 4.8 * mm);
        let img = GrayImage::from_fn(w, w, |x, y| {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            let r = dx.hypot(dy);
            let sector = ((dy.atan2(dx) / std::f32::consts::TAU + 0.5) * 16.0) as i32;
            let ink = (r - r_in).abs() <= half_stroke
                || (r - r_out).abs() <= half_stroke
                || (r > r_in && r < r_out && sector % 2 == 0);
            Luma([if ink { 30 } else { 220 }])
        });
        let edge = r_out + half_stroke;
        let outer = Ellipse {
            cx: f64::from(c),
            cy: f64::from(c),
            a: f64::from(edge),
            b: f64::from(edge),
            angle: 0.0,
        };
        let spec = MarkerSpecConfig {
            r_inner_expected: (r_in - half_stroke) / edge,
            ..MarkerSpecConfig::default()
        };
        (blur_gray(&img, sigma), outer, spec)
    }

    #[test]
    fn blurred_inner_edge_is_found_past_the_nominal_window() {
        let (img, outer, spec) = blurred_coded_ring(10.0, 3.0);
        let width = crate::ring::edge_quality::outer_edge_width_px(
            DistortionAwareSampler::new(&img, None),
            &outer,
        );
        // Blur pulled the inner edge's peak inside the nominal window: the
        // nominal search ends on its inner bound.
        let nominal =
            estimate_inner_scale_from_outer_with_mapper(&img, &outer, &spec, None, None, false);
        assert_eq!(
            nominal.failure,
            Some(InnerEstimateFailure::PeakAtSearchWindowEdge),
            "{nominal:?}"
        );
        // With the outer edge's measured width, the search reaches it.
        let est =
            estimate_inner_scale_from_outer_with_mapper(&img, &outer, &spec, width, None, false);
        assert_eq!(
            est.status,
            InnerStatus::Ok,
            "{est:?} (edge width {width:?})"
        );
        let r = est.r_inner_found.unwrap();
        assert!(
            r < nominal.search_window[0],
            "{r} vs {:?}",
            nominal.search_window
        );
        assert!(
            r > spec.r_inner_expected
                - spec.inner_search_halfwidth
                - blur_reach(&spec, width, outer.mean_axis() as f32),
            "{r}"
        );
    }

    #[test]
    fn sharp_inner_edge_ignores_the_edge_width() {
        // A sharp ring's nominal search succeeds; the edge width changes nothing.
        let (img, outer, spec) = blurred_coded_ring(12.0, 0.6);
        let width = crate::ring::edge_quality::outer_edge_width_px(
            DistortionAwareSampler::new(&img, None),
            &outer,
        );
        let nominal =
            estimate_inner_scale_from_outer_with_mapper(&img, &outer, &spec, None, None, false);
        let est =
            estimate_inner_scale_from_outer_with_mapper(&img, &outer, &spec, width, None, false);
        assert_eq!(nominal.status, InnerStatus::Ok, "{nominal:?}");
        assert_eq!(est.r_inner_found, nominal.r_inner_found);
        assert_eq!(est.search_window, nominal.search_window);
    }

    #[test]
    fn blur_reach_follows_the_ramp_sigma() {
        let spec = MarkerSpecConfig::default();
        assert_eq!(blur_reach(&spec, None, 10.0), 0.0);
        assert_eq!(blur_reach(&spec, Some(f32::NAN), 10.0), 0.0);
        // A ramp 2.5 σ wide (σ = 1 px) on a 10 px ring.
        let w = (2.0 * std::f32::consts::PI).sqrt();
        let reach = blur_reach(&spec, Some(w), 10.0);
        assert!(
            (reach - (1.0 + spec.r_inner_expected) * 0.1).abs() < 1e-6,
            "{reach}"
        );
    }
}
