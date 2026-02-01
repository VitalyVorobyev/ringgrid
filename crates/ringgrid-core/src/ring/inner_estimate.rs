//! Inner edge estimation anchored on the fitted outer ellipse.
//!
//! The inner edge is estimated in the outer-ellipse-normalized coordinate
//! system by aggregating radial edge responses over theta. This avoids fitting
//! an unconstrained inner ellipse from potentially-confusing code-band edges.

use image::GrayImage;

use crate::conic::Ellipse;
use crate::marker_spec::{AngularAggregator, InnerGradPolarity, MarkerSpec};

use super::edge_sample::bilinear_sample_u8_checked;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Pos,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerStatus {
    Ok,
    Rejected,
    Failed,
}

#[derive(Debug, Clone)]
pub struct InnerEstimate {
    pub r_inner_expected: f32,
    pub search_window: [f32; 2],
    pub r_inner_found: Option<f32>,
    pub polarity: Option<Polarity>,
    pub peak_strength: Option<f32>,
    pub theta_consistency: Option<f32>,
    pub status: InnerStatus,
    pub reason: Option<String>,
    pub radial_response_agg: Option<Vec<f32>>,
    pub r_samples: Option<Vec<f32>>,
}

pub fn estimate_inner_scale_from_outer(
    gray: &GrayImage,
    outer: &Ellipse,
    spec: &MarkerSpec,
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
            reason: Some("invalid_outer_ellipse".to_string()),
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
            reason: Some("invalid_search_window".to_string()),
            radial_response_agg: None,
            r_samples: None,
        };
    }

    let n_r = spec.radial_samples.max(5);
    let n_t = spec.theta_samples.max(8);

    let r_step = (window[1] - window[0]) / (n_r as f32 - 1.0);
    let r_samples: Vec<f32> = (0..n_r).map(|i| window[0] + i as f32 * r_step).collect();

    // Precompute rotation for ellipse sampling
    let cx = outer.cx as f32;
    let cy = outer.cy as f32;
    let a = outer.a as f32;
    let b = outer.b as f32;
    let ca = (outer.angle as f32).cos();
    let sa = (outer.angle as f32).sin();

    // Collect per-theta derivative curves (only for thetas fully in-bounds).
    let mut curves: Vec<Vec<f32>> = Vec::new();
    let mut theta_indices: Vec<usize> = Vec::new();

    for ti in 0..n_t {
        let theta = ti as f32 * 2.0 * std::f32::consts::PI / n_t as f32;
        let ct = theta.cos();
        let st = theta.sin();

        // Sample intensity along r
        let mut i_vals: Vec<f32> = Vec::with_capacity(n_r);
        let mut ok = true;
        for &r in &r_samples {
            // v = [a*r*cosθ, b*r*sinθ] then rotate by ellipse angle
            let vx = a * r * ct;
            let vy = b * r * st;
            let x = cx + ca * vx - sa * vy;
            let y = cy + sa * vx + ca * vy;

            let samp = match bilinear_sample_u8_checked(gray, x, y) {
                Some(v) => v,
                None => {
                    ok = false;
                    break;
                }
            };
            i_vals.push(samp);
        }
        if !ok {
            continue;
        }

        // Radial derivative dI/dr (central differences)
        let mut d = vec![0.0f32; n_r];
        for ri in 0..n_r {
            if ri == 0 {
                d[ri] = (i_vals[1] - i_vals[0]) / r_step;
            } else if ri + 1 == n_r {
                d[ri] = (i_vals[n_r - 1] - i_vals[n_r - 2]) / r_step;
            } else {
                d[ri] = (i_vals[ri + 1] - i_vals[ri - 1]) / (2.0 * r_step);
            }
        }

        // Light smoothing to reduce noise
        if n_r >= 5 {
            let mut d2 = d.clone();
            for ri in 1..(n_r - 1) {
                d2[ri] = (d[ri - 1] + d[ri] + d[ri + 1]) / 3.0;
            }
            d = d2;
        }

        curves.push(d);
        theta_indices.push(ti);
    }

    let coverage = curves.len() as f32 / n_t as f32;
    if coverage < spec.min_theta_coverage {
        return InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: window,
            r_inner_found: None,
            polarity: None,
            peak_strength: None,
            theta_consistency: Some(coverage),
            status: InnerStatus::Failed,
            reason: Some(format!(
                "insufficient_theta_coverage({:.2}<{:.2})",
                coverage, spec.min_theta_coverage
            )),
            radial_response_agg: None,
            r_samples: if store_response {
                Some(r_samples)
            } else {
                None
            },
        };
    }

    fn aggregate(values: &mut [f32], agg: &AngularAggregator) -> f32 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        match *agg {
            AngularAggregator::Median => values[values.len() / 2],
            AngularAggregator::TrimmedMean { trim_fraction } => {
                let tf = trim_fraction.clamp(0.0, 0.45);
                let k = (values.len() as f32 * tf).floor() as usize;
                let start = k.min(values.len());
                let end = values.len().saturating_sub(k).max(start);
                let slice = &values[start..end];
                if slice.is_empty() {
                    values[values.len() / 2]
                } else {
                    slice.iter().sum::<f32>() / slice.len() as f32
                }
            }
        }
    }

    fn find_peak_idx(agg: &[f32], pol: Polarity) -> (usize, f32) {
        match pol {
            Polarity::Pos => agg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap(),
            Polarity::Neg => agg
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap(),
        }
    }

    fn per_theta_peak_r(curves: &[Vec<f32>], r_samples: &[f32], pol: Polarity) -> Vec<f32> {
        let mut peaks = Vec::with_capacity(curves.len());
        for d in curves {
            let (idx, _) = match pol {
                Polarity::Pos => d
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, &v)| (i, v))
                    .unwrap(),
                Polarity::Neg => d
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, &v)| (i, v))
                    .unwrap(),
            };
            peaks.push(r_samples[idx]);
        }
        peaks
    }

    let polarity_candidates: Vec<Polarity> = match spec.inner_grad_polarity {
        InnerGradPolarity::DarkToLight => vec![Polarity::Pos],
        InnerGradPolarity::LightToDark => vec![Polarity::Neg],
        InnerGradPolarity::Auto => vec![Polarity::Neg, Polarity::Pos],
    };

    let mut best: Option<(InnerEstimate, f32)> = None;

    for pol in polarity_candidates {
        // Aggregate across theta at each radius sample
        let mut agg_resp = vec![0.0f32; n_r];
        let mut scratch: Vec<f32> = Vec::with_capacity(curves.len());
        for ri in 0..n_r {
            scratch.clear();
            for d in &curves {
                scratch.push(d[ri]);
            }
            agg_resp[ri] = aggregate(&mut scratch, &spec.aggregator);
        }

        let (peak_idx, peak_val) = find_peak_idx(&agg_resp, pol);
        let r_star = r_samples[peak_idx];

        // Consistency: how many per-theta peaks agree with r_star
        let per_theta = per_theta_peak_r(&curves, &r_samples, pol);
        let delta = (4.0 * r_step).max(0.02);
        let n_close = per_theta
            .iter()
            .filter(|&&r| (r - r_star).abs() <= delta)
            .count();
        let theta_consistency = n_close as f32 / per_theta.len().max(1) as f32;

        let peak_strength = peak_val.abs();

        let mut status = InnerStatus::Ok;
        let mut reason = None;
        if !(0.2..=0.9).contains(&r_star) {
            status = InnerStatus::Rejected;
            reason = Some(format!("scale_out_of_bounds({:.3})", r_star));
        } else if peak_idx == 0 || peak_idx + 1 == n_r {
            status = InnerStatus::Rejected;
            reason = Some("peak_at_search_window_edge".to_string());
        } else if theta_consistency < spec.min_theta_consistency {
            status = InnerStatus::Rejected;
            reason = Some(format!(
                "theta_inconsistent({:.2}<{:.2})",
                theta_consistency, spec.min_theta_consistency
            ));
        }

        let est = InnerEstimate {
            r_inner_expected: spec.r_inner_expected,
            search_window: window,
            r_inner_found: Some(r_star),
            polarity: Some(pol),
            peak_strength: Some(peak_strength),
            theta_consistency: Some(theta_consistency),
            status,
            reason,
            radial_response_agg: if store_response {
                Some(agg_resp.clone())
            } else {
                None
            },
            r_samples: if store_response {
                Some(r_samples.clone())
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
        reason: Some("no_polarity_candidates".to_string()),
        radial_response_agg: None,
        r_samples: if store_response {
            Some(r_samples)
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
            inner_grad_polarity: InnerGradPolarity::LightToDark,
            inner_search_halfwidth: 0.08,
            theta_samples: 64,
            radial_samples: 64,
            min_theta_coverage: 0.5,
            ..MarkerSpec::default()
        };

        let est = estimate_inner_scale_from_outer(&img, &outer_ellipse, &spec, true);
        assert_eq!(
            est.status,
            InnerStatus::Ok,
            "inner estimate should succeed: {:?}",
            est.reason
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
