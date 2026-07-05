//! Shared radial-profile aggregation and peak helpers.

use crate::marker::AngularAggregator;

/// Sign convention for radial derivative peaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Polarity {
    /// Positive radial derivative peak (`dI/dr > 0`).
    Pos,
    /// Negative radial derivative peak (`dI/dr < 0`).
    Neg,
}

/// Aggregate a per-theta response vector into one scalar profile value.
pub fn aggregate(values: &mut [f32], agg: &AngularAggregator) -> f32 {
    match *agg {
        AngularAggregator::Median => {
            let mid = values.len() / 2;
            // `total_cmp` is a total order over f32 (orders identically to
            // `partial_cmp` for finite values) so a stray NaN sorts to an end
            // instead of panicking the comparator.
            let (_, median, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            *median
        }
        AngularAggregator::TrimmedMean { trim_fraction } => {
            values.sort_by(|a, b| a.total_cmp(b));
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

/// Return index of the strongest peak for the requested polarity.
///
/// `values` must be non-empty (radial-scan curves always carry samples);
/// an empty slice yields index 0 in release builds.
pub fn peak_idx(values: &[f32], pol: Polarity) -> usize {
    debug_assert!(!values.is_empty(), "peak_idx requires a non-empty curve");
    let best = match pol {
        Polarity::Pos => values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)),
        Polarity::Neg => values.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)),
    };
    best.map_or(0, |(i, _)| i)
}

/// Compute per-theta peak radii from derivative curves.
#[allow(dead_code)]
pub fn per_theta_peak_r(curves: &[Vec<f32>], r_samples: &[f32], pol: Polarity) -> Vec<f32> {
    let mut peaks = Vec::with_capacity(curves.len());
    for d in curves {
        peaks.push(r_samples[peak_idx(d, pol)]);
    }
    peaks
}

/// Compute radial derivative `dI/dr` from sampled intensities using central differences.
///
/// Boundary samples use forward/backward differences.
pub fn radial_derivative_into(i_vals: &[f32], r_step: f32, out: &mut [f32]) {
    let n = i_vals.len();
    debug_assert_eq!(out.len(), n);
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = 0.0;
        return;
    }

    out[0] = (i_vals[1] - i_vals[0]) / r_step;
    for ri in 1..(n - 1) {
        out[ri] = (i_vals[ri + 1] - i_vals[ri - 1]) / (2.0 * r_step);
    }
    out[n - 1] = (i_vals[n - 1] - i_vals[n - 2]) / r_step;
}

/// Compute radial derivative `dI/dr` from sampled intensities using central differences.
///
/// Boundary samples use forward/backward differences.
#[allow(dead_code)]
pub fn radial_derivative(i_vals: &[f32], r_step: f32) -> Vec<f32> {
    let n = i_vals.len();
    let mut d = vec![0.0f32; n];
    radial_derivative_into(i_vals, r_step, &mut d);
    d
}

/// Apply 3-point moving average smoothing to a curve (in-place via copy).
///
/// Boundary values are left unchanged. No-op if the curve has fewer than 5 samples.
pub fn smooth_3point(d: &mut [f32]) {
    let n = d.len();
    if n < 5 {
        return;
    }

    // In-place rolling window equivalent to the previous copy-based implementation.
    let mut left = d[0];
    let mut mid = d[1];
    for ri in 1..(n - 1) {
        let right = d[ri + 1];
        d[ri] = (left + mid + right) / 3.0;
        left = mid;
        mid = right;
    }
}

/// Compute fraction of per-theta peaks that agree with a selected radius.
///
/// `min_delta` provides a floor for the tolerance window.
pub fn theta_consistency(per_theta_peaks: &[f32], r_star: f32, r_step: f32, min_delta: f32) -> f32 {
    let delta = (4.0 * r_step).max(min_delta);
    let n_close = per_theta_peaks
        .iter()
        .filter(|&&r| (r - r_star).abs() <= delta)
        .count();
    n_close as f32 / per_theta_peaks.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_median_returns_middle_value() {
        let mut v = [3.0f32, 1.0, 2.0];
        let m = aggregate(&mut v, &AngularAggregator::Median);
        assert_eq!(m, 2.0);
    }

    #[test]
    fn aggregate_trimmed_mean_drops_extremes() {
        let mut v = [0.0f32, 1.0, 2.0, 3.0, 100.0];
        let m = aggregate(
            &mut v,
            &AngularAggregator::TrimmedMean { trim_fraction: 0.2 },
        );
        // Trimming 20% off each end removes 0.0 and 100.0, leaving mean(1,2,3)=2.
        assert!((m - 2.0).abs() < 1e-6, "trimmed mean = {m}");
    }

    #[test]
    fn aggregate_does_not_panic_on_nan() {
        // Before the `total_cmp` fix these comparators panicked on NaN.
        let mut median_vals = [1.0f32, f32::NAN, 2.0];
        let _ = aggregate(&mut median_vals, &AngularAggregator::Median);
        let mut trimmed_vals = [1.0f32, f32::NAN, 2.0, 3.0];
        let _ = aggregate(
            &mut trimmed_vals,
            &AngularAggregator::TrimmedMean {
                trim_fraction: 0.25,
            },
        );
    }

    #[test]
    fn peak_idx_finds_extrema() {
        assert_eq!(peak_idx(&[0.1, 0.9, 0.3], Polarity::Pos), 1);
        assert_eq!(peak_idx(&[0.1, -0.9, 0.3], Polarity::Neg), 1);
    }

    #[test]
    fn peak_idx_does_not_panic_on_nan() {
        // A valid index is returned and no panic occurs with NaN present.
        let with_nan = [0.1f32, f32::NAN, 0.5];
        let i_pos = peak_idx(&with_nan, Polarity::Pos);
        let i_neg = peak_idx(&with_nan, Polarity::Neg);
        assert!(i_pos < with_nan.len());
        assert!(i_neg < with_nan.len());
    }

    #[test]
    fn radial_derivative_is_exact_for_a_linear_ramp() {
        // A linear intensity ramp has a constant slope; central differences
        // (and the forward/backward boundary differences) recover it exactly.
        let r_step = 0.5f32;
        let slope = 4.0f32;
        let i_vals: Vec<f32> = (0..6).map(|k| 1.0 + slope * k as f32 * r_step).collect();
        let d = radial_derivative(&i_vals, r_step);
        assert_eq!(d.len(), i_vals.len());
        for (k, &v) in d.iter().enumerate() {
            assert!((v - slope).abs() < 1e-4, "d[{k}] = {v}");
        }
    }

    #[test]
    fn radial_derivative_into_handles_degenerate_lengths() {
        let mut out = [0.0f32; 1];
        radial_derivative_into(&[3.0], 1.0, &mut out);
        assert_eq!(out[0], 0.0, "single sample has zero derivative");

        // Empty input is a no-op.
        let mut empty: [f32; 0] = [];
        radial_derivative_into(&[], 1.0, &mut empty);
    }

    #[test]
    fn smooth_3point_preserves_constant_and_endpoints() {
        // A constant curve is unchanged by averaging.
        let mut constant = [5.0f32; 6];
        smooth_3point(&mut constant);
        assert!(constant.iter().all(|&v| (v - 5.0).abs() < 1e-6));

        // A single spike is spread to its neighbors; endpoints stay fixed.
        let mut spike = [0.0f32, 3.0, 0.0, 0.0, 0.0];
        smooth_3point(&mut spike);
        assert_eq!(spike[0], 0.0, "left endpoint unchanged");
        assert_eq!(spike[4], 0.0, "right endpoint unchanged");
        assert!((spike[1] - 1.0).abs() < 1e-6, "spike[1] = {}", spike[1]);
        assert!((spike[2] - 1.0).abs() < 1e-6, "spike[2] = {}", spike[2]);
    }

    #[test]
    fn smooth_3point_is_noop_below_five_samples() {
        let mut short = [1.0f32, 9.0, 1.0, 9.0];
        smooth_3point(&mut short);
        assert_eq!(short, [1.0, 9.0, 1.0, 9.0]);
    }

    #[test]
    fn theta_consistency_counts_peaks_within_tolerance() {
        // delta = max(4·r_step, min_delta) = max(0.4, 0.02) = 0.4.
        // Three peaks lie within 0.4 of 10.0; the outlier at 20 does not.
        let peaks = [10.0f32, 10.2, 9.8, 20.0];
        let frac = theta_consistency(&peaks, 10.0, 0.1, 0.02);
        assert!((frac - 0.75).abs() < 1e-6, "consistency = {frac}");
    }

    #[test]
    fn theta_consistency_empty_is_zero() {
        assert_eq!(theta_consistency(&[], 10.0, 0.1, 0.02), 0.0);
    }

    #[test]
    fn per_theta_peak_r_maps_each_curve_to_its_extremum_radius() {
        let r_samples = [1.0f32, 2.0, 3.0];
        let curves = vec![vec![0.1f32, 0.9, 0.2], vec![0.5f32, 0.1, 0.8]];
        let pos = per_theta_peak_r(&curves, &r_samples, Polarity::Pos);
        assert_eq!(pos, vec![2.0, 3.0]);
        let neg = per_theta_peak_r(&curves, &r_samples, Polarity::Neg);
        assert_eq!(neg, vec![1.0, 2.0]);
    }
}
