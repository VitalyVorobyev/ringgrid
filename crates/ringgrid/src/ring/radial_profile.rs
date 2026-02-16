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
            let (_, median, _) =
                values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            *median
        }
        AngularAggregator::TrimmedMean { trim_fraction } => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
pub fn peak_idx(values: &[f32], pol: Polarity) -> usize {
    match pol {
        Polarity::Pos => values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap(),
        Polarity::Neg => values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap(),
    }
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
