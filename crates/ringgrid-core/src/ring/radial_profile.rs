//! Shared radial-profile aggregation and peak helpers.

use crate::marker_spec::AngularAggregator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Pos,
    Neg,
}

pub fn aggregate(values: &mut [f32], agg: &AngularAggregator) -> f32 {
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

pub fn per_theta_peak_r(curves: &[Vec<f32>], r_samples: &[f32], pol: Polarity) -> Vec<f32> {
    let mut peaks = Vec::with_capacity(curves.len());
    for d in curves {
        peaks.push(r_samples[peak_idx(d, pol)]);
    }
    peaks
}
