use crate::conic::{Ellipse, rms_sampson_distance};
use crate::ring::edge_sample::{DistortionAwareSampler, EdgeSampleResult};

use super::super::marker_build::fit_support_score;

/// Number of sampling directions for [`plain_ring_evidence`].
const EVIDENCE_RAYS: usize = 32;
/// Normalized radial half-width of the inner-edge contrast pair.
const EVIDENCE_EDGE_DELTA: f32 = 0.15;

/// Photometric evidence that a fitted outer ellipse encloses a plain (uncoded)
/// ring: a filled dark annulus from `r_inner_expected × R` out to `R` around a
/// bright hole.
///
/// Plain markers carry no code band, so this fills the decode-confidence slot
/// of [`score_outer_candidate`] with two intensity contrasts sampled along the
/// fitted ellipse, both normalized to `[0, 1]`:
///
/// 1. **Annulus contrast** — bright hole (`0.5 × ratio × R`) minus dark
///    mid-annulus (`(1 + ratio)/2 × R`).
/// 2. **Inner-edge agreement** — brightness drop across the *expected* inner
///    edge (`(ratio ∓ δ) × R`), which collapses when the true inner edge sits
///    at a different ratio than the target specifies.
///
/// Returns the mean of the two terms; `0.0` when too few samples land inside
/// the image.
pub(super) fn plain_ring_evidence(
    sampler: DistortionAwareSampler<'_>,
    outer: &Ellipse,
    r_inner_expected: f32,
) -> f32 {
    let ratio = r_inner_expected.clamp(0.05, 0.95);
    let stations = [
        0.5 * ratio,                             // bright hole
        0.5 * (1.0 + ratio),                     // dark mid-annulus
        (ratio - EVIDENCE_EDGE_DELTA).max(0.02), // bright side of inner edge
        (ratio + EVIDENCE_EDGE_DELTA).min(0.98), // dark side of inner edge
    ];

    let (sin_phi, cos_phi) = (outer.angle.sin(), outer.angle.cos());
    let mut sums = [0.0f32; 4];
    let mut counts = [0usize; 4];
    for i in 0..EVIDENCE_RAYS {
        let t = (i as f64) * std::f64::consts::TAU / (EVIDENCE_RAYS as f64);
        let (sin_t, cos_t) = (t.sin(), t.cos());
        for (station, fraction) in stations.iter().enumerate() {
            let f = f64::from(*fraction);
            let dx = f * (outer.a * cos_t * cos_phi - outer.b * sin_t * sin_phi);
            let dy = f * (outer.a * cos_t * sin_phi + outer.b * sin_t * cos_phi);
            if let Some(v) = sampler.sample_checked((outer.cx + dx) as f32, (outer.cy + dy) as f32)
            {
                sums[station] += v;
                counts[station] += 1;
            }
        }
    }

    let min_samples = EVIDENCE_RAYS / 2;
    if counts.iter().any(|&c| c < min_samples) {
        return 0.0;
    }
    // `sample_checked` yields intensities already normalized to [0, 1].
    let mean = |i: usize| sums[i] / counts[i] as f32;
    let annulus_contrast = (mean(0) - mean(1)).clamp(0.0, 1.0);
    let edge_contrast = (mean(2) - mean(3)).clamp(0.0, 1.0);
    0.5 * (annulus_contrast + edge_contrast)
}

pub(super) fn score_outer_candidate(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    decode_confidence: f32,
    r_expected: f32,
    size_score_weight: f32,
) -> f32 {
    let decode_score = decode_confidence.clamp(0.0, 1.0);
    let fit_support = fit_support_score(edge, outer_ransac);

    let mean_axis = ((outer.a + outer.b) * 0.5) as f32;
    let size_score = 1.0 - ((mean_axis - r_expected).abs() / r_expected.max(1.0)).min(1.0);

    let residual = rms_sampson_distance(outer, &edge.outer_points) as f32;
    let residual = if residual.is_finite() {
        residual.max(0.0)
    } else {
        f32::INFINITY
    };
    let residual_score = 1.0 / (1.0 + residual);

    let size_weight = size_score_weight.clamp(0.0, 1.0);
    let remaining = 1.0 - size_weight;
    // Preserve legacy non-size term ratios (0.55 : 0.25 : 0.05 = 0.85 total).
    let scale = remaining / 0.85;
    let decode_weight = 0.55 * scale;
    let fit_weight = 0.25 * scale;
    let residual_weight = 0.05 * scale;

    decode_weight * decode_score
        + fit_weight * fit_support
        + size_weight * size_score
        + residual_weight * residual_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::ring::edge_sample::EdgeSampleResult;

    fn circle_points(cx: f64, cy: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| {
                let t = (i as f64) * std::f64::consts::TAU / (n as f64);
                [cx + r * t.cos(), cy + r * t.sin()]
            })
            .collect()
    }

    fn edge_with_coverage(
        outer_points: Vec<[f64; 2]>,
        n_good: usize,
        n_total: usize,
    ) -> EdgeSampleResult {
        EdgeSampleResult {
            outer_points,
            inner_points: Vec::new(),
            outer_radii: Vec::new(),
            inner_radii: Vec::new(),
            n_good_rays: n_good,
            n_total_rays: n_total,
        }
    }

    #[test]
    fn default_size_weight_matches_legacy_formula() {
        let outer = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 10.0,
            b: 10.0,
            angle: 0.0,
        };
        let edge = edge_with_coverage(circle_points(0.0, 0.0, 10.0, 48), 36, 48);
        let score = score_outer_candidate(&edge, &outer, None, 0.7, 10.0, 0.15);
        let decode_score = 0.7f32;
        let fit_support = 36.0f32 / 48.0f32;
        let size_score = 1.0f32;
        let residual_score = 1.0f32
            / (1.0f32 + crate::conic::rms_sampson_distance(&outer, &edge.outer_points) as f32);
        let legacy =
            0.55 * decode_score + 0.25 * fit_support + 0.15 * size_score + 0.05 * residual_score;
        assert!((score - legacy).abs() < 1e-6);
    }

    #[test]
    fn size_weight_can_flip_candidate_ranking() {
        let r_expected = 16.0f32;

        // Candidate A: better decode/fit, but wrong size.
        let outer_a = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 10.0,
            b: 10.0,
            angle: 0.0,
        };
        let edge_a = edge_with_coverage(circle_points(0.0, 0.0, 10.0, 48), 48, 48);

        // Candidate B: weaker decode/fit, but size near expected.
        let outer_b = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 16.0,
            b: 16.0,
            angle: 0.0,
        };
        let edge_b = edge_with_coverage(circle_points(0.0, 0.0, 16.0, 48), 30, 48);

        let low_size_weight_a =
            score_outer_candidate(&edge_a, &outer_a, None, 0.95, r_expected, 0.05);
        let low_size_weight_b =
            score_outer_candidate(&edge_b, &outer_b, None, 0.70, r_expected, 0.05);
        assert!(
            low_size_weight_a > low_size_weight_b,
            "low size weight should favor stronger decode/fit"
        );

        let high_size_weight_a =
            score_outer_candidate(&edge_a, &outer_a, None, 0.95, r_expected, 0.70);
        let high_size_weight_b =
            score_outer_candidate(&edge_b, &outer_b, None, 0.70, r_expected, 0.70);
        assert!(
            high_size_weight_b > high_size_weight_a,
            "high size weight should favor size agreement"
        );
    }
}
