use crate::conic::{rms_sampson_distance, Ellipse};
use crate::ring::edge_sample::EdgeSampleResult;

pub(super) fn score_outer_candidate(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    decode_confidence: f32,
    r_expected: f32,
    size_score_weight: f32,
) -> f32 {
    let decode_score = decode_confidence.clamp(0.0, 1.0);
    let arc_cov = (edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32).clamp(0.0, 1.0);
    let inlier_ratio = outer_ransac
        .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let fit_support = (arc_cov * inlier_ratio).clamp(0.0, 1.0);

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
