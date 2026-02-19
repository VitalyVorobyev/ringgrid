use crate::conic::{rms_sampson_distance, Ellipse};
use crate::ring::edge_sample::EdgeSampleResult;

pub(super) fn score_outer_candidate(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    decode_confidence: f32,
    r_expected: f32,
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

    0.55 * decode_score + 0.25 * fit_support + 0.15 * size_score + 0.05 * residual_score
}
