//! Full ring detection pipeline: proposal → edge sampling → fit → decode.

use image::GrayImage;

use crate::conic::{fit_ellipse_direct, try_fit_ellipse_ransac, rms_sampson_distance, Ellipse, RansacConfig};
use crate::{DetectedMarker, DetectionResult, EllipseParams, FitMetrics, DecodeMetrics};

use super::proposal::{find_proposals, ProposalConfig};
use super::edge_sample::{sample_edges, EdgeSampleConfig, EdgeSampleResult};
use super::decode::{decode_marker, DecodeConfig};

/// Top-level detection configuration.
#[derive(Debug, Clone)]
pub struct DetectConfig {
    pub proposal: ProposalConfig,
    pub edge_sample: EdgeSampleConfig,
    pub decode: DecodeConfig,
    /// Minimum semi-axis for a valid outer ellipse.
    pub min_semi_axis: f64,
    /// Maximum semi-axis for a valid outer ellipse.
    pub max_semi_axis: f64,
    /// Maximum aspect ratio (a/b) for a valid ellipse.
    pub max_aspect_ratio: f64,
    /// NMS dedup radius for final markers (pixels).
    pub dedup_radius: f64,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            proposal: ProposalConfig::default(),
            edge_sample: EdgeSampleConfig::default(),
            decode: DecodeConfig::default(),
            min_semi_axis: 3.0,
            max_semi_axis: 15.0,
            max_aspect_ratio: 3.0,
            dedup_radius: 6.0,
        }
    }
}

/// Fit outer and inner ellipses from edge points.
///
/// Returns (outer, inner, outer_ransac_result, inner_ransac_result).
fn fit_ring_ellipses(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Option<(Ellipse, Option<Ellipse>, Option<crate::conic::RansacResult>, Option<crate::conic::RansacResult>)> {
    // Fit outer ellipse
    let ransac_config = RansacConfig {
        max_iters: 200,
        inlier_threshold: 1.5,
        min_inliers: 6,
        seed: 42,
    };

    let (outer, outer_ransac) = if edge.outer_points.len() >= 8 {
        match try_fit_ellipse_ransac(&edge.outer_points, &ransac_config) {
            Ok(r) => (r.ellipse, Some(r)),
            Err(_) => {
                // Fall back to direct fit
                match fit_ellipse_direct(&edge.outer_points) {
                    Some((_, e)) => (e, None),
                    None => return None,
                }
            }
        }
    } else if edge.outer_points.len() >= 6 {
        match fit_ellipse_direct(&edge.outer_points) {
            Some((_, e)) => (e, None),
            None => return None,
        }
    } else {
        return None;
    };

    // Validate outer ellipse
    if outer.a < config.min_semi_axis
        || outer.a > config.max_semi_axis
        || outer.b < config.min_semi_axis
        || outer.b > config.max_semi_axis
        || outer.aspect_ratio() > config.max_aspect_ratio
        || !outer.is_valid()
    {
        return None;
    }

    // Fit inner ellipse (optional)
    let (inner, inner_ransac) = if edge.inner_points.len() >= 8 {
        match try_fit_ellipse_ransac(&edge.inner_points, &ransac_config) {
            Ok(r) => {
                if r.ellipse.is_valid()
                    && r.ellipse.a >= 1.0
                    && r.ellipse.aspect_ratio() < config.max_aspect_ratio
                {
                    (Some(r.ellipse), Some(r))
                } else {
                    (None, None)
                }
            }
            Err(_) => {
                match fit_ellipse_direct(&edge.inner_points) {
                    Some((_, e)) if e.is_valid() && e.a >= 1.0 => (Some(e), None),
                    _ => (None, None),
                }
            }
        }
    } else if edge.inner_points.len() >= 6 {
        match fit_ellipse_direct(&edge.inner_points) {
            Some((_, e)) if e.is_valid() && e.a >= 1.0 => (Some(e), None),
            _ => (None, None),
        }
    } else {
        (None, None)
    };

    Some((outer, inner, outer_ransac, inner_ransac))
}

/// Compute the marker center as a weighted average of inner and outer ellipse centers.
fn compute_center(outer: &Ellipse, inner: Option<&Ellipse>, edge: &EdgeSampleResult) -> [f64; 2] {
    if let Some(inner) = inner {
        // Weight by number of points (more points = more reliable)
        let w_outer = edge.outer_points.len() as f64;
        let w_inner = edge.inner_points.len() as f64;
        let w_total = w_outer + w_inner;
        [
            (outer.cx * w_outer + inner.cx * w_inner) / w_total,
            (outer.cy * w_outer + inner.cy * w_inner) / w_total,
        ]
    } else {
        [outer.cx, outer.cy]
    }
}

/// Helper to create EllipseParams from a conic Ellipse.
fn ellipse_to_params(e: &Ellipse) -> EllipseParams {
    EllipseParams {
        center_xy: [e.cx, e.cy],
        semi_axes: [e.a, e.b],
        angle: e.angle,
    }
}

/// Run the full ring detection pipeline.
pub fn detect_rings(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let (w, h) = gray.dimensions();

    // Stage 1: Find candidate centers
    let proposals = find_proposals(gray, &config.proposal);
    tracing::info!("{} proposals found", proposals.len());

    // Stages 2-5: For each proposal, sample edges → fit → decode
    let mut markers: Vec<DetectedMarker> = Vec::new();

    for proposal in &proposals {
        // Stage 2: Sample radial edges
        let edge = match sample_edges(gray, [proposal.x, proposal.y], &config.edge_sample) {
            Some(er) => er,
            None => continue,
        };

        // Stage 3: Fit ellipses
        let (outer, inner, outer_ransac, inner_ransac) =
            match fit_ring_ellipses(&edge, config) {
                Some(r) => r,
                None => continue,
            };

        // Compute center
        let center = compute_center(&outer, inner.as_ref(), &edge);

        // Compute fit metrics
        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: edge.inner_points.len(),
            ransac_inlier_ratio_outer: outer_ransac.as_ref().map(|r| {
                r.num_inliers as f32 / edge.outer_points.len().max(1) as f32
            }),
            ransac_inlier_ratio_inner: inner_ransac.as_ref().map(|r| {
                r.num_inliers as f32 / edge.inner_points.len().max(1) as f32
            }),
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: inner.as_ref().map(|ie| {
                rms_sampson_distance(ie, &edge.inner_points)
            }),
        };

        // Stage 4: Decode
        let decode_result = decode_marker(gray, &outer, &config.decode);

        // Build marker
        let confidence = decode_result
            .as_ref()
            .map(|d| d.confidence)
            .unwrap_or(0.0);

        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let marker = DetectedMarker {
            id: decode_result.as_ref().map(|d| d.id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner.as_ref().map(|ie| ellipse_to_params(ie)),
            fit,
            decode: decode_metrics,
        };

        markers.push(marker);
    }

    // Stage 5: Dedup by center proximity
    markers = dedup_markers(markers, config.dedup_radius);

    tracing::info!("{} markers detected after dedup", markers.len());

    DetectionResult {
        detected_markers: markers,
        image_size: [w, h],
    }
}

/// Remove duplicate detections: keep the highest-confidence marker within dedup_radius.
fn dedup_markers(mut markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    // Sort by confidence descending
    markers.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; markers.len()];
    let r2 = radius * radius;

    for i in 0..markers.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..markers.len() {
            if !keep[j] {
                continue;
            }
            let dx = markers[i].center[0] - markers[j].center[0];
            let dy = markers[i].center[1] - markers[j].center[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
            }
        }
    }

    markers
        .into_iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(m, _)| m)
        .collect()
}
