//! Full ring detection pipeline: proposal → edge sampling → fit → decode → global filter.

use image::GrayImage;

use crate::conic::{fit_ellipse_direct, try_fit_ellipse_ransac, rms_sampson_distance, Ellipse, RansacConfig};
use crate::homography::{self, RansacHomographyConfig, project};
use crate::board_spec;
use crate::{DetectedMarker, DetectionResult, EllipseParams, FitMetrics, DecodeMetrics, RansacStats};

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
    /// Enable global homography filtering (requires board spec).
    pub use_global_filter: bool,
    /// RANSAC homography configuration.
    pub ransac_homography: RansacHomographyConfig,
    /// Enable one-iteration refinement using H.
    pub refine_with_h: bool,
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
            use_global_filter: true,
            ransac_homography: RansacHomographyConfig::default(),
            refine_with_h: true,
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

    // Stage 5b: Dedup by ID — keep best confidence per decoded ID
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());

    // Stage 6: Global homography filtering
    if !config.use_global_filter {
        return DetectionResult {
            detected_markers: markers,
            image_size: [w, h],
            homography: None,
            ransac: None,
        };
    }

    let (filtered, h_result, ransac_stats) =
        global_filter(&markers, &config.ransac_homography);

    let h_matrix = h_result.as_ref().map(|r| &r.h);

    // Stage 7: Optional refinement using H
    let final_markers = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                refine_with_homography(gray, &filtered, h, config)
            } else {
                filtered
            }
        } else {
            filtered
        }
    } else {
        filtered
    };

    // Refit H after refinement if we have enough markers
    let (final_h, final_ransac) = if config.refine_with_h && final_markers.len() >= 10 {
        refit_homography(&final_markers, &config.ransac_homography)
    } else {
        (
            h_result.map(|r| matrix3_to_array(&r.h)),
            ransac_stats,
        )
    };

    tracing::info!(
        "{} markers after global filter{}",
        final_markers.len(),
        if config.refine_with_h { " + refinement" } else { "" }
    );

    DetectionResult {
        detected_markers: final_markers,
        image_size: [w, h],
        homography: final_h,
        ransac: final_ransac,
    }
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    use std::collections::HashMap;
    let mut best_idx: HashMap<usize, usize> = HashMap::new();

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            match best_idx.get(&id) {
                Some(&prev) if markers[prev].confidence >= m.confidence => {}
                _ => { best_idx.insert(id, i); }
            }
        }
    }

    let keep_set: std::collections::HashSet<usize> = best_idx.values().copied().collect();
    let mut i = 0;
    markers.retain(|m| {
        let idx = i;
        i += 1;
        // Keep markers without ID (they'll be filtered by RANSAC anyway)
        m.id.is_none() || keep_set.contains(&idx)
    });
}

/// Apply global homography RANSAC filter.
///
/// Returns (filtered markers, RANSAC result, stats).
fn global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> (Vec<DetectedMarker>, Option<homography::RansacHomographyResult>, Option<RansacStats>) {
    // Build correspondences from decoded markers
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut candidate_indices = Vec::new(); // index into markers

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                candidate_indices.push(i);
            }
        }
    }

    tracing::info!(
        "Global filter: {} decoded candidates out of {} total detections",
        candidate_indices.len(),
        markers.len()
    );

    if candidate_indices.len() < 4 {
        tracing::warn!("Too few decoded candidates for homography ({} < 4)", candidate_indices.len());
        return (markers.to_vec(), None, None);
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Homography RANSAC failed: {}", e);
            return (markers.to_vec(), None, None);
        }
    };

    // Collect inlier markers
    let mut filtered = Vec::new();
    let mut inlier_errors = Vec::new();

    for (j, &marker_idx) in candidate_indices.iter().enumerate() {
        if result.inlier_mask[j] {
            filtered.push(markers[marker_idx].clone());
            inlier_errors.push(result.errors[j]);
        }
    }

    // Compute stats
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    let stats = RansacStats {
        n_candidates: candidate_indices.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    tracing::info!(
        "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
        result.n_inliers,
        candidate_indices.len(),
        mean_err,
        p95_err,
    );

    (filtered, Some(result), Some(stats))
}

/// Refine marker centers using H: project board coords through H as priors,
/// re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> Vec<DetectedMarker> {
    let mut refined = Vec::with_capacity(markers.len());

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board_spec::xy_mm(id) {
            Some(xy) => xy,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        // Project board coords through H to get refined center prior
        let prior = project(h, xy[0] as f64, xy[1] as f64);
        if prior[0].is_nan() || prior[1].is_nan() {
            refined.push(m.clone());
            continue;
        }

        // Re-run edge sampling around the H-projected center
        let edge = match sample_edges(gray, [prior[0] as f32, prior[1] as f32], &config.edge_sample) {
            Some(er) => er,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        // Re-fit ellipses
        let (outer, inner, outer_ransac, inner_ransac) = match fit_ring_ellipses(&edge, config) {
            Some(r) => r,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let center = compute_center(&outer, inner.as_ref(), &edge);

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

        // Re-decode with new ellipse
        let decode_result = decode_marker(gray, &outer, &config.decode);
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        // Keep original ID (validated by RANSAC), but update geometry
        refined.push(DetectedMarker {
            id: Some(id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner.as_ref().map(|ie| ellipse_to_params(ie)),
            fit,
            decode: decode_metrics,
        });
    }

    refined
}

/// Refit homography using refined marker centers, return (H_array, stats).
fn refit_homography(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> (Option<[[f64; 3]; 3]>, Option<RansacStats>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for m in markers {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src.push([xy[0] as f64, xy[1] as f64]);
                dst.push(m.center);
            }
        }
    }

    if src.len() < 4 {
        return (None, None);
    }

    // Use a light RANSAC (most outliers already removed)
    let light_config = RansacHomographyConfig {
        max_iters: 500,
        inlier_threshold: config.inlier_threshold,
        min_inliers: config.min_inliers,
        seed: config.seed + 1,
    };

    match homography::fit_homography_ransac(&src, &dst, &light_config) {
        Ok(result) => {
            let mut errors: Vec<f64> = result
                .inlier_mask
                .iter()
                .zip(&result.errors)
                .filter(|(&m, _)| m)
                .map(|(_, &e)| e)
                .collect();
            errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean_err = if errors.is_empty() {
                0.0
            } else {
                errors.iter().sum::<f64>() / errors.len() as f64
            };
            let p95_err = if errors.is_empty() {
                0.0
            } else {
                let idx = ((errors.len() as f64 * 0.95) as usize).min(errors.len() - 1);
                errors[idx]
            };

            let stats = RansacStats {
                n_candidates: src.len(),
                n_inliers: result.n_inliers,
                threshold_px: light_config.inlier_threshold,
                mean_err_px: mean_err,
                p95_err_px: p95_err,
            };

            (Some(matrix3_to_array(&result.h)), Some(stats))
        }
        Err(_) => (None, None),
    }
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
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
