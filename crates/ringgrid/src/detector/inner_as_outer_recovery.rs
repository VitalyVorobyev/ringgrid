//! Recovery stage for markers where the outer fit locked onto the inner ring edge.
//!
//! When a marker's outer radius is anomalously small relative to its neighbours
//! (the "inner-as-outer" substitution), this module re-attempts the outer fit using
//! the neighbour-median radius as a corrected prior. If the re-fit succeeds it
//! replaces the original marker in-place; otherwise the original is kept.
//!
//! This module also owns `annotate_neighbor_radius_ratios`, which populates
//! `FitMetrics.neighbor_radius_ratio` for every marker — the diagnostic field that
//! triggers the recovery.

use image::GrayImage;

use crate::conic::Ellipse;
use crate::marker::decode::DecodeResult;
use crate::pixelmap::PixelMapper;
use crate::ring::OuterEstimationConfig;
use crate::DetectedMarker;

use super::{
    inner_fit, marker_build,
    marker_build::compute_marker_confidence,
    median_outer_radius_from_neighbors_px,
    outer_fit::{self, OuterFitCandidate},
    DetectConfig,
};

/// Annotates each marker with the ratio of its outer radius to the median
/// outer radius of its k nearest neighbors. Values well below 1.0 (< 0.75)
/// indicate a potential inner-as-outer substitution.
pub(crate) fn annotate_neighbor_radius_ratios(markers: &mut [DetectedMarker], k: usize) {
    const WARN_THRESHOLD: f32 = 0.75;

    // Compute ratios in a separate immutable pass to satisfy the borrow checker.
    let ratios: Vec<Option<f32>> = {
        let m_ref: &[DetectedMarker] = markers;
        m_ref
            .iter()
            .map(|m| {
                let own_radius = m.ellipse_outer.as_ref()?.mean_axis() as f32;
                let median = median_outer_radius_from_neighbors_px(m.center, m_ref, k + 1)?;
                if median > 0.0 {
                    Some(own_radius / median)
                } else {
                    None
                }
            })
            .collect()
    };

    for (marker, ratio) in markers.iter_mut().zip(ratios) {
        marker.fit.neighbor_radius_ratio = ratio;
        if let Some(r) = ratio {
            if r < WARN_THRESHOLD {
                tracing::warn!(
                    ratio = r,
                    center_x = marker.center[0],
                    center_y = marker.center[1],
                    id = ?marker.id,
                    "marker outer radius anomalous vs neighbors (possible inner-as-outer)"
                );
            }
        }
    }
}

/// Attempts to recover markers where the outer fit locked onto the inner ring
/// edge. For each marker whose `neighbor_radius_ratio` is below the configured
/// threshold, re-attempts the outer fit using the neighbor-median radius as the
/// corrected expected radius. If the new fit succeeds (with a valid decode), the
/// marker is replaced in-place; otherwise the original is kept.
///
/// After this function the caller should re-run `annotate_neighbor_radius_ratios`
/// so the ratios reflect the recovered markers.
fn flagged_recovery_indices(markers: &[DetectedMarker], ratio_threshold: f32) -> Vec<usize> {
    markers
        .iter()
        .enumerate()
        .filter_map(|(i, marker)| {
            let ratio = marker.fit.neighbor_radius_ratio?;
            (ratio < ratio_threshold).then_some(i)
        })
        .collect()
}

fn marker_center_working(marker: &DetectedMarker) -> [f64; 2] {
    marker.center_mapped.unwrap_or(marker.center)
}

fn build_recovery_config(config: &DetectConfig) -> DetectConfig {
    let mut recovery_config = config.clone();
    let cfg = &config.inner_as_outer_recovery;
    recovery_config.outer_estimation.search_halfwidth_px =
        OuterEstimationConfig::default().search_halfwidth_px;
    recovery_config.outer_estimation.min_theta_consistency = cfg.min_theta_consistency;
    recovery_config.outer_estimation.min_theta_coverage = cfg.min_theta_coverage;
    recovery_config.outer_estimation.refine_halfwidth_px = cfg.refine_halfwidth_px;
    recovery_config.edge_sample.min_ring_depth = cfg.min_ring_depth;
    recovery_config
}

fn resolve_recovered_id(
    idx: usize,
    marker: &DetectedMarker,
    outer: &Ellipse,
    decode_result: Option<DecodeResult>,
    r_corrected: f32,
    size_gate_tolerance: f32,
) -> Option<(Option<usize>, Option<DecodeResult>)> {
    if let Some(decoded) = decode_result {
        return Some((Some(decoded.id), Some(decoded)));
    }

    let Some(orig_id) = marker.id else {
        tracing::debug!(
            idx,
            "inner-as-outer recovery: no decode and no validated id — skipping"
        );
        return None;
    };

    let orig_center_wf = marker_center_working(marker);
    let recovered_center = outer.center();
    let center_shift = (((recovered_center[0] - orig_center_wf[0]).powi(2)
        + (recovered_center[1] - orig_center_wf[1]).powi(2))
    .sqrt()) as f32;
    let max_shift = r_corrected * size_gate_tolerance;
    if center_shift > max_shift {
        tracing::debug!(
            idx,
            center_shift,
            max_shift,
            "inner-as-outer recovery: centre proximity gate failed"
        );
        return None;
    }
    tracing::debug!(
        idx,
        orig_id,
        center_shift,
        "inner-as-outer recovery: geometry-only (keeping id-corrected id)"
    );
    Some((Some(orig_id), None))
}

fn recover_marker_at_index(
    idx: usize,
    gray: &GrayImage,
    markers: &[DetectedMarker],
    config: &DetectConfig,
    recovery_config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> Option<DetectedMarker> {
    let cfg = &config.inner_as_outer_recovery;
    let center_wf = marker_center_working(&markers[idx]);
    let center_f32 = [center_wf[0] as f32, center_wf[1] as f32];

    let Some(r_corrected) =
        median_outer_radius_from_neighbors_px(markers[idx].center, markers, cfg.k_neighbors + 1)
    else {
        tracing::debug!(
            idx,
            "recovery skipped: could not compute neighbor median radius"
        );
        return None;
    };

    let candidate = match outer_fit::fit_outer_candidate_from_prior(
        gray,
        center_f32,
        r_corrected,
        recovery_config,
        mapper,
    ) {
        Ok(candidate) => candidate,
        Err(reject) => {
            tracing::debug!(
                idx,
                reject_reason = %reject.reason,
                "inner-as-outer recovery: outer fit failed"
            );
            return None;
        }
    };

    let recovered_r = candidate.outer.mean_axis() as f32;
    if (recovered_r - r_corrected).abs() / r_corrected > cfg.size_gate_tolerance {
        tracing::debug!(
            idx,
            recovered_r,
            r_corrected,
            "inner-as-outer recovery: size gate rejected (re-locked to wrong ring)"
        );
        return None;
    }

    let OuterFitCandidate {
        edge,
        outer,
        outer_ransac,
        decode_result,
        ..
    } = candidate;
    let (marker_id, decode_result) = resolve_recovered_id(
        idx,
        &markers[idx],
        &outer,
        decode_result,
        r_corrected,
        cfg.size_gate_tolerance,
    )?;

    let inner = inner_fit::fit_inner_ellipse_from_outer_hint(
        gray,
        &outer,
        &config.marker_spec,
        mapper,
        &config.inner_fit,
        false,
    );
    let fit_metrics =
        marker_build::fit_metrics_with_inner(&edge, &outer, outer_ransac.as_ref(), &inner);
    let confidence = compute_marker_confidence(
        decode_result.as_ref(),
        &edge,
        outer_ransac.as_ref(),
        &inner,
        &fit_metrics,
        &config.inner_fit,
    );
    let decode_metrics = marker_build::decode_metrics_from_result(decode_result.as_ref());
    let new_center_wf = outer.center();
    let new_center_image = mapper
        .and_then(|m| m.working_to_image_pixel(new_center_wf))
        .unwrap_or(new_center_wf);

    Some(DetectedMarker {
        id: marker_id,
        confidence,
        center: new_center_image,
        center_mapped: mapper.map(|_| new_center_wf),
        board_xy_mm: None, // populated by sync_marker_board_correspondence later
        ellipse_outer: Some(outer),
        ellipse_inner: inner.ellipse_inner,
        edge_points_outer: Some(edge.outer_points),
        edge_points_inner: Some(inner.points_inner),
        fit: fit_metrics,
        decode: decode_metrics,
    })
}

pub(crate) fn try_recover_inner_as_outer(
    gray: &GrayImage,
    markers: &mut [DetectedMarker],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) {
    let flagged = flagged_recovery_indices(markers, config.inner_as_outer_recovery.ratio_threshold);
    if flagged.is_empty() {
        return;
    }

    tracing::info!(
        n_flagged = flagged.len(),
        "attempting inner-as-outer recovery for flagged markers"
    );
    let recovery_config = build_recovery_config(config);
    let mut n_recovered = 0usize;
    for idx in flagged {
        let Some(recovered) =
            recover_marker_at_index(idx, gray, markers, config, &recovery_config, mapper)
        else {
            continue;
        };

        tracing::info!(
            idx,
            old_id = ?markers[idx].id,
            new_id = ?recovered.id,
            old_radius = markers[idx].ellipse_outer.as_ref().map(|e| e.mean_axis()),
            new_radius = recovered.ellipse_outer.as_ref().map(|e| e.mean_axis()),
            "inner-as-outer recovery: replaced marker"
        );
        markers[idx] = recovered;
        n_recovered += 1;
    }

    tracing::info!(n_recovered, "inner-as-outer recovery complete");
}
