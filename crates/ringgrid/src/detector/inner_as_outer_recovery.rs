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
use crate::detector::MarkerRecord;
use crate::marker::decode::DecodeResult;
use crate::pixelmap::PixelMapper;
use crate::ring::OuterEstimationConfig;

use super::{
    DetectConfig,
    marker_build::{DetectionSource, marker_from_outer_fit},
    median_neighbor_outer_radius_px, outer_fit,
};

/// Annotates each marker with the ratio of its outer radius to the median
/// outer radius of its k nearest neighbors. Values below `warn_threshold`
/// (the recovery `ratio_threshold`) indicate a potential inner-as-outer
/// substitution and are logged.
pub(crate) fn annotate_neighbor_radius_ratios(
    markers: &mut [MarkerRecord],
    k: usize,
    warn_threshold: f32,
) {
    // Compute ratios in a separate immutable pass to satisfy the borrow checker.
    let ratios: Vec<Option<f32>> = {
        let m_ref: &[MarkerRecord] = markers;
        m_ref
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let own_radius = m.ellipse_outer.as_ref()?.mean_axis();
                let median = median_neighbor_outer_radius_px(m.center, m_ref, k, Some(i), 1)?;
                if median > 0.0 {
                    Some((own_radius / median) as f32)
                } else {
                    None
                }
            })
            .collect()
    };

    for (marker, ratio) in markers.iter_mut().zip(ratios) {
        marker.fit.neighbor_radius_ratio = ratio;
        if let Some(r) = ratio
            && r < warn_threshold
        {
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

/// Attempts to recover markers where the outer fit locked onto the inner ring
/// edge. For each marker whose `neighbor_radius_ratio` is below the configured
/// threshold, re-attempts the outer fit using the neighbor-median radius as the
/// corrected expected radius. If the new fit succeeds (with a valid decode), the
/// marker is replaced in-place; otherwise the original is kept.
///
/// Runs on working-frame markers (centers not yet mapped to the image), before
/// the final homography refit. Callers re-annotate the ratios afterwards.
fn flagged_recovery_indices(markers: &[MarkerRecord], ratio_threshold: f32) -> Vec<usize> {
    markers
        .iter()
        .enumerate()
        .filter_map(|(i, marker)| {
            let ratio = marker.fit.neighbor_radius_ratio?;
            (ratio < ratio_threshold).then_some(i)
        })
        .collect()
}

/// Outer-radius estimation for recovery: the detector's settings with the
/// recovery-specific thresholds applied, and the default (unwidened)
/// hypothesis search window — the neighbour-median radius prior is already
/// close to the true outer radius. Edge sampling is the detector's own: its
/// depth gate is contrast-relative, so blur-softened edges need no separate,
/// relaxed floor.
fn recovery_estimation(config: &DetectConfig) -> OuterEstimationConfig {
    let cfg = &config.advanced.inner_as_outer_recovery;
    OuterEstimationConfig {
        search_halfwidth_px: OuterEstimationConfig::default().search_halfwidth_px,
        min_theta_consistency: cfg.min_theta_consistency,
        min_theta_coverage: cfg.min_theta_coverage,
        refine_halfwidth_px: cfg.refine_halfwidth_px,
        ..config.advanced.outer_estimation.clone()
    }
}

fn resolve_recovered_id(
    idx: usize,
    marker: &MarkerRecord,
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

    let orig_center_wf = marker.center;
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
    markers: &[MarkerRecord],
    config: &DetectConfig,
    estimation: &OuterEstimationConfig,
    mapper: Option<&dyn PixelMapper>,
) -> Option<MarkerRecord> {
    let cfg = &config.advanced.inner_as_outer_recovery;
    let center_wf = markers[idx].center;
    let center_f32 = [center_wf[0] as f32, center_wf[1] as f32];

    let Some(r_corrected) =
        median_neighbor_outer_radius_px(center_wf, markers, cfg.k_neighbors, Some(idx), 1)
            .map(|r| r as f32)
    else {
        tracing::debug!(
            idx,
            "recovery skipped: could not compute neighbor median radius"
        );
        return None;
    };

    let mut candidate = match outer_fit::fit_outer_candidate_with_sampling(
        gray,
        center_f32,
        r_corrected,
        config,
        mapper,
        estimation,
        &config.advanced.edge_sample,
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

    let (marker_id, decode_result) = resolve_recovered_id(
        idx,
        &markers[idx],
        &candidate.outer,
        candidate.decode_result.take(),
        r_corrected,
        cfg.size_gate_tolerance,
    )?;
    candidate.decode_result = decode_result;

    let marker = marker_from_outer_fit(gray, candidate, config, mapper);

    // `center` stays in the working frame like every marker at this stage;
    // finalize maps all centers to the image afterwards.
    Some(MarkerRecord {
        id: marker_id,
        // Recovery re-fits geometry in place (center proximity gated), so the
        // lattice cell is unchanged; coded-path sync re-derives it from the id.
        grid_coord: markers[idx].grid_coord,
        // board_xy_mm stays None: populated by sync_marker_board_correspondence.
        source: DetectionSource::FitDecoded,
        ..marker
    })
}

pub(crate) fn try_recover_inner_as_outer(
    gray: &GrayImage,
    markers: &mut [MarkerRecord],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) {
    let flagged = flagged_recovery_indices(
        markers,
        config.advanced.inner_as_outer_recovery.ratio_threshold,
    );
    if flagged.is_empty() {
        return;
    }

    tracing::info!(
        n_flagged = flagged.len(),
        "attempting inner-as-outer recovery for flagged markers"
    );
    let estimation = recovery_estimation(config);
    let mut n_recovered = 0usize;
    for idx in flagged {
        let Some(recovered) =
            recover_marker_at_index(idx, gray, markers, config, &estimation, mapper)
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
