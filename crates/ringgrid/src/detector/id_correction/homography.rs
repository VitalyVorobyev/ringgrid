use std::collections::{BTreeMap, HashMap, HashSet};

use crate::detector::marker_build::DetectedMarker;
use crate::homography::{fit_homography_ransac, homography_project, RansacHomographyConfig};

use super::consistency::candidate_passes_local_consistency_gate;
use super::local::candidate_reprojection_error;
use super::types::{HomographyAssignment, HomographyFallbackModel, RecoverySource, Trust};
use super::workspace::{
    apply_id_assignment, is_soft_locked_assignment, marker_center_is_finite, IdCorrectionWorkspace,
};

const HOMOGRAPHY_FALLBACK_SEED: u64 = 0x1DC0_11D0;

#[inline]
fn seed_allowed_for_homography(trust: Trust) -> bool {
    trust.is_anchor()
}

#[inline]
fn config_soft_lock_blocks_override(
    marker: &DetectedMarker,
    soft_lock_enable: bool,
    candidate_id: usize,
) -> bool {
    let current_id = marker.id;
    soft_lock_enable
        && is_soft_locked_assignment(marker, soft_lock_enable)
        && current_id.is_some()
        && current_id != Some(candidate_id)
}

pub(super) fn collect_best_trusted_by_id<F>(
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &super::index::BoardIndex,
    mut include: F,
) -> BTreeMap<usize, usize>
where
    F: FnMut(Trust) -> bool,
{
    let mut trusted_by_id = BTreeMap::<usize, usize>::new();
    for (i, m) in markers.iter().enumerate() {
        if !include(trust[i]) || !marker_center_is_finite(m) {
            continue;
        }
        let Some(id) = m.id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id) {
            continue;
        }
        match trusted_by_id.get_mut(&id) {
            Some(best_idx) => {
                if m.confidence > markers[*best_idx].confidence {
                    *best_idx = i;
                }
            }
            None => {
                trusted_by_id.insert(id, i);
            }
        }
    }
    trusted_by_id
}

fn build_homography_correspondences(
    trusted_by_id: &BTreeMap<usize, usize>,
    markers: &[DetectedMarker],
    board_index: &super::index::BoardIndex,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut src_board_mm = Vec::<[f64; 2]>::with_capacity(trusted_by_id.len());
    let mut dst_image_px = Vec::<[f64; 2]>::with_capacity(trusted_by_id.len());
    for (&id, &idx) in trusted_by_id {
        let Some(bxy) = board_index.id_to_xy.get(&id) else {
            continue;
        };
        src_board_mm.push([f64::from(bxy[0]), f64::from(bxy[1])]);
        dst_image_px.push(markers[idx].center);
    }
    (src_board_mm, dst_image_px)
}

fn fit_homography_model_from_trusted(
    trusted_by_id: BTreeMap<usize, usize>,
    markers: &[DetectedMarker],
    board_index: &super::index::BoardIndex,
    inlier_threshold: f64,
    min_inliers: usize,
    max_iters: usize,
    error_context: &'static str,
) -> Option<HomographyFallbackModel> {
    if trusted_by_id.len() < 4 {
        tracing::debug!(
            n_unique_ids = trusted_by_id.len(),
            "{error_context}: too few unique trusted IDs",
        );
        return None;
    }
    let (src_board_mm, dst_image_px) =
        build_homography_correspondences(&trusted_by_id, markers, board_index);
    let ransac_cfg = RansacHomographyConfig {
        max_iters,
        inlier_threshold,
        min_inliers: min_inliers.min(src_board_mm.len()).max(4),
        seed: HOMOGRAPHY_FALLBACK_SEED,
    };
    let h_result = match fit_homography_ransac(&src_board_mm, &dst_image_px, &ransac_cfg) {
        Ok(r) => r,
        Err(err) => {
            tracing::debug!(
                n_corr = src_board_mm.len(),
                "{error_context} fit failed: {}",
                err
            );
            return None;
        }
    };
    let Some(h_inv) = h_result.h.try_inverse() else {
        tracing::debug!("{error_context}: non-invertible H");
        return None;
    };
    Some(HomographyFallbackModel {
        trusted_by_id,
        h: h_result.h,
        h_inv,
        n_inliers: h_result.n_inliers,
    })
}

pub(super) fn fit_anchor_homography_for_local_stage(
    ws: &IdCorrectionWorkspace<'_>,
) -> Option<nalgebra::Matrix3<f64>> {
    let trusted_by_id = collect_best_trusted_by_id(ws.markers, &ws.trust, &ws.board_index, |t| {
        matches!(t, Trust::AnchorStrong | Trust::AnchorWeak)
    });
    fit_homography_model_from_trusted(
        trusted_by_id,
        ws.markers,
        &ws.board_index,
        ws.config.h_reproj_gate_px,
        ws.config.homography_min_inliers,
        1200,
        "id_correction anchor homography",
    )
    .map(|m| m.h)
}

fn collect_homography_assignments(
    ws: &IdCorrectionWorkspace<'_>,
    trusted_conf_by_id: &HashMap<usize, f32>,
    model: &HomographyFallbackModel,
    top_k: usize,
) -> Vec<HomographyAssignment> {
    let mut assignments = Vec::<HomographyAssignment>::new();
    for (i, m) in ws.markers.iter().enumerate() {
        let eligible = !matches!(ws.trust[i], Trust::AnchorStrong | Trust::AnchorWeak);
        if !eligible || !marker_center_is_finite(m) {
            continue;
        }
        if m.id.is_some() && is_soft_locked_assignment(m, ws.config.soft_lock_exact_decode) {
            continue;
        }
        let board_hint = homography_project(&model.h_inv, m.center[0], m.center[1]);
        if !(board_hint[0].is_finite() && board_hint[1].is_finite()) {
            continue;
        }

        let mut best: Option<(usize, f64)> = None;
        for (candidate_id, _) in ws.board_index.nearest_k_ids(board_hint, top_k) {
            if let Some(&trusted_conf) = trusted_conf_by_id.get(&candidate_id) {
                if trusted_conf >= m.confidence {
                    continue;
                }
            }
            if !candidate_passes_local_consistency_gate(ws, i, candidate_id) {
                continue;
            }
            let Some(err) = candidate_reprojection_error(
                Some(&model.h),
                &ws.board_index,
                candidate_id,
                m.center,
            ) else {
                continue;
            };
            match best {
                Some((best_id, best_err)) => {
                    if err < best_err || (err == best_err && candidate_id < best_id) {
                        best = Some((candidate_id, err));
                    }
                }
                None => best = Some((candidate_id, err)),
            }
        }

        let Some((id, reproj_err_px)) = best else {
            continue;
        };
        if reproj_err_px > ws.config.h_reproj_gate_px {
            continue;
        }
        let current_err = m.id.and_then(|cur_id| {
            candidate_reprojection_error(Some(&model.h), &ws.board_index, cur_id, m.center)
        });
        let should_apply = match m.id {
            None => true,
            Some(cur_id) if cur_id == id => false,
            Some(_) => current_err.is_none_or(|cur| reproj_err_px + 1.0 < cur),
        };
        if should_apply {
            assignments.push(HomographyAssignment {
                marker_index: i,
                id,
                reproj_err_px,
            });
        }
    }
    assignments
}

fn apply_homography_assignments(
    ws: &mut IdCorrectionWorkspace<'_>,
    assignments: &mut [HomographyAssignment],
    claimed_ids: &mut HashSet<usize>,
) -> usize {
    assignments.sort_by(|a, b| {
        a.reproj_err_px
            .total_cmp(&b.reproj_err_px)
            .then_with(|| a.marker_index.cmp(&b.marker_index))
            .then_with(|| a.id.cmp(&b.id))
    });
    let mut seeded = 0usize;
    for a in assignments.iter().copied() {
        if claimed_ids.contains(&a.id) {
            continue;
        }
        let i = a.marker_index;
        if matches!(ws.trust[i], Trust::AnchorStrong | Trust::AnchorWeak) {
            continue;
        }
        if config_soft_lock_blocks_override(&ws.markers[i], ws.config.soft_lock_exact_decode, a.id)
        {
            continue;
        }
        claimed_ids.insert(a.id);
        if apply_id_assignment(
            &mut ws.markers[i],
            a.id,
            &mut ws.stats,
            RecoverySource::Homography,
        ) {
            ws.trust[i] = Trust::RecoveredHomography;
            seeded += 1;
        }
    }
    seeded
}

pub(super) fn run_homography_fallback(ws: &mut IdCorrectionWorkspace<'_>) {
    if !ws.config.homography_fallback_enable {
        return;
    }
    let n_trusted = ws.trust.iter().filter(|&&t| t.is_trusted()).count();
    if n_trusted < ws.config.homography_min_trusted {
        tracing::debug!(
            n_trusted,
            min_required = ws.config.homography_min_trusted,
            "id_correction homography fallback skipped: insufficient trusted markers",
        );
        return;
    }

    let trusted_by_id = collect_best_trusted_by_id(ws.markers, &ws.trust, &ws.board_index, |t| {
        seed_allowed_for_homography(t)
    });
    let Some(model) = fit_homography_model_from_trusted(
        trusted_by_id,
        ws.markers,
        &ws.board_index,
        ws.config.h_reproj_gate_px,
        ws.config.homography_min_inliers,
        1200,
        "id_correction homography fallback",
    ) else {
        return;
    };

    let top_k = 19usize;
    let trusted_conf_by_id = model
        .trusted_by_id
        .iter()
        .map(|(&id, &idx)| (id, ws.markers[idx].confidence))
        .collect::<HashMap<_, _>>();
    let mut assignments = collect_homography_assignments(ws, &trusted_conf_by_id, &model, top_k);
    let mut claimed_ids = model.trusted_by_id.keys().copied().collect::<HashSet<_>>();
    let seeded = apply_homography_assignments(ws, &mut assignments, &mut claimed_ids);

    tracing::debug!(
        n_unique_trusted = model.trusted_by_id.len(),
        n_inliers = model.n_inliers,
        n_seeded = seeded,
        gate_px = ws.config.h_reproj_gate_px,
        top_k,
        "id_correction homography fallback summary",
    );
}
