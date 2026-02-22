use std::collections::HashMap;

use crate::detector::marker_build::DetectedMarker;
use crate::homography::homography_project;

use super::consistency::{
    candidate_passes_local_consistency_gate, local_edge_neighbor_ids, scrub_inconsistent_ids,
};
use super::index::dist2;
use super::types::{RecoverySource, ScrubStage, Trust};
use super::vote::{gather_trusted_neighbors_local_scale, vote_for_candidate, VoteOutcome};
use super::workspace::{
    apply_id_assignment, is_soft_locked_assignment, marker_center_is_finite,
    should_block_by_trusted_confidence, IdCorrectionWorkspace,
};

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

pub(super) fn candidate_reprojection_error(
    anchor_h: Option<&nalgebra::Matrix3<f64>>,
    board_index: &super::index::BoardIndex,
    id: usize,
    center: [f64; 2],
) -> Option<f64> {
    let h = anchor_h?;
    let bxy = board_index.id_to_xy.get(&id)?;
    let proj = homography_project(h, f64::from(bxy[0]), f64::from(bxy[1]));
    Some(dist2(proj, center).sqrt())
}

fn run_local_stage(
    ws: &mut IdCorrectionWorkspace<'_>,
    stage_name: &str,
    outer_mul: f64,
    max_iters: usize,
) {
    if max_iters == 0 {
        return;
    }

    for iter in 0..max_iters {
        let mut corrections = Vec::<(usize, usize)>::new();
        for i in 0..ws.markers.len() {
            if !marker_center_is_finite(&ws.markers[i]) || ws.trust[i].is_trusted() {
                continue;
            }
            let neighbors = gather_trusted_neighbors_local_scale(
                i,
                ws.markers,
                &ws.trust,
                &ws.board_index,
                &ws.outer_radii_px,
                outer_mul,
            );
            if neighbors.is_empty() {
                continue;
            }
            let effective_min_votes = ws.config.effective_min_votes(ws.markers[i].id.is_some());
            let vote = vote_for_candidate(
                ws.markers[i].center,
                ws.outer_radii_px[i],
                &neighbors,
                &ws.board_index,
                ws.board_index.pitch_mm * 0.6,
                effective_min_votes,
                ws.config.min_vote_weight_frac,
            );
            let candidate_id = match vote {
                VoteOutcome::Candidate { id, .. } => id,
                _ => continue,
            };
            if config_soft_lock_blocks_override(
                &ws.markers[i],
                ws.config.soft_lock_exact_decode,
                candidate_id,
            ) {
                continue;
            }
            if ws.markers[i].id.is_none() {
                if let (Some(h), Some(board_xy)) = (
                    ws.anchor_h.as_ref(),
                    ws.board_index.id_to_xy.get(&candidate_id),
                ) {
                    let proj =
                        homography_project(h, f64::from(board_xy[0]), f64::from(board_xy[1]));
                    let err = dist2(proj, ws.markers[i].center).sqrt();
                    if !err.is_finite() || err > ws.config.h_reproj_gate_px {
                        continue;
                    }
                }
            }
            if should_block_by_trusted_confidence(i, candidate_id, ws.markers, &ws.trust) {
                continue;
            }
            if !candidate_passes_local_consistency_gate(ws, i, candidate_id) {
                continue;
            }
            corrections.push((i, candidate_id));
        }

        let mut promoted = 0usize;
        for (i, candidate_id) in corrections {
            let _ = apply_id_assignment(
                &mut ws.markers[i],
                candidate_id,
                &mut ws.stats,
                RecoverySource::Local,
            );
            ws.trust[i] = Trust::RecoveredLocal;
            promoted += 1;
        }
        ws.stats.n_iterations += 1;

        tracing::debug!(
            stage = stage_name,
            outer_mul,
            iter = iter + 1,
            promoted,
            "id_correction local stage pass",
        );

        if promoted == 0 {
            break;
        }
    }
}

fn topology_support_counts(
    neighbor_ids: &[usize],
    board_index: &super::index::BoardIndex,
    current_id: Option<usize>,
) -> HashMap<usize, usize> {
    let mut support = HashMap::<usize, usize>::new();
    for neighbor_id in neighbor_ids {
        if let Some(board_nbrs) = board_index.board_neighbors.get(neighbor_id) {
            for &cand in board_nbrs {
                *support.entry(cand).or_insert(0) += 1;
            }
        }
    }
    if let Some(id) = current_id {
        support.entry(id).or_insert(0);
    }
    support
}

fn select_topology_candidate(
    ws: &IdCorrectionWorkspace<'_>,
    marker_index: usize,
    support: &HashMap<usize, usize>,
    neighbor_count: usize,
) -> Option<(usize, usize, f64)> {
    let mut best: Option<(usize, usize, usize, f64)> = None;
    for (&cand_id, &cand_support) in support {
        if cand_support < 2 {
            continue;
        }
        let contradiction = neighbor_count.saturating_sub(cand_support);
        if contradiction > cand_support {
            continue;
        }
        if config_soft_lock_blocks_override(
            &ws.markers[marker_index],
            ws.config.soft_lock_exact_decode,
            cand_id,
        ) {
            continue;
        }
        if should_block_by_trusted_confidence(marker_index, cand_id, ws.markers, &ws.trust) {
            continue;
        }
        let err = candidate_reprojection_error(
            ws.anchor_h.as_ref(),
            &ws.board_index,
            cand_id,
            ws.markers[marker_index].center,
        )
        .unwrap_or(0.0);
        match best {
            Some((best_id, best_support, best_contradiction, best_err)) => {
                let better = cand_support > best_support
                    || (cand_support == best_support && contradiction < best_contradiction)
                    || (cand_support == best_support
                        && contradiction == best_contradiction
                        && err < best_err)
                    || (cand_support == best_support
                        && contradiction == best_contradiction
                        && err == best_err
                        && cand_id < best_id);
                if better {
                    best = Some((cand_id, cand_support, contradiction, err));
                }
            }
            None => best = Some((cand_id, cand_support, contradiction, err)),
        }
    }
    best.map(|(id, support, _, err)| (id, support, err))
}

fn collect_topology_update_for_marker(
    ws: &IdCorrectionWorkspace<'_>,
    marker_index: usize,
) -> Option<(usize, usize)> {
    if !marker_center_is_finite(&ws.markers[marker_index])
        || matches!(
            ws.trust[marker_index],
            Trust::AnchorStrong | Trust::AnchorWeak
        )
    {
        return None;
    }
    let neighbor_ids = local_edge_neighbor_ids(
        marker_index,
        ws.markers,
        &ws.board_index,
        &ws.outer_radii_px,
        ws.config.consistency_outer_mul,
    );
    if neighbor_ids.len() < 2 {
        return None;
    }
    let current_id = ws.markers[marker_index].id;
    let support = topology_support_counts(&neighbor_ids, &ws.board_index, current_id);
    let (best_id, best_support, best_err) =
        select_topology_candidate(ws, marker_index, &support, neighbor_ids.len())?;

    let current_support = current_id
        .and_then(|id| support.get(&id).copied())
        .unwrap_or(0);
    let current_err = current_id.and_then(|id| {
        candidate_reprojection_error(
            ws.anchor_h.as_ref(),
            &ws.board_index,
            id,
            ws.markers[marker_index].center,
        )
    });
    let should_apply = match current_id {
        None => true,
        Some(id) if id == best_id => false,
        Some(_) => {
            best_support > current_support
                || (best_support == current_support
                    && current_err.is_some_and(|cur| best_err + 1.0 < cur))
        }
    };
    should_apply.then_some((marker_index, best_id))
}

fn run_topology_refinement(ws: &mut IdCorrectionWorkspace<'_>) {
    for pass in 0..2 {
        let updates = (0..ws.markers.len())
            .filter_map(|i| collect_topology_update_for_marker(ws, i))
            .collect::<Vec<_>>();
        if updates.is_empty() {
            break;
        }
        for (i, id) in updates {
            if apply_id_assignment(&mut ws.markers[i], id, &mut ws.stats, RecoverySource::Local) {
                ws.trust[i] = Trust::RecoveredLocal;
            }
        }
        tracing::debug!(
            pass = pass + 1,
            "id_correction topology refinement pass complete"
        );
    }
}

pub(super) fn run_adaptive_local_recovery(ws: &mut IdCorrectionWorkspace<'_>) {
    let outer_muls = ws.outer_muls.clone();
    for mul in outer_muls {
        run_local_stage(ws, "adaptive_local", mul, ws.config.max_iters);
    }
    run_topology_refinement(ws);
}

pub(super) fn run_post_consistency_refill(ws: &mut IdCorrectionWorkspace<'_>) {
    let first_outer_mul = ws.first_outer_mul();
    for _ in 0..2 {
        let _ = scrub_inconsistent_ids(ws, ScrubStage::Post);
        run_local_stage(ws, "post_consistency_refill", first_outer_mul, 1);
        run_topology_refinement(ws);
    }
}
