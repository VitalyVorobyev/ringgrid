//! Structural ID verification and correction using hex neighborhood consensus.
//!
//! This stage operates in image-space pixels (`DetectedMarker.center`) and
//! board-space millimeters (`BoardLayout` marker coordinates).
//!
//! ## Staged algorithm
//!
//! 1. Build board index and per-marker local scale (`ellipse_outer.mean_axis`).
//! 2. Bootstrap trusted anchors from decoded IDs.
//! 3. Pre-recovery consistency scrub (precision-first).
//! 4. Local iterative recovery with local-scale neighborhoods.
//! 5. Constrained rough-homography fallback seeding.
//! 6. Post-recovery consistency sweep + short local refill.
//! 7. Cleanup and deterministic conflict resolution.

mod index;
mod math;
mod vote;

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::board_layout::BoardLayout;
use crate::detector::config::IdCorrectionConfig;
use crate::detector::marker_build::DetectedMarker;
use crate::homography::{fit_homography_ransac, homography_project, RansacHomographyConfig};
use crate::marker::codebook::CODEBOOK_MIN_CYCLIC_DIST;

use index::{dist2, BoardIndex};
use vote::{
    gather_trusted_neighbors_local_scale, resolve_id_conflicts, vote_for_candidate, VoteOutcome,
};

const HOMOGRAPHY_FALLBACK_SEED: u64 = 0x1DC0_11D0;

// ── Internal types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trust {
    Untrusted,
    AnchorWeak,
    AnchorStrong,
    RecoveredLocal,
    RecoveredHomography,
}

impl Trust {
    pub(super) fn is_trusted(self) -> bool {
        self != Self::Untrusted
    }

    fn is_anchor(self) -> bool {
        matches!(self, Self::AnchorWeak | Self::AnchorStrong)
    }
}

#[derive(Debug, Clone, Copy)]
struct HomographyAssignment {
    marker_index: usize,
    id: usize,
    reproj_err_px: f64,
}

#[derive(Clone, Copy)]
struct LocalStageSpec<'a> {
    board_index: &'a BoardIndex,
    config: &'a IdCorrectionConfig,
    outer_radii_px: &'a [f64],
    anchor_h: Option<&'a nalgebra::Matrix3<f64>>,
    outer_mul: f64,
    max_iters: usize,
    stage_name: &'a str,
}

#[derive(Debug, Clone, Copy, Default)]
struct ConsistencyEvidence {
    n_neighbors: usize,
    support_edges: usize,
    contradiction_edges: usize,
    contradiction_frac: f64,
    vote_mismatch: bool,
    vote_winner_frac: f64,
}

#[derive(Debug, Clone, Copy)]
enum RecoverySource {
    Local,
    Homography,
}

#[derive(Debug, Clone, Copy)]
enum ScrubStage {
    Pre,
    Post,
}

// ── Public output ─────────────────────────────────────────────────────────────

/// Statistics produced by the ID verification and correction stage.
#[derive(Debug, Clone, Default)]
pub(crate) struct IdCorrectionStats {
    /// Markers whose decoded ID was replaced with a different, verified ID.
    pub n_ids_corrected: usize,
    /// Markers whose id was `None` and received a new ID.
    pub n_ids_recovered: usize,
    /// Markers assigned by rough-homography fallback.
    pub n_homography_seeded: usize,
    /// Markers whose ID was cleared (`id = None`) after failed verification.
    pub n_ids_cleared: usize,
    /// Markers removed entirely (only when `remove_unverified = true`).
    pub n_markers_removed: usize,
    /// Markers confirmed as structurally consistent with the board layout.
    pub n_verified: usize,
    /// Count of unresolved markers with no trusted neighbors in final diagnosis.
    pub n_unverified_no_neighbors: usize,
    /// Count of unresolved markers with no usable votes in final diagnosis.
    pub n_unverified_no_votes: usize,
    /// Count of unresolved markers blocked by vote-fraction gate in diagnosis.
    pub n_unverified_gate_rejects: usize,
    /// Number of local iterative passes executed across all local stages.
    pub n_iterations: usize,
    /// Estimated board pitch in image pixels (legacy diagnostic field).
    pub pitch_px_estimated: Option<f64>,
    /// IDs cleared by pre-recovery consistency scrub.
    pub n_ids_cleared_inconsistent_pre: usize,
    /// IDs cleared by post-recovery consistency sweep.
    pub n_ids_cleared_inconsistent_post: usize,
    /// Soft-locked exact decodes cleared on strict contradiction.
    pub n_soft_locked_cleared: usize,
    /// IDs recovered by local iterative stage.
    pub n_recovered_local: usize,
    /// IDs recovered by homography fallback stage.
    pub n_recovered_homography: usize,
    /// Remaining IDs that still violate consistency rules after full pipeline.
    pub n_inconsistent_remaining: usize,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn marker_center_is_finite(marker: &DetectedMarker) -> bool {
    marker.center[0].is_finite() && marker.center[1].is_finite()
}

#[inline]
fn marker_outer_radius_px(marker: &DetectedMarker) -> Option<f64> {
    marker
        .ellipse_outer
        .as_ref()
        .map(|e| e.mean_axis())
        .filter(|r| r.is_finite() && *r > 0.0)
}

#[inline]
fn is_exact_decode(marker: &DetectedMarker) -> bool {
    marker
        .decode
        .as_ref()
        .is_some_and(|d| d.best_dist == 0 && usize::from(d.margin) >= CODEBOOK_MIN_CYCLIC_DIST)
}

#[inline]
fn is_soft_locked_assignment(marker: &DetectedMarker, soft_lock_enable: bool) -> bool {
    if !soft_lock_enable {
        return false;
    }
    let Some(id) = marker.id else {
        return false;
    };
    marker.decode.as_ref().is_some_and(|d| {
        d.best_dist == 0 && usize::from(d.margin) >= CODEBOOK_MIN_CYCLIC_DIST && d.best_id == id
    })
}

fn compute_outer_radii_px(markers: &[DetectedMarker]) -> Vec<f64> {
    let mut valid = markers
        .iter()
        .filter_map(marker_outer_radius_px)
        .collect::<Vec<_>>();
    let median = if valid.is_empty() {
        20.0
    } else {
        valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let m = valid.len() / 2;
        if valid.len().is_multiple_of(2) {
            0.5 * (valid[m - 1] + valid[m])
        } else {
            valid[m]
        }
    };
    markers
        .iter()
        .map(|m| marker_outer_radius_px(m).unwrap_or(median))
        .collect()
}

fn estimate_adjacent_spacing_px(
    markers: &[DetectedMarker],
    board_index: &BoardIndex,
) -> Option<f64> {
    let mut dists = Vec::<f64>::new();
    for i in 0..markers.len() {
        let Some(id_i) = markers[i].id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id_i) || !marker_center_is_finite(&markers[i]) {
            continue;
        }
        for j in (i + 1)..markers.len() {
            let Some(id_j) = markers[j].id else {
                continue;
            };
            if !board_index.id_to_xy.contains_key(&id_j) || !marker_center_is_finite(&markers[j]) {
                continue;
            }
            if !board_index.are_neighbors(id_i, id_j) {
                continue;
            }
            let d = dist2(markers[i].center, markers[j].center).sqrt();
            if d.is_finite() && d > 1.0 {
                dists.push(d);
            }
        }
    }
    if dists.is_empty() {
        return None;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = dists.len() / 2;
    Some(if dists.len().is_multiple_of(2) {
        0.5 * (dists[mid - 1] + dists[mid])
    } else {
        dists[mid]
    })
}

fn build_outer_mul_schedule(config: &IdCorrectionConfig) -> Vec<f64> {
    let mut out: Vec<f64> = config
        .auto_search_radius_outer_muls
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if out.is_empty() {
        out.push(3.2);
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    out
}

fn should_block_by_trusted_confidence(
    marker_index: usize,
    candidate_id: usize,
    markers: &[DetectedMarker],
    trust: &[Trust],
) -> bool {
    markers.iter().enumerate().any(|(j, m)| {
        j != marker_index
            && trust[j].is_trusted()
            && m.id == Some(candidate_id)
            && m.confidence >= markers[marker_index].confidence
    })
}

fn apply_id_assignment(
    marker: &mut DetectedMarker,
    new_id: usize,
    stats: &mut IdCorrectionStats,
    source: RecoverySource,
) -> bool {
    let old_id = marker.id;
    let changed = old_id != Some(new_id);
    if changed {
        if old_id.is_some() {
            stats.n_ids_corrected += 1;
        } else {
            stats.n_ids_recovered += 1;
            match source {
                RecoverySource::Local => stats.n_recovered_local += 1,
                RecoverySource::Homography => {
                    stats.n_recovered_homography += 1;
                    stats.n_homography_seeded += 1;
                }
            }
        }
        marker.id = Some(new_id);
    }
    changed
}

fn clear_marker_id(
    i: usize,
    markers: &mut [DetectedMarker],
    trust: &mut [Trust],
    stats: &mut IdCorrectionStats,
    soft_lock_enable: bool,
    stage: ScrubStage,
) -> bool {
    if markers[i].id.is_none() {
        return false;
    }
    let was_soft_locked = is_soft_locked_assignment(&markers[i], soft_lock_enable);
    markers[i].id = None;
    trust[i] = Trust::Untrusted;
    stats.n_ids_cleared += 1;
    match stage {
        ScrubStage::Pre => stats.n_ids_cleared_inconsistent_pre += 1,
        ScrubStage::Post => stats.n_ids_cleared_inconsistent_post += 1,
    }
    if was_soft_locked {
        stats.n_soft_locked_cleared += 1;
    }
    true
}

fn local_edge_neighbor_ids(
    i: usize,
    markers: &[DetectedMarker],
    board_index: &BoardIndex,
    outer_radii_px: &[f64],
    outer_mul: f64,
) -> Vec<usize> {
    if !marker_center_is_finite(&markers[i]) || !outer_mul.is_finite() || outer_mul <= 0.0 {
        return Vec::new();
    }
    let center_i = markers[i].center;
    let radius_i = outer_radii_px[i];
    let mut ids = Vec::<usize>::new();
    for (j, m) in markers.iter().enumerate() {
        if i == j || !marker_center_is_finite(m) {
            continue;
        }
        let id_j = match m.id {
            Some(id) if board_index.id_to_xy.contains_key(&id) => id,
            _ => continue,
        };
        let gate = outer_mul * 0.5 * (radius_i + outer_radii_px[j]);
        if gate <= 0.0 || !gate.is_finite() {
            continue;
        }
        if dist2(center_i, m.center) <= gate * gate {
            ids.push(id_j);
        }
    }
    ids
}

fn anchor_edge_support_counts(
    i: usize,
    assumed_id: usize,
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
) -> (usize, usize) {
    if !marker_center_is_finite(&markers[i]) {
        return (0, 0);
    }
    let center_i = markers[i].center;
    let radius_i = outer_radii_px[i];
    let mut support_anchor = 0usize;
    let mut contradiction_anchor = 0usize;
    for (j, m) in markers.iter().enumerate() {
        if i == j || !marker_center_is_finite(m) {
            continue;
        }
        if !matches!(trust[j], Trust::AnchorStrong | Trust::AnchorWeak) {
            continue;
        }
        let Some(id_j) = m.id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id_j) {
            continue;
        }
        let gate = config.consistency_outer_mul * 0.5 * (radius_i + outer_radii_px[j]);
        if gate <= 0.0 || !gate.is_finite() {
            continue;
        }
        if dist2(center_i, m.center) > gate * gate {
            continue;
        }
        if board_index.are_neighbors(assumed_id, id_j) {
            support_anchor += 1;
        } else {
            contradiction_anchor += 1;
        }
    }
    (support_anchor, contradiction_anchor)
}

fn consistency_evidence_for_id(
    i: usize,
    assumed_id: usize,
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
) -> ConsistencyEvidence {
    let neighbor_ids = local_edge_neighbor_ids(
        i,
        markers,
        board_index,
        outer_radii_px,
        config.consistency_outer_mul,
    );
    let mut support_edges = 0usize;
    let mut contradiction_edges = 0usize;
    for &neighbor_id in &neighbor_ids {
        if board_index.are_neighbors(assumed_id, neighbor_id) {
            support_edges += 1;
        } else {
            contradiction_edges += 1;
        }
    }
    let n_neighbors = support_edges + contradiction_edges;
    let contradiction_frac = if n_neighbors == 0 {
        0.0
    } else {
        contradiction_edges as f64 / n_neighbors as f64
    };

    let vote_neighbors = gather_trusted_neighbors_local_scale(
        i,
        markers,
        trust,
        board_index,
        outer_radii_px,
        config.consistency_outer_mul,
    );
    let vote = vote_for_candidate(
        markers[i].center,
        outer_radii_px[i],
        &vote_neighbors,
        board_index,
        board_index.pitch_mm * 0.6,
        config.min_votes,
        config.min_vote_weight_frac,
    );

    let (vote_mismatch, vote_winner_frac) = match vote {
        VoteOutcome::Candidate {
            id,
            winner_weight_frac,
            ..
        } if id != assumed_id => (true, winner_weight_frac),
        _ => (false, 0.0),
    };

    ConsistencyEvidence {
        n_neighbors,
        support_edges,
        contradiction_edges,
        contradiction_frac,
        vote_mismatch,
        vote_winner_frac,
    }
}

fn should_clear_by_consistency(
    evidence: ConsistencyEvidence,
    soft_locked: bool,
    config: &IdCorrectionConfig,
) -> bool {
    if evidence.n_neighbors < config.consistency_min_neighbors {
        return false;
    }
    if soft_locked {
        // An exact decode with zero supporting neighbors and ≥ 2 contradictions
        // is almost certainly wrong — the edge evidence alone is sufficient.
        evidence.support_edges == 0 && evidence.contradiction_edges >= 2
    } else {
        let strong_vote_mismatch = evidence.vote_mismatch && evidence.vote_winner_frac >= 0.60;
        evidence.support_edges < config.consistency_min_support_edges
            || evidence.contradiction_frac > f64::from(config.consistency_max_contradiction_frac)
            || strong_vote_mismatch
    }
}

#[allow(clippy::too_many_arguments)]
fn scrub_inconsistent_ids(
    markers: &mut [DetectedMarker],
    trust: &mut [Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
    stats: &mut IdCorrectionStats,
    stage: ScrubStage,
) -> usize {
    // Phase 1: collect indices to clear (order-independent — evidence is computed
    // against the unmodified marker state so no clearing order bias).
    let mut to_clear = Vec::<usize>::new();
    for i in 0..markers.len() {
        let Some(id) = markers[i].id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id) {
            to_clear.push(i);
            continue;
        }
        let evidence =
            consistency_evidence_for_id(i, id, markers, trust, board_index, config, outer_radii_px);
        let (support_anchor, contradiction_anchor) =
            anchor_edge_support_counts(i, id, markers, trust, board_index, config, outer_radii_px);
        let recovered_two_neighbor_contradiction = matches!(stage, ScrubStage::Post)
            && matches!(trust[i], Trust::RecoveredLocal | Trust::RecoveredHomography)
            && ((evidence.support_edges == 0
                && evidence.contradiction_edges >= 2
                && evidence.vote_mismatch
                && evidence.vote_winner_frac >= 0.60)
                || (contradiction_anchor >= 1 && support_anchor == 0));
        if recovered_two_neighbor_contradiction {
            to_clear.push(i);
            continue;
        }
        let is_soft_locked = is_soft_locked_assignment(&markers[i], config.soft_lock_exact_decode);
        if should_clear_by_consistency(evidence, is_soft_locked, config) {
            to_clear.push(i);
        }
    }

    // Phase 2: apply clearings.
    let mut cleared = 0usize;
    for i in to_clear {
        if clear_marker_id(
            i,
            markers,
            trust,
            stats,
            config.soft_lock_exact_decode,
            stage,
        ) {
            cleared += 1;
        }
    }
    cleared
}

fn run_local_stage(
    markers: &mut [DetectedMarker],
    trust: &mut [Trust],
    stats: &mut IdCorrectionStats,
    spec: LocalStageSpec<'_>,
) {
    if spec.max_iters == 0 {
        return;
    }

    for iter in 0..spec.max_iters {
        let mut corrections = Vec::<(usize, usize)>::new();
        for i in 0..markers.len() {
            if !marker_center_is_finite(&markers[i]) || trust[i].is_trusted() {
                continue;
            }
            let neighbors = gather_trusted_neighbors_local_scale(
                i,
                markers,
                trust,
                spec.board_index,
                spec.outer_radii_px,
                spec.outer_mul,
            );
            if neighbors.is_empty() {
                continue;
            }
            let effective_min_votes = if markers[i].id.is_none() {
                spec.config.min_votes_recover
            } else {
                spec.config.min_votes
            };
            let vote = vote_for_candidate(
                markers[i].center,
                spec.outer_radii_px[i],
                &neighbors,
                spec.board_index,
                spec.board_index.pitch_mm * 0.6,
                effective_min_votes,
                spec.config.min_vote_weight_frac,
            );
            let candidate_id = match vote {
                VoteOutcome::Candidate { id, .. } => id,
                _ => continue,
            };
            if config_soft_lock_blocks_override(
                &markers[i],
                spec.config.soft_lock_exact_decode,
                candidate_id,
            ) {
                continue;
            }
            if markers[i].id.is_none() {
                if let (Some(h), Some(board_xy)) =
                    (spec.anchor_h, spec.board_index.id_to_xy.get(&candidate_id))
                {
                    let proj =
                        homography_project(h, f64::from(board_xy[0]), f64::from(board_xy[1]));
                    let err = dist2(proj, markers[i].center).sqrt();
                    if !err.is_finite() || err > spec.config.h_reproj_gate_px {
                        continue;
                    }
                }
            }
            if should_block_by_trusted_confidence(i, candidate_id, markers, trust) {
                continue;
            }
            if !candidate_passes_local_consistency_gate(
                i,
                candidate_id,
                markers,
                spec.board_index,
                spec.config,
                spec.outer_radii_px,
            ) {
                continue;
            }
            corrections.push((i, candidate_id));
        }

        let mut promoted = 0usize;
        for (i, candidate_id) in corrections {
            let _ =
                apply_id_assignment(&mut markers[i], candidate_id, stats, RecoverySource::Local);
            trust[i] = Trust::RecoveredLocal;
            promoted += 1;
        }
        stats.n_iterations += 1;

        tracing::debug!(
            stage = spec.stage_name,
            outer_mul = spec.outer_mul,
            iter = iter + 1,
            promoted,
            "id_correction local stage pass",
        );

        if promoted == 0 {
            break;
        }
    }
}

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

fn candidate_passes_local_consistency_gate(
    marker_index: usize,
    candidate_id: usize,
    markers: &[DetectedMarker],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
) -> bool {
    let neighbor_ids = local_edge_neighbor_ids(
        marker_index,
        markers,
        board_index,
        outer_radii_px,
        config.consistency_outer_mul,
    );
    let mut support_edges = 0usize;
    let mut contradiction_edges = 0usize;
    for id in neighbor_ids {
        if board_index.are_neighbors(candidate_id, id) {
            support_edges += 1;
        } else {
            contradiction_edges += 1;
        }
    }
    let total = support_edges + contradiction_edges;
    if support_edges < 1 {
        return false;
    }
    if total == 0 {
        return false;
    }
    let contradiction_frac = contradiction_edges as f64 / total as f64;
    contradiction_frac <= f64::from(config.consistency_max_contradiction_frac)
}

fn seed_allowed_for_homography(trust: Trust) -> bool {
    trust.is_anchor()
}

fn run_topology_refinement(
    markers: &mut [DetectedMarker],
    trust: &mut [Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
    anchor_h: Option<&nalgebra::Matrix3<f64>>,
    stats: &mut IdCorrectionStats,
) {
    for pass in 0..2 {
        let mut updates = Vec::<(usize, usize)>::new();
        for i in 0..markers.len() {
            if !marker_center_is_finite(&markers[i])
                || matches!(trust[i], Trust::AnchorStrong | Trust::AnchorWeak)
            {
                continue;
            }
            let neighbor_ids = local_edge_neighbor_ids(
                i,
                markers,
                board_index,
                outer_radii_px,
                config.consistency_outer_mul,
            );
            if neighbor_ids.len() < 2 {
                continue;
            }

            let mut support = HashMap::<usize, usize>::new();
            for neighbor_id in &neighbor_ids {
                if let Some(board_nbrs) = board_index.board_neighbors.get(neighbor_id) {
                    for &cand in board_nbrs {
                        *support.entry(cand).or_insert(0) += 1;
                    }
                }
            }
            if let Some(current_id) = markers[i].id {
                support.entry(current_id).or_insert(0);
            }

            let current_id = markers[i].id;
            let current_support = current_id
                .and_then(|id| support.get(&id).copied())
                .unwrap_or(0);
            let current_err = current_id.and_then(|id| {
                let h = anchor_h?;
                let bxy = board_index.id_to_xy.get(&id)?;
                let proj = homography_project(h, f64::from(bxy[0]), f64::from(bxy[1]));
                Some(dist2(proj, markers[i].center).sqrt())
            });

            let mut best: Option<(usize, usize, usize, f64)> = None;
            for (&cand_id, &cand_support) in &support {
                if cand_support < 2 {
                    continue;
                }
                let contradiction = neighbor_ids.len().saturating_sub(cand_support);
                if contradiction > cand_support {
                    continue;
                }
                if config_soft_lock_blocks_override(
                    &markers[i],
                    config.soft_lock_exact_decode,
                    cand_id,
                ) {
                    continue;
                }
                if should_block_by_trusted_confidence(i, cand_id, markers, trust) {
                    continue;
                }

                let err =
                    if let (Some(h), Some(bxy)) = (anchor_h, board_index.id_to_xy.get(&cand_id)) {
                        let proj = homography_project(h, f64::from(bxy[0]), f64::from(bxy[1]));
                        dist2(proj, markers[i].center).sqrt()
                    } else {
                        0.0
                    };

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

            let Some((best_id, best_support, _, best_err)) = best else {
                continue;
            };
            let should_apply = match current_id {
                None => true,
                Some(id) if id == best_id => false,
                Some(_) => {
                    best_support > current_support
                        || (best_support == current_support
                            && current_err.is_some_and(|cur| best_err + 1.0 < cur))
                }
            };
            if should_apply {
                updates.push((i, best_id));
            }
        }

        if updates.is_empty() {
            break;
        }
        for (i, id) in updates {
            if apply_id_assignment(&mut markers[i], id, stats, RecoverySource::Local) {
                trust[i] = Trust::RecoveredLocal;
            }
        }
        tracing::debug!(
            pass = pass + 1,
            "id_correction topology refinement pass complete"
        );
    }
}

fn fit_anchor_homography_for_local_stage(
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
) -> Option<nalgebra::Matrix3<f64>> {
    let mut trusted_by_id: BTreeMap<usize, usize> = BTreeMap::new();
    for (i, m) in markers.iter().enumerate() {
        if !matches!(trust[i], Trust::AnchorStrong | Trust::AnchorWeak)
            || !marker_center_is_finite(m)
        {
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
    if trusted_by_id.len() < 4 {
        return None;
    }
    let mut src = Vec::<[f64; 2]>::with_capacity(trusted_by_id.len());
    let mut dst = Vec::<[f64; 2]>::with_capacity(trusted_by_id.len());
    for (&id, &idx) in &trusted_by_id {
        let Some(bxy) = board_index.id_to_xy.get(&id) else {
            continue;
        };
        src.push([f64::from(bxy[0]), f64::from(bxy[1])]);
        dst.push(markers[idx].center);
    }
    let cfg = RansacHomographyConfig {
        max_iters: 1200,
        inlier_threshold: config.h_reproj_gate_px,
        min_inliers: config.homography_min_inliers.min(src.len()).max(4),
        seed: HOMOGRAPHY_FALLBACK_SEED,
    };
    fit_homography_ransac(&src, &dst, &cfg).ok().map(|r| r.h)
}

fn run_homography_fallback(
    markers: &mut [DetectedMarker],
    trust: &mut [Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
    stats: &mut IdCorrectionStats,
) {
    if !config.homography_fallback_enable {
        return;
    }

    let n_trusted = trust.iter().filter(|&&t| t.is_trusted()).count();
    if n_trusted < config.homography_min_trusted {
        tracing::debug!(
            n_trusted,
            min_required = config.homography_min_trusted,
            "id_correction homography fallback skipped: insufficient trusted markers",
        );
        return;
    }

    // Keep one trusted marker per ID (highest confidence), deterministic by ID.
    let mut trusted_by_id: BTreeMap<usize, usize> = BTreeMap::new();
    for (i, m) in markers.iter().enumerate() {
        if !seed_allowed_for_homography(trust[i]) || !marker_center_is_finite(m) {
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
    if trusted_by_id.len() < 4 {
        tracing::debug!(
            n_unique_ids = trusted_by_id.len(),
            "id_correction homography fallback skipped: too few unique trusted IDs",
        );
        return;
    }

    let mut src_board_mm = Vec::<[f64; 2]>::with_capacity(trusted_by_id.len());
    let mut dst_image_px = Vec::<[f64; 2]>::with_capacity(trusted_by_id.len());
    for (&id, &idx) in &trusted_by_id {
        let Some(bxy) = board_index.id_to_xy.get(&id) else {
            continue;
        };
        src_board_mm.push([f64::from(bxy[0]), f64::from(bxy[1])]);
        dst_image_px.push(markers[idx].center);
    }

    let ransac_cfg = RansacHomographyConfig {
        max_iters: 1200,
        inlier_threshold: config.h_reproj_gate_px,
        min_inliers: config.homography_min_inliers,
        seed: HOMOGRAPHY_FALLBACK_SEED,
    };
    let h_result = match fit_homography_ransac(&src_board_mm, &dst_image_px, &ransac_cfg) {
        Ok(r) => r,
        Err(err) => {
            tracing::debug!(
                n_corr = src_board_mm.len(),
                "id_correction homography fallback fit failed: {}",
                err
            );
            return;
        }
    };
    let Some(h_inv) = h_result.h.try_inverse() else {
        tracing::debug!("id_correction homography fallback skipped: non-invertible H");
        return;
    };

    let mut trusted_conf_by_id = HashMap::<usize, f32>::new();
    for (&id, &idx) in &trusted_by_id {
        trusted_conf_by_id.insert(id, markers[idx].confidence);
    }

    let mut assignments = Vec::<HomographyAssignment>::new();
    let top_k = 19usize;
    for (i, m) in markers.iter().enumerate() {
        let eligible = !matches!(trust[i], Trust::AnchorStrong | Trust::AnchorWeak);
        if !eligible || !marker_center_is_finite(m) {
            continue;
        }
        if m.id.is_some() && is_soft_locked_assignment(m, config.soft_lock_exact_decode) {
            continue;
        }

        let board_hint = homography_project(&h_inv, m.center[0], m.center[1]);
        if !(board_hint[0].is_finite() && board_hint[1].is_finite()) {
            continue;
        }

        let mut best: Option<(usize, f64)> = None;
        for (candidate_id, _) in board_index.nearest_k_ids(board_hint, top_k) {
            if let Some(&trusted_conf) = trusted_conf_by_id.get(&candidate_id) {
                if trusted_conf >= m.confidence {
                    continue;
                }
            }
            if !candidate_passes_local_consistency_gate(
                i,
                candidate_id,
                markers,
                board_index,
                config,
                outer_radii_px,
            ) {
                continue;
            }

            let Some(board_xy) = board_index.id_to_xy.get(&candidate_id) else {
                continue;
            };
            let proj =
                homography_project(&h_result.h, f64::from(board_xy[0]), f64::from(board_xy[1]));
            if !(proj[0].is_finite() && proj[1].is_finite()) {
                continue;
            }
            let err = dist2(proj, m.center).sqrt();
            match best {
                Some((best_id, best_err)) => {
                    if err < best_err || (err == best_err && candidate_id < best_id) {
                        best = Some((candidate_id, err));
                    }
                }
                None => {
                    best = Some((candidate_id, err));
                }
            }
        }

        if let Some((id, reproj_err_px)) = best {
            if reproj_err_px <= config.h_reproj_gate_px {
                let current_err = m.id.and_then(|cur_id| {
                    board_index.id_to_xy.get(&cur_id).map(|xy| {
                        let proj =
                            homography_project(&h_result.h, f64::from(xy[0]), f64::from(xy[1]));
                        dist2(proj, m.center).sqrt()
                    })
                });
                let should_apply = match m.id {
                    None => true,
                    Some(cur_id) if cur_id == id => false,
                    Some(_) => current_err.is_none_or(|cur| reproj_err_px + 1.0 < cur),
                };
                if !should_apply {
                    continue;
                }
                assignments.push(HomographyAssignment {
                    marker_index: i,
                    id,
                    reproj_err_px,
                });
            }
        }
    }

    assignments.sort_by(|a, b| {
        a.reproj_err_px
            .total_cmp(&b.reproj_err_px)
            .then_with(|| a.marker_index.cmp(&b.marker_index))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut claimed_ids = trusted_by_id.keys().copied().collect::<HashSet<_>>();
    let mut seeded = 0usize;
    for a in assignments {
        if claimed_ids.contains(&a.id) {
            continue;
        }
        let i = a.marker_index;
        if matches!(trust[i], Trust::AnchorStrong | Trust::AnchorWeak) {
            continue;
        }
        if config_soft_lock_blocks_override(&markers[i], config.soft_lock_exact_decode, a.id) {
            continue;
        }
        claimed_ids.insert(a.id);
        if apply_id_assignment(&mut markers[i], a.id, stats, RecoverySource::Homography) {
            trust[i] = Trust::RecoveredHomography;
            seeded += 1;
        }
    }

    tracing::debug!(
        n_unique_trusted = trusted_by_id.len(),
        n_inliers = h_result.n_inliers,
        n_seeded = seeded,
        gate_px = config.h_reproj_gate_px,
        top_k,
        "id_correction homography fallback summary",
    );
}

fn diagnose_unverified_reasons(
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
    final_outer_mul: f64,
    stats: &mut IdCorrectionStats,
) {
    let tolerance_mm = board_index.pitch_mm * 0.6;
    for i in 0..markers.len() {
        if trust[i].is_trusted() || !marker_center_is_finite(&markers[i]) {
            continue;
        }
        let neighbors = gather_trusted_neighbors_local_scale(
            i,
            markers,
            trust,
            board_index,
            outer_radii_px,
            final_outer_mul,
        );
        if neighbors.is_empty() {
            stats.n_unverified_no_neighbors += 1;
            continue;
        }
        let effective_min_votes = if markers[i].id.is_none() {
            config.min_votes_recover
        } else {
            config.min_votes
        };
        let out = vote_for_candidate(
            markers[i].center,
            outer_radii_px[i],
            &neighbors,
            board_index,
            tolerance_mm,
            effective_min_votes,
            config.min_vote_weight_frac,
        );
        match out {
            VoteOutcome::NoVotes | VoteOutcome::InsufficientVotes { .. } => {
                stats.n_unverified_no_votes += 1;
            }
            VoteOutcome::GateRejected { .. } => {
                stats.n_unverified_gate_rejects += 1;
            }
            VoteOutcome::Candidate { .. } => {}
        }
    }
}

fn count_inconsistent_remaining(
    markers: &[DetectedMarker],
    trust: &[Trust],
    board_index: &BoardIndex,
    config: &IdCorrectionConfig,
    outer_radii_px: &[f64],
) -> usize {
    let mut n = 0usize;
    for i in 0..markers.len() {
        let Some(id) = markers[i].id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id) {
            n += 1;
            continue;
        }
        let evidence =
            consistency_evidence_for_id(i, id, markers, trust, board_index, config, outer_radii_px);
        let is_soft_locked = is_soft_locked_assignment(&markers[i], config.soft_lock_exact_decode);
        if should_clear_by_consistency(evidence, is_soft_locked, config) {
            n += 1;
        }
    }
    n
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Verify and correct marker IDs using the board's hex neighborhood structure.
///
/// Mutates `markers` in-place. Returns statistics about the corrections made.
pub(crate) fn verify_and_correct_ids(
    markers: &mut Vec<DetectedMarker>,
    board: &BoardLayout,
    config: &IdCorrectionConfig,
) -> IdCorrectionStats {
    let mut stats = IdCorrectionStats::default();
    if markers.is_empty() {
        return stats;
    }

    // 1) Build index + diagnostics.
    let board_index = BoardIndex::build(board);
    let outer_muls = build_outer_mul_schedule(config);
    let final_outer_mul = outer_muls.last().copied().unwrap_or(3.2);
    let outer_radii_px = compute_outer_radii_px(markers);

    // 2) Bootstrap trust anchors.
    let mut trust = vec![Trust::Untrusted; markers.len()];
    let mut n_seeds = 0usize;
    for i in 0..markers.len() {
        let Some(id) = markers[i].id else {
            continue;
        };
        if !board_index.id_to_xy.contains_key(&id) {
            continue;
        }
        let exact = is_exact_decode(&markers[i]);
        let decode_conf = markers[i]
            .decode
            .as_ref()
            .map(|d| d.decode_confidence)
            .unwrap_or(markers[i].confidence);
        if exact {
            trust[i] = Trust::AnchorStrong;
            n_seeds += 1;
        } else if decode_conf >= config.seed_min_decode_confidence {
            trust[i] = Trust::AnchorWeak;
            n_seeds += 1;
        }
    }
    if n_seeds < 2 {
        tracing::debug!(n_seeds, "id_correction: too few seeds, skipping");
        return stats;
    }

    tracing::debug!(
        n_seeds,
        n_markers = markers.len(),
        outer_muls = ?outer_muls,
        "id_correction: starting local-scale consistency-first correction",
    );

    // 3) Pre-recovery consistency scrub.
    let n_pre_cleared = scrub_inconsistent_ids(
        markers,
        &mut trust,
        &board_index,
        config,
        &outer_radii_px,
        &mut stats,
        ScrubStage::Pre,
    );
    tracing::debug!(
        n_pre_cleared,
        "id_correction pre-consistency scrub complete",
    );
    let anchor_h = fit_anchor_homography_for_local_stage(markers, &trust, &board_index, config);

    // 4) Local iterative recovery (staged multipliers).
    for &mul in &outer_muls {
        run_local_stage(
            markers,
            &mut trust,
            &mut stats,
            LocalStageSpec {
                board_index: &board_index,
                config,
                outer_radii_px: &outer_radii_px,
                anchor_h: anchor_h.as_ref(),
                outer_mul: mul,
                max_iters: config.max_iters,
                stage_name: "adaptive_local",
            },
        );
    }
    run_topology_refinement(
        markers,
        &mut trust,
        &board_index,
        config,
        &outer_radii_px,
        anchor_h.as_ref(),
        &mut stats,
    );

    // 5) Constrained homography fallback.
    if trust.iter().any(|t| !t.is_trusted()) {
        run_homography_fallback(
            markers,
            &mut trust,
            &board_index,
            config,
            &outer_radii_px,
            &mut stats,
        );
    }

    // 6) Post-recovery consistency sweep + short refill.
    let first_outer_mul = outer_muls.first().copied().unwrap_or(3.2);
    for _ in 0..2 {
        let _ = scrub_inconsistent_ids(
            markers,
            &mut trust,
            &board_index,
            config,
            &outer_radii_px,
            &mut stats,
            ScrubStage::Post,
        );
        run_local_stage(
            markers,
            &mut trust,
            &mut stats,
            LocalStageSpec {
                board_index: &board_index,
                config,
                outer_radii_px: &outer_radii_px,
                anchor_h: anchor_h.as_ref(),
                outer_mul: first_outer_mul,
                max_iters: 1,
                stage_name: "post_consistency_refill",
            },
        );
        run_topology_refinement(
            markers,
            &mut trust,
            &board_index,
            config,
            &outer_radii_px,
            anchor_h.as_ref(),
            &mut stats,
        );
    }

    diagnose_unverified_reasons(
        markers,
        &trust,
        &board_index,
        config,
        &outer_radii_px,
        final_outer_mul,
        &mut stats,
    );

    // 7) Final cleanup of unresolved.
    if config.remove_unverified {
        let mut i = 0usize;
        while i < markers.len() {
            if !trust[i].is_trusted() && markers[i].id.is_some() {
                markers.remove(i);
                trust.remove(i);
                stats.n_markers_removed += 1;
            } else {
                i += 1;
            }
        }
    } else {
        for i in 0..markers.len() {
            if !trust[i].is_trusted() && markers[i].id.is_some() {
                markers[i].id = None;
                stats.n_ids_cleared += 1;
            }
        }
    }

    let n_conflicts_cleared = resolve_id_conflicts(markers);
    stats.n_ids_cleared += n_conflicts_cleared;
    for i in 0..markers.len() {
        if markers[i].id.is_none() {
            trust[i] = Trust::Untrusted;
        }
    }

    stats.n_verified = markers
        .iter()
        .enumerate()
        .filter(|(i, m)| m.id.is_some() && trust[*i].is_trusted())
        .count();
    stats.n_inconsistent_remaining =
        count_inconsistent_remaining(markers, &trust, &board_index, config, &outer_radii_px);
    // Diagnostic field now reports robust post-correction one-hop image spacing.
    stats.pitch_px_estimated = estimate_adjacent_spacing_px(markers, &board_index);

    stats
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::detector::id_correction::index::{BoardIndex, HEX_NEIGHBORS};
    use crate::detector::id_correction::math::{affine_to_board, fit_local_affine, solve_3x3};
    use crate::marker::decode::DecodeMetrics;

    fn marker_with_id(
        id: usize,
        center: [f64; 2],
        conf: f32,
        dist: u8,
        margin: u8,
    ) -> DetectedMarker {
        DetectedMarker {
            id: Some(id),
            center,
            confidence: conf,
            decode: Some(DecodeMetrics {
                observed_word: 0,
                best_id: id,
                best_rotation: 0,
                best_dist: dist,
                margin,
                decode_confidence: conf,
            }),
            ellipse_outer: Some(Ellipse {
                cx: center[0],
                cy: center[1],
                a: 22.0,
                b: 22.0,
                angle: 0.0,
            }),
            ..DetectedMarker::default()
        }
    }

    fn marker_no_id(center: [f64; 2], conf: f32) -> DetectedMarker {
        DetectedMarker {
            id: None,
            center,
            confidence: conf,
            ellipse_outer: Some(Ellipse {
                cx: center[0],
                cy: center[1],
                a: 22.0,
                b: 22.0,
                angle: 0.0,
            }),
            ..DetectedMarker::default()
        }
    }

    #[test]
    fn hex_neighbor_offsets_cover_all_six_directions() {
        let set: std::collections::HashSet<(i16, i16)> = HEX_NEIGHBORS.iter().copied().collect();
        assert_eq!(set.len(), 6, "all six neighbor offsets must be distinct");
        for &(q, r) in &HEX_NEIGHBORS {
            assert!(set.contains(&(-q, -r)), "missing opposite of ({q}, {r})");
        }
    }

    #[test]
    fn solve_3x3_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [3.0, 7.0, 2.0];
        let x = solve_3x3(&a, &b).expect("must solve identity");
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn solve_3x3_rejects_singular() {
        let a = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
        let b = [1.0, 2.0, 0.0];
        assert!(solve_3x3(&a, &b).is_none());
    }

    #[test]
    fn fit_affine_roundtrip() {
        let board_pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let image_pts = board_pts.map(|[x, y]| [2.0 * x + 100.0, 2.0 * y + 200.0]);
        let aff = fit_local_affine(&board_pts, &image_pts).expect("affine must fit");
        let query_board = [0.5, 0.5];
        let expected_img = [101.0, 201.0];
        let img_x = aff[0][0] * query_board[0] + aff[0][1] * query_board[1] + aff[0][2];
        let img_y = aff[1][0] * query_board[0] + aff[1][1] * query_board[1] + aff[1][2];
        assert!((img_x - expected_img[0]).abs() < 1e-8);
        assert!((img_y - expected_img[1]).abs() < 1e-8);
        let recovered = affine_to_board(&aff, expected_img).expect("must invert");
        assert!((recovered[0] - query_board[0]).abs() < 1e-8);
        assert!((recovered[1] - query_board[1]).abs() < 1e-8);
    }

    #[test]
    fn pre_consistency_scrub_clears_structural_outlier_ids() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (&center_id, neighbors) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| nbrs.len() >= 3)
            .expect("board must have marker with >=3 neighbors");
        let wrong_id = board_index
            .id_to_xy
            .keys()
            .copied()
            .find(|id| *id != center_id && !neighbors.contains(id))
            .expect("must find non-neighbor wrong id");

        let scale = 4.0f64;
        let mut markers = Vec::<DetectedMarker>::new();
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            markers.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
            ));
        }
        let cxy = board_index.id_to_xy[&center_id];
        markers.push(marker_with_id(
            wrong_id,
            [f64::from(cxy[0]) * scale, f64::from(cxy[1]) * scale],
            0.6,
            1,
            1,
        ));

        let cfg = IdCorrectionConfig {
            homography_fallback_enable: false,
            max_iters: 1,
            ..IdCorrectionConfig::default()
        };
        let stats = verify_and_correct_ids(&mut markers, &board, &cfg);
        assert!(stats.n_ids_cleared_inconsistent_pre >= 1);
    }

    #[test]
    fn soft_lock_exact_decode_only_clears_on_strict_contradiction() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let id_a = 0usize;
        let id_b = board_index.board_neighbors[&id_a][0];
        let id_far = board_index
            .id_to_xy
            .keys()
            .copied()
            .find(|id| *id != id_a && *id != id_b && !board_index.are_neighbors(id_a, *id))
            .expect("must find far id");

        let xa = board_index.id_to_xy[&id_a];
        let xb = board_index.id_to_xy[&id_b];
        let xf = board_index.id_to_xy[&id_far];
        let scale = 4.0f64;

        let mut markers = vec![
            marker_with_id(
                id_a,
                [f64::from(xa[0]) * scale, f64::from(xa[1]) * scale],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
            ),
            marker_with_id(
                id_b,
                [f64::from(xb[0]) * scale, f64::from(xb[1]) * scale],
                0.9,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
            ),
            marker_with_id(
                id_far,
                [f64::from(xf[0]) * scale, f64::from(xf[1]) * scale],
                0.9,
                1,
                1,
            ),
        ];

        let cfg = IdCorrectionConfig {
            homography_fallback_enable: false,
            max_iters: 1,
            ..IdCorrectionConfig::default()
        };
        let _ = verify_and_correct_ids(&mut markers, &board, &cfg);

        // Exact decode anchor should not be overridden in mild contradiction.
        assert!(markers[0].id.is_some());
    }

    #[test]
    fn two_local_contradictions_do_not_keep_wrong_recovered_id() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let id_a = 120usize;
        let id_b = 107usize;
        let wrong_id = 161usize; // not a neighbor of either id_a or id_b.

        let mut markers = vec![
            marker_with_id(id_a, [129.4, 454.7], 0.9, 0, CODEBOOK_MIN_CYCLIC_DIST as u8),
            marker_with_id(id_b, [158.9, 505.4], 0.9, 0, CODEBOOK_MIN_CYCLIC_DIST as u8),
            marker_with_id(wrong_id, [101.2, 504.2], 0.5, 1, 1),
        ];
        // Keep local radii valid.
        for m in &mut markers {
            m.ellipse_outer = Some(Ellipse {
                cx: m.center[0],
                cy: m.center[1],
                a: 22.0,
                b: 22.0,
                angle: 0.0,
            });
        }

        assert!(!board_index.are_neighbors(wrong_id, id_a));
        assert!(!board_index.are_neighbors(wrong_id, id_b));

        let cfg = IdCorrectionConfig {
            max_iters: 2,
            homography_fallback_enable: false,
            ..IdCorrectionConfig::default()
        };
        let _stats = verify_and_correct_ids(&mut markers, &board, &cfg);
        assert_ne!(markers[2].id, Some(wrong_id));
    }

    #[test]
    fn local_recovery_after_scrub_recovers_hole() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (&center_id, neighbors) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| nbrs.len() >= 3)
            .expect("board must have marker with >=3 neighbors");

        let scale = 4.0f64;
        let mut markers = Vec::<DetectedMarker>::new();
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            markers.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
            ));
        }
        let cxy = board_index.id_to_xy[&center_id];
        markers.push(marker_no_id(
            [f64::from(cxy[0]) * scale, f64::from(cxy[1]) * scale],
            0.6,
        ));

        let cfg = IdCorrectionConfig {
            homography_fallback_enable: false,
            auto_search_radius_outer_muls: vec![2.5, 4.0],
            max_iters: 3,
            ..IdCorrectionConfig::default()
        };
        let stats = verify_and_correct_ids(&mut markers, &board, &cfg);
        assert!(stats.n_recovered_local >= 1);
        assert!(markers.iter().any(|m| m.id == Some(center_id)));
    }

    #[test]
    fn homography_fallback_uses_topk_candidates_and_consistency_gate() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let seed_ids = [0usize, 1, 2, 14, 15, 16];

        let mut markers = Vec::<DetectedMarker>::new();
        for &id in &seed_ids {
            let bxy = board_index.id_to_xy[&id];
            markers.push(marker_with_id(
                id,
                [
                    f64::from(bxy[0]) * 3.8 + 200.0,
                    f64::from(bxy[1]) * 3.6 + 120.0,
                ],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
            ));
        }
        // Place unresolved marker near an already occupied trusted ID so
        // top-k expansion is required to consider alternates.
        let occupied_id = seed_ids[0];
        let txy = board_index.id_to_xy[&occupied_id];
        markers.push(marker_no_id(
            [
                f64::from(txy[0]) * 3.8 + 201.5,
                f64::from(txy[1]) * 3.6 + 120.0,
            ],
            0.5,
        ));

        let cfg = IdCorrectionConfig {
            auto_search_radius_outer_muls: vec![2.5],
            homography_fallback_enable: true,
            homography_min_trusted: 4,
            homography_min_inliers: 4,
            h_reproj_gate_px: 15.0,
            max_iters: 1,
            ..IdCorrectionConfig::default()
        };
        let mut result = markers.clone();
        let stats = verify_and_correct_ids(&mut result, &board, &cfg);

        let got = result.last().and_then(|m| m.id);
        assert!(
            got != Some(occupied_id),
            "fallback must not assign an ID already occupied by stronger trusted marker",
        );
        let _ = stats;
    }

    #[test]
    fn determinism_local_scale_pipeline() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (&center_id, neighbors) = board_index
            .board_neighbors
            .iter()
            .find(|(_, nbrs)| nbrs.len() >= 3)
            .expect("board must have marker with >=3 neighbors");
        let scale = 4.0f64;
        let mut base = Vec::<DetectedMarker>::new();
        for &nid in &neighbors[..3] {
            let xy = board_index.id_to_xy[&nid];
            base.push(marker_with_id(
                nid,
                [f64::from(xy[0]) * scale, f64::from(xy[1]) * scale],
                0.95,
                0,
                CODEBOOK_MIN_CYCLIC_DIST as u8,
            ));
        }
        let cxy = board_index.id_to_xy[&center_id];
        base.push(marker_no_id(
            [f64::from(cxy[0]) * scale, f64::from(cxy[1]) * scale],
            0.6,
        ));
        let cfg = IdCorrectionConfig {
            homography_fallback_enable: false,
            auto_search_radius_outer_muls: vec![2.4, 2.9, 3.5],
            max_iters: 3,
            ..IdCorrectionConfig::default()
        };
        let mut run_a = base.clone();
        let mut run_b = base.clone();
        let stats_a = verify_and_correct_ids(&mut run_a, &board, &cfg);
        let stats_b = verify_and_correct_ids(&mut run_b, &board, &cfg);
        assert_eq!(
            run_a.iter().map(|m| m.id).collect::<Vec<_>>(),
            run_b.iter().map(|m| m.id).collect::<Vec<_>>()
        );
        assert_eq!(stats_a.n_ids_recovered, stats_b.n_ids_recovered);
        assert_eq!(stats_a.n_ids_cleared, stats_b.n_ids_cleared);
    }

    #[test]
    fn pitch_diagnostic_reports_adjacent_spacing_px() {
        let board = BoardLayout::default();
        let board_index = BoardIndex::build(&board);
        let (id0, id1) = board_index
            .board_neighbors
            .iter()
            .find_map(|(&id, nbrs)| nbrs.first().copied().map(|n| (id, n)))
            .expect("board must have adjacent pair");

        let mut markers = vec![
            marker_with_id(id0, [100.0, 100.0], 0.9, 0, CODEBOOK_MIN_CYCLIC_DIST as u8),
            marker_with_id(id1, [150.0, 100.0], 0.9, 0, CODEBOOK_MIN_CYCLIC_DIST as u8),
        ];
        for m in &mut markers {
            m.ellipse_outer = Some(Ellipse {
                cx: m.center[0],
                cy: m.center[1],
                a: 22.0,
                b: 22.0,
                angle: 0.0,
            });
        }

        let cfg = IdCorrectionConfig {
            max_iters: 0,
            homography_fallback_enable: false,
            ..IdCorrectionConfig::default()
        };
        let stats = verify_and_correct_ids(&mut markers, &board, &cfg);
        let pitch = stats.pitch_px_estimated.expect("pitch diagnostic");
        assert!(
            (pitch - 50.0).abs() < 1e-6,
            "pitch diagnostic must be image spacing in px"
        );
    }
}
