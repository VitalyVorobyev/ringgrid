use std::cmp::Ordering;

use crate::board_layout::BoardLayout;
use crate::detector::config::IdCorrectionConfig;
use crate::detector::marker_build::DetectedMarker;
use crate::marker::codebook::CODEBOOK_MIN_CYCLIC_DIST;

use super::index::BoardIndex;
use super::types::{IdCorrectionStats, RecoverySource, ScrubStage, Trust};

pub(super) struct IdCorrectionWorkspace<'a> {
    pub(super) markers: &'a mut Vec<DetectedMarker>,
    pub(super) board_index: BoardIndex,
    pub(super) outer_radii_px: Vec<f64>,
    pub(super) outer_muls: Vec<f64>,
    pub(super) trust: Vec<Trust>,
    pub(super) stats: IdCorrectionStats,
    pub(super) config: &'a IdCorrectionConfig,
    pub(super) anchor_h: Option<nalgebra::Matrix3<f64>>,
}

impl<'a> IdCorrectionWorkspace<'a> {
    pub(super) fn new(
        markers: &'a mut Vec<DetectedMarker>,
        board: &BoardLayout,
        config: &'a IdCorrectionConfig,
    ) -> Self {
        let board_index = BoardIndex::build(board);
        let outer_radii_px = compute_outer_radii_px(markers);
        let outer_muls = build_outer_mul_schedule(config);
        let trust = vec![Trust::Untrusted; markers.len()];
        Self {
            markers,
            board_index,
            outer_radii_px,
            outer_muls,
            trust,
            stats: IdCorrectionStats::default(),
            config,
            anchor_h: None,
        }
    }

    #[inline]
    pub(super) fn first_outer_mul(&self) -> f64 {
        self.outer_muls.first().copied().unwrap_or(3.2)
    }

    #[inline]
    pub(super) fn final_outer_mul(&self) -> f64 {
        self.outer_muls.last().copied().unwrap_or(3.2)
    }
}

#[inline]
pub(super) fn marker_center_is_finite(marker: &DetectedMarker) -> bool {
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
pub(super) fn is_exact_decode(marker: &DetectedMarker) -> bool {
    marker
        .decode
        .as_ref()
        .is_some_and(|d| d.best_dist == 0 && usize::from(d.margin) >= CODEBOOK_MIN_CYCLIC_DIST)
}

#[inline]
pub(super) fn is_soft_locked_assignment(marker: &DetectedMarker, soft_lock_enable: bool) -> bool {
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

pub(super) fn should_block_by_trusted_confidence(
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

pub(super) fn apply_id_assignment(
    marker: &mut DetectedMarker,
    new_id: usize,
    stats: &mut IdCorrectionStats,
    source: RecoverySource,
) -> bool {
    match marker.id {
        Some(old_id) if old_id == new_id => false,
        Some(_) => {
            marker.id = Some(new_id);
            stats.n_ids_corrected += 1;
            true
        }
        None => {
            marker.id = Some(new_id);
            stats.n_ids_recovered += 1;
            match source {
                RecoverySource::Local => stats.n_recovered_local += 1,
                RecoverySource::Homography => {
                    stats.n_recovered_homography += 1;
                    stats.n_homography_seeded += 1;
                }
            }
            true
        }
    }
}

pub(super) fn clear_marker_id(
    marker_index: usize,
    markers: &mut [DetectedMarker],
    trust: &mut [Trust],
    stats: &mut IdCorrectionStats,
    soft_lock_enable: bool,
    stage: ScrubStage,
) -> bool {
    if markers[marker_index].id.is_none() {
        trust[marker_index] = Trust::Untrusted;
        return false;
    }
    let was_soft_locked = is_soft_locked_assignment(&markers[marker_index], soft_lock_enable);
    markers[marker_index].id = None;
    trust[marker_index] = Trust::Untrusted;
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
