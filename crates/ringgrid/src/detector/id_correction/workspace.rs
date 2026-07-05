use std::cmp::Ordering;

use crate::detector::config::IdCorrectionConfig;
use crate::detector::marker_build::MarkerRecord;
use crate::target::TargetLayout;

use super::index::BoardIndex;
use super::types::{IdCorrectionStats, RecoverySource, ScrubStage, Trust};

pub(super) struct IdCorrectionWorkspace<'a> {
    pub(super) markers: &'a mut Vec<MarkerRecord>,
    pub(super) board_index: BoardIndex,
    pub(super) outer_radii_px: Vec<f64>,
    pub(super) outer_muls: Vec<f64>,
    pub(super) trust: Vec<Trust>,
    pub(super) stats: IdCorrectionStats,
    pub(super) config: &'a IdCorrectionConfig,
    pub(super) anchor_h: Option<nalgebra::Matrix3<f64>>,
    pub(super) codebook_min_cyclic_dist: usize,
}

impl<'a> IdCorrectionWorkspace<'a> {
    pub(super) fn new(
        markers: &'a mut Vec<MarkerRecord>,
        target: &TargetLayout,
        config: &'a IdCorrectionConfig,
        codebook_min_cyclic_dist: usize,
    ) -> Self {
        let board_index = BoardIndex::build(target);
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
            codebook_min_cyclic_dist,
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
pub(super) fn marker_center_is_finite(marker: &MarkerRecord) -> bool {
    marker.center[0].is_finite() && marker.center[1].is_finite()
}

#[inline]
fn marker_outer_radius_px(marker: &MarkerRecord) -> Option<f64> {
    marker
        .ellipse_outer
        .as_ref()
        .map(|e| e.mean_axis())
        .filter(|r| r.is_finite() && *r > 0.0)
}

#[inline]
pub(super) fn is_exact_decode(marker: &MarkerRecord, codebook_min_cyclic_dist: usize) -> bool {
    marker
        .decode
        .as_ref()
        .is_some_and(|d| d.best_dist == 0 && usize::from(d.margin) >= codebook_min_cyclic_dist)
}

#[inline]
pub(super) fn is_soft_locked_assignment(
    marker: &MarkerRecord,
    soft_lock_enable: bool,
    codebook_min_cyclic_dist: usize,
) -> bool {
    if !soft_lock_enable {
        return false;
    }
    let Some(id) = marker.id else {
        return false;
    };
    marker.decode.as_ref().is_some_and(|d| {
        d.best_dist == 0 && usize::from(d.margin) >= codebook_min_cyclic_dist && d.best_id == id
    })
}

fn compute_outer_radii_px(markers: &[MarkerRecord]) -> Vec<f64> {
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
    markers: &[MarkerRecord],
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
    marker: &mut MarkerRecord,
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
    markers: &mut [MarkerRecord],
    trust: &mut [Trust],
    stats: &mut IdCorrectionStats,
    soft_lock_enable: bool,
    codebook_min_cyclic_dist: usize,
    stage: ScrubStage,
) -> bool {
    if markers[marker_index].id.is_none() {
        trust[marker_index] = Trust::Untrusted;
        return false;
    }
    let was_soft_locked = is_soft_locked_assignment(
        &markers[marker_index],
        soft_lock_enable,
        codebook_min_cyclic_dist,
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::marker::decode::DecodeMetrics;

    fn decode(best_id: usize, best_dist: u8, margin: u8) -> DecodeMetrics {
        DecodeMetrics {
            observed_word: 0,
            best_id,
            best_rotation: 0,
            best_dist,
            margin,
            decode_confidence: 1.0,
        }
    }

    fn marker_with_ellipse(a: f64, b: f64) -> MarkerRecord {
        MarkerRecord {
            ellipse_outer: Some(Ellipse {
                cx: 0.0,
                cy: 0.0,
                a,
                b,
                angle: 0.0,
            }),
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn marker_center_is_finite_detects_nan() {
        let mut m = MarkerRecord {
            center: [1.0, 2.0],
            ..MarkerRecord::default()
        };
        assert!(marker_center_is_finite(&m));
        m.center = [f64::NAN, 2.0];
        assert!(!marker_center_is_finite(&m));
        m.center = [1.0, f64::INFINITY];
        assert!(!marker_center_is_finite(&m));
    }

    #[test]
    fn is_exact_decode_requires_zero_dist_and_sufficient_margin() {
        let min_cyclic = 2usize;
        let exact = MarkerRecord {
            decode: Some(decode(7, 0, 2)),
            ..MarkerRecord::default()
        };
        assert!(is_exact_decode(&exact, min_cyclic));

        // dist > 0 is never exact.
        let noisy = MarkerRecord {
            decode: Some(decode(7, 1, 5)),
            ..MarkerRecord::default()
        };
        assert!(!is_exact_decode(&noisy, min_cyclic));

        // Margin below the codebook minimum cyclic distance fails.
        let low_margin = MarkerRecord {
            decode: Some(decode(7, 0, 1)),
            ..MarkerRecord::default()
        };
        assert!(!is_exact_decode(&low_margin, min_cyclic));

        // No decode → not exact.
        assert!(!is_exact_decode(&MarkerRecord::default(), min_cyclic));
    }

    #[test]
    fn is_soft_locked_requires_enable_and_matching_exact_decode() {
        let min_cyclic = 2usize;
        let locked = MarkerRecord {
            id: Some(7),
            decode: Some(decode(7, 0, 2)),
            ..MarkerRecord::default()
        };
        assert!(is_soft_locked_assignment(&locked, true, min_cyclic));
        // Disabled globally → never locked.
        assert!(!is_soft_locked_assignment(&locked, false, min_cyclic));

        // Decoded best_id must match the currently-assigned id.
        let mismatched = MarkerRecord {
            id: Some(9),
            decode: Some(decode(7, 0, 2)),
            ..MarkerRecord::default()
        };
        assert!(!is_soft_locked_assignment(&mismatched, true, min_cyclic));

        // No id → cannot be soft-locked.
        let no_id = MarkerRecord {
            id: None,
            decode: Some(decode(7, 0, 2)),
            ..MarkerRecord::default()
        };
        assert!(!is_soft_locked_assignment(&no_id, true, min_cyclic));
    }

    #[test]
    fn compute_outer_radii_uses_mean_axis_and_median_fallback() {
        // Two ellipses give radii 10 and 20; a marker without an ellipse gets
        // the median (15) of the valid radii.
        let markers = vec![
            marker_with_ellipse(10.0, 10.0),
            marker_with_ellipse(20.0, 20.0),
            MarkerRecord::default(),
        ];
        let radii = compute_outer_radii_px(&markers);
        assert_eq!(radii.len(), 3);
        assert!((radii[0] - 10.0).abs() < 1e-9);
        assert!((radii[1] - 20.0).abs() < 1e-9);
        assert!(
            (radii[2] - 15.0).abs() < 1e-9,
            "median fallback = {}",
            radii[2]
        );
    }

    #[test]
    fn compute_outer_radii_defaults_to_twenty_without_any_ellipse() {
        let markers = vec![MarkerRecord::default(), MarkerRecord::default()];
        let radii = compute_outer_radii_px(&markers);
        assert!(radii.iter().all(|&r| (r - 20.0).abs() < 1e-9));
    }

    #[test]
    fn build_outer_mul_schedule_filters_sorts_and_dedups() {
        let cfg = IdCorrectionConfig {
            auto_search_radius_outer_muls: vec![3.5, 2.4, 2.4, -1.0, f64::NAN, 2.9],
            ..IdCorrectionConfig::default()
        };
        let schedule = build_outer_mul_schedule(&cfg);
        assert_eq!(schedule, vec![2.4, 2.9, 3.5]);
    }

    #[test]
    fn build_outer_mul_schedule_falls_back_when_empty() {
        let cfg = IdCorrectionConfig {
            auto_search_radius_outer_muls: vec![-1.0, 0.0, f64::NAN],
            ..IdCorrectionConfig::default()
        };
        let schedule = build_outer_mul_schedule(&cfg);
        assert_eq!(schedule, vec![3.2]);
    }

    #[test]
    fn trusted_confidence_blocks_only_equal_or_higher_confidence() {
        let markers = vec![
            MarkerRecord {
                confidence: 0.5,
                ..MarkerRecord::default()
            },
            MarkerRecord {
                id: Some(5),
                confidence: 0.9,
                ..MarkerRecord::default()
            },
        ];
        // Trusted higher-confidence holder of id 5 blocks the override.
        let trust_hi = vec![Trust::Untrusted, Trust::AnchorStrong];
        assert!(should_block_by_trusted_confidence(
            0, 5, &markers, &trust_hi
        ));

        // Untrusted holder does not block.
        let trust_untrusted = vec![Trust::Untrusted, Trust::Untrusted];
        assert!(!should_block_by_trusted_confidence(
            0,
            5,
            &markers,
            &trust_untrusted
        ));

        // Lower-confidence trusted holder does not block.
        let markers_lo = vec![
            MarkerRecord {
                confidence: 0.5,
                ..MarkerRecord::default()
            },
            MarkerRecord {
                id: Some(5),
                confidence: 0.3,
                ..MarkerRecord::default()
            },
        ];
        assert!(!should_block_by_trusted_confidence(
            0,
            5,
            &markers_lo,
            &trust_hi
        ));
    }

    #[test]
    fn apply_id_assignment_is_a_state_machine() {
        // Same id → no-op, no stats.
        let mut m = MarkerRecord {
            id: Some(3),
            ..MarkerRecord::default()
        };
        let mut stats = IdCorrectionStats::default();
        assert!(!apply_id_assignment(
            &mut m,
            3,
            &mut stats,
            RecoverySource::Local
        ));
        assert_eq!(stats.n_ids_corrected, 0);
        assert_eq!(stats.n_ids_recovered, 0);

        // Different existing id → correction.
        assert!(apply_id_assignment(
            &mut m,
            4,
            &mut stats,
            RecoverySource::Local
        ));
        assert_eq!(m.id, Some(4));
        assert_eq!(stats.n_ids_corrected, 1);
        assert_eq!(stats.n_ids_recovered, 0);

        // None → recovery via Local.
        let mut none_marker = MarkerRecord::default();
        assert!(apply_id_assignment(
            &mut none_marker,
            8,
            &mut stats,
            RecoverySource::Local
        ));
        assert_eq!(none_marker.id, Some(8));
        assert_eq!(stats.n_ids_recovered, 1);
        assert_eq!(stats.n_recovered_local, 1);
        assert_eq!(stats.n_recovered_homography, 0);

        // None → recovery via Homography also bumps the homography-seeded count.
        let mut none_marker2 = MarkerRecord::default();
        assert!(apply_id_assignment(
            &mut none_marker2,
            9,
            &mut stats,
            RecoverySource::Homography
        ));
        assert_eq!(stats.n_ids_recovered, 2);
        assert_eq!(stats.n_recovered_homography, 1);
        assert_eq!(stats.n_homography_seeded, 1);
    }

    #[test]
    fn clear_marker_id_tracks_stage_and_soft_lock() {
        let min_cyclic = 2usize;
        // A soft-locked exact decode cleared in the Post stage.
        let mut markers = vec![MarkerRecord {
            id: Some(7),
            decode: Some(decode(7, 0, 2)),
            ..MarkerRecord::default()
        }];
        let mut trust = vec![Trust::AnchorStrong];
        let mut stats = IdCorrectionStats::default();
        let cleared = clear_marker_id(
            0,
            &mut markers,
            &mut trust,
            &mut stats,
            true,
            min_cyclic,
            ScrubStage::Post,
        );
        assert!(cleared);
        assert_eq!(markers[0].id, None);
        assert_eq!(trust[0], Trust::Untrusted);
        assert_eq!(stats.n_ids_cleared, 1);
        assert_eq!(stats.n_ids_cleared_inconsistent_post, 1);
        assert_eq!(stats.n_ids_cleared_inconsistent_pre, 0);
        assert_eq!(stats.n_soft_locked_cleared, 1);

        // Clearing an already-None marker is a no-op that only resets trust.
        let mut none_markers = vec![MarkerRecord::default()];
        let mut none_trust = vec![Trust::RecoveredLocal];
        let mut none_stats = IdCorrectionStats::default();
        let cleared = clear_marker_id(
            0,
            &mut none_markers,
            &mut none_trust,
            &mut none_stats,
            true,
            min_cyclic,
            ScrubStage::Pre,
        );
        assert!(!cleared);
        assert_eq!(none_trust[0], Trust::Untrusted);
        assert_eq!(none_stats.n_ids_cleared, 0);
    }
}
