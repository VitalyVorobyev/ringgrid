use std::collections::{HashMap, HashSet};

use crate::debug_dump as dbg;
use crate::DetectedMarker;

fn sort_by_confidence(
    markers: Vec<DetectedMarker>,
    mut cand_idx: Option<Vec<usize>>,
) -> (Vec<DetectedMarker>, Option<Vec<usize>>) {
    let mut order: Vec<usize> = (0..markers.len()).collect();
    order.sort_by(|&a, &b| {
        markers[b]
            .confidence
            .partial_cmp(&markers[a].confidence)
            .unwrap()
    });

    let sorted_markers = order.iter().map(|&i| markers[i].clone()).collect();
    let sorted_idx = cand_idx
        .as_mut()
        .map(|idx| order.iter().map(|&i| idx[i]).collect());

    (sorted_markers, sorted_idx)
}

fn dedup_by_proximity(
    markers: Vec<DetectedMarker>,
    cand_idx: Option<Vec<usize>>,
    radius: f64,
    collect_debug: bool,
) -> (
    Vec<DetectedMarker>,
    Option<Vec<usize>>,
    Vec<dbg::KeptByProximityDebugV1>,
) {
    let mut keep = vec![true; markers.len()];
    let r2 = radius * radius;
    let mut kept_by_proximity: Vec<dbg::KeptByProximityDebugV1> = Vec::new();

    for i in 0..markers.len() {
        if !keep[i] {
            continue;
        }

        let mut dropped_j: Vec<usize> = Vec::new();
        for j in (i + 1)..markers.len() {
            if !keep[j] {
                continue;
            }
            let dx = markers[i].center[0] - markers[j].center[0];
            let dy = markers[i].center[1] - markers[j].center[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
                dropped_j.push(j);
            }
        }

        if collect_debug && !dropped_j.is_empty() {
            if let Some(ref idx) = cand_idx {
                kept_by_proximity.push(dbg::KeptByProximityDebugV1 {
                    kept_cand_idx: idx[i],
                    dropped_cand_indices: dropped_j.into_iter().map(|j| idx[j]).collect(),
                    reasons: vec!["within_dedup_radius".to_string()],
                });
            }
        }
    }

    let mut markers_out: Vec<DetectedMarker> = Vec::new();
    let mut idx_out: Vec<usize> = Vec::new();
    for (i, m) in markers.into_iter().enumerate() {
        if keep[i] {
            markers_out.push(m);
            if let Some(ref idx) = cand_idx {
                idx_out.push(idx[i]);
            }
        }
    }

    (markers_out, cand_idx.map(|_| idx_out), kept_by_proximity)
}

fn dedup_by_id_core(
    markers: Vec<DetectedMarker>,
    cand_idx: Option<Vec<usize>>,
    collect_debug: bool,
) -> (
    Vec<DetectedMarker>,
    Option<Vec<usize>>,
    Vec<dbg::KeptByIdDebugV1>,
) {
    let mut best_idx: HashMap<usize, usize> = HashMap::new();

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            match best_idx.get(&id) {
                Some(&prev) if markers[prev].confidence >= m.confidence => {}
                _ => {
                    best_idx.insert(id, i);
                }
            }
        }
    }

    let keep_set: HashSet<usize> = best_idx.values().copied().collect();

    let mut kept_by_id: Vec<dbg::KeptByIdDebugV1> = Vec::new();
    if collect_debug {
        if let Some(ref idx) = cand_idx {
            for (&id, &kept_i) in best_idx.iter() {
                let mut dropped: Vec<usize> = Vec::new();
                for (i, m) in markers.iter().enumerate() {
                    if i == kept_i {
                        continue;
                    }
                    if m.id == Some(id) {
                        dropped.push(idx[i]);
                    }
                }
                if !dropped.is_empty() {
                    kept_by_id.push(dbg::KeptByIdDebugV1 {
                        id,
                        kept_cand_idx: idx[kept_i],
                        dropped_cand_indices: dropped,
                        reasons: vec!["lower_confidence".to_string()],
                    });
                }
            }
            kept_by_id.sort_by_key(|e| e.id);
        }
    }

    let mut markers_out: Vec<DetectedMarker> = Vec::new();
    let mut idx_out: Vec<usize> = Vec::new();

    for (i, m) in markers.into_iter().enumerate() {
        let keep_it = m.id.is_none() || keep_set.contains(&i);
        if keep_it {
            markers_out.push(m);
            if let Some(ref idx) = cand_idx {
                idx_out.push(idx[i]);
            }
        }
    }

    (markers_out, cand_idx.map(|_| idx_out), kept_by_id)
}

/// Remove duplicate detections: keep the highest-confidence marker within `radius`.
pub fn dedup_markers(markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    let (markers, _idx) = sort_by_confidence(markers, None);
    let (markers, _idx, _dbg) = dedup_by_proximity(markers, None, radius, false);
    markers
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
pub fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    let input = std::mem::take(markers);
    let (output, _idx, _dbg) = dedup_by_id_core(input, None, false);
    *markers = output;
}

pub fn dedup_with_debug(
    markers: Vec<DetectedMarker>,
    cand_idx: Vec<usize>,
    radius: f64,
) -> (Vec<DetectedMarker>, Vec<usize>, dbg::DedupDebugV1) {
    let (markers, cand_idx) = sort_by_confidence(markers, Some(cand_idx));
    let (markers, cand_idx, kept_by_proximity) =
        dedup_by_proximity(markers, cand_idx, radius, true);
    let (markers, cand_idx, kept_by_id) = dedup_by_id_core(markers, cand_idx, true);

    (
        markers,
        cand_idx.unwrap_or_default(),
        dbg::DedupDebugV1 {
            kept_by_proximity,
            kept_by_id,
            notes: Vec::new(),
        },
    )
}
