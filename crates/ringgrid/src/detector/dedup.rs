use std::collections::{HashMap, HashSet};

use crate::DetectedMarker;

fn sort_by_confidence(markers: Vec<DetectedMarker>) -> Vec<DetectedMarker> {
    let mut markers = markers;
    markers.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    markers
}

fn dedup_by_proximity(markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
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
        .enumerate()
        .filter_map(|(index, marker)| keep[index].then_some(marker))
        .collect()
}

fn dedup_by_id_core(markers: Vec<DetectedMarker>) -> Vec<DetectedMarker> {
    let mut best_idx: HashMap<usize, usize> = HashMap::new();

    for (index, marker) in markers.iter().enumerate() {
        if let Some(id) = marker.id {
            match best_idx.get(&id) {
                Some(&prev) if markers[prev].confidence >= marker.confidence => {}
                _ => {
                    best_idx.insert(id, index);
                }
            }
        }
    }

    let keep_set: HashSet<usize> = best_idx.values().copied().collect();
    markers
        .into_iter()
        .enumerate()
        .filter_map(|(index, marker)| {
            let keep = marker.id.is_none() || keep_set.contains(&index);
            keep.then_some(marker)
        })
        .collect()
}

/// Remove duplicate detections: keep the highest-confidence marker within `radius`.
pub fn dedup_markers(markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    let markers = sort_by_confidence(markers);
    dedup_by_proximity(markers, radius)
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
pub fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    let input = std::mem::take(markers);
    *markers = dedup_by_id_core(input);
}
