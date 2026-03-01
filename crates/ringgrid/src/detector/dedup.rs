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

// ---------------------------------------------------------------------------
// Multi-scale merge
// ---------------------------------------------------------------------------

fn outer_radius(m: &DetectedMarker) -> f64 {
    m.ellipse_outer
        .as_ref()
        .map(|e| e.mean_axis())
        .unwrap_or(0.0)
}

/// Compute a size-consistency score in `[0, 1]` for each marker.
///
/// The score measures how well the marker's outer radius matches the median
/// outer radius of its `k` nearest spatial neighbors. A score of 1.0 means
/// the radius equals the neighborhood median; lower values indicate
/// outliers (possible wrong-tier detections).
fn size_consistency_scores(markers: &[DetectedMarker], k_neighbors: usize) -> Vec<f64> {
    let n = markers.len();
    let k = k_neighbors.min(n.saturating_sub(1));
    let radii: Vec<f64> = markers.iter().map(outer_radius).collect();
    let mut scores = vec![1.0f64; n];

    if k == 0 || n <= 1 {
        return scores;
    }

    for i in 0..n {
        let ci = markers[i].center;

        // k nearest neighbors by squared Euclidean distance.
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let cj = markers[j].center;
                let dx = ci[0] - cj[0];
                let dy = ci[1] - cj[1];
                (dx * dx + dy * dy, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let neighbor_radii: Vec<f64> = dists[..k.min(dists.len())]
            .iter()
            .map(|(_, j)| radii[*j])
            .collect();

        if neighbor_radii.len() < 3 {
            // Too few neighbors for a reliable median; leave score neutral.
            continue;
        }

        let mut sorted_nr = neighbor_radii.clone();
        sorted_nr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_nr.len() / 2;
        let median_r = if sorted_nr.len().is_multiple_of(2) {
            (sorted_nr[mid - 1] + sorted_nr[mid]) * 0.5
        } else {
            sorted_nr[mid]
        };

        if median_r <= 0.0 {
            continue;
        }

        let r_i = radii[i];
        scores[i] = (1.0 - (r_i - median_r).abs() / median_r).max(0.0);
    }

    scores
}

/// Merge markers from multiple scale tiers using size-consistency-aware dedup.
///
/// For each pair of markers within `dedup_radius` pixels of each other
/// (spatial duplicates from overlapping tiers), keeps the one whose outer
/// radius is most consistent with the median outer radius of its `k_neighbors`
/// nearest neighbors. Ties are broken by decode confidence.
///
/// After spatial dedup, per-ID dedup keeps the highest-scoring match per
/// decoded ID.
///
/// `k_neighbors = 6` matches the hex-lattice valence and mirrors the
/// convention in [`InnerAsOuterRecoveryConfig`](super::config::InnerAsOuterRecoveryConfig).
pub(crate) fn merge_multiscale_markers(
    all_markers: Vec<DetectedMarker>,
    dedup_radius: f64,
    k_neighbors: usize,
) -> Vec<DetectedMarker> {
    if all_markers.is_empty() {
        return all_markers;
    }

    let n = all_markers.len();
    let scores = size_consistency_scores(&all_markers, k_neighbors);

    // Sort indices: highest size-consistency score first; confidence as tiebreak.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                all_markers[j]
                    .confidence
                    .partial_cmp(&all_markers[i].confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // NMS: iterate in score order; suppress lower-ranked markers within radius.
    let r2 = dedup_radius * dedup_radius;
    let mut keep = vec![true; n];

    for pos in 0..order.len() {
        let i = order[pos];
        if !keep[i] {
            continue;
        }
        let ci = all_markers[i].center;
        for &j in &order[pos + 1..] {
            if !keep[j] {
                continue;
            }
            let cj = all_markers[j].center;
            let dx = ci[0] - cj[0];
            let dy = ci[1] - cj[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
            }
        }
    }

    let surviving: Vec<DetectedMarker> = all_markers
        .into_iter()
        .enumerate()
        .filter_map(|(i, m)| keep[i].then_some(m))
        .collect();

    // Per-ID dedup: keep the best match per decoded ID (by confidence).
    dedup_by_id_core(surviving)
}
