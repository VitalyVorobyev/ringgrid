use std::collections::{HashMap, HashSet};

use crate::detector::MarkerRecord;

fn sort_by_confidence(markers: Vec<MarkerRecord>) -> Vec<MarkerRecord> {
    let mut markers = markers;
    markers.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    markers
}

fn dedup_by_proximity(markers: Vec<MarkerRecord>, radius: f64) -> Vec<MarkerRecord> {
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

fn dedup_by_id_core(markers: Vec<MarkerRecord>) -> Vec<MarkerRecord> {
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
pub fn dedup_markers(markers: Vec<MarkerRecord>, radius: f64) -> Vec<MarkerRecord> {
    let markers = sort_by_confidence(markers);
    dedup_by_proximity(markers, radius)
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
pub fn dedup_by_id(markers: &mut Vec<MarkerRecord>) {
    let input = std::mem::take(markers);
    *markers = dedup_by_id_core(input);
}

// ---------------------------------------------------------------------------
// Multi-scale merge
// ---------------------------------------------------------------------------

fn outer_radius(m: &MarkerRecord) -> f64 {
    m.ellipse_outer
        .as_ref()
        .map(|e| e.mean_axis())
        .unwrap_or(0.0)
}

/// Neighbor count for the multiscale-merge size-consistency score: the
/// hex-lattice valence, so the neighborhood median reflects the immediate
/// ring of surrounding markers.
pub(crate) const MERGE_SIZE_K_NEIGHBORS: usize = 6;

/// Minimum contributing neighbors for a reliable median; below this the
/// score stays neutral.
const MIN_NEIGHBORS_FOR_SCORE: usize = 3;

/// Compute a size-consistency score in `[0, 1]` for each marker.
///
/// The score measures how well the marker's outer radius matches the median
/// outer radius of its `k` nearest spatial neighbors (the shared statistic
/// from [`super::neighbors`]). A score of 1.0 means the radius equals the
/// neighborhood median; lower values indicate outliers (possible wrong-tier
/// detections).
fn size_consistency_scores(markers: &[MarkerRecord], k_neighbors: usize) -> Vec<f64> {
    let n = markers.len();
    let k = k_neighbors.min(n.saturating_sub(1));
    let mut scores = vec![1.0f64; n];

    if k == 0 || n <= 1 {
        return scores;
    }

    for (i, marker) in markers.iter().enumerate() {
        let Some(median_r) = super::median_neighbor_outer_radius_px(
            marker.center,
            markers,
            k,
            Some(i),
            MIN_NEIGHBORS_FOR_SCORE,
        ) else {
            // Too few neighbors for a reliable median; leave score neutral.
            continue;
        };
        if median_r <= 0.0 {
            continue;
        }

        let r_i = outer_radius(marker);
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
    all_markers: Vec<MarkerRecord>,
    dedup_radius: f64,
    k_neighbors: usize,
) -> Vec<MarkerRecord> {
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

    let surviving: Vec<MarkerRecord> = all_markers
        .into_iter()
        .enumerate()
        .filter_map(|(i, m)| keep[i].then_some(m))
        .collect();

    // Per-ID dedup: keep the best match per decoded ID (by confidence).
    dedup_by_id_core(surviving)
}
