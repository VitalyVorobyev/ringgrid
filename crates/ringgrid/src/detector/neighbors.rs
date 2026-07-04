//! Neighborhood radius statistics shared by the size-consistency gates.
//!
//! Three stages gate on "how does this radius compare to its spatial
//! neighborhood": multiscale-merge dedup, completion seeding, and
//! inner-as-outer recovery. Their *policies* (score, window, warn threshold)
//! are deliberately stage-local, but the underlying statistic — the median
//! outer radius of the k nearest markers — has exactly one definition here.

use crate::detector::MarkerRecord;

/// Median outer radius (mean semi-axis, pixels) of the `k` markers nearest to
/// `center`.
///
/// - f64 throughout, proper median (middle-pair average for even counts).
/// - Markers without a fitted outer ellipse do not contribute.
/// - `exclude_idx` removes the queried marker itself, so its own radius can
///   never bias its neighborhood statistic.
/// - Returns `None` when fewer than `min_neighbors` contributing markers
///   exist (callers pick the sufficiency policy).
pub(crate) fn median_neighbor_outer_radius_px(
    center: [f64; 2],
    markers: &[MarkerRecord],
    k: usize,
    exclude_idx: Option<usize>,
    min_neighbors: usize,
) -> Option<f64> {
    let mut candidates: Vec<(f64, f64)> = Vec::new();
    for (j, m) in markers.iter().enumerate() {
        if Some(j) == exclude_idx {
            continue;
        }
        let Some(e) = m.ellipse_outer.as_ref() else {
            continue;
        };
        let dx = m.center[0] - center[0];
        let dy = m.center[1] - center[1];
        let d2 = dx * dx + dy * dy;
        if d2.is_finite() {
            candidates.push((d2, e.mean_axis()));
        }
    }
    // Stable sort keeps equidistant candidates in marker order (deterministic).
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates.truncate(k.max(1));
    if candidates.len() < min_neighbors.max(1) {
        return None;
    }

    let mut radii: Vec<f64> = candidates.into_iter().map(|(_, r)| r).collect();
    radii.sort_by(|a, b| a.total_cmp(b));
    let mid = radii.len() / 2;
    Some(if radii.len().is_multiple_of(2) {
        (radii[mid - 1] + radii[mid]) * 0.5
    } else {
        radii[mid]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::detector::MarkerRecord;

    fn marker_at(x: f64, y: f64, radius: f64) -> MarkerRecord {
        MarkerRecord {
            center: [x, y],
            ellipse_outer: Some(Ellipse {
                cx: x,
                cy: y,
                a: radius,
                b: radius,
                angle: 0.0,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn proper_median_averages_middle_pair() {
        let markers = vec![
            marker_at(0.0, 0.0, 99.0), // the queried marker (excluded)
            marker_at(1.0, 0.0, 10.0),
            marker_at(2.0, 0.0, 12.0),
            marker_at(3.0, 0.0, 14.0),
            marker_at(4.0, 0.0, 16.0),
        ];
        let med = median_neighbor_outer_radius_px([0.0, 0.0], &markers, 4, Some(0), 1)
            .expect("median exists");
        assert!((med - 13.0).abs() < 1e-12, "median {med}");
    }

    #[test]
    fn exclude_idx_removes_self_contamination() {
        // The queried marker sits at distance 0 with an anomalous radius; when
        // self-inclusion was possible it always entered the k-window first.
        let markers = vec![
            marker_at(0.0, 0.0, 5.0),
            marker_at(1.0, 0.0, 20.0),
            marker_at(2.0, 0.0, 20.0),
            marker_at(3.0, 0.0, 20.0),
        ];
        let med = median_neighbor_outer_radius_px([0.0, 0.0], &markers, 3, Some(0), 1)
            .expect("median exists");
        assert!((med - 20.0).abs() < 1e-12, "self must not bias, got {med}");
    }

    #[test]
    fn min_neighbors_policy_returns_none_when_sparse() {
        let markers = vec![marker_at(0.0, 0.0, 5.0), marker_at(1.0, 0.0, 20.0)];
        assert!(median_neighbor_outer_radius_px([0.0, 0.0], &markers, 6, Some(0), 3).is_none());
        assert!(median_neighbor_outer_radius_px([0.0, 0.0], &markers, 6, Some(0), 1).is_some());
    }

    #[test]
    fn markers_without_outer_ellipse_do_not_contribute() {
        let no_ellipse = MarkerRecord {
            center: [1.0, 0.0],
            ..Default::default()
        };
        let markers = vec![
            marker_at(0.0, 0.0, 5.0),
            no_ellipse,
            marker_at(2.0, 0.0, 18.0),
        ];
        let med = median_neighbor_outer_radius_px([0.0, 0.0], &markers, 2, Some(0), 1)
            .expect("median exists");
        assert!((med - 18.0).abs() < 1e-12, "got {med}");
    }
}
