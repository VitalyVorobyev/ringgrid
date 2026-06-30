//! Axis-ratio consistency filter.
//!
//! Removes decoded markers whose inner/outer ellipse axis-ratio deviates
//! strongly from the fit-decoded median — a signal of a mis-fit (e.g. the inner
//! ring fitted as the outer). Like the geometric verification gate, it removes
//! such markers outright rather than leaving phantom `id: None` blobs.

use super::stats::median_f64;
use crate::detector::{DetectionSource, MarkerRecord};

const AXIS_RATIO_RELATIVE_TOLERANCE: f64 = 0.25;

fn marker_inner_outer_axis_ratio(marker: &MarkerRecord) -> Option<f64> {
    let inner = marker.ellipse_inner?;
    let outer = marker.ellipse_outer?;
    let inner_axis = inner.mean_axis();
    let outer_axis = outer.mean_axis();
    if !inner_axis.is_finite() || !outer_axis.is_finite() || inner_axis <= 0.0 || outer_axis <= 0.0
    {
        return None;
    }
    Some(inner_axis / outer_axis)
}

/// Remove markers whose inner/outer ellipse axis-ratio deviates from the
/// fit-decoded median by more than [`AXIS_RATIO_RELATIVE_TOLERANCE`].
///
/// A strongly anomalous ratio signals a mis-fit (e.g. the inner ring fitted as
/// the outer), so the detection is not a trustworthy correspondence. Markers are
/// removed outright — like the geometric verification gate — rather than left as
/// phantom `id: None` blobs. The reference median is taken over fit-decoded
/// markers only (completion markers, which copy a neighbor's geometry, are
/// excluded from the reference but still checked). Returns the number removed.
pub(super) fn remove_axis_ratio_outliers(markers: &mut Vec<MarkerRecord>) -> usize {
    let reference = median_f64(
        markers
            .iter()
            .filter(|marker| marker.id.is_some() && marker.source != DetectionSource::Completion)
            .filter_map(marker_inner_outer_axis_ratio)
            .collect(),
    );
    let Some(reference) = reference.filter(|ratio| ratio.is_finite() && *ratio > 0.0) else {
        return 0;
    };

    let before = markers.len();
    markers.retain(|marker| {
        let Some(id) = marker.id else {
            return true;
        };
        let Some(ratio) = marker_inner_outer_axis_ratio(marker) else {
            return true;
        };
        let rel_err = (ratio - reference).abs() / reference;
        if rel_err > AXIS_RATIO_RELATIVE_TOLERANCE {
            tracing::warn!(
                id,
                observed_ratio = ratio,
                reference_ratio = reference,
                rel_err,
                "removing marker due to inner/outer axis-ratio inconsistency"
            );
            false
        } else {
            true
        }
    });
    before - markers.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;

    fn marker_with_ratio(id: usize, ratio: f64, source: DetectionSource) -> MarkerRecord {
        let outer = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 20.0,
            b: 20.0,
            angle: 0.0,
        };
        let inner = Ellipse {
            cx: 0.0,
            cy: 0.0,
            a: 20.0 * ratio,
            b: 20.0 * ratio,
            angle: 0.0,
        };
        MarkerRecord {
            id: Some(id),
            confidence: 1.0,
            center: [id as f64, 0.0],
            ellipse_outer: Some(outer),
            ellipse_inner: Some(inner),
            source,
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn axis_ratio_filter_removes_strong_outliers() {
        let mut markers = vec![
            marker_with_ratio(0, 0.50, DetectionSource::FitDecoded),
            marker_with_ratio(1, 0.49, DetectionSource::FitDecoded),
            marker_with_ratio(2, 0.51, DetectionSource::SeededPass),
            marker_with_ratio(3, 0.30, DetectionSource::Completion),
        ];
        let removed = remove_axis_ratio_outliers(&mut markers);
        assert_eq!(removed, 1);
        assert_eq!(markers.len(), 3);
        assert!(!markers.iter().any(|marker| marker.id == Some(3)));
    }

    #[test]
    fn axis_ratio_filter_keeps_in_family_markers() {
        let mut markers = vec![
            marker_with_ratio(0, 0.50, DetectionSource::FitDecoded),
            marker_with_ratio(1, 0.49, DetectionSource::FitDecoded),
            marker_with_ratio(2, 0.52, DetectionSource::Completion),
        ];
        let removed = remove_axis_ratio_outliers(&mut markers);
        assert_eq!(removed, 0);
        assert_eq!(markers.len(), 3);
        assert!(markers.iter().all(|marker| marker.id.is_some()));
    }
}
