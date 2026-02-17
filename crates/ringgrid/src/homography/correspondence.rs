//! Shared homography correspondence assembly and reprojection-error helpers.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use nalgebra::Matrix3;

use crate::board_layout::BoardLayout;
use crate::detector::DetectedMarker;

use super::core::homography_reprojection_error;

/// Destination frame for homography correspondences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CorrespondenceDestinationFrame {
    /// Image-space pixel centers (`DetectedMarker::center`).
    Image,
    /// Working-space pixel centers (mapper output).
    Working,
}

/// Duplicate-ID handling policy for correspondence collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DuplicateIdPolicy {
    /// Keep every valid marker correspondence, including duplicate IDs.
    KeepAll,
    /// Keep at most one marker per ID: highest confidence wins; ties keep
    /// first-seen marker. Output is deterministic in ascending marker ID order.
    KeepHighestConfidencePerIdSorted,
}

/// Marker correspondences for board-mm -> destination-frame homography fitting.
#[derive(Debug, Clone)]
pub(crate) struct MarkerCorrespondences {
    /// Source points in board millimeters.
    pub src_board_mm: Vec<[f64; 2]>,
    /// Destination points in the selected destination frame (pixels).
    pub dst_points: Vec<[f64; 2]>,
    /// Index of each selected marker in the original marker slice.
    pub marker_indices: Vec<usize>,
    /// Marker IDs aligned with `src_board_mm` / `dst_points`.
    pub marker_ids: Vec<usize>,
    /// Destination frame label for `dst_points`.
    pub dst_frame: CorrespondenceDestinationFrame,
}

impl MarkerCorrespondences {
    fn new(dst_frame: CorrespondenceDestinationFrame) -> Self {
        Self {
            src_board_mm: Vec::new(),
            dst_points: Vec::new(),
            marker_indices: Vec::new(),
            marker_ids: Vec::new(),
            dst_frame,
        }
    }

    fn push(
        &mut self,
        marker_id: usize,
        marker_index: usize,
        src_board_mm: [f64; 2],
        dst_point: [f64; 2],
    ) {
        self.src_board_mm.push(src_board_mm);
        self.dst_points.push(dst_point);
        self.marker_indices.push(marker_index);
        self.marker_ids.push(marker_id);
    }

    pub(crate) fn len(&self) -> usize {
        self.src_board_mm.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct CandidateEntry {
    marker_index: usize,
    confidence: f32,
    src_board_mm: [f64; 2],
    dst_point: [f64; 2],
}

fn marker_candidate<F>(
    marker_index: usize,
    marker: &DetectedMarker,
    board: &BoardLayout,
    map_dst_point: &mut F,
) -> Option<(usize, CandidateEntry)>
where
    F: FnMut(&DetectedMarker) -> Option<[f64; 2]>,
{
    let id = marker.id?;
    let board_xy = board.xy_mm(id)?;
    let dst_point = map_dst_point(marker)?;
    Some((
        id,
        CandidateEntry {
            marker_index,
            confidence: marker.confidence,
            src_board_mm: [board_xy[0] as f64, board_xy[1] as f64],
            dst_point,
        },
    ))
}

/// Collect homography correspondences from detected markers.
///
/// Source frame is board millimeters. Destination frame is labeled by
/// `dst_frame` and populated via `map_dst_point`.
pub(crate) fn collect_marker_correspondences<F>(
    markers: &[DetectedMarker],
    board: &BoardLayout,
    dst_frame: CorrespondenceDestinationFrame,
    duplicate_policy: DuplicateIdPolicy,
    mut map_dst_point: F,
) -> MarkerCorrespondences
where
    F: FnMut(&DetectedMarker) -> Option<[f64; 2]>,
{
    let mut corr = MarkerCorrespondences::new(dst_frame);
    match duplicate_policy {
        DuplicateIdPolicy::KeepAll => {
            for (marker_index, marker) in markers.iter().enumerate() {
                let Some((id, candidate)) =
                    marker_candidate(marker_index, marker, board, &mut map_dst_point)
                else {
                    continue;
                };
                corr.push(
                    id,
                    candidate.marker_index,
                    candidate.src_board_mm,
                    candidate.dst_point,
                );
            }
        }
        DuplicateIdPolicy::KeepHighestConfidencePerIdSorted => {
            let mut by_id: BTreeMap<usize, CandidateEntry> = BTreeMap::new();
            for (marker_index, marker) in markers.iter().enumerate() {
                let Some((id, candidate)) =
                    marker_candidate(marker_index, marker, board, &mut map_dst_point)
                else {
                    continue;
                };
                match by_id.get_mut(&id) {
                    Some(best) => {
                        if candidate.confidence > best.confidence {
                            *best = candidate;
                        }
                    }
                    None => {
                        by_id.insert(id, candidate);
                    }
                }
            }

            for (id, candidate) in by_id {
                corr.push(
                    id,
                    candidate.marker_index,
                    candidate.src_board_mm,
                    candidate.dst_point,
                );
            }
        }
    }
    corr
}

/// Compute per-correspondence reprojection errors for a homography.
pub(crate) fn reprojection_errors(
    h: &Matrix3<f64>,
    correspondences: &MarkerCorrespondences,
) -> Vec<f64> {
    correspondences
        .src_board_mm
        .iter()
        .zip(correspondences.dst_points.iter())
        .map(|(src, dst)| homography_reprojection_error(h, src, dst))
        .collect()
}

/// Mean and P95 summary for a mutable error slice.
pub(crate) fn mean_and_p95(errors: &mut [f64]) -> (f64, f64) {
    if errors.is_empty() {
        return (0.0, 0.0);
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let idx = ((errors.len() as f64 * 0.95) as usize).min(errors.len() - 1);
    (mean, errors[idx])
}

/// Collect masked inlier errors (`inlier_mask[i] == true`) from the RANSAC
/// error vector.
pub(crate) fn collect_masked_inlier_errors(errors: &[f64], inlier_mask: &[bool]) -> Vec<f64> {
    inlier_mask
        .iter()
        .zip(errors.iter())
        .filter_map(|(&is_inlier, &err)| if is_inlier { Some(err) } else { None })
        .collect()
}

/// Mean finite inlier error and number of inliers used in the mean.
pub(crate) fn mean_finite_masked_inlier_error(
    errors: &[f64],
    inlier_mask: &[bool],
) -> Option<(f64, usize)> {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (i, &err) in errors.iter().enumerate() {
        if inlier_mask.get(i).copied().unwrap_or(false) && err.is_finite() {
            sum += err;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some((sum / n as f64, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FitMetrics;

    fn marker(id: Option<usize>, confidence: f32, center: [f64; 2]) -> DetectedMarker {
        DetectedMarker {
            id,
            confidence,
            center,
            center_mapped: None,
            ellipse_outer: None,
            ellipse_inner: None,
            edge_points_outer: None,
            edge_points_inner: None,
            fit: FitMetrics::default(),
            decode: None,
        }
    }

    #[test]
    fn collect_keep_all_skips_missing_id_out_of_board_and_invalid_centers_when_mapper_drops_them() {
        let board = BoardLayout::default();
        let markers = vec![
            marker(None, 0.9, [1.0, 1.0]),
            marker(Some(0), 0.5, [10.0, 20.0]),
            marker(Some(99_999), 0.8, [30.0, 40.0]),
            marker(Some(1), 0.7, [f64::NAN, 2.0]),
            marker(Some(2), 0.6, [50.0, 60.0]),
        ];

        let corr = collect_marker_correspondences(
            &markers,
            &board,
            CorrespondenceDestinationFrame::Image,
            DuplicateIdPolicy::KeepAll,
            |m| (m.center[0].is_finite() && m.center[1].is_finite()).then_some(m.center),
        );

        assert_eq!(corr.dst_frame, CorrespondenceDestinationFrame::Image);
        assert_eq!(corr.len(), 2);
        assert_eq!(corr.marker_ids, vec![0, 2]);
        assert_eq!(corr.marker_indices, vec![1, 4]);
        assert_eq!(corr.dst_points, vec![[10.0, 20.0], [50.0, 60.0]]);
        assert_eq!(corr.src_board_mm.len(), corr.dst_points.len());
    }

    #[test]
    fn collect_keep_highest_confidence_per_id_is_deterministic_and_tie_stable() {
        let board = BoardLayout::default();
        let markers = vec![
            marker(Some(2), 0.40, [100.0, 200.0]),
            marker(Some(1), 0.80, [11.0, 21.0]),
            marker(Some(2), 0.95, [120.0, 220.0]),
            marker(Some(1), 0.80, [13.0, 23.0]), // tie: keep first-seen id=1
        ];

        let corr = collect_marker_correspondences(
            &markers,
            &board,
            CorrespondenceDestinationFrame::Working,
            DuplicateIdPolicy::KeepHighestConfidencePerIdSorted,
            |m| Some(m.center),
        );

        assert_eq!(corr.marker_ids, vec![1, 2]);
        assert_eq!(corr.marker_indices, vec![1, 2]);
        assert_eq!(corr.dst_points, vec![[11.0, 21.0], [120.0, 220.0]]);
    }

    #[test]
    fn mean_finite_masked_inlier_error_skips_non_finite_and_non_inliers() {
        let errors = [0.5, f64::NAN, 0.25, 2.0];
        let mask = [true, true, true, false];
        let (mean, n) = mean_finite_masked_inlier_error(&errors, &mask).unwrap();
        assert_eq!(n, 2);
        assert!((mean - 0.375).abs() < 1e-12);
    }

    #[test]
    fn mean_and_p95_handles_empty_and_non_empty_errors() {
        let mut empty = Vec::<f64>::new();
        assert_eq!(mean_and_p95(&mut empty), (0.0, 0.0));

        let mut values = vec![0.4, 0.1, 0.2, 0.3, 0.5];
        let (mean, p95) = mean_and_p95(&mut values);
        assert!((mean - 0.3).abs() < 1e-12);
        assert!((p95 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn collect_with_mapper_closure_can_drop_points() {
        let board = BoardLayout::default();
        let markers = vec![
            marker(Some(0), 0.2, [3.0, 4.0]),
            marker(Some(1), 0.2, [5.0, 6.0]),
            marker(Some(2), 0.2, [7.0, 8.0]),
        ];

        let corr = collect_marker_correspondences(
            &markers,
            &board,
            CorrespondenceDestinationFrame::Working,
            DuplicateIdPolicy::KeepAll,
            |m| (m.center[0] > 4.0).then_some(m.center),
        );

        assert_eq!(corr.marker_ids, vec![1, 2]);
    }
}
