use crate::conic::Ellipse;
use crate::detector::{DetectionSource, FitMetrics, MarkerRecord};
use crate::marker::DecodeMetrics;
use crate::proposal::Proposal;
use std::cmp::Ordering;

/// Coordinate frame used by serialized detection outputs.
///
/// - `Image` — raw distorted pixel coordinates.
/// - `Working` — undistorted coordinates produced by a [`PixelMapper`](crate::PixelMapper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionFrame {
    /// Raw image pixel coordinates.
    #[default]
    Image,
    /// Working-frame (undistorted) pixel coordinates.
    Working,
}

/// A detected marker: refined geometry plus the decoded ID.
///
/// This is the slim, stable primary output. Algorithm internals (fit metrics,
/// decode metrics, raw edge sample points, stage provenance) live in the
/// separate opt-in [`MarkerDiagnostics`] channel — request them via
/// [`Detector::detect_with_diagnostics`](crate::Detector::detect_with_diagnostics).
///
/// The `center` field is always in image-pixel coordinates, regardless of
/// whether a [`PixelMapper`](crate::PixelMapper) was used. When a mapper is
/// active, `center_mapped` provides the working-frame (undistorted)
/// coordinates. `board_xy_mm` provides board-space marker coordinates in
/// millimeters when the decoded `id` is valid for the active
/// [`BoardLayout`](crate::BoardLayout). Ellipses are in the working frame when
/// a mapper is active.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct DetectedMarker {
    /// Decoded marker ID (codebook index), or None if decoding was rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    /// Combined detection + decode confidence in [0, 1].
    pub confidence: f32,
    /// Marker center in raw image pixel coordinates.
    ///
    /// This field is always image-space, independent of mapper usage.
    pub center: [f64; 2],
    /// Marker center in mapper working coordinates, when a mapper is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_mapped: Option<[f64; 2]>,
    /// Marker center on the physical board in millimeters `[x_mm, y_mm]`.
    ///
    /// Populated when `id` is present and valid for the active board layout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub board_xy_mm: Option<[f64; 2]>,
    /// Outer ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<Ellipse>,
    /// Inner ellipse parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<Ellipse>,
}

/// Detailed per-marker algorithm internals.
///
/// This is the opt-in diagnostics counterpart to [`DetectedMarker`]. Each
/// `MarkerDiagnostics` is positionally aligned 1:1 with the corresponding
/// [`DetectedMarker`] in [`DetectionResult::detected_markers`]: the
/// `MarkerDiagnostics` at index `i` in [`DetectionDiagnostics::markers`]
/// describes the `DetectedMarker` at index `i`.
///
/// Returned only via
/// [`Detector::detect_with_diagnostics`](crate::Detector::detect_with_diagnostics).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MarkerDiagnostics {
    /// Fit quality metrics (edge sampling and ellipse fit).
    pub fit: FitMetrics,
    /// Decode metrics (present if decoding was attempted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeMetrics>,
    /// Pipeline stage that produced this marker.
    pub source: DetectionSource,
    /// Raw sub-pixel outer edge inlier points used for ellipse fitting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_points_outer: Option<Vec<[f64; 2]>>,
    /// Raw sub-pixel inner edge inlier points used for ellipse fitting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_points_inner: Option<Vec<[f64; 2]>>,
}

/// Detection result for a single image.
///
/// Returned by [`Detector::detect`](crate::Detector::detect) and
/// [`Detector::detect_with_mapper`](crate::Detector::detect_with_mapper).
/// Contains detected markers, frame metadata, and an optional
/// board-to-image homography. Serializable to JSON via `serde`.
///
/// Algorithm internals and RANSAC statistics are not part of this type; obtain
/// them through the opt-in [`DetectionDiagnostics`] channel via
/// [`Detector::detect_with_diagnostics`](crate::Detector::detect_with_diagnostics).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct DetectionResult {
    /// Detected markers.
    pub detected_markers: Vec<DetectedMarker>,
    /// Coordinate frame of `DetectedMarker.center`.
    pub center_frame: DetectionFrame,
    /// Coordinate frame of `homography` output.
    pub homography_frame: DetectionFrame,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
    /// Fitted board-to-output-frame homography (3x3, row-major), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homography: Option<[[f64; 3]; 3]>,
    /// Estimated self-undistort division model, if self-undistort was run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_undistort: Option<crate::pixelmap::SelfUndistortResult>,
}

/// Opt-in detection diagnostics for debugging and tuning.
///
/// Returned alongside a [`DetectionResult`] by
/// [`Detector::detect_with_diagnostics`](crate::Detector::detect_with_diagnostics).
/// Carries per-marker algorithm internals and homography RANSAC statistics.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DetectionDiagnostics {
    /// Per-marker algorithm internals.
    ///
    /// Positionally aligned 1:1 with
    /// [`DetectionResult::detected_markers`]: `markers[i]` describes the
    /// `DetectedMarker` at the same index `i`.
    pub markers: Vec<MarkerDiagnostics>,
    /// Homography RANSAC statistics, if a homography was fitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac: Option<crate::homography::RansacStats>,
}

impl DetectionResult {
    /// Construct an empty result for an image with the provided dimensions.
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            detected_markers: Vec::new(),
            center_frame: DetectionFrame::Image,
            homography_frame: DetectionFrame::Image,
            image_size: [width, height],
            homography: None,
            self_undistort: None,
        }
    }
}

/// Internal pipeline result threaded through the detection stages.
///
/// Carries the rich [`MarkerRecord`] working markers and the homography RANSAC
/// statistics. Converted into the public `(DetectionResult, DetectionDiagnostics)`
/// pair at the [`Detector::detect`](crate::Detector::detect) boundary via
/// [`PipelineResult::split`].
#[derive(Debug, Clone, Default)]
pub(crate) struct PipelineResult {
    /// Rich working markers.
    pub markers: Vec<MarkerRecord>,
    /// Coordinate frame of `MarkerRecord.center`.
    pub center_frame: DetectionFrame,
    /// Coordinate frame of `homography` output.
    pub homography_frame: DetectionFrame,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
    /// Fitted board-to-output-frame homography (3x3, row-major), if available.
    pub homography: Option<[[f64; 3]; 3]>,
    /// Homography RANSAC statistics, if a homography was fitted.
    pub ransac: Option<crate::homography::RansacStats>,
    /// Estimated self-undistort division model, if self-undistort was run.
    pub self_undistort: Option<crate::pixelmap::SelfUndistortResult>,
}

impl PipelineResult {
    /// Split a finished pipeline result into the public slim
    /// [`DetectionResult`] plus its [`DetectionDiagnostics`] counterpart.
    ///
    /// The two `Vec`s remain positionally aligned 1:1.
    pub(crate) fn split(self) -> (DetectionResult, DetectionDiagnostics) {
        let mut detected_markers = Vec::with_capacity(self.markers.len());
        let mut diagnostics = Vec::with_capacity(self.markers.len());
        for record in self.markers {
            let (marker, diag) = split_marker_record(record);
            detected_markers.push(marker);
            diagnostics.push(diag);
        }

        let result = DetectionResult {
            detected_markers,
            center_frame: self.center_frame,
            homography_frame: self.homography_frame,
            image_size: self.image_size,
            homography: self.homography,
            self_undistort: self.self_undistort,
        };
        let diag = DetectionDiagnostics {
            markers: diagnostics,
            ransac: self.ransac,
        };
        (result, diag)
    }
}

/// Split one rich [`MarkerRecord`] into the public slim [`DetectedMarker`] and
/// its [`MarkerDiagnostics`] counterpart. Single source of truth for the
/// field partition.
fn split_marker_record(record: MarkerRecord) -> (DetectedMarker, MarkerDiagnostics) {
    let MarkerRecord {
        id,
        confidence,
        center,
        center_mapped,
        board_xy_mm,
        ellipse_outer,
        ellipse_inner,
        edge_points_outer,
        edge_points_inner,
        fit,
        decode,
        source,
    } = record;

    let marker = DetectedMarker {
        id,
        confidence,
        center,
        center_mapped,
        board_xy_mm,
        ellipse_outer,
        ellipse_inner,
    };
    let diagnostics = MarkerDiagnostics {
        fit,
        decode,
        source,
        edge_points_outer,
        edge_points_inner,
    };
    (marker, diagnostics)
}

/// Build pass-2 seed proposals from pass-1 detection markers.
///
/// Used internally for two-pass / self-undistort seeding.
///
/// Ordering is deterministic and confidence-ranked:
/// 1. higher `confidence` first (`NaN`/non-finite treated as `-inf`)
/// 2. decoded IDs before undecoded markers
/// 3. decoded ID ascending
/// 4. center `x` ascending, then `y` ascending
/// 5. original marker index (stable final tie-break)
pub(crate) fn seed_proposals(markers: &[MarkerRecord], max_seeds: Option<usize>) -> Vec<Proposal> {
    let mut candidates: Vec<SeedCandidate> = markers
        .iter()
        .enumerate()
        .filter_map(|(source_index, marker)| {
            let x = marker.center[0] as f32;
            let y = marker.center[1] as f32;
            if !(x.is_finite() && y.is_finite()) {
                return None;
            }

            let score = if marker.confidence.is_finite() {
                marker.confidence
            } else {
                f32::NEG_INFINITY
            };

            Some(SeedCandidate {
                proposal: Proposal { x, y, score },
                marker_id: marker.id,
                source_index,
            })
        })
        .collect();

    candidates.sort_by(compare_seed_candidate);

    let max = max_seeds.unwrap_or(candidates.len());
    candidates.truncate(max.min(candidates.len()));
    candidates.into_iter().map(|c| c.proposal).collect()
}

fn compare_seed_candidate(a: &SeedCandidate, b: &SeedCandidate) -> Ordering {
    b.proposal
        .score
        .total_cmp(&a.proposal.score)
        .then_with(|| b.marker_id.is_some().cmp(&a.marker_id.is_some()))
        .then_with(|| match (a.marker_id, b.marker_id) {
            (Some(aid), Some(bid)) => aid.cmp(&bid),
            _ => Ordering::Equal,
        })
        .then_with(|| a.proposal.x.total_cmp(&b.proposal.x))
        .then_with(|| a.proposal.y.total_cmp(&b.proposal.y))
        .then_with(|| a.source_index.cmp(&b.source_index))
}

#[derive(Clone)]
struct SeedCandidate {
    proposal: Proposal,
    marker_id: Option<usize>,
    source_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(id: Option<usize>, confidence: f32, center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            id,
            confidence,
            center,
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn seed_proposals_are_confidence_ordered_and_truncated() {
        let markers = vec![
            marker(Some(11), 0.1, [10.0, 10.0]),
            marker(Some(12), 0.8, [20.0, 20.0]),
            marker(Some(13), 0.6, [30.0, 30.0]),
        ];

        let proposals = seed_proposals(&markers, Some(2));
        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].score, 0.8);
        assert_eq!(proposals[1].score, 0.6);
    }

    #[test]
    fn seed_proposals_tie_break_is_deterministic() {
        let markers = vec![
            marker(Some(7), 0.8, [20.0, 10.0]),
            marker(None, 0.8, [1.0, 1.0]),
            marker(Some(3), 0.8, [8.0, 8.0]),
            marker(Some(3), 0.8, [4.0, 9.0]),
            marker(Some(2), 0.95, [100.0, 100.0]),
        ];

        let permuted = vec![
            markers[1].clone(),
            markers[3].clone(),
            markers[4].clone(),
            markers[0].clone(),
            markers[2].clone(),
        ];

        let pa = seed_proposals(&markers, None);
        let pb = seed_proposals(&permuted, None);
        assert_eq!(pa.len(), 5);
        let pa_xy_score: Vec<(f32, f32, f32)> = pa.iter().map(|p| (p.x, p.y, p.score)).collect();
        let pb_xy_score: Vec<(f32, f32, f32)> = pb.iter().map(|p| (p.x, p.y, p.score)).collect();
        assert_eq!(pa_xy_score, pb_xy_score);

        let ordered_centers: Vec<[f32; 2]> = pa.iter().map(|p| [p.x, p.y]).collect();
        assert_eq!(
            ordered_centers,
            vec![
                [100.0, 100.0], // highest confidence
                [4.0, 9.0],     // same confidence, lower id then lower x
                [8.0, 8.0],
                [20.0, 10.0],
                [1.0, 1.0], // undecoded marker sorted last in confidence tie
            ]
        );
    }

    #[test]
    fn seed_proposals_skip_non_finite_centers_and_demote_non_finite_confidence() {
        let markers = vec![
            marker(Some(1), 0.7, [10.0, 10.0]),
            marker(Some(2), f32::NAN, [11.0, 11.0]),
            marker(Some(3), 0.9, [f64::NAN, 12.0]),
        ];

        let proposals = seed_proposals(&markers, None);
        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].score, 0.7);
        assert!(proposals[1].score.is_infinite() && proposals[1].score.is_sign_negative());
    }

    #[test]
    fn split_partitions_records_into_aligned_slim_and_diagnostics() {
        let mut record = marker(Some(5), 0.9, [12.0, 34.0]);
        record.source = DetectionSource::Completion;
        record.edge_points_outer = Some(vec![[1.0, 2.0]]);
        record.fit.n_points_outer = 7;

        let pipeline = PipelineResult {
            markers: vec![record],
            image_size: [640, 480],
            ..PipelineResult::default()
        };
        let (result, diagnostics) = pipeline.split();

        assert_eq!(result.detected_markers.len(), diagnostics.markers.len());
        assert_eq!(result.detected_markers[0].id, Some(5));
        assert_eq!(result.detected_markers[0].center, [12.0, 34.0]);
        assert_eq!(diagnostics.markers[0].source, DetectionSource::Completion);
        assert_eq!(diagnostics.markers[0].fit.n_points_outer, 7);
        assert_eq!(
            diagnostics.markers[0].edge_points_outer,
            Some(vec![[1.0, 2.0]])
        );
    }
}
