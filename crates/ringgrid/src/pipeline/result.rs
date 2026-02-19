use crate::detector::proposal::Proposal;
use crate::detector::DetectedMarker;
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

/// Full detection result for a single image.
///
/// Returned by [`Detector::detect`](crate::Detector::detect) and
/// [`Detector::detect_with_mapper`](crate::Detector::detect_with_mapper).
/// Contains detected markers, an optional board-to-image homography,
/// and quality statistics. Serializable to JSON via `serde`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    /// RANSAC statistics, if homography was fitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ransac: Option<crate::homography::RansacStats>,
    /// Estimated self-undistort division model, if self-undistort was run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_undistort: Option<crate::pixelmap::SelfUndistortResult>,
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
            ransac: None,
            self_undistort: None,
        }
    }

    /// Build pass-2 seed proposals from pass-1 detections.
    ///
    /// Ordering is deterministic and confidence-ranked:
    /// 1. higher `confidence` first (`NaN`/non-finite treated as `-inf`)
    /// 2. decoded IDs before undecoded markers
    /// 3. decoded ID ascending
    /// 4. center `x` ascending, then `y` ascending
    /// 5. original marker index (stable final tie-break)
    pub fn seed_proposals(&self, max_seeds: Option<usize>) -> Vec<Proposal> {
        let mut candidates: Vec<SeedCandidate> = self
            .detected_markers
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

    fn marker(id: Option<usize>, confidence: f32, center: [f64; 2]) -> DetectedMarker {
        DetectedMarker {
            id,
            confidence,
            center,
            ..DetectedMarker::default()
        }
    }

    #[test]
    fn seed_proposals_are_confidence_ordered_and_truncated() {
        let result = DetectionResult {
            detected_markers: vec![
                marker(Some(11), 0.1, [10.0, 10.0]),
                marker(Some(12), 0.8, [20.0, 20.0]),
                marker(Some(13), 0.6, [30.0, 30.0]),
            ],
            ..DetectionResult::default()
        };

        let proposals = result.seed_proposals(Some(2));
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

        let a = DetectionResult {
            detected_markers: markers,
            ..DetectionResult::default()
        };
        let b = DetectionResult {
            detected_markers: permuted,
            ..DetectionResult::default()
        };

        let pa = a.seed_proposals(None);
        let pb = b.seed_proposals(None);
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
        let result = DetectionResult {
            detected_markers: vec![
                marker(Some(1), 0.7, [10.0, 10.0]),
                marker(Some(2), f32::NAN, [11.0, 11.0]),
                marker(Some(3), 0.9, [f64::NAN, 12.0]),
            ],
            ..DetectionResult::default()
        };

        let proposals = result.seed_proposals(None);
        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].score, 0.7);
        assert!(proposals[1].score.is_infinite() && proposals[1].score.is_sign_negative());
    }
}
