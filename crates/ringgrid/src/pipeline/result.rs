use crate::detector::proposal::Proposal;
use crate::detector::DetectedMarker;

/// Coordinate frame used by serialized detection outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionFrame {
    #[default]
    Image,
    Working,
}

/// Full detection result for a single image.
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

    pub fn seed_proposals(&self, max_seeds: Option<usize>) -> Vec<Proposal> {
        let max = max_seeds.unwrap_or(self.detected_markers.len());
        self.detected_markers
            .iter()
            .take(max.min(self.detected_markers.len()))
            .filter_map(|m| {
                let x = m.center[0] as f32;
                let y = m.center[1] as f32;
                if x.is_finite() && y.is_finite() {
                    Some(Proposal {
                        x,
                        y,
                        score: m.confidence,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}
