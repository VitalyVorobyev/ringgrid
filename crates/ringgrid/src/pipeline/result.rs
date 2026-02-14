use crate::detector::DetectedMarker;

/// Full detection result for a single image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionResult {
    /// Detected markers in detector working pixel coordinates.
    pub detected_markers: Vec<DetectedMarker>,
    /// Image dimensions [width, height].
    pub image_size: [u32; 2],
    /// Fitted board-to-working-frame homography (3x3, row-major), if available.
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
            image_size: [width, height],
            homography: None,
            ransac: None,
            self_undistort: None,
        }
    }

    pub fn seed_centers(&self, max_seeds: Option<usize>) -> Vec<[f32; 2]> {
        let max = max_seeds.unwrap_or(self.detected_markers.len());
        self.detected_markers
            .iter()
            .take(max.min(self.detected_markers.len()))
            .filter_map(|m| {
                let x = m.center[0] as f32;
                let y = m.center[1] as f32;
                if x.is_finite() && y.is_finite() {
                    Some([x, y])
                } else {
                    None
                }
            })
            .collect()
    }
}
