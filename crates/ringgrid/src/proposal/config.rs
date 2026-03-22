//! Standalone configuration for gradient-voting ellipse center detection.
//!
//! [`ProposalConfig`] is self-contained and knows nothing about ring markers,
//! calibration boards, or any other ringgrid-specific concepts.

/// Configuration for ellipse center detection via gradient-based radial
/// symmetry voting.
///
/// # Primary parameters
///
/// The three parameters that most strongly affect recall and performance:
///
/// - [`r_min`](Self::r_min) / [`r_max`](Self::r_max) — voting radius range
///   in pixels. Edges cast votes along the gradient direction at distances
///   `[r_min, r_max]` from the edge pixel. The range should bracket the
///   expected radii of the ellipses you are looking for.
/// - [`min_distance`](Self::min_distance) — minimum distance in pixels
///   between output proposals. Controls spatial suppression of nearby peaks.
///
/// # Secondary parameters
///
/// These have sensible defaults and rarely need tuning:
///
/// - [`grad_threshold`](Self::grad_threshold) — fraction of max gradient
///   magnitude; pixels below this are not used for voting.
/// - [`min_vote_frac`](Self::min_vote_frac) — fraction of the accumulator
///   peak; candidate centers below this are discarded.
/// - [`accum_sigma`](Self::accum_sigma) — Gaussian smoothing applied to the
///   vote accumulator before peak extraction.
/// - [`max_candidates`](Self::max_candidates) — optional hard cap on the
///   number of returned proposals.
/// - [`edge_thinning`](Self::edge_thinning) — Canny-style gradient-direction
///   NMS to thin edges before voting (reduces strong-edge count by 60-80%).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProposalConfig {
    /// Minimum voting radius (pixels).
    pub r_min: f32,
    /// Maximum voting radius (pixels).
    pub r_max: f32,
    /// Minimum distance between output proposals (pixels).
    ///
    /// Internally the module uses a two-step strategy:
    /// 1. NMS peak extraction with a capped radius for efficiency.
    /// 2. Greedy distance suppression at the full `min_distance`.
    pub min_distance: f32,
    /// Gradient magnitude threshold (fraction of max gradient).
    pub grad_threshold: f32,
    /// Minimum accumulator value for a proposal (fraction of max).
    pub min_vote_frac: f32,
    /// Gaussian sigma for accumulator smoothing.
    pub accum_sigma: f32,
    /// Optional cap on number of proposals returned (after score sorting).
    #[serde(default)]
    pub max_candidates: Option<usize>,
    /// Enable Canny-style edge thinning before voting.
    ///
    /// When `true`, non-maximum suppression along the gradient direction
    /// reduces multi-pixel edge bands to single-pixel ridges, typically
    /// cutting the strong-edge count by 60-80% and proportionally reducing
    /// the voting workload.
    pub edge_thinning: bool,
}

/// Cap on the internal NMS peak-extraction radius. The offset table scales
/// as `pi * r^2`, so keeping this bounded avoids quadratic blowup when
/// `min_distance` is large. The greedy distance-suppression pass (which
/// runs only on the handful of NMS survivors) handles the full
/// `min_distance` guarantee.
pub(crate) const MAX_NMS_RADIUS: f32 = 10.0;

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            r_min: 3.0,
            r_max: 12.0,
            min_distance: 7.0,
            grad_threshold: 0.05,
            min_vote_frac: 0.1,
            accum_sigma: 2.0,
            max_candidates: None,
            edge_thinning: true,
        }
    }
}

impl ProposalConfig {
    /// Effective NMS radius for vote-map peak extraction (capped for
    /// performance).
    pub(crate) fn nms_radius(&self) -> f32 {
        self.min_distance.min(MAX_NMS_RADIUS)
    }
}
