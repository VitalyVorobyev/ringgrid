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
/// - [`max_candidates`](Self::max_candidates) — optional hard cap on the
///   number of returned proposals.
/// - [`radius_step`](Self::radius_step) — stride between voting radii; raise
///   it to trade accumulator sensitivity for proposal-stage speed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
pub struct ProposalConfig {
    /// Minimum voting radius (pixels).
    /// Derived from `marker_scale` and the target layout; a value supplied in JSON is overwritten when the config is loaded.
    #[cfg_attr(feature = "schemars", schemars(extend("readOnly" = true, "x-unit" = "px")))]
    pub r_min: f32,
    /// Maximum voting radius (pixels).
    /// Derived from `marker_scale` and the target layout; a value supplied in JSON is overwritten when the config is loaded.
    #[cfg_attr(feature = "schemars", schemars(extend("readOnly" = true, "x-unit" = "px")))]
    pub r_max: f32,
    /// Minimum distance between output proposals (pixels).
    /// Derived from `marker_scale` and the target layout; a value supplied in JSON is overwritten when the config is loaded.
    ///
    /// Internally the module uses a two-step strategy:
    /// 1. NMS peak extraction with a capped radius for efficiency.
    /// 2. Greedy distance suppression at the full `min_distance`.
    #[cfg_attr(feature = "schemars", schemars(extend("readOnly" = true, "x-unit" = "px")))]
    pub min_distance: f32,
    /// Gradient magnitude threshold as a fraction (0 to 1) of the maximum gradient magnitude.
    pub grad_threshold: f32,
    /// Minimum accumulator value for a proposal, as a fraction (0 to 1) of the accumulator maximum.
    pub min_vote_frac: f32,
    /// Optional cap on the number of proposals returned (after score sorting); `null` means no cap.
    #[serde(default)]
    pub max_candidates: Option<usize>,
    /// Step (in pixels) between consecutive voting radii.
    ///
    /// `1` (the default) tests every integer radius in `[r_min, r_max]`. Values
    /// `> 1` subsample the radius set — `2` tests every other radius
    /// (≈ halves proposal voting cost), and so on — trading accumulator
    /// sensitivity for proposal-stage speed. The maximum radius is always
    /// included so the largest features still vote; values are clamped to a
    /// minimum of `1`.
    ///
    /// Subsampling is **opt-in**: on the regression suite, `radius_step = 2`
    /// cuts proposal time ~29 % but lowers recall on blurry / low-contrast and
    /// real-world scenes (rtv3d −2.9 %), so the default keeps full coverage.
    #[cfg_attr(feature = "schemars", schemars(range(min = 1), extend("x-unit" = "px")))]
    pub radius_step: u32,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            r_min: 3.0,
            r_max: 12.0,
            min_distance: 7.0,
            grad_threshold: 0.05,
            min_vote_frac: 0.1,
            max_candidates: None,
            radius_step: 1,
        }
    }
}
