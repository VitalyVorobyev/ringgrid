use serde::{Deserialize, Serialize};

/// Configuration for self-undistort estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
pub struct SelfUndistortConfig {
    /// Enable self-undistort refinement.
    pub enable: bool,
    /// Search range `[lambda_min, lambda_max]` for the division-model parameter lambda (units 1/px^2, with radius measured in pixels from the distortion center).
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "1/px^2")))]
    pub lambda_range: [f64; 2],
    /// Maximum objective evaluations of the 1D lambda optimizer.
    pub max_evals: usize,
    /// Minimum number of markers with both inner and outer edge points required to run the estimation.
    pub min_markers: usize,
    /// Relative improvement threshold: accept only if
    /// `(baseline - optimum) / baseline > improvement_threshold`.
    pub improvement_threshold: f64,
    /// Minimum absolute objective improvement required to apply the model.
    ///
    /// This prevents applying when the objective is near numerical noise floor.
    pub min_abs_improvement: f64,
    /// Trim fraction for robust aggregation of per-marker objective values.
    ///
    /// `0.1` means drop 10% low and 10% high scores before averaging. Values are clamped to `[0, 0.49]`.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0, max = 0.49)))]
    pub trim_fraction: f64,
    /// Minimum |lambda| (1/px^2) required for applying the model.
    ///
    /// Very small lambda values are effectively identity and are treated as
    /// "no correction" even if relative improvement is non-zero.
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "1/px^2")))]
    pub min_lambda_abs: f64,
    /// Reject solutions that land too close to lambda-range boundaries.
    pub reject_range_edge: bool,
    /// Relative margin (fraction of the lambda range width) treated as unstable boundary area; values are clamped to `[0, 0.49]`.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0, max = 0.49)))]
    pub range_edge_margin_frac: f64,
    /// Minimum decoded-ID correspondences needed for homography validation.
    pub validation_min_markers: usize,
    /// Minimum absolute homography self-error improvement (pixels) required.
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub validation_abs_improvement_px: f64,
    /// Minimum relative homography self-error improvement required.
    pub validation_rel_improvement: f64,
}

impl Default for SelfUndistortConfig {
    fn default() -> Self {
        Self {
            enable: false,
            lambda_range: [-8e-7, 8e-7],
            max_evals: 40,
            min_markers: 6,
            improvement_threshold: 0.01,
            min_abs_improvement: 1e-4,
            trim_fraction: 0.1,
            min_lambda_abs: 5e-9,
            reject_range_edge: true,
            range_edge_margin_frac: 0.02,
            validation_min_markers: 24,
            validation_abs_improvement_px: 0.05,
            validation_rel_improvement: 0.03,
        }
    }
}
