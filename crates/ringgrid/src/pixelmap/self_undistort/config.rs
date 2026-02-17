use serde::{Deserialize, Serialize};

/// Configuration for self-undistort estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfUndistortConfig {
    /// Enable self-undistort refinement.
    pub enable: bool,
    /// Search range for lambda: [lambda_min, lambda_max].
    pub lambda_range: [f64; 2],
    /// Maximum function evaluations for the 1D optimizer.
    pub max_evals: usize,
    /// Minimum number of markers with both inner+outer edge points required.
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
    /// `0.1` means drop 10% low and 10% high scores before averaging.
    pub trim_fraction: f64,
    /// Minimum |lambda| required for applying the model.
    ///
    /// Very small lambda values are effectively identity and are treated as
    /// "no correction" even if relative improvement is non-zero.
    pub min_lambda_abs: f64,
    /// Reject solutions that land too close to lambda-range boundaries.
    pub reject_range_edge: bool,
    /// Relative margin of the lambda range treated as unstable boundary area.
    pub range_edge_margin_frac: f64,
    /// Minimum decoded-ID correspondences needed for homography validation.
    pub validation_min_markers: usize,
    /// Minimum absolute homography self-error improvement (pixels) required.
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
