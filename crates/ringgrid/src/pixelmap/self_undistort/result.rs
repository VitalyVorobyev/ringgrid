use serde::{Deserialize, Serialize};

use crate::pixelmap::DivisionModel;

/// Result of self-undistort estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfUndistortResult {
    /// Estimated division model.
    pub model: DivisionModel,
    /// Objective value at the estimated lambda.
    pub objective_at_lambda: f64,
    /// Objective value at lambda=0 (baseline).
    pub objective_at_zero: f64,
    /// Number of markers used in estimation.
    pub n_markers_used: usize,
    /// Whether the estimated model was applied (improvement exceeded threshold).
    pub applied: bool,
}
