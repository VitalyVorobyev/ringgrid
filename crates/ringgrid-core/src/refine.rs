//! Per-marker robust refinement via shared-center dual-ring fitting.
//!
//! Each marker consists of two concentric circles (inner/outer ring).
//! Under projection these become two ellipses sharing the same center
//! (in affine-rectified space) but with different semi-axes.
//!
//! The refinement stage solves a nonlinear least-squares problem:
//!   minimize Σ ρ(d_i²)
//! where d_i is the geometric distance from edge point i to the nearest
//! ring boundary, and ρ is a robust loss function (e.g., Huber or Tukey).
//!
//! Parameters: shared center (cx, cy), outer ellipse (a1, b1, θ1),
//! inner ellipse ratio (r_inner), plus optional per-ring refinement.
//!
//! TODO Milestone 3:
//! - Implement Levenberg–Marquardt solver (or use a small LM crate).
//! - Robust loss functions (Huber, Tukey bisquare).
//! - Shared-center parameterization.
//! - Convergence criteria and iteration limits.
//! - Covariance estimation for uncertainty propagation.

use crate::conic::Ellipse;

/// Parameters for the dual-ring refinement.
#[derive(Debug, Clone)]
pub struct DualRingParams {
    /// Shared center (after affine rectification correction).
    pub center: [f64; 2],
    /// Outer ring ellipse.
    pub outer: Ellipse,
    /// Ratio of inner ring semi-axes to outer ring semi-axes.
    pub inner_ratio: f64,
    /// RMS residual of the fit.
    pub residual: f64,
}

/// Refine a marker's center and ring parameters.
///
/// Stub — returns initial ellipse parameters unchanged.
pub fn refine_marker(
    outer_ellipse: &Ellipse,
    _inner_ellipse: Option<&Ellipse>,
    _edge_points: &[[f64; 2]],
) -> DualRingParams {
    // TODO Milestone 3: implement LM refinement
    DualRingParams {
        center: [outer_ellipse.cx, outer_ellipse.cy],
        outer: *outer_ellipse,
        inner_ratio: 0.5, // default assumption
        residual: f64::NAN,
    }
}
