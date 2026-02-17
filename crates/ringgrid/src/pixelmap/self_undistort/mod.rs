//! Self-undistort: intrinsics-free distortion estimation from ring markers.
//!
//! Uses a 1-parameter division model to estimate lens distortion from
//! conic-consistency of detected inner/outer ring edge points.
//! The division model maps distorted → undistorted coordinates:
//!
//!   x_u = cx + (x_d - cx) / (1 + λ r²)
//!   y_u = cy + (y_d - cy) / (1 + λ r²)
//!
//! where r² = (x_d - cx)² + (y_d - cy)² and (cx, cy) is the distortion center
//! (typically image center).

mod config;
mod estimator;
mod objective;
mod optimizer;
mod policy;
mod result;

pub use config::SelfUndistortConfig;
pub use estimator::estimate_self_undistort;
pub use result::SelfUndistortResult;

#[cfg(test)]
mod tests;
