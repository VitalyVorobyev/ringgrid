//! Ring geometry primitives shared across the detector pipeline.

pub(crate) mod edge_quality;
pub(crate) mod edge_sample;
pub(crate) mod inner_estimate;
pub(crate) mod outer_estimate;
pub(crate) mod photometric_center;
pub(crate) mod projective_center;
pub(crate) mod radial_estimator;
pub(crate) mod radial_profile;

/// Rescales a median absolute deviation to a Gaussian-consistent sigma
/// (`1 / Φ⁻¹(3/4)`). The one definition shared by every robust-scale estimate
/// in the crate (photometric fits, geometric verification).
pub(crate) const MAD_TO_SIGMA: f64 = 1.4826;

pub use edge_sample::EdgeSampleConfig;
pub use outer_estimate::OuterEstimationConfig;
