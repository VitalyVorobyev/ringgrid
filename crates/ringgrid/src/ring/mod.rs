//! Ring geometry primitives shared across the detector pipeline.

pub(crate) mod edge_sample;
pub(crate) mod inner_estimate;
pub(crate) mod outer_estimate;
pub(crate) mod projective_center;
pub(crate) mod radial_estimator;
pub(crate) mod radial_profile;

pub use edge_sample::EdgeSampleConfig;
pub use outer_estimate::OuterEstimationConfig;
