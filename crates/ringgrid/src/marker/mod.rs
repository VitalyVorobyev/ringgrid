//! Marker coding and geometry primitives.

pub(crate) mod decode;
mod marker_spec;

pub mod codebook;
pub mod codec;

pub use decode::{DecodeConfig, DecodeMetrics};
pub use marker_spec::{AngularAggregator, GradPolarity, MarkerSpec};
