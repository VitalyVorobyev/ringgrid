//! Marker coding and geometry primitives.

pub(crate) mod decode;
mod marker_spec;

#[cfg(feature = "cli-internal")]
pub mod codebook;
#[cfg(not(feature = "cli-internal"))]
mod codebook;

#[cfg(feature = "cli-internal")]
pub mod codec;
#[cfg(not(feature = "cli-internal"))]
mod codec;

pub use decode::DecodeConfig;
pub use marker_spec::{AngularAggregator, GradPolarity, MarkerSpec};
