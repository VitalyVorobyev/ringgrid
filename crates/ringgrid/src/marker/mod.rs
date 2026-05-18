//! Marker coding and geometry primitives.

pub(crate) mod decode;
mod diagnostics;
mod marker_spec;

pub mod codebook;
pub mod codec;

pub use codec::CodebookProfile;
pub use decode::{DecodeConfig, DecodeMetrics};
pub use diagnostics::{CodebookInfo, CodewordMatch, codebook_info, decode_word};
pub use marker_spec::{AngularAggregator, GradPolarity, MarkerSpec};
