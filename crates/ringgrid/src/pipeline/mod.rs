//! High-level detection pipeline.
//!
//! This module is the internal "glue" layer that wires together detector stages:
//! proposal generation -> fit/decode -> dedup -> global filter/refine -> completion.
//!
//! Algorithmic primitives live in `crate::detector`, `crate::ring`, and `crate::pixelmap`.
//! The pipeline layer focuses on stage boundaries, call order, and data flow.
//!
//! Entry points:
//! - `detect_single_pass`: baseline detection (optional mapper for distortion-aware sampling)
//! - `detect_two_pass`: baseline pass + seeded mapper pass
//! - `detect_with_self_undistort`: baseline pass + estimate division-model mapper + optional rerun
//! - `detect_single_pass_with_debug`: debug collection for single-pass only
//!
//! See `docs/pipeline_analysis.md` for a detailed architecture write-up.

mod finalize;
mod fit_decode;
mod prelude;
mod result;
mod run;

pub use result::DetectionResult;

pub(crate) use prelude::*;

#[cfg(feature = "cli-internal")]
pub use run::detect_single_pass_with_debug;
pub(crate) use run::{detect_single_pass, detect_with_mapper, detect_with_self_undistort};
