//! End-to-end ring marker detection pipeline.
//!
//! Stages:
//! 1. **Proposal** – gradient-voting radial symmetry to find candidate centers.
//! 2. **Edge sampling** – radial intensity profiles to locate inner/outer ring edges.
//! 3. **Ellipse fitting** – robust conic fit to edge points (uses `crate::conic`).
//! 4. **Decode** – ellipse-guided 16-sector sampling + codebook matching.
//! 5. **Detect** – orchestrates the above into `DetectionResult`.

pub mod decode;
pub mod detect;
pub mod edge_sample;
pub mod inner_estimate;
pub mod outer_estimate;
pub mod proposal;

pub use detect::{detect_rings, detect_rings_with_debug, DebugCollectConfig, DetectConfig};
