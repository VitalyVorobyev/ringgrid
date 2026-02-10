//! End-to-end ring marker detection pipeline.
//!
//! Stages:
//! 1. **Proposal** – gradient-voting radial symmetry to find candidate centers.
//! 2. **Edge sampling** – radial intensity profiles to locate inner/outer ring edges.
//! 3. **Ellipse fitting** – robust conic fit to edge points (uses `crate::conic`).
//! 4. **Decode** – ellipse-guided 16-sector sampling + codebook matching.
//! 5. **Detect** – orchestrates the above into `DetectionResult`.

pub(crate) mod decode;
pub(crate) mod detect;
pub(crate) mod edge_sample;
pub(crate) mod inner_estimate;
pub(crate) mod outer_estimate;
pub(crate) mod proposal;
pub(crate) mod radial_profile;
