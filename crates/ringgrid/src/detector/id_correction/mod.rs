//! Structural ID verification and correction using hex neighborhood consensus.
//!
//! This stage operates in image-space pixels (`DetectedMarker.center`) and
//! board-space millimeters (`BoardLayout` marker coordinates).
//!
//! ## Staged algorithm
//!
//! 1. Build board index and per-marker local scale (`ellipse_outer.mean_axis`).
//! 2. Bootstrap trusted anchors from decoded IDs.
//! 3. Pre-recovery consistency scrub (precision-first).
//! 4. Local iterative recovery with local-scale neighborhoods.
//! 5. Constrained rough-homography fallback seeding.
//! 6. Post-recovery consistency sweep + short local refill.
//! 7. Cleanup and deterministic conflict resolution.

mod bootstrap;
mod cleanup;
mod consistency;
mod diagnostics;
mod engine;
mod homography;
mod index;
mod local;
mod math;
mod types;
mod vote;
mod workspace;

pub(crate) use engine::verify_and_correct_ids;
pub(crate) use types::IdCorrectionStats;
