//! Top-level pipeline orchestrator: fit_decode → finalize.

use super::*;
use crate::detector::proposal::Proposal;

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
    proposals: Vec<Proposal>,
    debug_cfg: Option<&DebugCollectConfig>,
) -> (DetectionResult, Option<crate::debug_dump::DebugDump>) {
    let (w, h) = gray.dimensions();
    let fit_out = super::fit_decode::run(gray, config, mapper, proposals, debug_cfg);
    super::finalize::run(gray, fit_out, [w, h], config, mapper, debug_cfg)
}
