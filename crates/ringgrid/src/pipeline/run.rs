//! Top-level pipeline orchestrator: fit_decode → finalize.

use super::*;

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
    debug_cfg: Option<&DebugCollectConfig>,
) -> (DetectionResult, Option<crate::debug_dump::DebugDump>) {
    let (w, h) = gray.dimensions();
    let fit_out = super::fit_decode::run(
        gray,
        config,
        mapper,
        seed_centers_image,
        seed_cfg,
        debug_cfg,
    );
    super::finalize::run(gray, fit_out, [w, h], config, mapper, debug_cfg)
}
