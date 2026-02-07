use super::*;

mod stage_finalize;
mod stage_fit_decode;

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> DetectionResult {
    let (w, h) = gray.dimensions();
    let markers = stage_fit_decode::run(gray, config, mapper, seed_centers_image, seed_cfg);
    stage_finalize::run(gray, markers, [w, h], config, mapper)
}

#[cfg(feature = "debug-trace")]
pub(super) fn run_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    seed_centers_image: &[[f32; 2]],
    seed_cfg: &SeedProposalParams,
) -> (DetectionResult, crate::debug_dump::DebugDumpV1) {
    let (w, h) = gray.dimensions();
    let fit_out = stage_fit_decode::run_with_debug(
        gray,
        config,
        mapper,
        seed_centers_image,
        seed_cfg,
        debug_cfg,
    );
    stage_finalize::run_with_debug(gray, fit_out, [w, h], config, mapper, debug_cfg)
}
