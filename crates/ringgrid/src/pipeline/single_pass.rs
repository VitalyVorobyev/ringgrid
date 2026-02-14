use image::GrayImage;

use crate::debug_dump as dbg;
use crate::detector::proposal::find_proposals;
use crate::detector::{DebugCollectConfig, DetectConfig};
use crate::pixelmap::PixelMapper;

use super::run;
use super::DetectionResult;

/// Single-pass detection. Mapper (if provided) is used for distortion-aware
/// sampling within the single pass — it does NOT trigger two-pass detection.
pub(crate) fn detect_single_pass(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    let proposals = find_proposals(gray, &config.proposal);
    run::run(gray, config, mapper, proposals, None).0
}

/// Single-pass detection with debug dump collection.
pub fn detect_single_pass_with_debug(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (DetectionResult, dbg::DebugDump) {
    let proposals = find_proposals(gray, &config.proposal);
    let (result, dump) = run::run(gray, config, mapper, proposals, Some(debug_cfg));
    (
        result,
        dump.expect("debug dump present when debug_cfg is provided"),
    )
}
