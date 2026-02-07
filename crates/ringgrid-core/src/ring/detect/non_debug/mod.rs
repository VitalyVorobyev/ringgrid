use super::*;

mod stage_finalize;
mod stage_fit_decode;

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
) -> DetectionResult {
    let (w, h) = gray.dimensions();
    let markers = stage_fit_decode::run(gray, config, mapper);
    stage_finalize::run(gray, markers, [w, h], config, mapper)
}
