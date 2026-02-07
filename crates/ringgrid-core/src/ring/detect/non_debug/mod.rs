use super::*;

mod stage_finalize;
mod stage_fit_decode;

pub(super) fn run(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let (w, h) = gray.dimensions();
    let markers = stage_fit_decode::run(gray, config);
    stage_finalize::run(gray, markers, [w, h], config)
}
