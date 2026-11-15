//! Photometric centre refinement: the pipeline's last stage.
//!
//! Each public entry point in [`super::run`] finishes its result here, exactly
//! once per call. The per-marker fits cost milliseconds each, so the stage runs
//! on the finished result only — after every pass, scale tier and adaptive
//! candidate has been decided — never inside them.

use super::time_compat::{Instant, duration_ms};
use super::*;
use crate::detector::photometric_refine::{
    PhotometricResidualMarker, apply_photometric_centers, mark_unrefined,
};
use crate::pixelmap::PixelMapper;

#[cfg(test)]
thread_local! {
    /// Times the stage ran on this thread (tests only: the "exactly once per
    /// public call" contract).
    pub(super) static RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Refine the centres of a finished result when the configuration asks for
/// it; otherwise return it unchanged.
///
/// `mapper` is the caller's external mapper, if any; a self-undistort model
/// the result applied is used when there is none. `residuals`, when given,
/// receives every fitted marker's per-pixel residuals.
pub(super) fn finish(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    mut result: PipelineResult,
    residuals: Option<&mut Vec<PhotometricResidualMarker>>,
) -> PipelineResult {
    if !config.circle_refinement.uses_photometric() {
        return result;
    }
    #[cfg(test)]
    RUNS.with(|n| n.set(n.get() + 1));

    let start = Instant::now();
    refine(gray, config, mapper, &mut result, residuals);
    // The stage is part of the end-to-end total.
    if let Some(timings) = result.timings.as_mut() {
        timings.total_ms += duration_ms(start.elapsed());
    }
    result
}

fn refine(
    gray: &GrayImage,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    result: &mut PipelineResult,
    residuals: Option<&mut Vec<PhotometricResidualMarker>>,
) {
    let self_undistort = result
        .self_undistort
        .as_ref()
        .filter(|s| s.applied)
        .map(|s| s.model);
    let mapper: Option<&dyn PixelMapper> = match result.homography_frame {
        DetectionFrame::Image => None,
        DetectionFrame::Working => {
            match mapper.or(self_undistort.as_ref().map(|m| m as &dyn PixelMapper)) {
                Some(m) => Some(m),
                None => {
                    // No way back to image pixels: nothing can be placed.
                    mark_unrefined(&mut result.markers);
                    return;
                }
            }
        }
    };
    let h = result.homography;
    result.photometric_print_radii_mm = apply_photometric_centers(
        gray,
        &mut result.markers,
        h.as_ref(),
        result.board_frame,
        mapper,
        config,
        residuals,
    );
}
