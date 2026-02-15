use image::GrayImage;

use super::{
    apply_projective_centers, complete_with_h, compute_h_stats, global_filter, matrix3_to_array,
    mean_reproj_error_px, reapply_projective_centers, refine_with_homography,
    refit_homography_matrix, warn_center_correction_without_intrinsics, CompletionStats,
    DetectConfig,
};
use crate::detector::DetectedMarker;
use crate::homography::RansacStats;
use crate::pipeline::DetectionResult;
use crate::pixelmap::PixelMapper;

#[derive(Clone, Copy)]
struct FinalizeFlags {
    use_projective_center: bool,
}

struct FilterPhaseOutput {
    markers: Vec<DetectedMarker>,
    h_current: Option<nalgebra::Matrix3<f64>>,
    ransac_stats: Option<RansacStats>,
    short_circuit_no_h: bool,
}

fn finalize_flags(config: &DetectConfig) -> FinalizeFlags {
    FinalizeFlags {
        use_projective_center: config.circle_refinement.uses_projective_center()
            && config.projective_center.enable,
    }
}

fn phase_filter_and_refine_h(
    gray: &GrayImage,
    mut fit_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    flags: FinalizeFlags,
) -> FilterPhaseOutput {
    if flags.use_projective_center {
        apply_projective_centers(&mut fit_markers, config);
    }

    if !config.use_global_filter {
        return FilterPhaseOutput {
            markers: fit_markers,
            h_current: None,
            ransac_stats: None,
            short_circuit_no_h: true,
        };
    }

    let (filtered, h_result, stats) =
        global_filter(&fit_markers, &config.ransac_homography, &config.board);
    let h_current = h_result.as_ref().map(|r| r.h);
    let h_matrix = h_result.as_ref().map(|r| &r.h);

    let (mut markers, did_refine) = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                let refined =
                    refine_with_homography(gray, &filtered, h, config, &config.board, mapper);
                (refined, true)
            } else {
                (filtered, false)
            }
        } else {
            (filtered, false)
        }
    } else {
        (filtered, false)
    };

    if flags.use_projective_center && did_refine {
        reapply_projective_centers(&mut markers, config);
    }

    FilterPhaseOutput {
        markers,
        h_current,
        ransac_stats: stats,
        short_circuit_no_h: false,
    }
}

fn phase_completion(
    gray: &GrayImage,
    final_markers: &mut Vec<DetectedMarker>,
    h_current: Option<&nalgebra::Matrix3<f64>>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> CompletionStats {
    if !config.completion.enable {
        return CompletionStats::default();
    }

    let Some(h) = h_current else {
        return CompletionStats::default();
    };

    complete_with_h(gray, h, final_markers, config, &config.board, mapper)
}

fn phase_final_h(
    final_markers: &[DetectedMarker],
    h_current: Option<nalgebra::Matrix3<f64>>,
    mut ransac_stats: Option<RansacStats>,
    config: &DetectConfig,
) -> (Option<[[f64; 3]; 3]>, Option<RansacStats>) {
    let did_refit = config.refine_with_h && final_markers.len() >= 10;
    let final_h_matrix = if did_refit {
        let h_refit =
            refit_homography_matrix(final_markers, &config.ransac_homography, &config.board)
                .map(|(h, _)| h);
        match (h_current, h_refit) {
            (Some(h_cur), Some(h_new)) => {
                let cur_err = mean_reproj_error_px(&h_cur, final_markers, &config.board);
                let new_err = mean_reproj_error_px(&h_new, final_markers, &config.board);
                if new_err.is_finite() && (new_err < cur_err || !cur_err.is_finite()) {
                    Some(h_new)
                } else {
                    Some(h_cur)
                }
            }
            (None, Some(h_new)) => Some(h_new),
            (Some(h_cur), None) => Some(h_cur),
            (None, None) => None,
        }
    } else {
        h_current
    };

    let final_h = final_h_matrix.as_ref().map(matrix3_to_array);
    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| {
            compute_h_stats(
                h,
                final_markers,
                config.ransac_homography.inlier_threshold,
                &config.board,
            )
        })
        .or_else(|| ransac_stats.take());

    (final_h, final_ransac)
}

fn build_result(
    final_markers: Vec<DetectedMarker>,
    image_size: [u32; 2],
    final_h: Option<[[f64; 3]; 3]>,
    final_ransac: Option<RansacStats>,
) -> DetectionResult {
    DetectionResult {
        detected_markers: final_markers,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        self_undistort: None,
    }
}

pub(super) fn run(
    gray: &GrayImage,
    fit_out: super::fit_decode::FitDecodeCoreOutput,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];

    warn_center_correction_without_intrinsics(config, mapper.is_some());
    let flags = finalize_flags(config);

    let mut filter_phase = phase_filter_and_refine_h(gray, fit_out.markers, config, mapper, flags);

    if filter_phase.short_circuit_no_h {
        return build_result(filter_phase.markers, image_size, None, None);
    }

    let mut final_markers = filter_phase.markers;
    let h_current = filter_phase.h_current;
    let completion_stats =
        phase_completion(gray, &mut final_markers, h_current.as_ref(), config, mapper);

    if flags.use_projective_center {
        apply_projective_centers(&mut final_markers, config);
    }

    let (final_h, final_ransac) = phase_final_h(
        &final_markers,
        h_current,
        filter_phase.ransac_stats.take(),
        config,
    );

    tracing::info!(
        "{} markers after global filter{}",
        final_markers.len(),
        if config.refine_with_h {
            " + refinement"
        } else {
            ""
        }
    );
    tracing::debug!(
        "Completion summary: attempted={}, added={}, failed_fit={}, failed_gate={}",
        completion_stats.n_attempted,
        completion_stats.n_added,
        completion_stats.n_failed_fit,
        completion_stats.n_failed_gate
    );

    build_result(final_markers, image_size, final_h, final_ransac)
}
