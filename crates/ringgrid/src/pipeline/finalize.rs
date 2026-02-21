use image::GrayImage;

use super::{
    apply_projective_centers, complete_with_h, compute_h_stats, global_filter, matrix3_to_array,
    mean_reproj_error_px, refit_homography_matrix, verify_and_correct_ids,
    warn_center_correction_without_intrinsics, CompletionStats, DetectConfig,
};
use crate::detector::DetectedMarker;
use crate::homography::RansacStats;
use crate::pipeline::{DetectionFrame, DetectionResult};
use crate::pixelmap::PixelMapper;

struct FilterPhaseOutput {
    markers: Vec<DetectedMarker>,
    h_current: Option<nalgebra::Matrix3<f64>>,
    ransac_stats: Option<RansacStats>,
}

fn filter_with_h(fit_markers: Vec<DetectedMarker>, config: &DetectConfig) -> FilterPhaseOutput {
    if !config.use_global_filter {
        return FilterPhaseOutput {
            markers: fit_markers,
            h_current: None,
            ransac_stats: None,
        };
    }

    let (filtered, h_result, stats) =
        global_filter(&fit_markers, &config.ransac_homography, &config.board);
    let h_current = h_result.as_ref().map(|r| r.h);

    FilterPhaseOutput {
        markers: filtered,
        h_current,
        ransac_stats: stats,
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
    let final_h_matrix = if final_markers.len() >= 10 {
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

fn drop_unmappable_markers(markers: &mut Vec<DetectedMarker>, mapper: &dyn PixelMapper) -> usize {
    let before = markers.len();
    markers.retain(|m| {
        mapper
            .working_to_image_pixel(m.center)
            .map(|p| p[0].is_finite() && p[1].is_finite())
            .unwrap_or(false)
    });
    before.saturating_sub(markers.len())
}

fn map_centers_to_image(markers: &mut [DetectedMarker], mapper: &dyn PixelMapper) {
    for marker in markers.iter_mut() {
        let center_mapped = marker.center;
        marker.center_mapped = Some(center_mapped);
        if let Some(center_image) = mapper.working_to_image_pixel(center_mapped) {
            marker.center = center_image;
        }
    }
}

pub(super) fn run(
    gray: &GrayImage,
    fit_markers: Vec<DetectedMarker>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
) -> DetectionResult {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    let homography_frame = if mapper.is_some() {
        DetectionFrame::Working
    } else {
        DetectionFrame::Image
    };

    warn_center_correction_without_intrinsics(config, mapper.is_some());

    let mut corrected_markers = fit_markers;
    if config.projective_center.enable {
        apply_projective_centers(&mut corrected_markers, config);
    }

    if config.id_correction.enable {
        let stats =
            verify_and_correct_ids(&mut corrected_markers, &config.board, &config.id_correction);
        tracing::info!(
            n_ids_corrected = stats.n_ids_corrected,
            n_ids_recovered = stats.n_ids_recovered,
            n_ids_cleared = stats.n_ids_cleared,
            n_verified = stats.n_verified,
            n_iterations = stats.n_iterations,
            pitch_px = stats.pitch_px_estimated,
            "id_correction complete"
        );
    }

    if !config.use_global_filter {
        if let Some(mapper) = mapper {
            let dropped = drop_unmappable_markers(&mut corrected_markers, mapper);
            if dropped > 0 {
                tracing::warn!(
                    dropped,
                    "dropping markers whose mapped centers cannot be converted to image frame"
                );
            }
            map_centers_to_image(&mut corrected_markers, mapper);
        }
        return DetectionResult {
            detected_markers: corrected_markers,
            center_frame: DetectionFrame::Image,
            homography_frame,
            image_size,
            ..DetectionResult::default()
        };
    }

    let mut filter_phase = filter_with_h(corrected_markers, config);

    let mut final_markers = filter_phase.markers;
    let h_current = filter_phase.h_current;
    let n_before_completion = final_markers.len();
    let completion_stats =
        phase_completion(gray, &mut final_markers, h_current.as_ref(), config, mapper);

    if config.projective_center.enable && final_markers.len() > n_before_completion {
        apply_projective_centers(&mut final_markers[n_before_completion..], config);
    }

    if let Some(mapper) = mapper {
        let dropped = drop_unmappable_markers(&mut final_markers, mapper);
        if dropped > 0 {
            tracing::warn!(
                dropped,
                "dropping markers whose mapped centers cannot be converted to image frame"
            );
        }
    }

    let (final_h, final_ransac) = phase_final_h(
        &final_markers,
        h_current,
        filter_phase.ransac_stats.take(),
        config,
    );

    if let Some(mapper) = mapper {
        map_centers_to_image(&mut final_markers, mapper);
    }

    tracing::info!(
        "{} markers after global filter/completion",
        final_markers.len(),
    );
    tracing::debug!(
        "Completion summary: attempted={}, added={}, failed_fit={}, failed_gate={}",
        completion_stats.n_attempted,
        completion_stats.n_added,
        completion_stats.n_failed_fit,
        completion_stats.n_failed_gate
    );

    DetectionResult {
        detected_markers: final_markers,
        center_frame: DetectionFrame::Image,
        homography_frame,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        ..DetectionResult::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OffsetMapper;

    impl PixelMapper for OffsetMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some([image_xy[0] + 5.0, image_xy[1] + 7.0])
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some([working_xy[0] - 5.0, working_xy[1] - 7.0])
        }
    }

    struct RejectingMapper;

    impl PixelMapper for RejectingMapper {
        fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]> {
            Some(image_xy)
        }

        fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]> {
            if working_xy[0] > 50.0 {
                None
            } else {
                Some([working_xy[0], working_xy[1]])
            }
        }
    }

    fn marker(center: [f64; 2]) -> DetectedMarker {
        DetectedMarker {
            confidence: 1.0,
            center,
            ..DetectedMarker::default()
        }
    }

    fn no_global_filter_config() -> DetectConfig {
        DetectConfig {
            use_global_filter: false,
            projective_center: crate::ProjectiveCenterParams {
                enable: false,
                ..crate::ProjectiveCenterParams::default()
            },
            ..DetectConfig::default()
        }
    }

    #[test]
    fn no_mapper_keeps_image_space_centers() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([12.0, 22.0])];
        let result = run(&img, markers, &no_global_filter_config(), None);

        assert_eq!(result.center_frame, DetectionFrame::Image);
        assert_eq!(result.homography_frame, DetectionFrame::Image);
        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [12.0, 22.0]);
        assert!(result.detected_markers[0].center_mapped.is_none());
    }

    #[test]
    fn mapper_outputs_image_center_and_preserves_mapped_center() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([20.0, 30.0])];
        let mapper = OffsetMapper;
        let result = run(&img, markers, &no_global_filter_config(), Some(&mapper));

        assert_eq!(result.center_frame, DetectionFrame::Image);
        assert_eq!(result.homography_frame, DetectionFrame::Working);
        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [15.0, 23.0]);
        assert_eq!(result.detected_markers[0].center_mapped, Some([20.0, 30.0]));
    }

    #[test]
    fn mapper_drops_unmappable_centers() {
        let img = GrayImage::new(100, 80);
        let markers = vec![marker([10.0, 10.0]), marker([60.0, 10.0])];
        let mapper = RejectingMapper;
        let result = run(&img, markers, &no_global_filter_config(), Some(&mapper));

        assert_eq!(result.detected_markers.len(), 1);
        assert_eq!(result.detected_markers[0].center, [10.0, 10.0]);
        assert_eq!(result.detected_markers[0].center_mapped, Some([10.0, 10.0]));
    }
}
