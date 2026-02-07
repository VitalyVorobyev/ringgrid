use super::super::*;

pub(super) fn run(
    gray: &GrayImage,
    markers: Vec<DetectedMarker>,
    image_size: [u32; 2],
    config: &DetectConfig,
) -> DetectionResult {
    warn_center_correction_without_intrinsics(config);
    let use_nl_refine = config.circle_refinement.uses_nl_refine() && config.nl_refine.enabled;
    let use_projective_center =
        config.circle_refinement.uses_projective_center() && config.projective_center.enable;

    // Stage 6: Global homography filtering
    if !config.use_global_filter {
        let mut markers = markers;
        if use_projective_center {
            apply_projective_centers(&mut markers, config);
        }
        return DetectionResult {
            detected_markers: markers,
            image_size,
            homography: None,
            ransac: None,
            camera: config.camera,
        };
    }

    let (filtered, h_result, ransac_stats) = global_filter(&markers, &config.ransac_homography);
    let h_matrix = h_result.as_ref().map(|r| &r.h);

    // Stage 7: Optional refinement using H
    let mut final_markers = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                refine_with_homography(gray, &filtered, h, config)
            } else {
                filtered
            }
        } else {
            filtered
        }
    } else {
        filtered
    };

    // Stage 8: Homography-guided completion (only when H exists)
    if config.completion.enable {
        if let Some(h) = h_matrix {
            let (_stats, _attempts) =
                complete_with_h(gray, h, &mut final_markers, config, false, false);
        }
    }

    // Stage 9: Non-linear refinement in board plane (optional).
    let mut h_current: Option<nalgebra::Matrix3<f64>> = h_result.as_ref().map(|r| r.h);
    if use_nl_refine {
        if let Some(h0) = h_current {
            let _ = refine::refine_markers_circle_board_with_camera(
                gray,
                &h0,
                &mut final_markers,
                &config.nl_refine,
                config.camera.as_ref(),
                false,
            );

            if config.nl_refine.enable_h_refit && final_markers.len() >= 10 {
                let max_iters = config.nl_refine.h_refit_iters.clamp(1, 3);
                let mut h_prev = h0;
                let mut mean_prev = mean_reproj_error_px(&h_prev, &final_markers);
                for _ in 0..max_iters {
                    let Some((h_next, _stats1)) =
                        refit_homography_matrix(&final_markers, &config.ransac_homography)
                    else {
                        break;
                    };

                    let mean_next = mean_reproj_error_px(&h_next, &final_markers);
                    if mean_next.is_finite() && (mean_next < mean_prev || !mean_prev.is_finite()) {
                        h_current = Some(h_next);
                        h_prev = h_next;
                        mean_prev = mean_next;

                        let _ = refine::refine_markers_circle_board_with_camera(
                            gray,
                            &h_prev,
                            &mut final_markers,
                            &config.nl_refine,
                            config.camera.as_ref(),
                            false,
                        );
                    } else {
                        break;
                    }
                }
            }
        } else {
            tracing::warn!(
                "nl_board center correction selected but homography is unavailable; \
                 keeping uncorrected centers"
            );
        }
    }

    // Projective center correction must be reflected in the final H statistics/refit.
    if use_projective_center {
        apply_projective_centers(&mut final_markers, config);
    }

    // Final H: refit after refinement if enabled (or keep the original RANSAC H).
    let final_h_matrix = if config.refine_with_h && final_markers.len() >= 10 {
        refit_homography_matrix(&final_markers, &config.ransac_homography)
            .map(|(h, _stats)| h)
            .or(h_current)
    } else {
        h_current
    };
    let final_h = final_h_matrix.as_ref().map(matrix3_to_array);
    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| compute_h_stats(h, &final_markers, config.ransac_homography.inlier_threshold))
        .or(ransac_stats);

    tracing::info!(
        "{} markers after global filter{}",
        final_markers.len(),
        if config.refine_with_h {
            " + refinement"
        } else {
            ""
        }
    );

    DetectionResult {
        detected_markers: final_markers,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        camera: config.camera,
    }
}
