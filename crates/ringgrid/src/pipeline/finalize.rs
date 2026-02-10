use image::GrayImage;

use super::{
    apply_projective_centers, complete_with_h, compute_h_stats, global_filter,
    global_filter_with_debug, matrix3_to_array, mean_reproj_error_px, refine_with_homography,
    refine_with_homography_with_debug, refit_homography_matrix,
    warn_center_correction_without_intrinsics, CompletionAttemptRecord, CompletionDebugOptions,
    CompletionStats, DebugCollectConfig, DetectConfig,
};
use crate::debug_dump as dbg;
use crate::detector::DetectedMarker;
use crate::homography::RansacStats;
use crate::pipeline::DetectionResult;
use crate::pixelmap::PixelMapper;

#[derive(Clone, Copy)]
struct FinalizeFlags {
    use_projective_center: bool,
    collect_debug: bool,
    store_points: bool,
}

struct FilterPhaseOutput {
    markers: Vec<DetectedMarker>,
    h_current: Option<nalgebra::Matrix3<f64>>,
    ransac_stats: Option<RansacStats>,
    ransac_debug: Option<dbg::RansacDebug>,
    refine_debug: Option<dbg::RefineDebug>,
    short_circuit_no_h_no_debug: bool,
}

struct CompletionPhaseOutput {
    stats: CompletionStats,
    attempts: Option<Vec<CompletionAttemptRecord>>,
    h_available: bool,
}

fn finalize_flags(config: &DetectConfig, debug_cfg: Option<&DebugCollectConfig>) -> FinalizeFlags {
    FinalizeFlags {
        use_projective_center: config.circle_refinement.uses_projective_center()
            && config.projective_center.enable,
        collect_debug: debug_cfg.is_some(),
        store_points: debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false),
    }
}

fn phase_filter_and_refine_h(
    gray: &GrayImage,
    fit_markers: Vec<DetectedMarker>,
    marker_cand_idx: &[usize],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    flags: FinalizeFlags,
) -> FilterPhaseOutput {
    if !config.use_global_filter {
        let ransac_debug = if flags.collect_debug {
            Some(dbg::RansacDebug {
                enabled: false,
                result: None,
                stats: None,
                correspondences_used: 0,
                inlier_ids: Vec::new(),
                outlier_ids: Vec::new(),
                per_id_error_px: Vec::new(),
                notes: vec!["global_filter_disabled".to_string()],
            })
        } else {
            None
        };

        return FilterPhaseOutput {
            markers: fit_markers,
            h_current: None,
            ransac_stats: None,
            ransac_debug,
            refine_debug: None,
            short_circuit_no_h_no_debug: !flags.collect_debug,
        };
    }

    if flags.collect_debug {
        let (filtered, h_result, stats, rdbg) = global_filter_with_debug(
            &fit_markers,
            marker_cand_idx,
            &config.ransac_homography,
            &config.board,
        );
        let mut h_current = h_result.as_ref().map(|r| r.h);
        let h_matrix = h_result.as_ref().map(|r| &r.h);

        let (markers, refine_debug) = if config.refine_with_h {
            if let Some(h) = h_matrix {
                if filtered.len() >= 10 {
                    let (refined, refine_dbg) = refine_with_homography_with_debug(
                        gray,
                        &filtered,
                        h,
                        config,
                        &config.board,
                        mapper,
                    );
                    (refined, Some(refine_dbg))
                } else {
                    (filtered, None)
                }
            } else {
                (filtered, None)
            }
        } else {
            (filtered, None)
        };

        if h_current.is_none() {
            h_current = h_result.map(|r| r.h);
        }

        return FilterPhaseOutput {
            markers,
            h_current,
            ransac_stats: stats,
            ransac_debug: Some(rdbg),
            refine_debug,
            short_circuit_no_h_no_debug: false,
        };
    }

    let (filtered, h_result, stats) =
        global_filter(&fit_markers, &config.ransac_homography, &config.board);
    let h_current = h_result.as_ref().map(|r| r.h);
    let h_matrix = h_result.as_ref().map(|r| &r.h);

    let markers = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                refine_with_homography(gray, &filtered, h, config, &config.board, mapper)
            } else {
                filtered
            }
        } else {
            filtered
        }
    } else {
        filtered
    };

    FilterPhaseOutput {
        markers,
        h_current,
        ransac_stats: stats,
        ransac_debug: None,
        refine_debug: None,
        short_circuit_no_h_no_debug: false,
    }
}

fn phase_completion(
    gray: &GrayImage,
    final_markers: &mut Vec<DetectedMarker>,
    h_current: Option<&nalgebra::Matrix3<f64>>,
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    flags: FinalizeFlags,
) -> CompletionPhaseOutput {
    if !config.completion.enable {
        return CompletionPhaseOutput {
            stats: CompletionStats::default(),
            attempts: if flags.collect_debug {
                Some(Vec::new())
            } else {
                None
            },
            h_available: h_current.is_some(),
        };
    }

    let Some(h) = h_current else {
        return CompletionPhaseOutput {
            stats: CompletionStats::default(),
            attempts: if flags.collect_debug {
                Some(Vec::new())
            } else {
                None
            },
            h_available: false,
        };
    };

    let (stats, attempts) = complete_with_h(
        gray,
        h,
        final_markers,
        config,
        &config.board,
        mapper,
        CompletionDebugOptions {
            store_points: flags.store_points,
            record: flags.collect_debug,
        },
    );

    CompletionPhaseOutput {
        stats,
        attempts,
        h_available: true,
    }
}

fn phase_center_correction(
    final_markers: &mut [DetectedMarker],
    config: &DetectConfig,
    flags: FinalizeFlags,
) {
    if flags.use_projective_center {
        apply_projective_centers(final_markers, config);
    }
}

fn phase_final_h(
    final_markers: &[DetectedMarker],
    h_current: Option<nalgebra::Matrix3<f64>>,
    mut ransac_stats: Option<RansacStats>,
    config: &DetectConfig,
    refine_debug: &mut Option<dbg::RefineDebug>,
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

    if did_refit {
        if let Some(rd) = refine_debug.as_mut() {
            rd.h_refit = final_h;
        }
    }

    (final_h, final_ransac)
}

fn build_result(
    final_markers: Vec<DetectedMarker>,
    image_size: [u32; 2],
    final_h: Option<[[f64; 3]; 3]>,
    final_ransac: Option<RansacStats>,
    config: &DetectConfig,
) -> DetectionResult {
    DetectionResult {
        detected_markers: final_markers,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        camera: config.camera,
        self_undistort: None,
    }
}

struct DumpBuildInput<'a> {
    debug_cfg: &'a DebugCollectConfig,
    config: &'a DetectConfig,
    image_size: [u32; 2],
    stage0: Option<dbg::StageDebug>,
    stage1: Option<dbg::StageDebug>,
    stage2: Option<dbg::DedupDebug>,
    ransac_debug: dbg::RansacDebug,
    refine_debug: Option<dbg::RefineDebug>,
    completion: CompletionPhaseOutput,
    result: &'a DetectionResult,
}

fn build_debug_dump(input: DumpBuildInput<'_>) -> dbg::DebugDump {
    let DumpBuildInput {
        debug_cfg,
        config,
        image_size,
        stage0,
        stage1,
        stage2,
        ransac_debug,
        refine_debug,
        completion,
        result,
    } = input;
    let completion_added = completion.stats.n_added;
    let completion_debug = dbg::CompletionDebug {
        enabled: config.completion.enable && completion.h_available,
        params: config.completion.clone(),
        attempted: completion.attempts.unwrap_or_default(),
        stats: completion.stats,
        notes: Vec::new(),
    };

    dbg::DebugDump {
        schema_version: dbg::DEBUG_SCHEMA_V7.to_string(),
        image: dbg::ImageDebug {
            path: debug_cfg.image_path.clone(),
            width: image_size[0],
            height: image_size[1],
        },
        detect_config: dbg::DetectConfigSnapshot::from_config(config, debug_cfg),
        stages: dbg::StagesDebug {
            stage0_proposals: stage0.expect("stage0 debug should be present"),
            stage1_fit_decode: stage1.expect("stage1 debug should be present"),
            stage2_dedup: stage2.expect("stage2 debug should be present"),
            stage3_ransac: ransac_debug,
            stage4_refine: refine_debug,
            stage5_completion: Some(completion_debug),
            final_: dbg::FinalDebug {
                h_final: result.homography,
                detections: result.detected_markers.clone(),
                notes: if completion_added > 0 {
                    vec![format!("completion_added={}", completion_added)]
                } else {
                    Vec::new()
                },
            },
        },
    }
}

pub(super) fn run(
    gray: &GrayImage,
    fit_out: super::fit_decode::FitDecodeCoreOutput,
    image_size: [u32; 2],
    config: &DetectConfig,
    mapper: Option<&dyn PixelMapper>,
    debug_cfg: Option<&DebugCollectConfig>,
) -> (DetectionResult, Option<dbg::DebugDump>) {
    warn_center_correction_without_intrinsics(config, mapper.is_some());
    let flags = finalize_flags(config, debug_cfg);

    let super::fit_decode::FitDecodeCoreOutput {
        markers: fit_markers,
        marker_cand_idx,
        stage0,
        stage1,
        stage2,
    } = fit_out;

    let mut filter_phase =
        phase_filter_and_refine_h(gray, fit_markers, &marker_cand_idx, config, mapper, flags);

    if filter_phase.short_circuit_no_h_no_debug {
        phase_center_correction(&mut filter_phase.markers, config, flags);
        let result = build_result(filter_phase.markers, image_size, None, None, config);
        return (result, None);
    }

    let mut final_markers = filter_phase.markers;
    let h_current = filter_phase.h_current;
    let mut refine_debug = filter_phase.refine_debug;

    let completion = phase_completion(
        gray,
        &mut final_markers,
        h_current.as_ref(),
        config,
        mapper,
        flags,
    );

    phase_center_correction(&mut final_markers, config, flags);

    let (final_h, final_ransac) = phase_final_h(
        &final_markers,
        h_current,
        filter_phase.ransac_stats.take(),
        config,
        &mut refine_debug,
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

    let result = build_result(final_markers, image_size, final_h, final_ransac, config);

    let dump = if let Some(debug_cfg) = debug_cfg {
        Some(build_debug_dump(DumpBuildInput {
            debug_cfg,
            config,
            image_size,
            stage0,
            stage1,
            stage2,
            ransac_debug: filter_phase
                .ransac_debug
                .expect("ransac debug should be present when debug is enabled"),
            refine_debug,
            completion,
            result: &result,
        }))
    } else {
        None
    };

    (result, dump)
}
