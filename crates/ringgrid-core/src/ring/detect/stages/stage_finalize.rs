use super::super::debug_conv;
use super::super::*;

use crate::debug_dump as dbg;

pub(super) fn run(
    gray: &GrayImage,
    fit_out: super::stage_fit_decode::FitDecodeCoreOutput,
    image_size: [u32; 2],
    config: &DetectConfig,
    mapper: Option<&dyn crate::camera::PixelMapper>,
    debug_cfg: Option<&DebugCollectConfig>,
) -> (DetectionResult, Option<dbg::DebugDumpV1>) {
    let (w, h) = gray.dimensions();
    warn_center_correction_without_intrinsics(config, mapper.is_some());
    let use_projective_center =
        config.circle_refinement.uses_projective_center() && config.projective_center.enable;
    let use_nl_refine = config.circle_refinement.uses_nl_refine() && config.nl_refine.enabled;
    let collect_debug = debug_cfg.is_some();

    let mut final_markers;
    let mut refine_debug: Option<dbg::RefineDebugV1>;
    let mut h_current: Option<nalgebra::Matrix3<f64>>;
    let ransac_debug: Option<dbg::RansacDebugV1>;
    let mut ransac_stats: Option<RansacStats>;

    if !config.use_global_filter {
        final_markers = fit_out.markers;
        h_current = None;
        ransac_stats = None;
        refine_debug = None;
        ransac_debug = if collect_debug {
            Some(dbg::RansacDebugV1 {
                enabled: false,
                h_best: None,
                correspondences_used: 0,
                inlier_ids: Vec::new(),
                outlier_ids: Vec::new(),
                per_id_error_px: None,
                stats: dbg::RansacStatsDebugV1 {
                    iters: 0,
                    thresh_px: config.ransac_homography.inlier_threshold,
                    n_corr: 0,
                    n_inliers: 0,
                    mean_err_inliers: 0.0,
                    p95_err_inliers: 0.0,
                },
                notes: vec!["global_filter_disabled".to_string()],
            })
        } else {
            None
        };

        if !collect_debug {
            if use_projective_center {
                apply_projective_centers(&mut final_markers, config);
            }
            return (
                DetectionResult {
                    detected_markers: final_markers,
                    image_size,
                    homography: None,
                    ransac: None,
                    camera: config.camera,
                },
                None,
            );
        }
    } else if collect_debug {
        let (filtered, h_result, stats, rdbg) = global_filter_with_debug(
            &fit_out.markers,
            &fit_out.marker_cand_idx,
            &config.ransac_homography,
            &config.board,
        );
        ransac_debug = Some(rdbg);
        ransac_stats = stats;
        h_current = h_result.as_ref().map(|r| r.h);
        let h_matrix = h_result.as_ref().map(|r| &r.h);

        let (refined, rd) = if config.refine_with_h {
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
        final_markers = refined;
        refine_debug = rd;
    } else {
        // Non-debug global filter path
        let (filtered, h_result, stats) =
            global_filter(&fit_out.markers, &config.ransac_homography, &config.board);
        ransac_debug = None;
        ransac_stats = stats;
        h_current = h_result.as_ref().map(|r| r.h);
        let h_matrix = h_result.as_ref().map(|r| &r.h);

        final_markers = if config.refine_with_h {
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
        refine_debug = None;
    }

    // Stage 8: Homography-guided completion
    let store_points = debug_cfg.map(|cfg| cfg.store_points).unwrap_or(false);
    let (completion_stats, completion_attempts) = if config.completion.enable {
        if let Some(h) = h_current.as_ref() {
            complete_with_h(
                gray,
                h,
                &mut final_markers,
                config,
                &config.board,
                mapper,
                store_points,
                collect_debug,
            )
        } else {
            (
                CompletionStats::default(),
                if collect_debug {
                    Some(Vec::new())
                } else {
                    None
                },
            )
        }
    } else {
        (
            CompletionStats::default(),
            if collect_debug {
                Some(Vec::new())
            } else {
                None
            },
        )
    };

    // Stage 9: Non-linear refinement in board plane (optional).
    let mut nl_refine_debug: Option<dbg::NlRefineDebugV1> = if collect_debug {
        Some(dbg::NlRefineDebugV1 {
            enabled: use_nl_refine && h_current.is_some(),
            params: dbg::NlRefineParamsV1 {
                enabled: use_nl_refine,
                solver: match config.nl_refine.solver {
                    refine::CircleCenterSolver::Irls => dbg::NlRefineSolverV1::Irls,
                    refine::CircleCenterSolver::Lm => dbg::NlRefineSolverV1::Lm,
                },
                max_iters: config.nl_refine.max_iters,
                huber_delta_mm: config.nl_refine.huber_delta_mm,
                min_points: config.nl_refine.min_points,
                reject_shift_mm: config.nl_refine.reject_thresh_mm,
                enable_h_refit: config.nl_refine.enable_h_refit,
                h_refit_iters: config.nl_refine.h_refit_iters,
                marker_outer_radius_mm: config.board.marker_outer_radius_mm() as f64,
            },
            h_used: h_current.as_ref().map(matrix3_to_array),
            h_refit: None,
            stats: dbg::NlRefineStatsDebugV1 {
                n_inliers: 0,
                n_refined: 0,
                n_failed: 0,
                mean_before_mm: 0.0,
                mean_after_mm: 0.0,
                p95_before_mm: 0.0,
                p95_after_mm: 0.0,
            },
            refined_markers: Vec::new(),
            notes: Vec::new(),
        })
    } else {
        None
    };

    if use_nl_refine {
        if let Some(h0) = h_current {
            let (stats0, records0) = refine::refine_markers_circle_board_with_mapper(
                gray,
                &h0,
                &mut final_markers,
                &config.nl_refine,
                &config.board,
                mapper,
                store_points,
            );

            if let Some(nld) = nl_refine_debug.as_mut() {
                nld.stats = debug_conv::nl_stats(&stats0);
                nld.refined_markers = records0.into_iter().map(debug_conv::nl_record).collect();
            }

            if config.nl_refine.enable_h_refit && final_markers.len() >= 10 {
                let max_iters = config.nl_refine.h_refit_iters.clamp(1, 3);
                let mut h_prev = h0;
                let mut mean_prev = mean_reproj_error_px(&h_prev, &final_markers, &config.board);
                for iter in 0..max_iters {
                    let Some((h_next, _stats1)) = refit_homography_matrix(
                        &final_markers,
                        &config.ransac_homography,
                        &config.board,
                    ) else {
                        if let Some(nld) = nl_refine_debug.as_mut() {
                            nld.notes.push(format!("h_refit_iter{}:refit_failed", iter));
                        }
                        break;
                    };

                    let mean_next = mean_reproj_error_px(&h_next, &final_markers, &config.board);
                    if mean_next.is_finite() && (mean_next < mean_prev || !mean_prev.is_finite()) {
                        if let Some(nld) = nl_refine_debug.as_mut() {
                            nld.h_refit = Some(matrix3_to_array(&h_next));
                            nld.notes.push(format!(
                                "h_refit_iter{}:accepted mean_err_px {:.3} -> {:.3}",
                                iter, mean_prev, mean_next
                            ));
                        }

                        h_current = Some(h_next);
                        h_prev = h_next;
                        mean_prev = mean_next;

                        let (stats_i, records_i) = refine::refine_markers_circle_board_with_mapper(
                            gray,
                            &h_prev,
                            &mut final_markers,
                            &config.nl_refine,
                            &config.board,
                            mapper,
                            store_points,
                        );
                        if let Some(nld) = nl_refine_debug.as_mut() {
                            nld.stats = debug_conv::nl_stats(&stats_i);
                            nld.refined_markers =
                                records_i.into_iter().map(debug_conv::nl_record).collect();
                        }
                    } else {
                        if let Some(nld) = nl_refine_debug.as_mut() {
                            nld.notes.push(format!(
                                "h_refit_iter{}:rejected mean_err_px {:.3} -> {:.3}",
                                iter, mean_prev, mean_next
                            ));
                        }
                        break;
                    }
                }
            }
        } else {
            tracing::warn!(
                "nl_board center correction selected but homography is unavailable; \
                 keeping uncorrected centers"
            );
            if let Some(nld) = nl_refine_debug.as_mut() {
                nld.notes.push("skipped_no_homography".to_string());
            }
        }
    } else if let Some(nld) = nl_refine_debug.as_mut() {
        nld.notes
            .push("disabled_by_circle_refinement_method".to_string());
    }

    // Projective center correction
    if use_projective_center {
        apply_projective_centers(&mut final_markers, config);
    }

    // Final H: refit after refinement if enabled
    let did_refit = config.refine_with_h && final_markers.len() >= 10;
    let final_h_matrix = if did_refit {
        refit_homography_matrix(&final_markers, &config.ransac_homography, &config.board)
            .map(|(h, _stats)| h)
            .or(h_current)
    } else {
        h_current
    };
    let final_h = final_h_matrix.as_ref().map(matrix3_to_array);
    let final_ransac = final_h_matrix
        .as_ref()
        .and_then(|h| {
            compute_h_stats(
                h,
                &final_markers,
                config.ransac_homography.inlier_threshold,
                &config.board,
            )
        })
        .or(ransac_stats.take());

    if did_refit {
        if let Some(ref mut rd) = refine_debug {
            rd.h_refit = final_h;
        }
    }

    tracing::info!(
        "{} markers after global filter{}",
        final_markers.len(),
        if config.refine_with_h {
            " + refinement"
        } else {
            ""
        }
    );

    let result = DetectionResult {
        detected_markers: final_markers,
        image_size,
        homography: final_h,
        ransac: final_ransac,
        camera: config.camera,
    };

    let dump = if let Some(debug_cfg) = debug_cfg {
        use crate::codebook::{CODEBOOK_BITS, CODEBOOK_N};

        let completion_debug = dbg::CompletionDebugV1 {
            enabled: config.completion.enable && h_current.is_some(),
            params: dbg::CompletionParamsDebugV1 {
                roi_radius_px: config.completion.roi_radius_px,
                reproj_gate_px: config.completion.reproj_gate_px,
                min_fit_confidence: config.completion.min_fit_confidence,
                min_arc_coverage: config.completion.min_arc_coverage,
                max_attempts: config.completion.max_attempts,
                image_margin_px: config.completion.image_margin_px,
            },
            attempted: completion_attempts
                .unwrap_or_default()
                .into_iter()
                .map(debug_conv::completion_attempt)
                .collect(),
            stats: debug_conv::completion_stats(&completion_stats),
            notes: Vec::new(),
        };

        Some(dbg::DebugDumpV1 {
            schema_version: dbg::DEBUG_SCHEMA_V1.to_string(),
            image: dbg::ImageDebugV1 {
                path: debug_cfg.image_path.clone(),
                width: w,
                height: h,
            },
            board: dbg::BoardDebugV1 {
                pitch_mm: config.board.pitch_mm,
                board_mm: config.board.board_size_mm[0],
                board_size_mm: config.board.board_size_mm,
                marker_count: config.board.n_markers(),
                codebook_bits: CODEBOOK_BITS,
                codebook_n: CODEBOOK_N,
            },
            params: dbg::ParamsDebugV1 {
                marker_diameter_px: config.marker_diameter_px as f64,
                proposal: dbg::ProposalParamsV1 {
                    r_min: config.proposal.r_min,
                    r_max: config.proposal.r_max,
                    grad_threshold: config.proposal.grad_threshold,
                    nms_radius: config.proposal.nms_radius,
                    min_vote_frac: config.proposal.min_vote_frac,
                    accum_sigma: config.proposal.accum_sigma,
                },
                edge_sample: dbg::EdgeSampleParamsV1 {
                    n_rays: config.edge_sample.n_rays,
                    r_max: config.edge_sample.r_max,
                    r_min: config.edge_sample.r_min,
                    r_step: config.edge_sample.r_step,
                    min_ring_depth: config.edge_sample.min_ring_depth,
                    min_rays_with_ring: config.edge_sample.min_rays_with_ring,
                },
                outer_estimation: Some(dbg::OuterEstimationParamsV1 {
                    search_halfwidth_px: config.outer_estimation.search_halfwidth_px,
                    radial_samples: config.outer_estimation.radial_samples,
                    theta_samples: config.outer_estimation.theta_samples,
                    aggregator: config.outer_estimation.aggregator,
                    grad_polarity: config.outer_estimation.grad_polarity,
                    min_theta_coverage: config.outer_estimation.min_theta_coverage,
                    min_theta_consistency: config.outer_estimation.min_theta_consistency,
                    allow_two_hypotheses: config.outer_estimation.allow_two_hypotheses,
                    second_peak_min_rel: config.outer_estimation.second_peak_min_rel,
                    refine_halfwidth_px: config.outer_estimation.refine_halfwidth_px,
                }),
                decode: dbg::DecodeParamsV1 {
                    code_band_ratio: config.decode.code_band_ratio,
                    samples_per_sector: config.decode.samples_per_sector,
                    n_radial_rings: config.decode.n_radial_rings,
                    max_decode_dist: config.decode.max_decode_dist,
                    min_decode_confidence: config.decode.min_decode_confidence,
                },
                marker_spec: config.marker_spec.clone(),
                camera: config.camera,
                circle_refinement_method: Some(match config.circle_refinement {
                    CircleRefinementMethod::None => dbg::CircleRefinementMethodV1::None,
                    CircleRefinementMethod::ProjectiveCenter => {
                        dbg::CircleRefinementMethodV1::ProjectiveCenter
                    }
                    CircleRefinementMethod::NlBoard => dbg::CircleRefinementMethodV1::NlBoard,
                }),
                projective_center: Some(dbg::ProjectiveCenterParamsV1 {
                    enabled: config.projective_center.enable,
                    use_expected_ratio: config.projective_center.use_expected_ratio,
                    ratio_penalty_weight: config.projective_center.ratio_penalty_weight,
                    max_center_shift_px: config.projective_center.max_center_shift_px,
                    max_selected_residual: config.projective_center.max_selected_residual,
                    min_eig_separation: config.projective_center.min_eig_separation,
                }),
                nl_refine: nl_refine_debug.as_ref().map(|nld| nld.params.clone()),
                min_semi_axis: config.min_semi_axis,
                max_semi_axis: config.max_semi_axis,
                max_aspect_ratio: config.max_aspect_ratio,
                dedup_radius: config.dedup_radius,
                use_global_filter: config.use_global_filter,
                ransac_homography: dbg::RansacHomographyParamsV1 {
                    max_iters: config.ransac_homography.max_iters,
                    inlier_threshold: config.ransac_homography.inlier_threshold,
                    min_inliers: config.ransac_homography.min_inliers,
                    seed: config.ransac_homography.seed,
                },
                refine_with_h: config.refine_with_h,
                debug: dbg::DebugOptionsV1 {
                    max_candidates: debug_cfg.max_candidates,
                    store_points: debug_cfg.store_points,
                },
            },
            stages: dbg::StagesDebugV1 {
                stage0_proposals: fit_out.stage0.expect("stage0 debug should be present"),
                stage1_fit_decode: fit_out.stage1.expect("stage1 debug should be present"),
                stage2_dedup: fit_out.stage2.expect("stage2 debug should be present"),
                stage3_ransac: ransac_debug.expect("ransac debug should be present"),
                stage4_refine: refine_debug,
                stage5_completion: Some(completion_debug),
                stage6_nl_refine: nl_refine_debug,
                final_: dbg::FinalDebugV1 {
                    h_final: result.homography,
                    detections: result.detected_markers.clone(),
                    notes: if completion_stats.n_added > 0 {
                        vec![format!("completion_added={}", completion_stats.n_added)]
                    } else {
                        Vec::new()
                    },
                },
            },
        })
    } else {
        None
    };

    (result, dump)
}
