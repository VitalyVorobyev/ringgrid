use super::*;
use crate::debug_dump as dbg;
use crate::ring::inner_estimate::Polarity;
use crate::ring::outer_estimate::{OuterGradPolarity, OuterStatus};

pub(super) fn run(
    gray: &GrayImage,
    config: &DetectConfig,
    debug_cfg: &DebugCollectConfig,
) -> (DetectionResult, dbg::DebugDumpV1) {
    use crate::board_spec::{BOARD_N, BOARD_PITCH_MM, BOARD_SIZE_MM};
    use crate::codebook::{CODEBOOK_BITS, CODEBOOK_N};

    let (w, h) = gray.dimensions();

    // Stage 0: proposals
    let proposals = find_proposals(gray, &config.proposal);

    let n_rec = proposals.len().min(debug_cfg.max_candidates);
    let mut stage0 = dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    };
    for (i, p) in proposals.iter().take(n_rec).enumerate() {
        stage0.candidates.push(dbg::CandidateDebugV1 {
            cand_idx: i,
            proposal: dbg::ProposalDebugV1 {
                center_xy: [p.x, p.y],
                score: p.score,
            },
            ring_fit: None,
            decode: None,
            decision: dbg::DecisionDebugV1 {
                status: dbg::DecisionStatusV1::Accepted,
                reason: "proposal".to_string(),
            },
            derived: dbg::DerivedDebugV1 {
                id: None,
                confidence: None,
                center_xy: None,
            },
        });
    }

    // Stage 1: per-proposal fit + decode
    let mut stage1 = dbg::StageDebugV1 {
        n_total: proposals.len(),
        n_recorded: n_rec,
        candidates: Vec::with_capacity(n_rec),
        notes: Vec::new(),
    };

    let mut markers: Vec<DetectedMarker> = Vec::new();
    let mut marker_cand_idx: Vec<usize> = Vec::new(); // parallel to markers

    for (i, proposal) in proposals.iter().enumerate() {
        let mut cand_debug = if i < n_rec {
            Some(dbg::CandidateDebugV1 {
                cand_idx: i,
                proposal: dbg::ProposalDebugV1 {
                    center_xy: [proposal.x, proposal.y],
                    score: proposal.score,
                },
                ring_fit: None,
                decode: None,
                decision: dbg::DecisionDebugV1 {
                    status: dbg::DecisionStatusV1::Rejected,
                    reason: "unprocessed".to_string(),
                },
                derived: dbg::DerivedDebugV1 {
                    id: None,
                    confidence: None,
                    center_xy: None,
                },
            })
        } else {
            None
        };

        let fit = match fit_outer_ellipse_robust_with_reason(
            gray,
            [proposal.x, proposal.y],
            marker_outer_radius_expected_px(config),
            config,
            &config.edge_sample,
            debug_cfg.store_points,
        ) {
            Ok(v) => v,
            Err(reason) => {
                if let Some(cd) = cand_debug.as_mut() {
                    cd.decision = dbg::DecisionDebugV1 {
                        status: dbg::DecisionStatusV1::Rejected,
                        reason,
                    };
                }
                if let Some(cd) = cand_debug {
                    stage1.candidates.push(cd);
                }
                continue;
            }
        };

        let OuterFitCandidate {
            edge,
            outer,
            outer_ransac,
            outer_estimate,
            chosen_hypothesis,
            decode_result,
            decode_diag,
            ..
        } = fit;

        let center = compute_center(&outer);

        let fit_metrics = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: 0,
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: None,
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: None,
        };

        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let derived_id = decode_result.as_ref().map(|d| d.id);

        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let inner_est = if decode_result.is_some() {
            Some(estimate_inner_scale_from_outer(
                gray,
                &outer,
                &config.marker_spec,
                debug_cfg.store_points,
            ))
        } else {
            None
        };
        let inner_params = inner_est.as_ref().and_then(|est| {
            if est.status == InnerStatus::Ok {
                let s = est
                    .r_inner_found
                    .unwrap_or(config.marker_spec.r_inner_expected) as f64;
                Some(EllipseParams {
                    center_xy: [outer.cx, outer.cy],
                    semi_axes: [outer.a * s, outer.b * s],
                    angle: outer.angle,
                })
            } else {
                None
            }
        });

        let marker = DetectedMarker {
            id: derived_id,
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner_params.clone(),
            fit: fit_metrics.clone(),
            decode: decode_metrics,
        };

        markers.push(marker);
        marker_cand_idx.push(i);

        if let Some(cd) = cand_debug.as_mut() {
            let arc_cov = (edge.n_good_rays as f32) / (edge.n_total_rays.max(1) as f32);
            cd.ring_fit = Some(dbg::RingFitDebugV1 {
                center_xy_fit: [center[0] as f32, center[1] as f32],
                edges: dbg::RingEdgesDebugV1 {
                    n_angles_total: edge.n_total_rays,
                    n_angles_with_both: edge.n_good_rays,
                    inner_peak_r: if edge.inner_radii.is_empty() {
                        None
                    } else {
                        Some(edge.inner_radii.clone())
                    },
                    outer_peak_r: Some(edge.outer_radii.clone()),
                },
                outer_estimation: Some({
                    let chosen = outer_estimate.hypotheses.get(chosen_hypothesis);
                    dbg::OuterEstimationDebugV1 {
                        r_outer_expected_px: outer_estimate.r_outer_expected_px,
                        search_window_px: outer_estimate.search_window_px,
                        r_outer_found_px: chosen.map(|h| h.r_outer_px),
                        polarity: outer_estimate.polarity.map(|p| match p {
                            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                        }),
                        peak_strength: chosen.map(|h| h.peak_strength),
                        theta_consistency: chosen.map(|h| h.theta_consistency),
                        status: match outer_estimate.status {
                            OuterStatus::Ok => dbg::OuterEstimationStatusDebugV1::Ok,
                            OuterStatus::Rejected => dbg::OuterEstimationStatusDebugV1::Rejected,
                            OuterStatus::Failed => dbg::OuterEstimationStatusDebugV1::Failed,
                        },
                        reason: outer_estimate.reason.clone(),
                        hypotheses: outer_estimate
                            .hypotheses
                            .iter()
                            .map(|h| dbg::OuterHypothesisDebugV1 {
                                r_outer_px: h.r_outer_px,
                                peak_strength: h.peak_strength,
                                theta_consistency: h.theta_consistency,
                            })
                            .collect(),
                        chosen_hypothesis: Some(chosen_hypothesis),
                        radial_response_agg: outer_estimate.radial_response_agg.clone(),
                        r_samples: outer_estimate.r_samples.clone(),
                    }
                }),
                ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                    center_xy: [outer.cx as f32, outer.cy as f32],
                    semi_axes: [outer.a as f32, outer.b as f32],
                    angle: outer.angle as f32,
                }),
                ellipse_inner: inner_params.as_ref().map(|p| dbg::EllipseParamsDebugV1 {
                    center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
                    semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
                    angle: p.angle as f32,
                }),
                inner_estimation: inner_est.as_ref().map(|est| dbg::InnerEstimationDebugV1 {
                    r_inner_expected: est.r_inner_expected,
                    search_window: est.search_window,
                    r_inner_found: est.r_inner_found,
                    polarity: est.polarity.map(|p| match p {
                        Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
                        Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
                    }),
                    peak_strength: est.peak_strength,
                    theta_consistency: est.theta_consistency,
                    status: match est.status {
                        InnerStatus::Ok => dbg::InnerEstimationStatusDebugV1::Ok,
                        InnerStatus::Rejected => dbg::InnerEstimationStatusDebugV1::Rejected,
                        InnerStatus::Failed => dbg::InnerEstimationStatusDebugV1::Failed,
                    },
                    reason: est.reason.clone(),
                    radial_response_agg: est.radial_response_agg.clone(),
                    r_samples: est.r_samples.clone(),
                }),
                metrics: dbg::RingFitMetricsDebugV1 {
                    inlier_ratio_inner: fit_metrics.ransac_inlier_ratio_inner,
                    inlier_ratio_outer: fit_metrics.ransac_inlier_ratio_outer,
                    mean_resid_inner: fit_metrics.rms_residual_inner.map(|v| v as f32),
                    mean_resid_outer: fit_metrics.rms_residual_outer.map(|v| v as f32),
                    arc_coverage: arc_cov,
                    valid_inner: inner_params.is_some(),
                    valid_outer: true,
                },
                points_outer: if debug_cfg.store_points {
                    Some(
                        edge.outer_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
                points_inner: if debug_cfg.store_points {
                    Some(
                        edge.inner_points
                            .iter()
                            .map(|p| [p[0] as f32, p[1] as f32])
                            .collect(),
                    )
                } else {
                    None
                },
            });

            cd.decode = Some(dbg::DecodeDebugV1 {
                sector_means: decode_diag.sector_intensities,
                threshold: decode_diag.threshold,
                observed_word_hex: format!("0x{:04X}", decode_diag.used_word),
                inverted_used: decode_diag.inverted_used,
                r#match: dbg::DecodeMatchDebugV1 {
                    best_id: decode_diag.best_id,
                    best_rotation: decode_diag.best_rotation,
                    best_dist: decode_diag.best_dist,
                    margin: decode_diag.margin,
                    decode_confidence: decode_diag.decode_confidence,
                },
                accepted: Some(decode_result.is_some()),
                reject_reason: decode_diag.reject_reason.clone(),
            });

            cd.decision = dbg::DecisionDebugV1 {
                status: dbg::DecisionStatusV1::Accepted,
                reason: if let Some(r) = decode_diag.reject_reason {
                    format!("ok_with_decode_reject:{}", r)
                } else {
                    "ok".to_string()
                },
            };
            cd.derived = dbg::DerivedDebugV1 {
                id: derived_id,
                confidence: Some(confidence),
                center_xy: Some([center[0] as f32, center[1] as f32]),
            };
        }

        if let Some(cd) = cand_debug {
            stage1.candidates.push(cd);
        }
    }

    // Stage 2: dedup (proximity + id)
    let (markers_dedup, cand_idx_dedup, dedup_debug) =
        dedup_with_debug(markers, marker_cand_idx, config.dedup_radius);

    // Stage 3: global filter
    let (filtered, h_result, ransac_stats, ransac_debug) = if !config.use_global_filter {
        (
            markers_dedup,
            None,
            None,
            dbg::RansacDebugV1 {
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
            },
        )
    } else {
        global_filter_with_debug(&markers_dedup, &cand_idx_dedup, &config.ransac_homography)
    };

    let h_matrix = h_result.as_ref().map(|r| &r.h);

    // Stage 4: refine (optional)
    let (mut final_markers, mut refine_debug) = if config.refine_with_h {
        if let Some(h) = h_matrix {
            if filtered.len() >= 10 {
                let (refined, refine_dbg) =
                    refine_with_homography_with_debug(gray, &filtered, h, config);
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

    // Stage 5: completion (optional, only when H exists)
    let (completion_stats, completion_attempts) = if config.completion.enable {
        if let Some(h) = h_matrix {
            complete_with_h(
                gray,
                h,
                &mut final_markers,
                config,
                debug_cfg.store_points,
                true,
            )
        } else {
            (CompletionStats::default(), Some(Vec::new()))
        }
    } else {
        (CompletionStats::default(), Some(Vec::new()))
    };

    let completion_debug = dbg::CompletionDebugV1 {
        enabled: config.completion.enable && h_matrix.is_some(),
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
            .map(|a| dbg::CompletionAttemptDebugV1 {
                id: a.id,
                projected_center_xy: a.projected_center_xy,
                status: match a.status {
                    CompletionAttemptStatus::Added => dbg::CompletionAttemptStatusDebugV1::Added,
                    CompletionAttemptStatus::SkippedPresent => {
                        dbg::CompletionAttemptStatusDebugV1::SkippedPresent
                    }
                    CompletionAttemptStatus::SkippedOob => {
                        dbg::CompletionAttemptStatusDebugV1::SkippedOob
                    }
                    CompletionAttemptStatus::FailedFit => {
                        dbg::CompletionAttemptStatusDebugV1::FailedFit
                    }
                    CompletionAttemptStatus::FailedGate => {
                        dbg::CompletionAttemptStatusDebugV1::FailedGate
                    }
                },
                reason: a.reason,
                reproj_err_px: a.reproj_err_px,
                fit_confidence: a.fit_confidence,
                fit: a.fit,
            })
            .collect(),
        stats: dbg::CompletionStatsDebugV1 {
            n_candidates_total: completion_stats.n_candidates_total,
            n_in_image: completion_stats.n_in_image,
            n_attempted: completion_stats.n_attempted,
            n_added: completion_stats.n_added,
            n_failed_fit: completion_stats.n_failed_fit,
            n_failed_gate: completion_stats.n_failed_gate,
        },
        notes: Vec::new(),
    };

    // Stage 6: Non-linear refinement in board plane (optional).
    let mut h_current: Option<nalgebra::Matrix3<f64>> = h_result.as_ref().map(|r| r.h);
    let mut nl_refine_debug = dbg::NlRefineDebugV1 {
        enabled: config.nl_refine.enabled && h_current.is_some(),
        params: dbg::NlRefineParamsV1 {
            enabled: config.nl_refine.enabled,
            max_iters: config.nl_refine.max_iters,
            huber_delta_mm: config.nl_refine.huber_delta_mm,
            min_points: config.nl_refine.min_points,
            reject_shift_mm: config.nl_refine.reject_thresh_mm,
            enable_h_refit: config.nl_refine.enable_h_refit,
            h_refit_iters: config.nl_refine.h_refit_iters,
            marker_outer_radius_mm: crate::board_spec::marker_outer_radius_mm() as f64,
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
    };

    if config.nl_refine.enabled {
        if let Some(h0) = h_current {
            let (stats0, records0) = refine::refine_markers_circle_board(
                gray,
                &h0,
                &mut final_markers,
                &config.nl_refine,
                debug_cfg.store_points,
            );

            nl_refine_debug.stats = dbg::NlRefineStatsDebugV1 {
                n_inliers: stats0.n_inliers,
                n_refined: stats0.n_refined,
                n_failed: stats0.n_failed,
                mean_before_mm: stats0.mean_before_mm,
                mean_after_mm: stats0.mean_after_mm,
                p95_before_mm: stats0.p95_before_mm,
                p95_after_mm: stats0.p95_after_mm,
            };
            nl_refine_debug.refined_markers = records0
                .into_iter()
                .map(|r| dbg::NlRefinedMarkerDebugV1 {
                    id: r.id,
                    n_points: r.n_points,
                    init_center_board_mm: r.init_center_board_mm,
                    refined_center_board_mm: r.refined_center_board_mm,
                    center_img_before: r.center_img_before,
                    center_img_after: r.center_img_after,
                    before_rms_mm: r.before_rms_mm,
                    after_rms_mm: r.after_rms_mm,
                    delta_center_mm: r.delta_center_mm,
                    edge_points_img: r.edge_points_img,
                    edge_points_board_mm: r.edge_points_board_mm,
                    status: match r.status {
                        refine::MarkerRefineStatus::Ok => dbg::NlRefineStatusDebugV1::Ok,
                        refine::MarkerRefineStatus::Rejected => {
                            dbg::NlRefineStatusDebugV1::Rejected
                        }
                        refine::MarkerRefineStatus::Failed => dbg::NlRefineStatusDebugV1::Failed,
                        refine::MarkerRefineStatus::Skipped => dbg::NlRefineStatusDebugV1::Skipped,
                    },
                    reason: r.reason,
                })
                .collect();

            if config.nl_refine.enable_h_refit && final_markers.len() >= 10 {
                let max_iters = config.nl_refine.h_refit_iters.clamp(1, 3);
                let mut h_prev = h0;
                let mut mean_prev = mean_reproj_error_px(&h_prev, &final_markers);
                for iter in 0..max_iters {
                    let Some((h_next, _stats1)) =
                        refit_homography_matrix(&final_markers, &config.ransac_homography)
                    else {
                        nl_refine_debug
                            .notes
                            .push(format!("h_refit_iter{}:refit_failed", iter));
                        break;
                    };

                    let mean_next = mean_reproj_error_px(&h_next, &final_markers);
                    if mean_next.is_finite() && (mean_next < mean_prev || !mean_prev.is_finite()) {
                        nl_refine_debug.h_refit = Some(matrix3_to_array(&h_next));
                        nl_refine_debug.notes.push(format!(
                            "h_refit_iter{}:accepted mean_err_px {:.3} -> {:.3}",
                            iter, mean_prev, mean_next
                        ));

                        h_current = Some(h_next);
                        h_prev = h_next;
                        mean_prev = mean_next;

                        let (stats_i, records_i) = refine::refine_markers_circle_board(
                            gray,
                            &h_prev,
                            &mut final_markers,
                            &config.nl_refine,
                            debug_cfg.store_points,
                        );
                        nl_refine_debug.stats = dbg::NlRefineStatsDebugV1 {
                            n_inliers: stats_i.n_inliers,
                            n_refined: stats_i.n_refined,
                            n_failed: stats_i.n_failed,
                            mean_before_mm: stats_i.mean_before_mm,
                            mean_after_mm: stats_i.mean_after_mm,
                            p95_before_mm: stats_i.p95_before_mm,
                            p95_after_mm: stats_i.p95_after_mm,
                        };
                        nl_refine_debug.refined_markers = records_i
                            .into_iter()
                            .map(|r| dbg::NlRefinedMarkerDebugV1 {
                                id: r.id,
                                n_points: r.n_points,
                                init_center_board_mm: r.init_center_board_mm,
                                refined_center_board_mm: r.refined_center_board_mm,
                                center_img_before: r.center_img_before,
                                center_img_after: r.center_img_after,
                                before_rms_mm: r.before_rms_mm,
                                after_rms_mm: r.after_rms_mm,
                                delta_center_mm: r.delta_center_mm,
                                edge_points_img: r.edge_points_img,
                                edge_points_board_mm: r.edge_points_board_mm,
                                status: match r.status {
                                    refine::MarkerRefineStatus::Ok => {
                                        dbg::NlRefineStatusDebugV1::Ok
                                    }
                                    refine::MarkerRefineStatus::Rejected => {
                                        dbg::NlRefineStatusDebugV1::Rejected
                                    }
                                    refine::MarkerRefineStatus::Failed => {
                                        dbg::NlRefineStatusDebugV1::Failed
                                    }
                                    refine::MarkerRefineStatus::Skipped => {
                                        dbg::NlRefineStatusDebugV1::Skipped
                                    }
                                },
                                reason: r.reason,
                            })
                            .collect();
                    } else {
                        nl_refine_debug.notes.push(format!(
                            "h_refit_iter{}:rejected mean_err_px {:.3} -> {:.3}",
                            iter, mean_prev, mean_next
                        ));
                        break;
                    }
                }
            }
        } else {
            nl_refine_debug
                .notes
                .push("skipped_no_homography".to_string());
        }
    }

    // Final H: refit after refinement if we have enough markers.
    let did_refit = config.refine_with_h && final_markers.len() >= 10;
    let final_h_matrix = if did_refit {
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

    if did_refit {
        if let Some(ref mut rd) = refine_debug {
            rd.h_refit = final_h;
        }
    }

    let result = DetectionResult {
        detected_markers: final_markers.clone(),
        image_size: [w, h],
        homography: final_h,
        ransac: final_ransac,
    };

    let dump = dbg::DebugDumpV1 {
        schema_version: dbg::DEBUG_SCHEMA_V1.to_string(),
        image: dbg::ImageDebugV1 {
            path: debug_cfg.image_path.clone(),
            width: w,
            height: h,
        },
        board: dbg::BoardDebugV1 {
            pitch_mm: BOARD_PITCH_MM,
            board_mm: BOARD_SIZE_MM[0],
            board_size_mm: BOARD_SIZE_MM,
            marker_count: BOARD_N,
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
                grad_polarity: match config.outer_estimation.grad_polarity {
                    OuterGradPolarity::DarkToLight => dbg::OuterGradPolarityParamsV1::DarkToLight,
                    OuterGradPolarity::LightToDark => dbg::OuterGradPolarityParamsV1::LightToDark,
                    OuterGradPolarity::Auto => dbg::OuterGradPolarityParamsV1::Auto,
                },
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
            nl_refine: Some(nl_refine_debug.params.clone()),
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
            stage0_proposals: stage0,
            stage1_fit_decode: stage1,
            stage2_dedup: dedup_debug,
            stage3_ransac: ransac_debug,
            stage4_refine: refine_debug,
            stage5_completion: Some(completion_debug),
            stage6_nl_refine: Some(nl_refine_debug),
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
    };

    (result, dump)
}
