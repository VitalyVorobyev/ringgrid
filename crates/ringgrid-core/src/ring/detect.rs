//! Full ring detection pipeline: proposal → edge sampling → fit → decode → global filter.

use image::GrayImage;

use crate::board_spec;
use crate::conic::{
    fit_ellipse_direct, rms_sampson_distance, try_fit_ellipse_ransac, Ellipse, RansacConfig,
};
use crate::debug_dump as dbg;
use crate::homography::{self, project, RansacHomographyConfig};
use crate::{
    DecodeMetrics, DetectedMarker, DetectionResult, EllipseParams, FitMetrics, RansacStats,
};

use super::decode::{decode_marker, decode_marker_with_diagnostics, DecodeConfig};
use super::edge_sample::{sample_edges, EdgeSampleConfig, EdgeSampleResult};
use super::proposal::{find_proposals, ProposalConfig};

/// Debug collection options for `detect_rings_with_debug`.
#[derive(Debug, Clone)]
pub struct DebugCollectConfig {
    pub image_path: Option<String>,
    pub marker_diameter_px: f64,
    pub max_candidates: usize,
    pub store_points: bool,
}

/// Top-level detection configuration.
#[derive(Debug, Clone)]
pub struct DetectConfig {
    pub proposal: ProposalConfig,
    pub edge_sample: EdgeSampleConfig,
    pub decode: DecodeConfig,
    /// Minimum semi-axis for a valid outer ellipse.
    pub min_semi_axis: f64,
    /// Maximum semi-axis for a valid outer ellipse.
    pub max_semi_axis: f64,
    /// Maximum aspect ratio (a/b) for a valid ellipse.
    pub max_aspect_ratio: f64,
    /// NMS dedup radius for final markers (pixels).
    pub dedup_radius: f64,
    /// Enable global homography filtering (requires board spec).
    pub use_global_filter: bool,
    /// RANSAC homography configuration.
    pub ransac_homography: RansacHomographyConfig,
    /// Enable one-iteration refinement using H.
    pub refine_with_h: bool,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            proposal: ProposalConfig::default(),
            edge_sample: EdgeSampleConfig::default(),
            decode: DecodeConfig::default(),
            min_semi_axis: 3.0,
            max_semi_axis: 15.0,
            max_aspect_ratio: 3.0,
            dedup_radius: 6.0,
            use_global_filter: true,
            ransac_homography: RansacHomographyConfig::default(),
            refine_with_h: true,
        }
    }
}

/// Fit outer and inner ellipses from edge points.
///
/// Returns (outer, inner, outer_ransac_result, inner_ransac_result).
fn fit_ring_ellipses(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Option<(
    Ellipse,
    Option<Ellipse>,
    Option<crate::conic::RansacResult>,
    Option<crate::conic::RansacResult>,
)> {
    fit_ring_ellipses_with_reason(edge, config).ok()
}

fn fit_ring_ellipses_with_reason(
    edge: &EdgeSampleResult,
    config: &DetectConfig,
) -> Result<
    (
        Ellipse,
        Option<Ellipse>,
        Option<crate::conic::RansacResult>,
        Option<crate::conic::RansacResult>,
    ),
    String,
> {
    // Fit outer ellipse
    let ransac_config = RansacConfig {
        max_iters: 200,
        inlier_threshold: 1.5,
        min_inliers: 6,
        seed: 42,
    };

    let (outer, outer_ransac) = if edge.outer_points.len() >= 8 {
        match try_fit_ellipse_ransac(&edge.outer_points, &ransac_config) {
            Ok(r) => (r.ellipse, Some(r)),
            Err(_) => {
                // Fall back to direct fit
                match fit_ellipse_direct(&edge.outer_points) {
                    Some((_, e)) => (e, None),
                    None => return Err("fit_outer:direct_failed".to_string()),
                }
            }
        }
    } else if edge.outer_points.len() >= 6 {
        match fit_ellipse_direct(&edge.outer_points) {
            Some((_, e)) => (e, None),
            None => return Err("fit_outer:direct_failed".to_string()),
        }
    } else {
        return Err("fit_outer:too_few_points".to_string());
    };

    // Validate outer ellipse
    if outer.a < config.min_semi_axis
        || outer.a > config.max_semi_axis
        || outer.b < config.min_semi_axis
        || outer.b > config.max_semi_axis
        || outer.aspect_ratio() > config.max_aspect_ratio
        || !outer.is_valid()
    {
        return Err("fit_outer:invalid_ellipse".to_string());
    }

    // Fit inner ellipse (optional)
    let (inner, inner_ransac) = if edge.inner_points.len() >= 8 {
        match try_fit_ellipse_ransac(&edge.inner_points, &ransac_config) {
            Ok(r) => {
                if r.ellipse.is_valid()
                    && r.ellipse.a >= 1.0
                    && r.ellipse.aspect_ratio() < config.max_aspect_ratio
                {
                    (Some(r.ellipse), Some(r))
                } else {
                    (None, None)
                }
            }
            Err(_) => match fit_ellipse_direct(&edge.inner_points) {
                Some((_, e)) if e.is_valid() && e.a >= 1.0 => (Some(e), None),
                _ => (None, None),
            },
        }
    } else if edge.inner_points.len() >= 6 {
        match fit_ellipse_direct(&edge.inner_points) {
            Some((_, e)) if e.is_valid() && e.a >= 1.0 => (Some(e), None),
            _ => (None, None),
        }
    } else {
        (None, None)
    };

    Ok((outer, inner, outer_ransac, inner_ransac))
}

/// Compute the marker center as a weighted average of inner and outer ellipse centers.
fn compute_center(outer: &Ellipse, inner: Option<&Ellipse>, edge: &EdgeSampleResult) -> [f64; 2] {
    if let Some(inner) = inner {
        // Weight by number of points (more points = more reliable)
        let w_outer = edge.outer_points.len() as f64;
        let w_inner = edge.inner_points.len() as f64;
        let w_total = w_outer + w_inner;
        [
            (outer.cx * w_outer + inner.cx * w_inner) / w_total,
            (outer.cy * w_outer + inner.cy * w_inner) / w_total,
        ]
    } else {
        [outer.cx, outer.cy]
    }
}

/// Helper to create EllipseParams from a conic Ellipse.
fn ellipse_to_params(e: &Ellipse) -> EllipseParams {
    EllipseParams {
        center_xy: [e.cx, e.cy],
        semi_axes: [e.a, e.b],
        angle: e.angle,
    }
}

/// Run the full ring detection pipeline.
pub fn detect_rings(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    let (w, h) = gray.dimensions();

    // Stage 1: Find candidate centers
    let proposals = find_proposals(gray, &config.proposal);
    tracing::info!("{} proposals found", proposals.len());

    // Stages 2-5: For each proposal, sample edges → fit → decode
    let mut markers: Vec<DetectedMarker> = Vec::new();

    for proposal in &proposals {
        // Stage 2: Sample radial edges
        let edge = match sample_edges(gray, [proposal.x, proposal.y], &config.edge_sample) {
            Some(er) => er,
            None => continue,
        };

        // Stage 3: Fit ellipses
        let (outer, inner, outer_ransac, inner_ransac) = match fit_ring_ellipses(&edge, config) {
            Some(r) => r,
            None => continue,
        };

        // Compute center
        let center = compute_center(&outer, inner.as_ref(), &edge);

        // Compute fit metrics
        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: edge.inner_points.len(),
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: inner_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.inner_points.len().max(1) as f32),
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: inner
                .as_ref()
                .map(|ie| rms_sampson_distance(ie, &edge.inner_points)),
        };

        // Stage 4: Decode
        let decode_result = decode_marker(gray, &outer, &config.decode);

        // Build marker
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);

        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let marker = DetectedMarker {
            id: decode_result.as_ref().map(|d| d.id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner.as_ref().map(|ie| ellipse_to_params(ie)),
            fit,
            decode: decode_metrics,
        };

        markers.push(marker);
    }

    // Stage 5: Dedup by center proximity
    markers = dedup_markers(markers, config.dedup_radius);

    // Stage 5b: Dedup by ID — keep best confidence per decoded ID
    dedup_by_id(&mut markers);

    tracing::info!("{} markers detected after dedup", markers.len());

    // Stage 6: Global homography filtering
    if !config.use_global_filter {
        return DetectionResult {
            detected_markers: markers,
            image_size: [w, h],
            homography: None,
            ransac: None,
        };
    }

    let (filtered, h_result, ransac_stats) = global_filter(&markers, &config.ransac_homography);

    let h_matrix = h_result.as_ref().map(|r| &r.h);

    // Stage 7: Optional refinement using H
    let final_markers = if config.refine_with_h {
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

    // Refit H after refinement if we have enough markers
    let (final_h, final_ransac) = if config.refine_with_h && final_markers.len() >= 10 {
        refit_homography(&final_markers, &config.ransac_homography)
    } else {
        (h_result.map(|r| matrix3_to_array(&r.h)), ransac_stats)
    };

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
        image_size: [w, h],
        homography: final_h,
        ransac: final_ransac,
    }
}

/// Run the full ring detection pipeline and collect a versioned debug dump.
pub fn detect_rings_with_debug(
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

        // Edge sampling
        let edge = match sample_edges(gray, [proposal.x, proposal.y], &config.edge_sample) {
            Some(er) => er,
            None => {
                if let Some(cd) = cand_debug.as_mut() {
                    cd.decision = dbg::DecisionDebugV1 {
                        status: dbg::DecisionStatusV1::Rejected,
                        reason: "edge_sample:insufficient_ring_rays".to_string(),
                    };
                }
                if let Some(cd) = cand_debug {
                    stage1.candidates.push(cd);
                }
                continue;
            }
        };

        // Fit
        let (outer, inner, outer_ransac, inner_ransac) =
            match fit_ring_ellipses_with_reason(&edge, config) {
                Ok(r) => r,
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

        let center = compute_center(&outer, inner.as_ref(), &edge);

        let fit_metrics = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: edge.inner_points.len(),
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: inner_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.inner_points.len().max(1) as f32),
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: inner
                .as_ref()
                .map(|ie| rms_sampson_distance(ie, &edge.inner_points)),
        };

        // Decode + diagnostics
        let (decode_result, decode_diag) =
            decode_marker_with_diagnostics(gray, &outer, &config.decode);
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

        let marker = DetectedMarker {
            id: derived_id,
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner.as_ref().map(|ie| ellipse_to_params(ie)),
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
                    inner_peak_r: Some(edge.inner_radii.clone()),
                    outer_peak_r: Some(edge.outer_radii.clone()),
                },
                ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                    center_xy: [outer.cx as f32, outer.cy as f32],
                    semi_axes: [outer.a as f32, outer.b as f32],
                    angle: outer.angle as f32,
                }),
                ellipse_inner: inner.as_ref().map(|ie| dbg::EllipseParamsDebugV1 {
                    center_xy: [ie.cx as f32, ie.cy as f32],
                    semi_axes: [ie.a as f32, ie.b as f32],
                    angle: ie.angle as f32,
                }),
                metrics: dbg::RingFitMetricsDebugV1 {
                    inlier_ratio_inner: fit_metrics.ransac_inlier_ratio_inner,
                    inlier_ratio_outer: fit_metrics.ransac_inlier_ratio_outer,
                    mean_resid_inner: fit_metrics.rms_residual_inner.map(|v| v as f32),
                    mean_resid_outer: fit_metrics.rms_residual_outer.map(|v| v as f32),
                    arc_coverage: arc_cov,
                    valid_inner: inner.is_some(),
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
    let (final_markers, mut refine_debug) = if config.refine_with_h {
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

    // Refit H after refinement if we have enough markers
    let did_refit = config.refine_with_h && final_markers.len() >= 10;
    let (final_h, final_ransac) = if did_refit {
        let (h_arr, stats) = refit_homography(&final_markers, &config.ransac_homography);
        (h_arr, stats)
    } else {
        (h_result.map(|r| matrix3_to_array(&r.h)), ransac_stats)
    };
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
            marker_diameter_px: debug_cfg.marker_diameter_px,
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
            decode: dbg::DecodeParamsV1 {
                code_band_ratio: config.decode.code_band_ratio,
                samples_per_sector: config.decode.samples_per_sector,
                n_radial_rings: config.decode.n_radial_rings,
                max_decode_dist: config.decode.max_decode_dist,
                min_decode_confidence: config.decode.min_decode_confidence,
            },
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
            final_: dbg::FinalDebugV1 {
                h_final: result.homography,
                detections: result.detected_markers.clone(),
                notes: Vec::new(),
            },
        },
    };

    (result, dump)
}

fn dedup_with_debug(
    mut markers: Vec<DetectedMarker>,
    mut cand_idx: Vec<usize>,
    radius: f64,
) -> (Vec<DetectedMarker>, Vec<usize>, dbg::DedupDebugV1) {
    // Sort by confidence descending (keep cand_idx in sync)
    let mut order: Vec<usize> = (0..markers.len()).collect();
    order.sort_by(|&a, &b| {
        markers[b]
            .confidence
            .partial_cmp(&markers[a].confidence)
            .unwrap()
    });

    markers = order.iter().map(|&i| markers[i].clone()).collect();
    cand_idx = order.iter().map(|&i| cand_idx[i]).collect();

    let mut keep = vec![true; markers.len()];
    let r2 = radius * radius;
    let mut kept_by_proximity: Vec<dbg::KeptByProximityDebugV1> = Vec::new();

    for i in 0..markers.len() {
        if !keep[i] {
            continue;
        }
        let mut dropped: Vec<usize> = Vec::new();
        for j in (i + 1)..markers.len() {
            if !keep[j] {
                continue;
            }
            let dx = markers[i].center[0] - markers[j].center[0];
            let dy = markers[i].center[1] - markers[j].center[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
                dropped.push(cand_idx[j]);
            }
        }
        if !dropped.is_empty() {
            kept_by_proximity.push(dbg::KeptByProximityDebugV1 {
                kept_cand_idx: cand_idx[i],
                dropped_cand_indices: dropped,
                reasons: vec!["within_dedup_radius".to_string()],
            });
        }
    }

    let mut markers2: Vec<DetectedMarker> = Vec::new();
    let mut cand2: Vec<usize> = Vec::new();
    for ((m, k), ci) in markers
        .into_iter()
        .zip(keep.into_iter())
        .zip(cand_idx.into_iter())
    {
        if k {
            markers2.push(m);
            cand2.push(ci);
        }
    }

    // Dedup by ID (keep best confidence)
    use std::collections::{HashMap, HashSet};
    let mut best_idx: HashMap<usize, usize> = HashMap::new();
    for (i, m) in markers2.iter().enumerate() {
        if let Some(id) = m.id {
            match best_idx.get(&id) {
                Some(&prev) if markers2[prev].confidence >= m.confidence => {}
                _ => {
                    best_idx.insert(id, i);
                }
            }
        }
    }

    let keep_set: HashSet<usize> = best_idx.values().copied().collect();
    let mut kept_by_id: Vec<dbg::KeptByIdDebugV1> = Vec::new();
    for (&id, &kept_i) in best_idx.iter() {
        let mut dropped: Vec<usize> = Vec::new();
        for (i, m) in markers2.iter().enumerate() {
            if i == kept_i {
                continue;
            }
            if m.id == Some(id) {
                dropped.push(cand2[i]);
            }
        }
        if !dropped.is_empty() {
            kept_by_id.push(dbg::KeptByIdDebugV1 {
                id,
                kept_cand_idx: cand2[kept_i],
                dropped_cand_indices: dropped,
                reasons: vec!["lower_confidence".to_string()],
            });
        }
    }
    kept_by_id.sort_by_key(|e| e.id);

    let mut markers3: Vec<DetectedMarker> = Vec::new();
    let mut cand3: Vec<usize> = Vec::new();
    for (i, m) in markers2.into_iter().enumerate() {
        let keep_it = m.id.is_none() || keep_set.contains(&i);
        if keep_it {
            markers3.push(m);
            cand3.push(cand2[i]);
        }
    }

    (
        markers3,
        cand3,
        dbg::DedupDebugV1 {
            kept_by_proximity,
            kept_by_id,
            notes: Vec::new(),
        },
    )
}

fn global_filter_with_debug(
    markers: &[DetectedMarker],
    _cand_idx: &[usize],
    config: &RansacHomographyConfig,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
    dbg::RansacDebugV1,
) {
    // Build correspondences from decoded markers
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut corr_ids: Vec<usize> = Vec::new();

    for m in markers {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                corr_ids.push(id);
            }
        }
    }

    if src_pts.len() < 4 {
        let dbg = dbg::RansacDebugV1 {
            enabled: true,
            h_best: None,
            correspondences_used: src_pts.len(),
            inlier_ids: Vec::new(),
            outlier_ids: Vec::new(),
            per_id_error_px: None,
            stats: dbg::RansacStatsDebugV1 {
                iters: config.max_iters,
                thresh_px: config.inlier_threshold,
                n_corr: src_pts.len(),
                n_inliers: 0,
                mean_err_inliers: 0.0,
                p95_err_inliers: 0.0,
            },
            notes: vec![format!("too_few_correspondences({}<4)", src_pts.len())],
        };
        return (markers.to_vec(), None, None, dbg);
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            let dbg = dbg::RansacDebugV1 {
                enabled: true,
                h_best: None,
                correspondences_used: src_pts.len(),
                inlier_ids: Vec::new(),
                outlier_ids: Vec::new(),
                per_id_error_px: None,
                stats: dbg::RansacStatsDebugV1 {
                    iters: config.max_iters,
                    thresh_px: config.inlier_threshold,
                    n_corr: src_pts.len(),
                    n_inliers: 0,
                    mean_err_inliers: 0.0,
                    p95_err_inliers: 0.0,
                },
                notes: vec![format!("ransac_failed:{}", e)],
            };
            return (markers.to_vec(), None, None, dbg);
        }
    };

    // Collect inliers/outliers and per-id errors
    let mut filtered: Vec<DetectedMarker> = Vec::new();
    let mut inlier_errors: Vec<f64> = Vec::new();
    let mut inlier_ids: Vec<usize> = Vec::new();
    let mut outlier_ids: Vec<usize> = Vec::new();
    let mut per_id_error: Vec<dbg::PerIdErrorDebugV1> = Vec::new();

    for (j, &id) in corr_ids.iter().enumerate() {
        let err = result.errors[j];
        per_id_error.push(dbg::PerIdErrorDebugV1 {
            id,
            reproj_err_px: err,
        });
        if result.inlier_mask[j] {
            inlier_ids.push(id);
            inlier_errors.push(err);
        } else {
            outlier_ids.push(id);
        }
    }

    // Filter markers to inliers only (by matching id list)
    use std::collections::HashSet;
    let inlier_set: HashSet<usize> = inlier_ids.iter().copied().collect();
    for m in markers {
        if let Some(id) = m.id {
            if inlier_set.contains(&id) {
                filtered.push(m.clone());
            }
        }
    }

    // Stats
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    let stats = RansacStats {
        n_candidates: src_pts.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    let dbg = dbg::RansacDebugV1 {
        enabled: true,
        h_best: Some(matrix3_to_array(&result.h)),
        correspondences_used: src_pts.len(),
        inlier_ids,
        outlier_ids,
        per_id_error_px: Some(per_id_error),
        stats: dbg::RansacStatsDebugV1 {
            iters: config.max_iters,
            thresh_px: config.inlier_threshold,
            n_corr: src_pts.len(),
            n_inliers: result.n_inliers,
            mean_err_inliers: mean_err,
            p95_err_inliers: p95_err,
        },
        notes: Vec::new(),
    };

    (filtered, Some(result), Some(stats), dbg)
}

fn refine_with_homography_with_debug(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> (Vec<DetectedMarker>, dbg::RefineDebugV1) {
    let mut refined = Vec::with_capacity(markers.len());
    let mut refined_dbg = Vec::with_capacity(markers.len());

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board_spec::xy_mm(id) {
            Some(xy) => xy,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let prior = project(h, xy[0] as f64, xy[1] as f64);
        if prior[0].is_nan() || prior[1].is_nan() {
            refined.push(m.clone());
            continue;
        }

        let edge = match sample_edges(
            gray,
            [prior[0] as f32, prior[1] as f32],
            &config.edge_sample,
        ) {
            Some(er) => er,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let (outer, inner, outer_ransac, inner_ransac) =
            match fit_ring_ellipses_with_reason(&edge, config) {
                Ok(r) => r,
                Err(_) => {
                    refined.push(m.clone());
                    continue;
                }
            };

        let center = compute_center(&outer, inner.as_ref(), &edge);

        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: edge.inner_points.len(),
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: inner_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.inner_points.len().max(1) as f32),
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: inner
                .as_ref()
                .map(|ie| rms_sampson_distance(ie, &edge.inner_points)),
        };

        let decode_result = decode_marker(gray, &outer, &config.decode);
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        let updated = DetectedMarker {
            id: Some(id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner.as_ref().map(|ie| ellipse_to_params(ie)),
            fit: fit.clone(),
            decode: decode_metrics,
        };

        refined_dbg.push(dbg::RefinedMarkerDebugV1 {
            id,
            prior_center_xy: [prior[0] as f32, prior[1] as f32],
            refined_center_xy: [center[0] as f32, center[1] as f32],
            ellipse_outer: Some(dbg::EllipseParamsDebugV1 {
                center_xy: [outer.cx as f32, outer.cy as f32],
                semi_axes: [outer.a as f32, outer.b as f32],
                angle: outer.angle as f32,
            }),
            ellipse_inner: inner.as_ref().map(|ie| dbg::EllipseParamsDebugV1 {
                center_xy: [ie.cx as f32, ie.cy as f32],
                semi_axes: [ie.a as f32, ie.b as f32],
                angle: ie.angle as f32,
            }),
            fit,
        });

        refined.push(updated);
    }

    (
        refined,
        dbg::RefineDebugV1 {
            h_prior: matrix3_to_array(h),
            refined_markers: refined_dbg,
            h_refit: None,
            notes: Vec::new(),
        },
    )
}

/// Dedup by ID: if the same decoded ID appears multiple times, keep the
/// one with the highest confidence.
fn dedup_by_id(markers: &mut Vec<DetectedMarker>) {
    use std::collections::HashMap;
    let mut best_idx: HashMap<usize, usize> = HashMap::new();

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            match best_idx.get(&id) {
                Some(&prev) if markers[prev].confidence >= m.confidence => {}
                _ => {
                    best_idx.insert(id, i);
                }
            }
        }
    }

    let keep_set: std::collections::HashSet<usize> = best_idx.values().copied().collect();
    let mut i = 0;
    markers.retain(|m| {
        let idx = i;
        i += 1;
        // Keep markers without ID (they'll be filtered by RANSAC anyway)
        m.id.is_none() || keep_set.contains(&idx)
    });
}

/// Apply global homography RANSAC filter.
///
/// Returns (filtered markers, RANSAC result, stats).
fn global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    // Build correspondences from decoded markers
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut candidate_indices = Vec::new(); // index into markers

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                candidate_indices.push(i);
            }
        }
    }

    tracing::info!(
        "Global filter: {} decoded candidates out of {} total detections",
        candidate_indices.len(),
        markers.len()
    );

    if candidate_indices.len() < 4 {
        tracing::warn!(
            "Too few decoded candidates for homography ({} < 4)",
            candidate_indices.len()
        );
        return (markers.to_vec(), None, None);
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Homography RANSAC failed: {}", e);
            return (markers.to_vec(), None, None);
        }
    };

    // Collect inlier markers
    let mut filtered = Vec::new();
    let mut inlier_errors = Vec::new();

    for (j, &marker_idx) in candidate_indices.iter().enumerate() {
        if result.inlier_mask[j] {
            filtered.push(markers[marker_idx].clone());
            inlier_errors.push(result.errors[j]);
        }
    }

    // Compute stats
    inlier_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_err = if inlier_errors.is_empty() {
        0.0
    } else {
        inlier_errors.iter().sum::<f64>() / inlier_errors.len() as f64
    };
    let p95_err = if inlier_errors.is_empty() {
        0.0
    } else {
        let idx = ((inlier_errors.len() as f64 * 0.95) as usize).min(inlier_errors.len() - 1);
        inlier_errors[idx]
    };

    let stats = RansacStats {
        n_candidates: candidate_indices.len(),
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    tracing::info!(
        "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
        result.n_inliers,
        candidate_indices.len(),
        mean_err,
        p95_err,
    );

    (filtered, Some(result), Some(stats))
}

/// Refine marker centers using H: project board coords through H as priors,
/// re-run local ring fit around those priors.
fn refine_with_homography(
    gray: &GrayImage,
    markers: &[DetectedMarker],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) -> Vec<DetectedMarker> {
    let mut refined = Vec::with_capacity(markers.len());

    for m in markers {
        let id = match m.id {
            Some(id) => id,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let xy = match board_spec::xy_mm(id) {
            Some(xy) => xy,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        // Project board coords through H to get refined center prior
        let prior = project(h, xy[0] as f64, xy[1] as f64);
        if prior[0].is_nan() || prior[1].is_nan() {
            refined.push(m.clone());
            continue;
        }

        // Re-run edge sampling around the H-projected center
        let edge = match sample_edges(
            gray,
            [prior[0] as f32, prior[1] as f32],
            &config.edge_sample,
        ) {
            Some(er) => er,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        // Re-fit ellipses
        let (outer, inner, outer_ransac, inner_ransac) = match fit_ring_ellipses(&edge, config) {
            Some(r) => r,
            None => {
                refined.push(m.clone());
                continue;
            }
        };

        let center = compute_center(&outer, inner.as_ref(), &edge);

        let fit = FitMetrics {
            n_angles_total: edge.n_total_rays,
            n_angles_with_both_edges: edge.n_good_rays,
            n_points_outer: edge.outer_points.len(),
            n_points_inner: edge.inner_points.len(),
            ransac_inlier_ratio_outer: outer_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.outer_points.len().max(1) as f32),
            ransac_inlier_ratio_inner: inner_ransac
                .as_ref()
                .map(|r| r.num_inliers as f32 / edge.inner_points.len().max(1) as f32),
            rms_residual_outer: Some(rms_sampson_distance(&outer, &edge.outer_points)),
            rms_residual_inner: inner
                .as_ref()
                .map(|ie| rms_sampson_distance(ie, &edge.inner_points)),
        };

        // Re-decode with new ellipse
        let decode_result = decode_marker(gray, &outer, &config.decode);
        let confidence = decode_result.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let decode_metrics = decode_result.as_ref().map(|d| DecodeMetrics {
            observed_word: d.raw_word,
            best_id: d.id,
            best_rotation: d.rotation,
            best_dist: d.dist,
            margin: d.margin,
            decode_confidence: d.confidence,
        });

        // Keep original ID (validated by RANSAC), but update geometry
        refined.push(DetectedMarker {
            id: Some(id),
            confidence,
            center,
            ellipse_outer: Some(ellipse_to_params(&outer)),
            ellipse_inner: inner.as_ref().map(|ie| ellipse_to_params(ie)),
            fit,
            decode: decode_metrics,
        });
    }

    refined
}

/// Refit homography using refined marker centers, return (H_array, stats).
fn refit_homography(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
) -> (Option<[[f64; 3]; 3]>, Option<RansacStats>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for m in markers {
        if let Some(id) = m.id {
            if let Some(xy) = board_spec::xy_mm(id) {
                src.push([xy[0] as f64, xy[1] as f64]);
                dst.push(m.center);
            }
        }
    }

    if src.len() < 4 {
        return (None, None);
    }

    // Use a light RANSAC (most outliers already removed)
    let light_config = RansacHomographyConfig {
        max_iters: 500,
        inlier_threshold: config.inlier_threshold,
        min_inliers: config.min_inliers,
        seed: config.seed + 1,
    };

    match homography::fit_homography_ransac(&src, &dst, &light_config) {
        Ok(result) => {
            let mut errors: Vec<f64> = result
                .inlier_mask
                .iter()
                .zip(&result.errors)
                .filter(|(&m, _)| m)
                .map(|(_, &e)| e)
                .collect();
            errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean_err = if errors.is_empty() {
                0.0
            } else {
                errors.iter().sum::<f64>() / errors.len() as f64
            };
            let p95_err = if errors.is_empty() {
                0.0
            } else {
                let idx = ((errors.len() as f64 * 0.95) as usize).min(errors.len() - 1);
                errors[idx]
            };

            let stats = RansacStats {
                n_candidates: src.len(),
                n_inliers: result.n_inliers,
                threshold_px: light_config.inlier_threshold,
                mean_err_px: mean_err,
                p95_err_px: p95_err,
            };

            (Some(matrix3_to_array(&result.h)), Some(stats))
        }
        Err(_) => (None, None),
    }
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

/// Remove duplicate detections: keep the highest-confidence marker within dedup_radius.
fn dedup_markers(mut markers: Vec<DetectedMarker>, radius: f64) -> Vec<DetectedMarker> {
    // Sort by confidence descending
    markers.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; markers.len()];
    let r2 = radius * radius;

    for i in 0..markers.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..markers.len() {
            if !keep[j] {
                continue;
            }
            let dx = markers[i].center[0] - markers[j].center[0];
            let dy = markers[i].center[1] - markers[j].center[1];
            if dx * dx + dy * dy < r2 {
                keep[j] = false;
            }
        }
    }

    markers
        .into_iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(m, _)| m)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    fn debug_dump_does_not_panic_when_stages_skipped() {
        let img = GrayImage::new(64, 64);
        let mut cfg = DetectConfig::default();
        cfg.use_global_filter = false;
        cfg.refine_with_h = false;

        let dbg_cfg = DebugCollectConfig {
            image_path: Some("dummy.png".to_string()),
            marker_diameter_px: 32.0,
            max_candidates: 10,
            store_points: false,
        };

        let (res, dump) = detect_rings_with_debug(&img, &cfg, &dbg_cfg);
        assert_eq!(res.image_size, [64, 64]);
        assert_eq!(dump.schema_version, crate::debug_dump::DEBUG_SCHEMA_V1);
        assert_eq!(dump.stages.stage0_proposals.n_total, 0);
        assert!(!dump.stages.stage3_ransac.enabled);
    }
}
