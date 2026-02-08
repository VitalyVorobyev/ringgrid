use crate::board_layout::BoardLayout;
use crate::debug_dump as dbg;
use crate::homography::{self, RansacHomographyConfig};
use crate::{DetectedMarker, RansacStats};

struct GlobalFilterOutcome {
    filtered: Vec<DetectedMarker>,
    result: Option<homography::RansacHomographyResult>,
    stats: Option<RansacStats>,
    #[allow(dead_code)]
    debug: Option<dbg::RansacDebugV1>,
}

fn matrix3_to_array(m: &nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

fn run_global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
    collect_debug: bool,
    log_messages: bool,
) -> GlobalFilterOutcome {
    // Build correspondences from decoded markers.
    let mut src_pts = Vec::new(); // board coords (mm)
    let mut dst_pts = Vec::new(); // image coords (px)
    let mut corr_ids: Vec<usize> = Vec::new();
    let mut candidate_indices: Vec<usize> = Vec::new();

    for (i, m) in markers.iter().enumerate() {
        if let Some(id) = m.id {
            if let Some(xy) = board.xy_mm(id) {
                src_pts.push([xy[0] as f64, xy[1] as f64]);
                dst_pts.push(m.center);
                corr_ids.push(id);
                candidate_indices.push(i);
            }
        }
    }

    if log_messages {
        tracing::info!(
            "Global filter: {} decoded candidates out of {} total detections",
            candidate_indices.len(),
            markers.len()
        );
    }

    if src_pts.len() < 4 {
        if log_messages {
            tracing::warn!(
                "Too few decoded candidates for homography ({} < 4)",
                src_pts.len()
            );
        }

        let debug = if collect_debug {
            Some(dbg::RansacDebugV1 {
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
            })
        } else {
            None
        };

        return GlobalFilterOutcome {
            filtered: markers.to_vec(),
            result: None,
            stats: None,
            debug,
        };
    }

    let result = match homography::fit_homography_ransac(&src_pts, &dst_pts, config) {
        Ok(r) => r,
        Err(e) => {
            if log_messages {
                tracing::warn!("Homography RANSAC failed: {}", e);
            }

            let debug = if collect_debug {
                Some(dbg::RansacDebugV1 {
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
                })
            } else {
                None
            };

            return GlobalFilterOutcome {
                filtered: markers.to_vec(),
                result: None,
                stats: None,
                debug,
            };
        }
    };

    // Collect inliers/outliers and per-id errors.
    let mut filtered: Vec<DetectedMarker> = Vec::new();
    let mut inlier_errors: Vec<f64> = Vec::new();
    let mut inlier_ids: Vec<usize> = Vec::new();
    let mut outlier_ids: Vec<usize> = Vec::new();
    let mut per_id_error: Vec<dbg::PerIdErrorDebugV1> = Vec::new();

    for (j, &id) in corr_ids.iter().enumerate() {
        let err = result.errors[j];
        if collect_debug {
            per_id_error.push(dbg::PerIdErrorDebugV1 {
                id,
                reproj_err_px: err,
            });
        }

        if result.inlier_mask[j] {
            inlier_ids.push(id);
            inlier_errors.push(err);
            filtered.push(markers[candidate_indices[j]].clone());
        } else {
            outlier_ids.push(id);
        }
    }

    // Compute stats.
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

    if log_messages {
        tracing::info!(
            "Homography RANSAC: {}/{} inliers, mean_err={:.2}px, p95={:.2}px",
            result.n_inliers,
            src_pts.len(),
            mean_err,
            p95_err,
        );
    }

    let debug = if collect_debug {
        Some(dbg::RansacDebugV1 {
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
        })
    } else {
        None
    };

    GlobalFilterOutcome {
        filtered,
        result: Some(result),
        stats: Some(stats),
        debug,
    }
}

#[allow(dead_code)]
pub fn global_filter_with_debug(
    markers: &[DetectedMarker],
    _cand_idx: &[usize],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
    dbg::RansacDebugV1,
) {
    let out = run_global_filter(markers, config, board, true, false);
    (
        out.filtered,
        out.result,
        out.stats,
        out.debug.expect("debug is present when collect_debug=true"),
    )
}

/// Apply global homography RANSAC filter.
///
/// Returns `(filtered markers, RANSAC result, stats)`.
pub fn global_filter(
    markers: &[DetectedMarker],
    config: &RansacHomographyConfig,
    board: &BoardLayout,
) -> (
    Vec<DetectedMarker>,
    Option<homography::RansacHomographyResult>,
    Option<RansacStats>,
) {
    let out = run_global_filter(markers, config, board, false, true);
    (out.filtered, out.result, out.stats)
}
