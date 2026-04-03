use std::collections::HashMap;

use image::GrayImage;
use nalgebra::Point2;
use projective_grid::hex::hex_predict_grid_position;
use projective_grid::GridIndex;

use crate::conic::Ellipse;
use crate::detector::id_correction::{affine_to_image, fit_local_affine};
use crate::detector::marker_build::DetectionSource;
use crate::homography::homography_project as project;
use crate::marker::codec::Codebook;
use crate::ring::edge_sample::EdgeSampleResult;
use crate::DetectedMarker;

use super::{
    fit_outer_candidate_from_prior_for_completion,
    marker_build::{decode_metrics_from_result, fit_metrics_with_inner, fit_support_score},
    median_outer_radius_from_neighbors_px, CompletionParams, DetectConfig, OuterFitCandidate,
};

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CompletionStats {
    pub n_candidates_total: usize,
    pub n_in_image: usize,
    pub n_attempted: usize,
    pub n_added: usize,
    pub n_failed_fit: usize,
    pub n_failed_gate: usize,
    /// Number of accepted completion markers whose re-decoded ID did not match
    /// the expected board ID (decode mismatch accepted).
    pub n_decode_mismatch: usize,
}

struct CandidateQuality {
    center: [f64; 2],
    arc_cov: f32,
    fit_confidence: f32,
    mean_axis: f32,
    scale_ok: bool,
    reproj_err: f32,
    max_angular_gap_outer: f64,
    /// Coefficient of variation (std_dev / mean) of per-ray outer radii.
    /// Set to 0.0 when fewer than 2 rays are available or the mean is degenerate.
    radii_cv: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompletionGateRejectReason {
    ArcCoverageLow,
    FitConfidenceLow,
    ReprojectionTooHigh,
    ScaleOutOfRange,
    PerfectDecodeRequired,
    AngularGapTooLarge,
    RadiiScatterTooHigh,
}

impl CompletionGateRejectReason {
    const fn code(self) -> &'static str {
        match self {
            Self::ArcCoverageLow => "arc_coverage_low",
            Self::FitConfidenceLow => "fit_confidence_low",
            Self::ReprojectionTooHigh => "reprojection_too_high",
            Self::ScaleOutOfRange => "scale_out_of_range",
            Self::PerfectDecodeRequired => "perfect_decode_required",
            Self::AngularGapTooLarge => "angular_gap_too_large",
            Self::RadiiScatterTooHigh => "radii_scatter_too_high",
        }
    }
}

impl std::fmt::Display for CompletionGateRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CompletionGateRejectContext {
    ArcCoverageLow {
        observed_arc_coverage: f32,
        min_required_arc_coverage: f32,
    },
    FitConfidenceLow {
        observed_fit_confidence: f32,
        min_required_fit_confidence: f32,
    },
    ReprojectionTooHigh {
        observed_reproj_error_px: f32,
        max_allowed_reproj_error_px: f32,
    },
    ScaleOutOfRange {
        observed_mean_axis_px: f32,
        expected_radius_px: f32,
        min_allowed_axis_px: f32,
        max_allowed_axis_px: f32,
    },
    AngularGapTooLarge {
        observed_gap_rad: f64,
        max_allowed_gap_rad: f64,
    },
    RadiiScatterTooHigh {
        observed_radii_cv: f32,
        max_allowed_radii_cv: f32,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CompletionGateReject {
    reason: CompletionGateRejectReason,
    context: CompletionGateRejectContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompletionDecodeNoticeReason {
    DecodeMismatchAccepted,
}

impl CompletionDecodeNoticeReason {
    const fn code(self) -> &'static str {
        match self {
            Self::DecodeMismatchAccepted => "decode_mismatch_accepted",
        }
    }
}

impl std::fmt::Display for CompletionDecodeNoticeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CompletionDecodeNotice {
    reason: CompletionDecodeNoticeReason,
    expected_id: usize,
    observed_id: usize,
}

fn radii_coefficient_of_variation(radii: &[f32]) -> f32 {
    if radii.len() < 2 {
        return 0.0;
    }
    let mean = radii.iter().sum::<f32>() / radii.len() as f32;
    if mean < 1.0 {
        return 0.0;
    }
    let variance = radii.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / radii.len() as f32;
    variance.sqrt() / mean
}

fn local_affine_completion_seed(
    target_board_xy: [f64; 2],
    markers: &[DetectedMarker],
    board: &crate::board_layout::BoardLayout,
) -> Option<[f64; 2]> {
    const MAX_NEIGHBORS: usize = 4;

    let mut neighbors: Vec<([f64; 2], [f64; 2], f64)> = markers
        .iter()
        .filter_map(|marker| {
            let id = marker.id?;
            let board_xy = board.xy_mm(id)?;
            let center = marker.center;
            if !(center[0].is_finite() && center[1].is_finite()) {
                return None;
            }
            let board_xy = [f64::from(board_xy[0]), f64::from(board_xy[1])];
            let dx = board_xy[0] - target_board_xy[0];
            let dy = board_xy[1] - target_board_xy[1];
            let d2 = dx * dx + dy * dy;
            d2.is_finite().then_some((board_xy, center, d2))
        })
        .collect();
    if neighbors.len() < 3 {
        return None;
    }

    neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    neighbors.truncate(MAX_NEIGHBORS);

    let board_pts: Vec<[f64; 2]> = neighbors.iter().map(|(board_xy, _, _)| *board_xy).collect();
    let image_pts: Vec<[f64; 2]> = neighbors.iter().map(|(_, center, _)| *center).collect();
    let affine = fit_local_affine(&board_pts, &image_pts)?;
    let seed = affine_to_image(&affine, target_board_xy);
    (seed[0].is_finite() && seed[1].is_finite()).then_some(seed)
}

fn hex_neighbor_seed(
    id: usize,
    hex_grid: &HashMap<GridIndex, Point2<f32>>,
    board: &crate::board_layout::BoardLayout,
) -> Option<[f64; 2]> {
    let bm = board.marker(id)?;
    let q = bm.q? as i32;
    let r = bm.r? as i32;
    let predicted = hex_predict_grid_position(hex_grid, GridIndex { i: q, j: r })?;
    Some([f64::from(predicted.x), f64::from(predicted.y)])
}

fn projected_completion_seed(
    id: usize,
    h: &nalgebra::Matrix3<f64>,
    markers: &[DetectedMarker],
    board: &crate::board_layout::BoardLayout,
    hex_grid: &HashMap<GridIndex, Point2<f32>>,
) -> Option<[f64; 2]> {
    let board_xy = board.xy_mm(id)?;
    let target_board_xy = [f64::from(board_xy[0]), f64::from(board_xy[1])];
    hex_neighbor_seed(id, hex_grid, board)
        .or_else(|| local_affine_completion_seed(target_board_xy, markers, board))
        .or_else(|| Some(project(h, target_board_xy[0], target_board_xy[1])))
}

fn compute_candidate_quality(
    edge: &EdgeSampleResult,
    outer: &Ellipse,
    outer_ransac: Option<&crate::conic::RansacResult>,
    projected_center: [f64; 2],
    r_expected: f32,
) -> CandidateQuality {
    let center = outer.center();
    let arc_cov = edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32;
    let fit_confidence = fit_support_score(edge, outer_ransac);
    let mean_axis = ((outer.a + outer.b) * 0.5) as f32;
    let scale_ok = mean_axis.is_finite()
        && mean_axis >= (r_expected * 0.75)
        && mean_axis <= (r_expected * 1.33);
    let reproj_err = {
        let dx = center[0] - projected_center[0];
        let dy = center[1] - projected_center[1];
        (dx * dx + dy * dy).sqrt() as f32
    };
    let max_angular_gap_outer = super::outer_fit::max_angular_gap(center, &edge.outer_points);
    let radii_cv = radii_coefficient_of_variation(&edge.outer_radii);
    CandidateQuality {
        center,
        arc_cov,
        fit_confidence,
        mean_axis,
        scale_ok,
        reproj_err,
        max_angular_gap_outer,
        radii_cv,
    }
}

fn check_decode_gate(
    decode_result: Option<&crate::marker::decode::DecodeResult>,
    expected_id: usize,
) -> Option<CompletionDecodeNotice> {
    if let Some(d) = decode_result {
        if d.id != expected_id {
            return Some(CompletionDecodeNotice {
                reason: CompletionDecodeNoticeReason::DecodeMismatchAccepted,
                expected_id,
                observed_id: d.id,
            });
        }
    }
    None
}

fn check_quality_gates(
    quality: &CandidateQuality,
    params: &CompletionParams,
    r_expected: f32,
    max_angular_gap_rad: f64,
) -> Result<(), CompletionGateReject> {
    if quality.arc_cov < params.min_arc_coverage {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::ArcCoverageLow,
            context: CompletionGateRejectContext::ArcCoverageLow {
                observed_arc_coverage: quality.arc_cov,
                min_required_arc_coverage: params.min_arc_coverage,
            },
        });
    }
    if quality.radii_cv > params.max_radii_std_ratio {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::RadiiScatterTooHigh,
            context: CompletionGateRejectContext::RadiiScatterTooHigh {
                observed_radii_cv: quality.radii_cv,
                max_allowed_radii_cv: params.max_radii_std_ratio,
            },
        });
    }
    if quality.fit_confidence < params.min_fit_confidence {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::FitConfidenceLow,
            context: CompletionGateRejectContext::FitConfidenceLow {
                observed_fit_confidence: quality.fit_confidence,
                min_required_fit_confidence: params.min_fit_confidence,
            },
        });
    }
    if (quality.reproj_err as f64) > (params.reproj_gate_px as f64) {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::ReprojectionTooHigh,
            context: CompletionGateRejectContext::ReprojectionTooHigh {
                observed_reproj_error_px: quality.reproj_err,
                max_allowed_reproj_error_px: params.reproj_gate_px,
            },
        });
    }
    if !quality.scale_ok {
        let min_allowed_axis_px = r_expected * 0.75;
        let max_allowed_axis_px = r_expected * 1.33;
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::ScaleOutOfRange,
            context: CompletionGateRejectContext::ScaleOutOfRange {
                observed_mean_axis_px: quality.mean_axis,
                expected_radius_px: r_expected,
                min_allowed_axis_px,
                max_allowed_axis_px,
            },
        });
    }
    if quality.max_angular_gap_outer > max_angular_gap_rad {
        return Err(CompletionGateReject {
            reason: CompletionGateRejectReason::AngularGapTooLarge,
            context: CompletionGateRejectContext::AngularGapTooLarge {
                observed_gap_rad: quality.max_angular_gap_outer,
                max_allowed_gap_rad: max_angular_gap_rad,
            },
        });
    }
    Ok(())
}

/// Evaluate the quality gates for a completion candidate.
///
/// Returns `Ok(quality)` if all gates pass, or `Err(())` if the candidate
/// should be rejected (with appropriate stats and tracing already applied).
fn evaluate_completion_candidate(
    id: usize,
    cand: &OuterFitCandidate,
    projected_center: [f64; 2],
    r_expected: f32,
    config: &DetectConfig,
    active_codebook_min_cyclic_dist: u8,
    stats: &mut CompletionStats,
) -> Result<CandidateQuality, ()> {
    let params = &config.completion;
    let quality = compute_candidate_quality(
        &cand.edge,
        &cand.outer,
        cand.outer_ransac.as_ref(),
        projected_center,
        r_expected,
    );

    if let Err(reject) = check_quality_gates(
        &quality,
        params,
        r_expected,
        config.outer_fit.max_angular_gap_rad,
    ) {
        tracing::trace!(
            "Completion id={} gate_reject={} context={:?}",
            id,
            reject.reason,
            reject.context
        );
        stats.n_failed_gate += 1;
        return Err(());
    }

    if params.require_perfect_decode {
        let is_perfect = cand
            .decode_result
            .as_ref()
            .is_some_and(|d| d.dist == 0 && d.margin >= active_codebook_min_cyclic_dist);
        if !is_perfect {
            tracing::trace!(
                "Completion id={} gate_reject={} (dist={:?}, margin={:?})",
                id,
                CompletionGateRejectReason::PerfectDecodeRequired.code(),
                cand.decode_result.as_ref().map(|d| d.dist),
                cand.decode_result.as_ref().map(|d| d.margin),
            );
            stats.n_failed_gate += 1;
            return Err(());
        }
    }

    Ok(quality)
}

/// Build the final `DetectedMarker` for an accepted completion candidate.
///
/// Performs inner ellipse fit, computes fit/decode metrics, and assembles the
/// marker struct.
fn assemble_completion_marker(
    gray: &GrayImage,
    id: usize,
    cand: &OuterFitCandidate,
    quality: &CandidateQuality,
    config: &DetectConfig,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
) -> DetectedMarker {
    let inner_fit = super::inner_fit::fit_inner_ellipse_from_outer_hint(
        gray,
        &cand.outer,
        &config.marker_spec,
        mapper,
        &config.inner_fit,
        false,
    );
    let fit = fit_metrics_with_inner(
        &cand.edge,
        &cand.outer,
        cand.outer_ransac.as_ref(),
        &inner_fit,
    );
    let decode_metrics =
        decode_metrics_from_result(cand.decode_result.as_ref().filter(|d| d.id == id));
    let confidence = decode_metrics
        .as_ref()
        .map(|d| d.decode_confidence)
        .unwrap_or(quality.fit_confidence);

    DetectedMarker {
        id: Some(id),
        confidence,
        center: quality.center,
        ellipse_outer: Some(cand.outer),
        ellipse_inner: inner_fit.ellipse_inner,
        edge_points_outer: Some(cand.edge.outer_points.clone()),
        edge_points_inner: Some(inner_fit.points_inner.clone()),
        fit,
        decode: decode_metrics,
        source: DetectionSource::Completion,
        ..DetectedMarker::default()
    }
}

/// Check whether a projected center is valid and inside the image with margin.
///
/// Returns `true` if the center is finite and at least `safe_margin` pixels
/// away from every image edge.
fn is_center_in_bounds(center: [f64; 2], img_w: f64, img_h: f64, safe_margin: f64) -> bool {
    center[0].is_finite()
        && center[1].is_finite()
        && center[0] >= safe_margin
        && center[0] < (img_w - safe_margin)
        && center[1] >= safe_margin
        && center[1] < (img_h - safe_margin)
}

/// Attempt to fit and validate a single completion marker at a projected center.
fn try_complete_marker(
    gray: &GrayImage,
    id: usize,
    projected_center: [f64; 2],
    markers: &[DetectedMarker],
    config: &DetectConfig,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
    stats: &mut CompletionStats,
) -> Option<DetectedMarker> {
    let active_codebook_min_cyclic_dist =
        Codebook::from_profile(config.decode.codebook_profile).min_cyclic_dist() as u8;
    let r_expected = median_outer_radius_from_neighbors_px(projected_center, markers, 12)
        .unwrap_or(config.marker_scale.nominal_outer_radius_px());

    let cand = match fit_outer_candidate_from_prior_for_completion(
        gray,
        [projected_center[0] as f32, projected_center[1] as f32],
        r_expected,
        config,
        mapper,
    ) {
        Ok(v) => v,
        Err(_) => {
            stats.n_failed_fit += 1;
            return None;
        }
    };

    let quality = match evaluate_completion_candidate(
        id,
        &cand,
        projected_center,
        r_expected,
        config,
        active_codebook_min_cyclic_dist,
        stats,
    ) {
        Ok(q) => q,
        Err(()) => return None,
    };

    if let Some(notice) = check_decode_gate(cand.decode_result.as_ref(), id) {
        tracing::info!(
            "Completion id={} {} expected={} observed={}",
            id,
            notice.reason,
            notice.expected_id,
            notice.observed_id
        );
        stats.n_decode_mismatch += 1;
    }

    tracing::debug!(
        "Completion added id={} reproj_err={:.2}px",
        id,
        quality.reproj_err
    );

    Some(assemble_completion_marker(
        gray, id, &cand, &quality, config, mapper,
    ))
}

pub(crate) fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<DetectedMarker>,
    config: &DetectConfig,
    board: &crate::board_layout::BoardLayout,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
) -> CompletionStats {
    use std::collections::HashSet;

    let params = &config.completion;
    if !params.enable {
        return CompletionStats::default();
    }

    let (w, h_img) = gray.dimensions();
    let w_f = w as f64;
    let h_f = h_img as f64;

    let roi_radius = params.roi_radius_px.clamp(8.0, 200.0) as f64;
    let safe_margin = roi_radius + params.image_margin_px.max(0.0) as f64;

    let present_ids: HashSet<usize> = markers.iter().filter_map(|m| m.id).collect();

    let hex_grid = crate::pipeline::build_hex_grid_map(markers, board);

    let mut stats = CompletionStats {
        n_candidates_total: board.n_markers(),
        ..Default::default()
    };
    let mut attempted_fits = 0usize;

    for id in board.marker_ids() {
        let projected_center = match projected_completion_seed(id, h, markers, board, &hex_grid) {
            Some(center) => center,
            None => continue,
        };

        if present_ids.contains(&id) {
            continue;
        }

        if !is_center_in_bounds(projected_center, w_f, h_f, safe_margin) {
            continue;
        }
        stats.n_in_image += 1;

        if let Some(max) = params.max_attempts {
            if attempted_fits >= max {
                break;
            }
        }
        attempted_fits += 1;
        stats.n_attempted += 1;

        let marker = match try_complete_marker(
            gray,
            id,
            projected_center,
            markers,
            config,
            mapper,
            &mut stats,
        ) {
            Some(m) => m,
            None => continue,
        };
        markers.push(marker);
        stats.n_added += 1;
    }

    tracing::info!(
        "Completion: added {} markers (attempted {}, in_image {})",
        stats.n_added,
        stats.n_attempted,
        stats.n_in_image
    );

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::detector::marker_build::FitMetrics;

    #[test]
    fn completion_gate_reason_serialization_is_stable() {
        let reason = CompletionGateRejectReason::ScaleOutOfRange;
        assert_eq!(reason.to_string(), "scale_out_of_range");
        let json = serde_json::to_string(&reason).expect("serialize completion gate reason");
        assert_eq!(json, "\"scale_out_of_range\"");
    }

    #[test]
    fn completion_quality_gate_reports_typed_arc_coverage_context() {
        let quality = CandidateQuality {
            center: [0.0, 0.0],
            arc_cov: 0.2,
            fit_confidence: 0.9,
            mean_axis: 20.0,
            scale_ok: true,
            reproj_err: 0.5,
            max_angular_gap_outer: 0.1,
            radii_cv: 0.0,
        };
        let params = CompletionParams::default();
        let reject = check_quality_gates(&quality, &params, 20.0, std::f64::consts::FRAC_PI_2)
            .expect_err("expected gate fail");
        assert_eq!(reject.reason, CompletionGateRejectReason::ArcCoverageLow);
        match reject.context {
            CompletionGateRejectContext::ArcCoverageLow {
                observed_arc_coverage,
                min_required_arc_coverage,
            } => {
                assert!(observed_arc_coverage < min_required_arc_coverage);
            }
            other => panic!("unexpected completion gate context: {other:?}"),
        }
    }

    fn marker_with_id(id: usize, center: [f64; 2]) -> DetectedMarker {
        DetectedMarker {
            id: Some(id),
            confidence: 1.0,
            center,
            ellipse_outer: Some(Ellipse {
                cx: center[0],
                cy: center[1],
                a: 10.0,
                b: 10.0,
                angle: 0.0,
            }),
            fit: FitMetrics::default(),
            source: DetectionSource::FitDecoded,
            ..DetectedMarker::default()
        }
    }

    #[test]
    fn local_affine_completion_seed_uses_nearest_decoded_neighbors() {
        let board = crate::BoardLayout::default();
        let target_id = 16usize;
        let neighbor_ids = [0usize, 1usize, 14usize, 15usize];
        let affine = [[2.0, 0.1, 5.0], [-0.2, 1.5, 7.0]];
        let markers: Vec<DetectedMarker> = neighbor_ids
            .iter()
            .map(|&id| {
                let board_xy = board.xy_mm(id).expect("board xy");
                let center =
                    affine_to_image(&affine, [f64::from(board_xy[0]), f64::from(board_xy[1])]);
                marker_with_id(id, center)
            })
            .collect();

        let target_board_xy = board.xy_mm(target_id).expect("target board xy");
        let seed = local_affine_completion_seed(
            [f64::from(target_board_xy[0]), f64::from(target_board_xy[1])],
            &markers,
            &board,
        )
        .expect("local affine seed");
        let expected = affine_to_image(
            &affine,
            [f64::from(target_board_xy[0]), f64::from(target_board_xy[1])],
        );
        assert!((seed[0] - expected[0]).abs() < 1e-6);
        assert!((seed[1] - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn projected_completion_seed_falls_back_to_h_with_fewer_than_three_neighbors() {
        let board = crate::BoardLayout::default();
        let target_id = 16usize;
        let markers = vec![
            marker_with_id(0, [11.0, 7.0]),
            marker_with_id(1, [19.0, 7.5]),
        ];
        let h = nalgebra::Matrix3::new(1.0, 0.0, 3.0, 0.0, 1.0, -4.0, 0.0, 0.0, 1.0);
        let target_board_xy = board.xy_mm(target_id).expect("target board xy");
        let hex_grid = crate::pipeline::build_hex_grid_map(&markers, &board);
        let seed =
            projected_completion_seed(target_id, &h, &markers, &board, &hex_grid).expect("seed");
        let expected = project(
            &h,
            f64::from(target_board_xy[0]),
            f64::from(target_board_xy[1]),
        );
        assert!((seed[0] - expected[0]).abs() < 1e-9);
        assert!((seed[1] - expected[1]).abs() < 1e-9);
    }
}
