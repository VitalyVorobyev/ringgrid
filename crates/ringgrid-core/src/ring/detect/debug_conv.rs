use crate::conic::Ellipse;
use crate::debug_dump as dbg;
use crate::refine::{MarkerRefineRecord, MarkerRefineStatus, RefineStats};
use crate::ring::inner_estimate::{InnerEstimate, InnerStatus, Polarity};
use crate::ring::outer_estimate::{OuterEstimate, OuterStatus};
use crate::{EllipseParams, FitMetrics};

use super::completion::{CompletionAttemptRecord, CompletionAttemptStatus, CompletionStats};

pub(super) fn ellipse_from_params(p: &EllipseParams) -> dbg::EllipseParamsDebugV1 {
    dbg::EllipseParamsDebugV1 {
        center_xy: [p.center_xy[0] as f32, p.center_xy[1] as f32],
        semi_axes: [p.semi_axes[0] as f32, p.semi_axes[1] as f32],
        angle: p.angle as f32,
    }
}

pub(super) fn ellipse_from_conic(e: &Ellipse) -> dbg::EllipseParamsDebugV1 {
    dbg::EllipseParamsDebugV1 {
        center_xy: [e.cx as f32, e.cy as f32],
        semi_axes: [e.a as f32, e.b as f32],
        angle: e.angle as f32,
    }
}

pub(super) fn ring_fit_metrics(
    fit: &FitMetrics,
    arc_coverage: f32,
    valid_inner: bool,
    valid_outer: bool,
) -> dbg::RingFitMetricsDebugV1 {
    dbg::RingFitMetricsDebugV1 {
        inlier_ratio_inner: fit.ransac_inlier_ratio_inner,
        inlier_ratio_outer: fit.ransac_inlier_ratio_outer,
        mean_resid_inner: fit.rms_residual_inner.map(|v| v as f32),
        mean_resid_outer: fit.rms_residual_outer.map(|v| v as f32),
        arc_coverage,
        valid_inner,
        valid_outer,
    }
}

pub(super) fn outer_estimation_debug(
    outer_estimate: &OuterEstimate,
    chosen_hypothesis: usize,
) -> dbg::OuterEstimationDebugV1 {
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
}

pub(super) fn inner_estimation_debug(estimate: &InnerEstimate) -> dbg::InnerEstimationDebugV1 {
    dbg::InnerEstimationDebugV1 {
        r_inner_expected: estimate.r_inner_expected,
        search_window: estimate.search_window,
        r_inner_found: estimate.r_inner_found,
        polarity: estimate.polarity.map(|p| match p {
            Polarity::Pos => dbg::InnerPolarityDebugV1::Pos,
            Polarity::Neg => dbg::InnerPolarityDebugV1::Neg,
        }),
        peak_strength: estimate.peak_strength,
        theta_consistency: estimate.theta_consistency,
        status: match estimate.status {
            InnerStatus::Ok => dbg::InnerEstimationStatusDebugV1::Ok,
            InnerStatus::Rejected => dbg::InnerEstimationStatusDebugV1::Rejected,
            InnerStatus::Failed => dbg::InnerEstimationStatusDebugV1::Failed,
        },
        reason: estimate.reason.clone(),
        radial_response_agg: estimate.radial_response_agg.clone(),
        r_samples: estimate.r_samples.clone(),
    }
}

pub(super) fn completion_attempt_status(
    status: CompletionAttemptStatus,
) -> dbg::CompletionAttemptStatusDebugV1 {
    match status {
        CompletionAttemptStatus::Added => dbg::CompletionAttemptStatusDebugV1::Added,
        CompletionAttemptStatus::SkippedPresent => {
            dbg::CompletionAttemptStatusDebugV1::SkippedPresent
        }
        CompletionAttemptStatus::SkippedOob => dbg::CompletionAttemptStatusDebugV1::SkippedOob,
        CompletionAttemptStatus::FailedFit => dbg::CompletionAttemptStatusDebugV1::FailedFit,
        CompletionAttemptStatus::FailedGate => dbg::CompletionAttemptStatusDebugV1::FailedGate,
    }
}

pub(super) fn completion_attempt(a: CompletionAttemptRecord) -> dbg::CompletionAttemptDebugV1 {
    dbg::CompletionAttemptDebugV1 {
        id: a.id,
        projected_center_xy: a.projected_center_xy,
        status: completion_attempt_status(a.status),
        reason: a.reason,
        reproj_err_px: a.reproj_err_px,
        fit_confidence: a.fit_confidence,
        fit: a.fit,
    }
}

pub(super) fn completion_stats(stats: &CompletionStats) -> dbg::CompletionStatsDebugV1 {
    dbg::CompletionStatsDebugV1 {
        n_candidates_total: stats.n_candidates_total,
        n_in_image: stats.n_in_image,
        n_attempted: stats.n_attempted,
        n_added: stats.n_added,
        n_failed_fit: stats.n_failed_fit,
        n_failed_gate: stats.n_failed_gate,
    }
}

pub(super) fn nl_status(status: MarkerRefineStatus) -> dbg::NlRefineStatusDebugV1 {
    match status {
        MarkerRefineStatus::Ok => dbg::NlRefineStatusDebugV1::Ok,
        MarkerRefineStatus::Rejected => dbg::NlRefineStatusDebugV1::Rejected,
        MarkerRefineStatus::Failed => dbg::NlRefineStatusDebugV1::Failed,
        MarkerRefineStatus::Skipped => dbg::NlRefineStatusDebugV1::Skipped,
    }
}

pub(super) fn nl_stats(stats: &RefineStats) -> dbg::NlRefineStatsDebugV1 {
    dbg::NlRefineStatsDebugV1 {
        n_inliers: stats.n_inliers,
        n_refined: stats.n_refined,
        n_failed: stats.n_failed,
        mean_before_mm: stats.mean_before_mm,
        mean_after_mm: stats.mean_after_mm,
        p95_before_mm: stats.p95_before_mm,
        p95_after_mm: stats.p95_after_mm,
    }
}

pub(super) fn nl_record(r: MarkerRefineRecord) -> dbg::NlRefinedMarkerDebugV1 {
    dbg::NlRefinedMarkerDebugV1 {
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
        status: nl_status(r.status),
        reason: r.reason,
    }
}
