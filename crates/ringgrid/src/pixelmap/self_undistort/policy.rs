use crate::board_layout::BoardLayout;
use crate::DetectedMarker;

use super::super::DivisionModel;
use super::config::SelfUndistortConfig;
use super::objective::homography_self_error_px;

#[derive(Debug, Clone, Copy)]
pub(super) struct EstimateCandidate {
    pub lambda_opt: f64,
    pub objective_at_zero: f64,
    pub objective_at_lambda: f64,
}

pub(super) fn should_apply_model(
    candidate: EstimateCandidate,
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
    board: Option<&BoardLayout>,
) -> bool {
    passes_primary_gates(candidate, config)
        && passes_range_edge_gate(candidate.lambda_opt, config)
        && passes_homography_validation(candidate.lambda_opt, markers, image_size, config, board)
}

fn passes_primary_gates(candidate: EstimateCandidate, config: &SelfUndistortConfig) -> bool {
    let abs_improvement = candidate.objective_at_zero - candidate.objective_at_lambda;
    let rel_improvement = if candidate.objective_at_zero > 1e-18 {
        abs_improvement / candidate.objective_at_zero
    } else {
        0.0
    };

    candidate.objective_at_lambda.is_finite()
        && abs_improvement.is_finite()
        && abs_improvement > config.min_abs_improvement
        && rel_improvement > config.improvement_threshold
        && candidate.lambda_opt.abs() >= config.min_lambda_abs
}

fn passes_range_edge_gate(lambda_opt: f64, config: &SelfUndistortConfig) -> bool {
    if !config.reject_range_edge {
        return true;
    }

    let lo = config.lambda_range[0].min(config.lambda_range[1]);
    let hi = config.lambda_range[0].max(config.lambda_range[1]);
    let span = (hi - lo).abs();
    if span == 0.0 {
        return true;
    }

    let margin = span * config.range_edge_margin_frac.clamp(0.0, 0.49);
    !((lambda_opt - lo).abs() <= margin || (hi - lambda_opt).abs() <= margin)
}

fn passes_homography_validation(
    lambda_opt: f64,
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
    board: Option<&BoardLayout>,
) -> bool {
    let Some(board) = board else {
        return true;
    };

    let zero_model = DivisionModel::centered(0.0, image_size[0], image_size[1]);
    let opt_model = DivisionModel::centered(lambda_opt, image_size[0], image_size[1]);
    let err0 = homography_self_error_px(markers, board, &zero_model);
    let err1 = homography_self_error_px(markers, board, &opt_model);

    let (Some(err0), Some(err1)) = (err0, err1) else {
        return true;
    };

    let enough_ids = err0.n_inliers >= config.validation_min_markers
        && err1.n_inliers >= config.validation_min_markers;
    if !enough_ids {
        return true;
    }

    let abs_gain = err0.mean_error_px - err1.mean_error_px;
    let rel_gain = if err0.mean_error_px > 1e-12 {
        abs_gain / err0.mean_error_px
    } else {
        0.0
    };

    let by_abs = abs_gain >= config.validation_abs_improvement_px;
    let by_rel = rel_gain >= config.validation_rel_improvement;
    by_abs && by_rel
}
