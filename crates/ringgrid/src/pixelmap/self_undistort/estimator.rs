use crate::board_layout::BoardLayout;
use crate::DetectedMarker;

use super::super::DivisionModel;
use super::config::SelfUndistortConfig;
use super::objective::{
    collect_marker_edge_data, conic_consistency_objective, homography_self_error_px, MarkerEdgeData,
};
use super::optimizer::golden_section_minimize;
use super::policy::{should_apply_model, EstimateCandidate};
use super::result::SelfUndistortResult;

#[derive(Debug)]
enum ObjectiveStrategy<'a> {
    Homography {
        board: &'a BoardLayout,
        baseline_error_px: f64,
        n_markers_used: usize,
    },
    Conic {
        marker_edge_data: Vec<MarkerEdgeData>,
    },
}

#[derive(Debug, Clone, Copy)]
struct OptimizationOutcome {
    objective_at_zero: f64,
    lambda_opt: f64,
    objective_at_lambda: f64,
    n_markers_used: usize,
}

/// Estimate a division-model distortion parameter from detected markers.
///
/// Uses a robust mean of Sampson residuals of fitted inner/outer ellipses:
/// correct distortion makes ring boundaries more conic-like.
///
/// Returns `None` if fewer than `config.min_markers` have both inner and
/// outer edge points with sufficient count and homography-based objective is
/// unavailable.
pub fn estimate_self_undistort(
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
    board: Option<&BoardLayout>,
) -> Option<SelfUndistortResult> {
    let strategy = select_objective_strategy(markers, image_size, config, board)?;
    let outcome = optimize_strategy(markers, image_size, config, strategy)?;

    let candidate = EstimateCandidate {
        lambda_opt: outcome.lambda_opt,
        objective_at_zero: outcome.objective_at_zero,
        objective_at_lambda: outcome.objective_at_lambda,
    };

    let applied = should_apply_model(candidate, markers, image_size, config, board);

    Some(SelfUndistortResult {
        model: DivisionModel::centered(outcome.lambda_opt, image_size[0], image_size[1]),
        objective_at_lambda: outcome.objective_at_lambda,
        objective_at_zero: outcome.objective_at_zero,
        n_markers_used: outcome.n_markers_used,
        applied,
    })
}

fn select_objective_strategy<'a>(
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
    board: Option<&'a BoardLayout>,
) -> Option<ObjectiveStrategy<'a>> {
    if let Some(board) = board {
        let zero_model = DivisionModel::centered(0.0, image_size[0], image_size[1]);
        if let Some(err0) = homography_self_error_px(markers, board, &zero_model) {
            if err0.n_inliers >= config.validation_min_markers {
                return Some(ObjectiveStrategy::Homography {
                    board,
                    baseline_error_px: err0.mean_error_px,
                    n_markers_used: err0.n_inliers,
                });
            }
        }
    }

    let marker_edge_data = collect_marker_edge_data(markers);
    if marker_edge_data.len() < config.min_markers {
        return None;
    }

    Some(ObjectiveStrategy::Conic { marker_edge_data })
}

fn optimize_strategy(
    markers: &[DetectedMarker],
    image_size: [u32; 2],
    config: &SelfUndistortConfig,
    strategy: ObjectiveStrategy<'_>,
) -> Option<OptimizationOutcome> {
    let image_center = [image_size[0] as f64 / 2.0, image_size[1] as f64 / 2.0];

    match strategy {
        ObjectiveStrategy::Homography {
            board,
            baseline_error_px,
            n_markers_used,
        } => {
            let (lambda_opt, objective_at_lambda) = golden_section_minimize(
                |lambda| {
                    let model = DivisionModel::centered(lambda, image_size[0], image_size[1]);
                    homography_self_error_px(markers, board, &model)
                        .map(|v| v.mean_error_px)
                        .unwrap_or(f64::MAX)
                },
                config.lambda_range[0],
                config.lambda_range[1],
                config.max_evals,
            );

            Some(OptimizationOutcome {
                objective_at_zero: baseline_error_px,
                lambda_opt,
                objective_at_lambda,
                n_markers_used,
            })
        }
        ObjectiveStrategy::Conic { marker_edge_data } => {
            let objective_at_zero = conic_consistency_objective(
                0.0,
                &marker_edge_data,
                image_center,
                config.trim_fraction,
            );
            if !objective_at_zero.is_finite() {
                return None;
            }

            let (lambda_opt, objective_at_lambda) = golden_section_minimize(
                |lambda| {
                    conic_consistency_objective(
                        lambda,
                        &marker_edge_data,
                        image_center,
                        config.trim_fraction,
                    )
                },
                config.lambda_range[0],
                config.lambda_range[1],
                config.max_evals,
            );

            Some(OptimizationOutcome {
                objective_at_zero,
                lambda_opt,
                objective_at_lambda,
                n_markers_used: marker_edge_data.len(),
            })
        }
    }
}
