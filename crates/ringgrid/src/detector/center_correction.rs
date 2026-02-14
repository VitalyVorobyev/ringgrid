use crate::conic::Conic2D;
use crate::DetectedMarker;

use super::{CircleRefinementMethod, DetectConfig};

pub(crate) fn warn_center_correction_without_intrinsics(config: &DetectConfig, has_mapper: bool) {
    if config.circle_refinement == CircleRefinementMethod::None || has_mapper {
        return;
    }

    tracing::warn!(
        "center correction is running without camera intrinsics/undistortion; \
         lens distortion can still bias corrected centers"
    );
}

/// Apply projective center correction, skipping markers that already have it.
///
/// Use this for the initial pass and for completion markers that need correction
/// without disturbing markers that were already corrected.
pub(crate) fn apply_projective_centers(markers: &mut [DetectedMarker], config: &DetectConfig) {
    apply_projective_centers_impl(markers, config, false);
}

/// Reapply projective center correction to all markers, clearing previous results.
///
/// Use this after refine-H, which produces new ellipses that invalidate the
/// previous projective center estimate.
pub(crate) fn reapply_projective_centers(markers: &mut [DetectedMarker], config: &DetectConfig) {
    apply_projective_centers_impl(markers, config, true);
}

fn apply_projective_centers_impl(
    markers: &mut [DetectedMarker],
    config: &DetectConfig,
    force: bool,
) {
    use crate::ring::projective_center::{
        ring_center_projective_with_debug, RingCenterProjectiveOptions,
    };

    if force {
        for m in markers.iter_mut() {
            m.center_projective = None;
            m.center_projective_residual = None;
        }
    }

    if !config.projective_center.enable {
        return;
    }

    let expected_ratio = if config.projective_center.use_expected_ratio {
        Some(config.marker_spec.r_inner_expected as f64)
    } else {
        None
    };
    let opts = RingCenterProjectiveOptions {
        expected_ratio,
        ratio_penalty_weight: config.projective_center.ratio_penalty_weight,
        ..Default::default()
    };

    let mut n_skipped = 0usize;
    let mut n_missing_conics = 0usize;
    let mut n_solver_failed = 0usize;
    let mut n_rejected_shift = 0usize;
    let mut n_rejected_residual = 0usize;
    let mut n_rejected_eig_sep = 0usize;
    let mut n_applied = 0usize;

    for m in markers.iter_mut() {
        if !force && m.center_projective.is_some() {
            n_skipped += 1;
            continue;
        }

        let (Some(inner), Some(outer)) = (m.ellipse_inner.as_ref(), m.ellipse_outer.as_ref())
        else {
            n_missing_conics += 1;
            continue;
        };

        let center_before = m.center;
        let q_inner = Conic2D::from_ellipse(inner).mat;
        let q_outer = Conic2D::from_ellipse(outer).mat;
        let Ok(res) = ring_center_projective_with_debug(&q_inner, &q_outer, opts) else {
            n_solver_failed += 1;
            continue;
        };

        if let Some(max_residual) = config.projective_center.max_selected_residual {
            if !res.debug.selected_residual.is_finite()
                || res.debug.selected_residual > max_residual
            {
                n_rejected_residual += 1;
                continue;
            }
        }

        if let Some(min_sep) = config.projective_center.min_eig_separation {
            if !res.debug.selected_eig_separation.is_finite()
                || res.debug.selected_eig_separation < min_sep
            {
                n_rejected_eig_sep += 1;
                continue;
            }
        }

        let center_projective = [res.center.x, res.center.y];
        let dx = center_projective[0] - center_before[0];
        let dy = center_projective[1] - center_before[1];
        let center_shift = (dx * dx + dy * dy).sqrt();
        if let Some(max_shift_px) = config.projective_center.max_center_shift_px {
            if !center_shift.is_finite() || center_shift > max_shift_px {
                n_rejected_shift += 1;
                continue;
            }
        }

        m.center = center_projective;
        m.center_projective = Some(center_projective);
        m.center_projective_residual = Some(res.debug.selected_residual);
        n_applied += 1;
    }

    tracing::debug!(
        applied = n_applied,
        skipped = n_skipped,
        missing_conics = n_missing_conics,
        solver_failed = n_solver_failed,
        rejected_shift = n_rejected_shift,
        rejected_residual = n_rejected_residual,
        rejected_eig_sep = n_rejected_eig_sep,
        "projective-center application summary (force={})",
        force,
    );
}
