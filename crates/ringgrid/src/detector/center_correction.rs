use crate::conic::Conic2D;
use crate::detector::MarkerRecord;

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

/// Largest accepted centre shift of a correction: `None` in the config means
/// "auto", the nominal marker diameter — a correction jumping further is a
/// solver artifact.
fn max_correction_shift_px(config: &DetectConfig) -> f64 {
    config
        .advanced
        .projective_center
        .max_correction_shift_px
        .unwrap_or(f64::from(config.marker_scale.nominal_diameter_px()))
}

/// Apply projective center correction to the provided marker slice.
pub(crate) fn apply_projective_centers(markers: &mut [MarkerRecord], config: &DetectConfig) {
    use crate::ring::projective_center::{
        RingCenterProjectiveOptions, ring_center_projective_with_debug,
    };

    if !config.circle_refinement.uses_projective_center() {
        return;
    }

    let expected_ratio = if config.advanced.projective_center.use_expected_ratio {
        Some(config.advanced.marker_spec.r_inner_expected as f64)
    } else {
        None
    };
    let opts = RingCenterProjectiveOptions {
        expected_ratio,
        ratio_penalty_weight: config.advanced.projective_center.ratio_penalty_weight,
        ..Default::default()
    };

    let max_shift_px = max_correction_shift_px(config);

    let mut n_missing_conics = 0usize;
    let mut n_solver_failed = 0usize;
    let mut n_rejected_shift = 0usize;
    let mut n_rejected_residual = 0usize;
    let mut n_rejected_eig_sep = 0usize;
    let mut n_applied = 0usize;

    for m in markers.iter_mut() {
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

        if let Some(max_residual) = config.advanced.projective_center.max_selected_residual
            && (!res.debug.selected_residual.is_finite()
                || res.debug.selected_residual > max_residual)
        {
            n_rejected_residual += 1;
            continue;
        }

        if let Some(min_sep) = config.advanced.projective_center.min_eig_separation
            && (!res.debug.selected_eig_separation.is_finite()
                || res.debug.selected_eig_separation < min_sep)
        {
            n_rejected_eig_sep += 1;
            continue;
        }

        let center_projective = [res.center.x, res.center.y];
        let dx = center_projective[0] - center_before[0];
        let dy = center_projective[1] - center_before[1];
        let center_shift = (dx * dx + dy * dy).sqrt();
        if !center_shift.is_finite() || center_shift > max_shift_px {
            n_rejected_shift += 1;
            continue;
        }

        m.center = center_projective;
        m.center_source = crate::detector::CenterSource::ProjectiveCenter;
        n_applied += 1;
    }

    tracing::debug!(
        applied = n_applied,
        missing_conics = n_missing_conics,
        solver_failed = n_solver_failed,
        rejected_shift = n_rejected_shift,
        rejected_residual = n_rejected_residual,
        rejected_eig_sep = n_rejected_eig_sep,
        "projective-center application summary",
    );
}

/// Perspective-correct, from the board homography `h` (board plane → the
/// markers' frame), the centre of every labeled marker that still carries
/// its outer-ellipse centre — no inner ellipse, or the two-conic correction
/// was rejected. The image of a circle's centre is the pole of the plane's
/// vanishing line `h⁻ᵀ (0, 0, 1)ᵀ` with respect to the circle's image
/// ([`ring_center_from_vanishing_line`]); the outer ellipse alone then
/// suffices. Same shift gate as the two-conic correction.
///
/// [`ring_center_from_vanishing_line`]: crate::ring::projective_center::ring_center_from_vanishing_line
pub(crate) fn apply_vanishing_line_centres(
    markers: &mut [MarkerRecord],
    h: &nalgebra::Matrix3<f64>,
    config: &DetectConfig,
) {
    use crate::detector::CenterSource;
    use crate::ring::projective_center::ring_center_from_vanishing_line;

    if !config.circle_refinement.uses_projective_center()
        || std::env::var("RINGGRID_EXP_OFF").is_ok()
    {
        return;
    }
    let Some(h_inv) = h.try_inverse() else {
        return;
    };
    let global_line = h_inv.transpose() * nalgebra::Vector3::new(0.0, 0.0, 1.0);
    let max_shift_px = max_correction_shift_px(config);
    let local = std::env::var("RINGGRID_EXP_LOCAL_H").is_ok();
    let anchors: Vec<([f64; 2], [f64; 2])> = markers
        .iter()
        .filter(|m| m.center_source == CenterSource::ProjectiveCenter)
        .filter_map(|m| Some((m.board_xy_mm?, m.center)))
        .collect();
    let radius_mm = 2.05 * f64::from(config.target.min_center_spacing_mm());
    let (mut n_applied, mut n_rejected) = (0usize, 0usize);
    for m in markers.iter_mut() {
        if m.center_source != CenterSource::OuterEllipse {
            continue;
        }
        let (Some(ellipse), Some(xy)) = (m.ellipse_outer.as_ref(), m.board_xy_mm) else {
            continue;
        };
        let line = if local {
            let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = anchors
                .iter()
                .filter(|(p, _)| (p[0] - xy[0]).hypot(p[1] - xy[1]) <= radius_mm)
                .copied()
                .unzip();
            if src.len() < 8 {
                n_rejected += 1;
                continue;
            }
            let Some(hl) = crate::homography::estimate_homography_dlt(&src, &dst)
                .ok()
                .and_then(|hl| hl.try_inverse())
            else {
                n_rejected += 1;
                continue;
            };
            hl.transpose() * nalgebra::Vector3::new(0.0, 0.0, 1.0)
        } else {
            global_line
        };
        let accepted = ring_center_from_vanishing_line(ellipse, &line).filter(|c| {
            let shift = (c[0] - m.center[0]).hypot(c[1] - m.center[1]);
            shift.is_finite() && shift <= max_shift_px
        });
        let Some(center) = accepted else {
            n_rejected += 1;
            continue;
        };
        m.center = center;
        m.center_source = CenterSource::OuterWithHomography;
        n_applied += 1;
    }
    tracing::debug!(
        applied = n_applied,
        rejected = n_rejected,
        "vanishing-line centre summary"
    );
}
