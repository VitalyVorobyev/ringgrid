//! Plain-target grid assignment.
//!
//! Plain (uncoded) rings carry no IDs, so the coded path's decode → global
//! RANSAC filter cannot label them. Instead, fitted ring centers are labeled
//! with lattice coordinates by `projective_grid::detect_grid` (topological
//! labeling only — its detection facade is `f32`), and the frame homography is
//! then refit in `f64` with ringgrid's homography RANSAC over the labeled
//! correspondences. Labels start in a canonical *relative* frame; resolving
//! them to absolute board cells is `pipeline::anchor`'s job.

use std::collections::HashMap;

use nalgebra::{Matrix3, Point2};
use projective_grid::{
    Coord, DetectionParams, DetectionRequest, Evidence, GridDimensions, GridEntry, LatticeKind,
    PointFeature, detect_grid,
};

use crate::conic::RansacConfig;
use crate::detector::MarkerRecord;
use crate::homography::{
    RansacStats, collect_masked_inlier_errors, fit_homography_ransac, mean_and_p95,
};
use crate::target::{LatticeGeometry, TargetLayout};

/// Assignment-frame position (mm) of a lattice coordinate.
///
/// The model-plane point scaled by the target's nearest-neighbor spacing, so
/// the relative frame has the board's physical scale: an anchored labeling is
/// congruent to the board cell set (rotation + translation away from
/// `cell_xy_mm`).
pub(crate) fn frame_xy_mm(target: &TargetLayout, coord: Coord) -> [f64; 2] {
    let p = target.lattice_kind().model_point(coord);
    let s = f64::from(target.min_center_spacing_mm());
    [s * f64::from(p.x), s * f64::from(p.y)]
}

/// Successful plain-grid assignment. Labeling counts are traced, not carried.
pub(crate) struct GridAssignment {
    /// Frame-mm → working-px homography fitted in `f64` over labeling inliers.
    pub h: Matrix3<f64>,
    /// RANSAC statistics of the frame homography fit.
    pub ransac: RansacStats,
}

/// Label plain-ring centers with lattice coordinates and fit the frame
/// homography.
///
/// On success, labeled homography-inlier markers survive with
/// `grid_coord` set (canonical relative frame) and `board_xy_mm` set to the
/// assignment-frame mm position; unlabeled and RANSAC-outlier markers are
/// removed — mirroring the coded global filter, which also keeps homography
/// inliers only. On failure (`None`) `markers` is left untouched.
pub(crate) fn assign_plain_grid(
    markers: &mut Vec<MarkerRecord>,
    target: &TargetLayout,
    config: &RansacConfig,
) -> Option<GridAssignment> {
    let features: Vec<PointFeature> = markers
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let x = m.center[0] as f32;
            let y = m.center[1] as f32;
            (x.is_finite() && y.is_finite()).then(|| PointFeature::new(i, Point2::new(x, y)))
        })
        .collect();
    let n_input = features.len();
    if n_input < 4 {
        tracing::warn!(n_input, "too few ring centers for plain grid labeling");
        return None;
    }

    let dimensions = match target.lattice() {
        LatticeGeometry::Rect(r) => Some(GridDimensions::new(r.cols, r.rows)),
        // Hex axial extents do not map onto GridDimensions' width/height
        // semantics; let the finder work unconstrained.
        LatticeGeometry::Hex(_) => None,
    };

    let request = DetectionRequest::new(
        target.lattice_kind(),
        Evidence::Positions(&features),
        dimensions,
        DetectionParams::default(),
    );
    let mut solution = match detect_grid(request) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "plain grid labeling failed");
            return None;
        }
    };

    match target.lattice_kind() {
        LatticeKind::Square => solution.grid.normalize(),
        // `LabelledGrid::normalize` sign-flips axes independently, which is not
        // a hex-lattice automorphism; canonicalize hex labels with a rotation
        // from the lattice symmetry group instead.
        _ => canonicalize_hex_entries(&mut solution.grid.entries),
    }

    let labeled: Vec<(usize, Coord)> = solution
        .grid
        .entries
        .iter()
        .map(|e| (e.source_index, e.coord))
        .collect();
    let n_labeled = labeled.len();
    let n_rejected = solution.rejected.len();
    if n_labeled < 4 {
        tracing::warn!(
            n_labeled,
            "too few labeled ring centers for frame homography"
        );
        return None;
    }

    let src: Vec<[f64; 2]> = labeled
        .iter()
        .map(|(_, c)| frame_xy_mm(target, *c))
        .collect();
    let dst: Vec<[f64; 2]> = labeled.iter().map(|(i, _)| markers[*i].center).collect();
    let result = match fit_homography_ransac(&src, &dst, config) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "frame homography RANSAC failed");
            return None;
        }
    };

    let mut inlier_errors = collect_masked_inlier_errors(&result.errors, &result.inlier_mask);
    let (mean_err, p95_err) = mean_and_p95(&mut inlier_errors);
    let ransac = RansacStats {
        n_candidates: n_labeled,
        n_inliers: result.n_inliers,
        threshold_px: config.inlier_threshold,
        mean_err_px: mean_err,
        p95_err_px: p95_err,
    };

    apply_inlier_labels(markers, &labeled, &result.inlier_mask, target);

    tracing::info!(
        n_input,
        n_labeled,
        n_rejected,
        n_inliers = result.n_inliers,
        mean_err_px = mean_err,
        "plain grid assignment complete"
    );

    Some(GridAssignment {
        h: result.h,
        ransac,
    })
}

/// Keep labeled homography inliers only, writing each survivor's `grid_coord`
/// and assignment-frame `board_xy_mm` in place; all other markers are removed
/// (the coded global filter drops its outliers the same way).
fn apply_inlier_labels(
    markers: &mut Vec<MarkerRecord>,
    labeled: &[(usize, Coord)],
    inlier_mask: &[bool],
    target: &TargetLayout,
) {
    let keep: HashMap<usize, Coord> = labeled
        .iter()
        .enumerate()
        .filter(|(j, _)| inlier_mask.get(*j).copied().unwrap_or(false))
        .map(|(_, (i, c))| (*i, *c))
        .collect();
    let old = std::mem::take(markers);
    *markers = old
        .into_iter()
        .enumerate()
        .filter_map(|(i, mut m)| {
            let coord = keep.get(&i)?;
            m.grid_coord = Some([coord.u, coord.v]);
            m.board_xy_mm = Some(frame_xy_mm(target, *coord));
            Some(m)
        })
        .collect();
}

/// Canonicalize hex axial labels: apply the rotation from the hex symmetry
/// group that best aligns the `+u` axis with image `+x` and `+v` with image
/// `+y`, then rebase the bounding-box minimum to `(0, 0)`.
///
/// Only rotations (determinant +1) are considered — an opaque planar target
/// images through an orientation-preserving homography, so a reflected
/// labeling cannot occur.
fn canonicalize_hex_entries(entries: &mut [GridEntry]) {
    if entries.len() < 2 {
        rebase_entries(entries);
        return;
    }

    let rotations = LatticeKind::Hex
        .symmetry_transforms()
        .iter()
        .filter(|t| t.determinant() > 0);

    let mut best: Option<(f32, projective_grid::GridTransform)> = None;
    for &rot in rotations {
        let pos_by_coord: HashMap<(i32, i32), (f32, f32)> = entries
            .iter()
            .map(|e| {
                let c = rot.apply(e.coord);
                ((c.u, c.v), (e.image_position.x, e.image_position.y))
            })
            .collect();
        let mut keys: Vec<(i32, i32)> = pos_by_coord.keys().copied().collect();
        keys.sort_unstable();
        let mut u_step = (0.0f32, 0.0f32);
        let mut v_step = (0.0f32, 0.0f32);
        let (mut nu, mut nv) = (0u32, 0u32);
        for &(u, v) in &keys {
            let (x, y) = pos_by_coord[&(u, v)];
            if let Some(&(xn, yn)) = pos_by_coord.get(&(u + 1, v)) {
                u_step = (u_step.0 + xn - x, u_step.1 + yn - y);
                nu += 1;
            }
            if let Some(&(xn, yn)) = pos_by_coord.get(&(u, v + 1)) {
                v_step = (v_step.0 + xn - x, v_step.1 + yn - y);
                nv += 1;
            }
        }
        if nu == 0 || nv == 0 {
            continue;
        }
        let norm = |(x, y): (f32, f32), n: u32| {
            let (x, y) = (x / n as f32, y / n as f32);
            let len = (x * x + y * y).sqrt().max(1e-6);
            (x / len, y / len)
        };
        let (ux, _) = norm(u_step, nu);
        let (_, vy) = norm(v_step, nv);
        let score = ux + vy;
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, rot));
        }
    }

    if let Some((_, rot)) = best {
        for e in entries.iter_mut() {
            e.coord = rot.apply(e.coord);
        }
    }
    rebase_entries(entries);
}

/// Shift entry coordinates so the bounding-box minimum is `(0, 0)`. Lattice
/// translations are automorphisms for both families, so this is always safe.
fn rebase_entries(entries: &mut [GridEntry]) {
    let Some(first) = entries.first() else { return };
    let mut min = first.coord;
    for e in entries.iter() {
        min.u = min.u.min(e.coord.u);
        min.v = min.v.min(e.coord.v);
    }
    if min.u != 0 || min.v != 0 {
        for e in entries.iter_mut() {
            e.coord.u -= min.u;
            e.coord.v -= min.v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{HexGeometry, RectGeometry, RingGeometry};

    fn rect_target(rows: usize, cols: usize, pitch_mm: f32) -> TargetLayout {
        TargetLayout::new(
            "test_rect",
            LatticeGeometry::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm,
            }),
            RingGeometry {
                outer_radius_mm: 0.35 * pitch_mm,
                inner_radius_mm: 0.175 * pitch_mm,
            },
            crate::target::MarkerCoding::Plain,
            None,
        )
        .expect("valid rect target")
    }

    fn marker_at(center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
            confidence: 1.0,
            center,
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn frame_xy_matches_rect_cell_geometry() {
        let target = rect_target(4, 5, 14.0);
        let xy = frame_xy_mm(&target, Coord::new(3, 2));
        assert!((xy[0] - 42.0).abs() < 1e-9);
        assert!((xy[1] - 28.0).abs() < 1e-9);
    }

    #[test]
    fn frame_xy_matches_hex_cell_spacing() {
        let hex = TargetLayout::default_hex();
        // Two axially adjacent cells must sit one nearest-neighbor spacing apart.
        let a = frame_xy_mm(&hex, Coord::new(0, 0));
        let b = frame_xy_mm(&hex, Coord::new(1, 0));
        let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        assert!((d - f64::from(hex.min_center_spacing_mm())).abs() < 1e-4);
    }

    #[test]
    fn assign_labels_full_rect_grid_and_fits_h() {
        let target = rect_target(5, 6, 10.0);
        // Synthetic image positions: affine map of the cell grid.
        let (s, tx, ty) = (7.0f64, 40.0f64, 60.0f64);
        let mut markers: Vec<MarkerRecord> = target
            .cells()
            .iter()
            .map(|cell| {
                marker_at([
                    s * f64::from(cell.xy_mm[0]) + tx,
                    s * f64::from(cell.xy_mm[1]) + ty,
                ])
            })
            .collect();
        let n = markers.len();

        let assignment = assign_plain_grid(&mut markers, &target, &RansacConfig::default())
            .expect("assignment must succeed on a clean grid");

        assert_eq!(markers.len(), n, "no clean marker may be dropped");
        assert!(markers.iter().all(|m| m.grid_coord.is_some()));
        assert!(assignment.ransac.mean_err_px < 0.1);

        // Labels must be a canonical relabeling: the frame H must map each
        // marker's frame position onto its center.
        for m in &markers {
            let xy = m.board_xy_mm.expect("frame xy set");
            let p = crate::homography::homography_project(&assignment.h, xy[0], xy[1]);
            let err = ((p[0] - m.center[0]).powi(2) + (p[1] - m.center[1]).powi(2)).sqrt();
            assert!(err < 0.5, "frame reprojection error {err} too large");
        }

        // Canonical frame: coords non-negative, +u aligned with image +x.
        let mut by_coord: HashMap<(i32, i32), [f64; 2]> = HashMap::new();
        for m in &markers {
            let c = m.grid_coord.unwrap();
            assert!(c[0] >= 0 && c[1] >= 0);
            by_coord.insert((c[0], c[1]), m.center);
        }
        let a = by_coord[&(0, 0)];
        let b = by_coord[&(1, 0)];
        assert!(b[0] > a[0], "+u must point toward image +x");
    }

    #[test]
    fn assign_returns_none_and_keeps_markers_when_too_few() {
        let target = rect_target(5, 6, 10.0);
        let mut markers = vec![marker_at([10.0, 10.0]), marker_at([20.0, 10.0])];
        assert!(assign_plain_grid(&mut markers, &target, &RansacConfig::default()).is_none());
        assert_eq!(markers.len(), 2, "failure must leave markers untouched");
        assert!(markers.iter().all(|m| m.grid_coord.is_none()));
    }

    #[test]
    fn hex_positions_label_and_canonicalize() {
        let hex = TargetLayout::new(
            "test_hex",
            LatticeGeometry::Hex(HexGeometry {
                rows: 7,
                long_row_cols: 7,
                pitch_mm: 8.0,
            }),
            RingGeometry {
                outer_radius_mm: 4.8,
                inner_radius_mm: 2.4,
            },
            crate::target::MarkerCoding::Plain,
            None,
        )
        .expect("valid hex target");

        let (s, tx, ty) = (5.0f64, 30.0f64, 30.0f64);
        let mut markers: Vec<MarkerRecord> = hex
            .cells()
            .iter()
            .map(|cell| {
                marker_at([
                    s * f64::from(cell.xy_mm[0]) + tx,
                    s * f64::from(cell.xy_mm[1]) + ty,
                ])
            })
            .collect();
        let n = markers.len();

        let assignment = assign_plain_grid(&mut markers, &hex, &RansacConfig::default())
            .expect("hex assignment must succeed on a clean grid");
        // Upstream hex labeling is topological-only: interior nodes recover,
        // convex-hull boundary slivers may drop (documented recall floor 3n/5,
        // zero wrong labels). Ringgrid's completion stage recovers the rest.
        assert!(
            markers.len() >= (n * 3) / 5,
            "labeled only {}/{n} hex cells",
            markers.len()
        );
        assert!(assignment.ransac.mean_err_px < 0.5);

        // Zero wrong labels: the frame H must reproject every kept marker.
        for m in &markers {
            let xy = m.board_xy_mm.expect("frame xy set");
            let p = crate::homography::homography_project(&assignment.h, xy[0], xy[1]);
            let err = ((p[0] - m.center[0]).powi(2) + (p[1] - m.center[1]).powi(2)).sqrt();
            assert!(err < 0.5, "hex frame reprojection error {err} too large");
        }

        // Canonical frame: rebased to (0, 0) and +u aligned with image +x.
        let min_u = markers
            .iter()
            .map(|m| m.grid_coord.unwrap()[0])
            .min()
            .unwrap();
        let min_v = markers
            .iter()
            .map(|m| m.grid_coord.unwrap()[1])
            .min()
            .unwrap();
        assert_eq!((min_u, min_v), (0, 0));
    }
}
