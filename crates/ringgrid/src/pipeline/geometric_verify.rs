//! Final precision-first geometric verification gate.
//!
//! After the final homography, each decoded marker is checked against the hex
//! lattice — via its neighbor-midpoint prediction (locally affine, so
//! distortion-robust) and its final-H reprojection residual — and removed when
//! geometrically inconsistent. Only trusted board correspondences reach the
//! output, which is what sensor calibration needs. See [`geometric_verify_filter`].

use std::collections::{HashMap, HashSet};

use nalgebra::{Matrix3, Point2};
use projective_grid::{Coord, LatticeKind, predict_grid_position};

use super::stats::median_f64;
use crate::detector::MarkerRecord;
use crate::target::TargetLayout;

/// Build a hex grid map from decoded markers, mapping `(q, r)` → center.
pub(crate) fn build_hex_grid_map(
    markers: &[MarkerRecord],
    target: &TargetLayout,
) -> HashMap<Coord, Point2<f32>> {
    markers
        .iter()
        .filter_map(|m| {
            let id = m.id?;
            let coord = target.coord_of_id(id)?;
            Some((coord, Point2::new(m.center[0] as f32, m.center[1] as f32)))
        })
        .collect()
}

/// 1.4826 rescales a median-absolute-deviation to a Gaussian-consistent sigma.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Minimum residual samples before the adaptive MAD term is trusted; below this
/// the threshold falls back to its floor alone.
const MIN_SAMPLES: usize = 8;
/// Floor (px) for the local hex-midpoint residual threshold. An order of
/// magnitude below the ~1-pitch residual of a mis-celled marker, several times
/// the sub-pixel center-estimation noise of a true marker.
const FLOOR_LOCAL_PX: f64 = 2.0;
/// MAD multiplier (~4σ, one-sided) for the local hex-midpoint test.
const K_LOCAL: f64 = 4.0;
/// Floor multiplier (× RANSAC inlier threshold) for the global final-H residual
/// test. Default `2 × 5px = 10px` — a deliberately loose gross-blunder backstop.
const C1_GLOBAL: f64 = 2.0;
/// MAD multiplier for the global final-H residual test. Looser than the local
/// multiplier so legitimately distorted peripheral markers survive.
const K_GLOBAL: f64 = 5.0;

/// Outcome of [`geometric_verify_filter`].
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct GeometricVerifyStats {
    /// Number of decoded markers inspected by the gate.
    pub(super) n_decoded_checked: usize,
    /// Markers flagged by the local hex-midpoint test (reason tally; may overlap
    /// with `n_removed_global`).
    pub(super) n_removed_local: usize,
    /// Markers flagged by the global final-H residual test (reason tally).
    pub(super) n_removed_global: usize,
    /// Actual number of markers removed (deduplicated union of both tests).
    pub(super) n_removed_total: usize,
    /// Local residual threshold used (px).
    pub(super) t_local_px: f64,
    /// Global residual threshold used (px); `None` when no final homography.
    pub(super) t_global_px: Option<f64>,
}

/// Robust adaptive threshold: `max(floor, median + k · 1.4826 · MAD)`.
///
/// Returns `floor` when fewer than [`MIN_SAMPLES`] finite values are available
/// (MAD unreliable). The MAD term only ever *raises* the threshold, so the floor
/// dominates in low-scatter regimes (clean boards) while distorted boards
/// auto-loosen — keeping the gate recall-safe in both.
fn adaptive_threshold(values: &[f64], floor: f64, k: f64) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < MIN_SAMPLES {
        return floor;
    }
    let Some(median) = median_f64(finite.clone()) else {
        return floor;
    };
    let deviations: Vec<f64> = finite.iter().map(|v| (v - median).abs()).collect();
    let mad = median_f64(deviations).unwrap_or(0.0);
    floor.max(median + k * MAD_TO_SIGMA * mad)
}

/// Populate [`FitMetrics::h_reproj_err_px`](crate::FitMetrics) for every decoded
/// marker as the working-frame distance between its center and its board position
/// projected through `final_h`. The **sole writer** of that diagnostic.
///
/// Runs whether or not the hard gate is enabled, so callers that opt out of
/// rejection (`geometric_verify = false`) still receive the residual for their
/// own filtering. The value is `None` for markers without a board id and for
/// every marker when there is no `final_h`.
pub(super) fn annotate_h_reproj_err_px(
    markers: &mut [MarkerRecord],
    final_h: Option<&Matrix3<f64>>,
) {
    let Some(h) = final_h else {
        for m in markers.iter_mut() {
            m.fit.h_reproj_err_px = None;
        }
        return;
    };
    for m in markers.iter_mut() {
        let (Some(_id), Some(board_xy)) = (m.id, m.board_xy_mm) else {
            m.fit.h_reproj_err_px = None;
            continue;
        };
        let (x, y) = (board_xy[0], board_xy[1]);
        let pw = h[(0, 0)] * x + h[(0, 1)] * y + h[(0, 2)];
        let ph = h[(1, 0)] * x + h[(1, 1)] * y + h[(1, 2)];
        let pz = h[(2, 0)] * x + h[(2, 1)] * y + h[(2, 2)];
        if pz.abs() <= 1e-15 {
            m.fit.h_reproj_err_px = None;
            continue;
        }
        let dx = pw / pz - m.center[0];
        let dy = ph / pz - m.center[1];
        m.fit.h_reproj_err_px = Some((dx * dx + dy * dy).sqrt() as f32);
    }
}

/// Final precision-first geometric verification gate.
///
/// Runs in the working frame (where `final_h` was fit), after completion and the
/// final H refit, over **all** decoded markers including completed ones. Removes
/// markers the hex lattice judges geometrically inconsistent via two
/// complementary tests, rejecting on their union:
///
/// 1. **Local hex-midpoint** (H-free, distortion-robust primary): each marker's
///    center vs the midpoint predicted by its hex neighbors. Affine-exact, so it
///    sees only second-difference curvature under smooth lens distortion while a
///    wrong-cell marker sits ~1 pitch away.
/// 2. **Global final-H reprojection** (gross-blunder backstop): each marker's
///    center vs its board position projected through `final_h`. Catches boundary
///    markers that lack a complete neighbor pair for the local test.
///
/// Delegates to [`annotate_h_reproj_err_px`] to populate
/// [`FitMetrics::h_reproj_err_px`](crate::FitMetrics) before reading it for the
/// global test, so the diagnostic is identical whether or not the gate runs.
pub(super) fn geometric_verify_filter(
    markers: &mut Vec<MarkerRecord>,
    final_h: Option<&Matrix3<f64>>,
    target: &TargetLayout,
    ransac_inlier_threshold_px: f64,
) -> GeometricVerifyStats {
    let n_decoded_checked = markers.iter().filter(|m| m.id.is_some()).count();

    // --- Local hex-midpoint test (H-free) ---
    let grid = build_hex_grid_map(markers, target);
    let local_by_coord: Vec<(Coord, f64)> = grid
        .iter()
        .filter_map(|(&idx, &pos)| {
            let pred = predict_grid_position(&grid, idx, LatticeKind::Hex)?.position;
            let dx = (pos.x - pred.x) as f64;
            let dy = (pos.y - pred.y) as f64;
            Some((idx, (dx * dx + dy * dy).sqrt()))
        })
        .collect();
    let local_residuals: Vec<f64> = local_by_coord.iter().map(|(_, r)| *r).collect();
    let t_local = adaptive_threshold(&local_residuals, FLOOR_LOCAL_PX, K_LOCAL);
    let flagged_coords: HashSet<Coord> = local_by_coord
        .iter()
        .filter(|(_, r)| *r > t_local)
        .map(|(c, _)| *c)
        .collect();

    // --- Global final-H residual test (reads the annotated diagnostic) ---
    // Annotation is the sole writer of h_reproj_err_px and runs even when the
    // gate is disabled; here the global backstop just consumes those residuals.
    annotate_h_reproj_err_px(markers, final_h);
    let mut t_global: Option<f64> = None;
    let mut flagged_global_ids: HashSet<usize> = HashSet::new();
    if final_h.is_some() {
        let global_by_id: Vec<(usize, f64)> = markers
            .iter()
            .filter_map(|m| Some((m.id?, f64::from(m.fit.h_reproj_err_px?))))
            .collect();
        let global_residuals: Vec<f64> = global_by_id.iter().map(|(_, r)| *r).collect();
        let tg = adaptive_threshold(
            &global_residuals,
            C1_GLOBAL * ransac_inlier_threshold_px,
            K_GLOBAL,
        );
        flagged_global_ids = global_by_id
            .iter()
            .filter(|(_, r)| *r > tg)
            .map(|(id, _)| *id)
            .collect();
        t_global = Some(tg);
    }

    // --- Reject = union, single retain pass ---
    let n_before = markers.len();
    let mut n_removed_local = 0usize;
    let mut n_removed_global = 0usize;
    markers.retain(|m| {
        let Some(id) = m.id else { return true };
        let Some(coord) = target.coord_of_id(id) else {
            return true;
        };
        let local_bad = flagged_coords.contains(&coord);
        let global_bad = flagged_global_ids.contains(&id);
        if local_bad {
            n_removed_local += 1;
        }
        if global_bad {
            n_removed_global += 1;
        }
        !(local_bad || global_bad)
    });

    let stats = GeometricVerifyStats {
        n_decoded_checked,
        n_removed_local,
        n_removed_global,
        n_removed_total: n_before - markers.len(),
        t_local_px: t_local,
        t_global_px: t_global,
    };
    if stats.n_removed_total > 0 {
        tracing::info!(
            n_decoded_checked = stats.n_decoded_checked,
            n_removed_total = stats.n_removed_total,
            n_removed_local = stats.n_removed_local,
            n_removed_global = stats.n_removed_global,
            t_local_px = stats.t_local_px,
            t_global_px = stats.t_global_px,
            "geometric verification removed lattice-inconsistent markers"
        );
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DetectConfig;
    use crate::detector::DetectionSource;

    fn affine_h(s: f64, tx: f64, ty: f64) -> Matrix3<f64> {
        Matrix3::new(s, 0.0, tx, 0.0, s, ty, 0.0, 0.0, 1.0)
    }

    fn project(h: &Matrix3<f64>, xy: [f64; 2]) -> [f64; 2] {
        let pw = h[(0, 0)] * xy[0] + h[(0, 1)] * xy[1] + h[(0, 2)];
        let ph = h[(1, 0)] * xy[0] + h[(1, 1)] * xy[1] + h[(1, 2)];
        let pz = h[(2, 0)] * xy[0] + h[(2, 1)] * xy[1] + h[(2, 2)];
        [pw / pz, ph / pz]
    }

    fn board() -> TargetLayout {
        DetectConfig::default().target
    }

    fn lattice_ids(board: &TargetLayout) -> Vec<usize> {
        (0..=board.max_marker_id())
            .filter(|&id| board.coord_of_id(id).is_some() && board.xy_mm_of_id(id).is_some())
            .collect()
    }

    fn marker_at(board: &TargetLayout, id: usize, center: [f64; 2]) -> MarkerRecord {
        let xy = board.xy_mm_of_id(id).expect("board xy");
        MarkerRecord {
            id: Some(id),
            confidence: 1.0,
            center,
            board_xy_mm: Some([xy[0] as f64, xy[1] as f64]),
            ..MarkerRecord::default()
        }
    }

    /// Full clean lattice mapped through `h` (true markers, zero residual).
    fn clean_lattice(board: &TargetLayout, h: &Matrix3<f64>) -> Vec<MarkerRecord> {
        lattice_ids(board)
            .into_iter()
            .map(|id| {
                let xy = board.xy_mm_of_id(id).unwrap();
                let center = project(h, [xy[0] as f64, xy[1] as f64]);
                marker_at(board, id, center)
            })
            .collect()
    }

    fn centroid(markers: &[MarkerRecord]) -> [f64; 2] {
        let n = markers.len().max(1) as f64;
        let (sx, sy) = markers.iter().fold((0.0, 0.0), |(ax, ay), m| {
            (ax + m.center[0], ay + m.center[1])
        });
        [sx / n, sy / n]
    }

    /// Board id whose center is nearest the lattice centroid — deep interior, so
    /// all six hex neighbors exist with complete opposite pairs.
    fn most_interior_id(board: &TargetLayout, h: &Matrix3<f64>) -> usize {
        let markers = clean_lattice(board, h);
        let c = centroid(&markers);
        markers
            .iter()
            .min_by(|a, b| {
                let da = (a.center[0] - c[0]).powi(2) + (a.center[1] - c[1]).powi(2);
                let db = (b.center[0] - c[0]).powi(2) + (b.center[1] - c[1]).powi(2);
                da.total_cmp(&db)
            })
            .and_then(|m| m.id)
            .expect("interior id")
    }

    #[test]
    fn clean_lattice_has_no_removals() {
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let mut markers = clean_lattice(&board, &h);
        let n = markers.len();
        assert!(n > 20, "test board should expose a full lattice");
        let stats = geometric_verify_filter(&mut markers, Some(&h), &board, 5.0);
        assert_eq!(
            stats.n_removed_total, 0,
            "clean lattice must not be filtered"
        );
        assert_eq!(markers.len(), n);
        assert!(markers.iter().all(|m| m.fit.h_reproj_err_px.is_some()));
    }

    #[test]
    fn smoothly_distorted_lattice_survives() {
        // Radial cubic distortion δ = k·ρ³ on true-marker centers; the homography
        // stays the clean affine, so the *global* residual grows to a few px at
        // the periphery (what a fixed small gate would wrongly reject) while the
        // *local* midpoint residual stays sub-pixel. Nothing must be removed.
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let mut markers = clean_lattice(&board, &h);
        let c = centroid(&markers);
        let k_dist = 3.2e-8;
        for m in markers.iter_mut() {
            let dx = m.center[0] - c[0];
            let dy = m.center[1] - c[1];
            let rho = (dx * dx + dy * dy).sqrt();
            if rho > 1e-9 {
                let delta = k_dist * rho.powi(3);
                m.center[0] += delta * dx / rho;
                m.center[1] += delta * dy / rho;
            }
        }
        let n = markers.len();
        let stats = geometric_verify_filter(&mut markers, Some(&h), &board, 5.0);
        assert_eq!(
            stats.n_removed_total, 0,
            "distorted true markers must survive"
        );
        assert_eq!(markers.len(), n);
        // The distortion is real: some peripheral marker carries a global residual
        // a fixed 1px gate would have rejected, yet it survived.
        let max_err = markers
            .iter()
            .filter_map(|m| m.fit.h_reproj_err_px)
            .fold(0.0f32, f32::max);
        assert!(
            (1.5..8.0).contains(&max_err),
            "expected a meaningful but sub-floor peripheral residual, got {max_err}"
        );
    }

    #[test]
    fn displaced_interior_marker_is_removed_locally() {
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let target = most_interior_id(&board, &h);
        let far = *lattice_ids(&board)
            .iter()
            .max_by(|&&a, &&b| {
                let bt = board.xy_mm_of_id(target).unwrap();
                let ba = board.xy_mm_of_id(a).unwrap();
                let bb = board.xy_mm_of_id(b).unwrap();
                let da = (ba[0] - bt[0]).powi(2) + (ba[1] - bt[1]).powi(2);
                let db = (bb[0] - bt[0]).powi(2) + (bb[1] - bt[1]).powi(2);
                da.total_cmp(&db)
            })
            .unwrap();

        let mut markers = clean_lattice(&board, &h);
        for m in markers.iter_mut() {
            if m.id == Some(target) {
                m.center[0] += 30.0;
                m.center[1] += 30.0;
            }
        }
        let n = markers.len();
        let stats = geometric_verify_filter(&mut markers, Some(&h), &board, 5.0);

        assert!(
            !markers.iter().any(|m| m.id == Some(target)),
            "displaced marker must be removed"
        );
        assert!(
            markers.iter().any(|m| m.id == Some(far)),
            "a far marker must be retained"
        );
        assert!(stats.n_removed_total >= 1);
        // Local, not a global cascade: at most the target plus its six neighbors.
        assert!(
            stats.n_removed_total <= 7,
            "removed {} — cascade too large",
            stats.n_removed_total
        );
        assert!(markers.len() >= n - 7);
        // The gate removes; it never manufactures an id:None blob.
        assert!(markers.iter().all(|m| m.id.is_some()));
    }

    #[test]
    fn boundary_blunder_removed_by_global_backstop() {
        // Two markers only ⇒ neither has a complete opposite hex pair, so the
        // local test cannot flag them. A gross global-H residual must still be
        // caught by the backstop.
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let ids = lattice_ids(&board);
        let (a, b) = (ids[0], ids[1]);
        let bxy_a = board.xy_mm_of_id(a).unwrap();
        let bxy_b = board.xy_mm_of_id(b).unwrap();
        let bad = marker_at(&board, a, {
            let p = project(&h, [bxy_a[0] as f64, bxy_a[1] as f64]);
            [p[0] + 20.0, p[1]] // 20px ≫ 10px global floor
        });
        let good = marker_at(&board, b, project(&h, [bxy_b[0] as f64, bxy_b[1] as f64]));
        let mut markers = vec![bad, good];
        let stats = geometric_verify_filter(&mut markers, Some(&h), &board, 5.0);

        assert_eq!(
            stats.n_removed_local, 0,
            "no complete pair ⇒ local cannot flag"
        );
        assert_eq!(stats.n_removed_global, 1);
        assert_eq!(stats.n_removed_total, 1);
        assert!(!markers.iter().any(|m| m.id == Some(a)));
        assert!(markers.iter().any(|m| m.id == Some(b)));
    }

    #[test]
    fn no_homography_runs_local_only_without_panic() {
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let mut markers = clean_lattice(&board, &h);
        let n = markers.len();
        let stats = geometric_verify_filter(&mut markers, None, &board, 5.0);
        assert!(stats.t_global_px.is_none());
        assert_eq!(
            stats.n_removed_total, 0,
            "clean lattice, no H ⇒ nothing removed"
        );
        assert_eq!(markers.len(), n);
        assert!(markers.iter().all(|m| m.fit.h_reproj_err_px.is_none()));
    }

    #[test]
    fn completion_marker_is_verified() {
        // Regression: the old topology filter ran *before* completion and never
        // saw completed markers. The gate runs after, so an off-lattice
        // completion is rejected like any other.
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let target = most_interior_id(&board, &h);
        let mut markers = clean_lattice(&board, &h);
        for m in markers.iter_mut() {
            if m.id == Some(target) {
                m.source = DetectionSource::Completion;
                m.center[0] += 30.0;
                m.center[1] += 30.0;
            }
        }
        geometric_verify_filter(&mut markers, Some(&h), &board, 5.0);
        assert!(
            !markers.iter().any(|m| m.id == Some(target)),
            "off-lattice completion marker must be removed"
        );
    }

    #[test]
    fn removals_are_deterministic() {
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let target = most_interior_id(&board, &h);
        let build = || {
            let mut markers = clean_lattice(&board, &h);
            for m in markers.iter_mut() {
                if m.id == Some(target) {
                    m.center[0] += 30.0;
                    m.center[1] += 30.0;
                }
            }
            geometric_verify_filter(&mut markers, Some(&h), &board, 5.0);
            markers.iter().filter_map(|m| m.id).collect::<Vec<_>>()
        };
        assert_eq!(build(), build(), "gate output must be order-independent");
    }

    #[test]
    fn adaptive_threshold_falls_back_to_floor_when_sparse() {
        assert_eq!(adaptive_threshold(&[], 2.0, 4.0), 2.0);
        assert_eq!(adaptive_threshold(&[1.0, 1.0, 1.0], 2.0, 4.0), 2.0);
    }

    #[test]
    fn adaptive_threshold_floor_dominates_zero_scatter() {
        // 8 identical samples ⇒ MAD = 0 ⇒ max(floor, median).
        let vals = [5.0f64; 8];
        assert_eq!(adaptive_threshold(&vals, 2.0, 4.0), 5.0);
    }

    #[test]
    fn adaptive_threshold_uses_median_plus_k_mad() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // median = 5.5, MAD = 2.5 ⇒ 5.5 + 1·1.4826·2.5 = 9.2065
        let got = adaptive_threshold(&vals, 0.0, 1.0);
        assert!((got - 9.2065).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn annotate_populates_diagnostic_without_rejecting() {
        // Opt-out path (geometric_verify = false): annotation runs independently
        // of rejection, so callers still get h_reproj_err_px for their own filter.
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let mut markers = clean_lattice(&board, &h);
        let n = markers.len();
        annotate_h_reproj_err_px(&mut markers, Some(&h));
        assert_eq!(markers.len(), n, "annotation must not remove markers");
        assert!(
            markers.iter().all(|m| m.fit.h_reproj_err_px.is_some()),
            "every decoded marker receives a reprojection residual"
        );
    }

    #[test]
    fn annotate_clears_diagnostic_without_homography() {
        let board = board();
        let h = affine_h(5.0, 100.0, 100.0);
        let mut markers = clean_lattice(&board, &h);
        for m in markers.iter_mut() {
            m.fit.h_reproj_err_px = Some(3.0); // stale value from a prior frame
        }
        annotate_h_reproj_err_px(&mut markers, None);
        assert!(markers.iter().all(|m| m.fit.h_reproj_err_px.is_none()));
    }
}
