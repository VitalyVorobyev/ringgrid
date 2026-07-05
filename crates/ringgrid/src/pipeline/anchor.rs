//! Origin resolution for plain targets.
//!
//! Plain-grid labeling (`pipeline::assign`) produces cell coordinates in a
//! canonical *relative* frame: the true board labeling differs by a rotation
//! from the lattice symmetry group plus a lattice translation. When the target
//! carries origin fiducials (small filled dots at known board positions), each
//! candidate (rotation × translation) labeling is scored by projecting the dot
//! positions through the candidate's board→image homography and measuring dot
//! darkness against the local background — verify-at-predicted-positions, no
//! separate dot detector needed. The winner must clear an absolute contrast
//! threshold and a margin over the runner-up; otherwise the origin stays
//! unresolved and outputs remain in the relative frame.
//!
//! Only rotations (determinant +1 elements of the coordinate symmetry group)
//! are enumerated: an opaque planar target always images through an
//! orientation-preserving homography, so reflected labelings are physically
//! impossible. This is also why fiducial validation (`crate::target`) only
//! rejects rotationally symmetric dot arrangements.

use std::collections::HashSet;

use image::GrayImage;
use nalgebra::Matrix3;
use projective_grid::{Coord, GridTransform};

use crate::detector::MarkerRecord;
use crate::homography::{estimate_homography_dlt, homography_project};
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;
use crate::target::TargetLayout;

/// Upper bound on (rotation × translation) candidates before giving up.
/// Small detected patches on a large board explode the translation count and
/// cannot be disambiguated reliably anyway.
const MAX_CANDIDATES: usize = 512;
/// Minimum normalized (background − dot) contrast for the winning candidate.
const MIN_DOT_CONTRAST: f32 = 0.10;
/// Minimum score margin of the winner over the runner-up.
const MIN_MARGIN: f32 = 0.05;
/// Dot-disk sampling radius as a fraction of the projected dot radius.
const DISK_FRACTION: f64 = 0.55;
/// Background annulus sampling radii as fractions of the projected dot radius.
/// Kept small enough that the annulus stays clear of neighboring ring ink.
const BG_FRACTIONS: [f64; 2] = [1.6, 2.2];
/// Samples per sampling ring.
const RING_SAMPLES: usize = 8;

/// Integer coordinate map `coord → M·coord + t` from the relative labeling
/// frame to absolute board coordinates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoordMap {
    matrix: [[i32; 2]; 2],
    offset: [i32; 2],
}

impl CoordMap {
    fn new(rotation: GridTransform, offset: [i32; 2]) -> Self {
        Self {
            matrix: rotation.matrix,
            offset,
        }
    }

    /// Apply the map to a relative-frame coordinate.
    pub(crate) fn apply(&self, c: Coord) -> Coord {
        Coord::new(
            self.matrix[0][0] * c.u + self.matrix[0][1] * c.v + self.offset[0],
            self.matrix[1][0] * c.u + self.matrix[1][1] * c.v + self.offset[1],
        )
    }
}

/// Accepted origin resolution.
pub(crate) struct OriginResolution {
    /// Relative-frame → board-frame coordinate map that won.
    pub coord_map: CoordMap,
    /// Board-mm → working-px homography of the winning candidate.
    pub h: Matrix3<f64>,
}

struct Candidate {
    coord_map: CoordMap,
    h: Matrix3<f64>,
    score: f32,
}

/// Resolve the board origin for labeled plain markers.
///
/// `markers` must carry relative-frame `grid_coord`s (from
/// [`assign_plain_grid`](super::assign::assign_plain_grid)); centers are
/// working-frame. Returns `None` — leaving the caller in the relative frame —
/// when the target has no fiducials, the candidate set is empty, too large, or
/// unscorable (dots out of view), or the best score fails the absolute
/// threshold or runner-up margin.
pub(crate) fn resolve_origin(
    gray: &GrayImage,
    markers: &[MarkerRecord],
    target: &TargetLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Option<OriginResolution> {
    let fiducials = target.fiducials()?;

    let entries: Vec<(Coord, [f64; 2])> = markers
        .iter()
        .filter_map(|m| {
            let c = m.grid_coord?;
            Some((Coord::new(c[0], c[1]), m.center))
        })
        .collect();
    if entries.len() < 4 {
        return None;
    }

    let board_coords: HashSet<Coord> = target.cells().iter().map(|cell| cell.coord).collect();
    let candidates = enumerate_candidates(&entries, &board_coords, target)?;
    if candidates.is_empty() {
        tracing::info!("origin anchor: no placement admits the labeled patch");
        return None;
    }

    let sampler = DistortionAwareSampler::new(gray, mapper);
    let mut scored: Vec<Candidate> = Vec::new();
    for coord_map in candidates {
        let Some((h, score)) = score_candidate(&entries, coord_map, target, sampler, fiducials)
        else {
            continue;
        };
        scored.push(Candidate {
            coord_map,
            h,
            score,
        });
    }
    if scored.is_empty() {
        tracing::info!("origin anchor: no candidate could be scored (dots out of view?)");
        return None;
    }

    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    let best = &scored[0];
    let runner_up = scored.get(1).map(|c| c.score);

    if best.score < MIN_DOT_CONTRAST {
        tracing::info!(
            best_score = best.score,
            "origin anchor: best candidate below contrast threshold"
        );
        return None;
    }
    if let Some(second) = runner_up
        && best.score - second < MIN_MARGIN
    {
        tracing::info!(
            best_score = best.score,
            runner_up = second,
            "origin anchor: ambiguous — margin over runner-up too small"
        );
        return None;
    }

    tracing::info!(
        best_score = best.score,
        ?runner_up,
        "origin anchor: resolved board origin"
    );
    Some(OriginResolution {
        coord_map: best.coord_map,
        h: best.h,
    })
}

/// Enumerate all (symmetry × translation) maps embedding every labeled
/// coordinate into the board cell set. Returns `None` when the count exceeds
/// [`MAX_CANDIDATES`].
///
/// The full dihedral group is used — rotations **and** reflections — because the
/// relative labeling frame's handedness relative to the board's axial frame is
/// not guaranteed (hex canonicalization mirrors it). The physical
/// orientation-preserving constraint is enforced downstream by
/// [`score_candidate`]'s Jacobian check, which is what actually selects the
/// correct handedness; pre-filtering to `det > 0` here would wrongly drop the
/// only valid hex embeddings.
fn enumerate_candidates(
    entries: &[(Coord, [f64; 2])],
    board_coords: &HashSet<Coord>,
    target: &TargetLayout,
) -> Option<Vec<CoordMap>> {
    let (board_min, board_max) = coord_bbox(board_coords.iter().copied())?;

    let transforms = target.lattice_kind().symmetry_transforms();

    let mut candidates: Vec<CoordMap> = Vec::new();
    for &rot in transforms.iter() {
        let mapped: Vec<Coord> = entries.iter().map(|(c, _)| rot.apply(*c)).collect();
        let Some((min, max)) = coord_bbox(mapped.iter().copied()) else {
            continue;
        };
        for du in (board_min.u - min.u)..=(board_max.u - max.u) {
            for dv in (board_min.v - min.v)..=(board_max.v - max.v) {
                let all_in = mapped
                    .iter()
                    .all(|c| board_coords.contains(&Coord::new(c.u + du, c.v + dv)));
                if all_in {
                    candidates.push(CoordMap::new(rot, [du, dv]));
                    if candidates.len() > MAX_CANDIDATES {
                        tracing::info!(
                            max = MAX_CANDIDATES,
                            "origin anchor: too many placement candidates"
                        );
                        return None;
                    }
                }
            }
        }
    }
    Some(candidates)
}

fn coord_bbox(coords: impl Iterator<Item = Coord>) -> Option<(Coord, Coord)> {
    let mut it = coords;
    let first = it.next()?;
    let (mut min, mut max) = (first, first);
    for c in it {
        min.u = min.u.min(c.u);
        min.v = min.v.min(c.v);
        max.u = max.u.max(c.u);
        max.v = max.v.max(c.v);
    }
    Some((min, max))
}

/// Fit the candidate's board→image homography and score its fiducial dots.
/// Returns `None` for degenerate, orientation-reversing, or unscorable
/// candidates.
fn score_candidate(
    entries: &[(Coord, [f64; 2])],
    coord_map: CoordMap,
    target: &TargetLayout,
    sampler: DistortionAwareSampler<'_>,
    fiducials: &crate::target::OriginFiducials,
) -> Option<(Matrix3<f64>, f32)> {
    let mut src: Vec<[f64; 2]> = Vec::with_capacity(entries.len());
    let mut dst: Vec<[f64; 2]> = Vec::with_capacity(entries.len());
    for (coord, center) in entries {
        let xy = target.cell_xy_mm(coord_map.apply(*coord))?;
        src.push([f64::from(xy[0]), f64::from(xy[1])]);
        dst.push(*center);
    }
    let h = estimate_homography_dlt(&src, &dst).ok()?;

    // Physical filter: the board→image map must preserve orientation. This is
    // also what disambiguates a mirrored candidate frame from the true one.
    let center_mm = [
        src.iter().map(|p| p[0]).sum::<f64>() / src.len() as f64,
        src.iter().map(|p| p[1]).sum::<f64>() / src.len() as f64,
    ];
    if jacobian_det(&h, center_mm) <= 0.0 {
        return None;
    }

    let mut min_contrast = f32::INFINITY;
    for dot in &fiducials.dots_mm {
        let contrast = dot_contrast(
            sampler,
            &h,
            [f64::from(dot[0]), f64::from(dot[1])],
            f64::from(fiducials.dot_radius_mm),
        )?;
        min_contrast = min_contrast.min(contrast);
    }
    min_contrast.is_finite().then_some((h, min_contrast))
}

/// Determinant of the homography's Jacobian at a board point (numeric, 1 mm
/// steps). Positive iff the local board→image map preserves orientation.
fn jacobian_det(h: &Matrix3<f64>, p: [f64; 2]) -> f64 {
    let o = homography_project(h, p[0], p[1]);
    let px = homography_project(h, p[0] + 1.0, p[1]);
    let py = homography_project(h, p[0], p[1] + 1.0);
    let (ax, ay) = (px[0] - o[0], px[1] - o[1]);
    let (bx, by) = (py[0] - o[0], py[1] - o[1]);
    ax * by - ay * bx
}

/// Normalized (background − dot) intensity contrast at a projected dot.
///
/// Samples a small disk at the predicted dot center against a clear background
/// annulus around it. `None` when the projected dot is sub-pixel or any sample
/// falls outside the image — an unverifiable candidate, not a dark one.
fn dot_contrast(
    sampler: DistortionAwareSampler<'_>,
    h: &Matrix3<f64>,
    dot_mm: [f64; 2],
    dot_radius_mm: f64,
) -> Option<f32> {
    let c = homography_project(h, dot_mm[0], dot_mm[1]);
    if !(c[0].is_finite() && c[1].is_finite()) {
        return None;
    }
    // Local projected dot radius from 1-radius steps along both board axes.
    let px = homography_project(h, dot_mm[0] + dot_radius_mm, dot_mm[1]);
    let py = homography_project(h, dot_mm[0], dot_mm[1] + dot_radius_mm);
    let r_px = 0.5
        * (((px[0] - c[0]).powi(2) + (px[1] - c[1]).powi(2)).sqrt()
            + ((py[0] - c[0]).powi(2) + (py[1] - c[1]).powi(2)).sqrt());
    if !r_px.is_finite() || r_px < 1.0 {
        return None;
    }

    let mut dot_sum = 0.0f32;
    let mut dot_n = 0usize;
    dot_sum += sampler.sample_checked(c[0] as f32, c[1] as f32)?;
    dot_n += 1;
    for i in 0..RING_SAMPLES {
        let t = (i as f64) * std::f64::consts::TAU / (RING_SAMPLES as f64);
        let x = c[0] + DISK_FRACTION * r_px * t.cos();
        let y = c[1] + DISK_FRACTION * r_px * t.sin();
        dot_sum += sampler.sample_checked(x as f32, y as f32)?;
        dot_n += 1;
    }

    let mut bg_sum = 0.0f32;
    let mut bg_n = 0usize;
    for fraction in BG_FRACTIONS {
        for i in 0..RING_SAMPLES {
            // Offset alternate rings by half a step for better angular coverage.
            let t = ((i as f64) + 0.5 * ((fraction != BG_FRACTIONS[0]) as u8 as f64))
                * std::f64::consts::TAU
                / (RING_SAMPLES as f64);
            let x = c[0] + fraction * r_px * t.cos();
            let y = c[1] + fraction * r_px * t.sin();
            bg_sum += sampler.sample_checked(x as f32, y as f32)?;
            bg_n += 1;
        }
    }

    // `sample_checked` yields intensities already normalized to [0, 1].
    Some(bg_sum / bg_n as f32 - dot_sum / dot_n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::TargetLayout;

    /// Render the rect target through an affine map: filled annuli plus
    /// the three origin dots, on a white background.
    fn render_plain_target(
        target: &TargetLayout,
        scale: f64,
        tx: f64,
        ty: f64,
        rot_rad: f64,
        w: u32,
        h: u32,
    ) -> (GrayImage, Matrix3<f64>) {
        let (sin, cos) = rot_rad.sin_cos();
        let h_mat = Matrix3::new(
            scale * cos,
            -scale * sin,
            tx,
            scale * sin,
            scale * cos,
            ty,
            0.0,
            0.0,
            1.0,
        );

        let mut img = GrayImage::from_pixel(w, h, image::Luma([255u8]));
        let ring = target.ring();
        let (r_out, r_in) = (
            f64::from(ring.outer_radius_mm) * scale,
            f64::from(ring.inner_radius_mm) * scale,
        );
        let mut disks: Vec<([f64; 2], f64, f64)> = target
            .cells()
            .iter()
            .map(|cell| {
                let p =
                    homography_project(&h_mat, f64::from(cell.xy_mm[0]), f64::from(cell.xy_mm[1]));
                (p, r_out, r_in)
            })
            .collect();
        if let Some(fid) = target.fiducials() {
            let r_dot = f64::from(fid.dot_radius_mm) * scale;
            for dot in &fid.dots_mm {
                let p = homography_project(&h_mat, f64::from(dot[0]), f64::from(dot[1]));
                disks.push((p, r_dot, 0.0));
            }
        }
        for (c, ro, ri) in disks {
            let (x0, x1) = (
                (c[0] - ro - 1.0).floor().max(0.0) as u32,
                ((c[0] + ro + 1.0).ceil() as u32).min(w - 1),
            );
            let (y0, y1) = (
                (c[1] - ro - 1.0).floor().max(0.0) as u32,
                ((c[1] + ro + 1.0).ceil() as u32).min(h - 1),
            );
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d2 = (f64::from(x) - c[0]).powi(2) + (f64::from(y) - c[1]).powi(2);
                    if d2 <= ro * ro && d2 >= ri * ri {
                        img.put_pixel(x, y, image::Luma([20u8]));
                    }
                }
            }
        }
        (img, h_mat)
    }

    /// Markers labeled with the *true* board coords remapped by a known
    /// relabeling, so the resolver must recover the inverse.
    fn labeled_markers(
        target: &TargetLayout,
        h_mat: &Matrix3<f64>,
        relabel: impl Fn(Coord) -> Coord,
    ) -> Vec<MarkerRecord> {
        target
            .cells()
            .iter()
            .map(|cell| {
                let p =
                    homography_project(h_mat, f64::from(cell.xy_mm[0]), f64::from(cell.xy_mm[1]));
                let rel = relabel(cell.coord);
                MarkerRecord {
                    grid_coord: Some([rel.u, rel.v]),
                    confidence: 1.0,
                    center: p,
                    ..MarkerRecord::default()
                }
            })
            .collect()
    }

    fn small_rect_like() -> TargetLayout {
        // A compact plain rect target with the L-shaped dot triple,
        // small enough to render quickly.
        use crate::target::{
            LatticeGeometry, MarkerCoding, OriginFiducials, RectGeometry, RingGeometry,
        };
        TargetLayout::new(
            "test_rect_small",
            LatticeGeometry::Rect(RectGeometry {
                rows: 8,
                cols: 8,
                pitch_mm: 14.0,
            }),
            RingGeometry {
                outer_radius_mm: 5.6,
                inner_radius_mm: 2.8,
            },
            MarkerCoding::Plain,
            Some(OriginFiducials {
                dot_radius_mm: 1.4,
                dots_mm: vec![[49.0, 49.0], [35.0, 49.0], [49.0, 63.0]],
            }),
        )
        .expect("valid test target")
    }

    #[test]
    fn identity_labeling_resolves_to_identity() {
        let target = small_rect_like();
        let (img, h_mat) = render_plain_target(&target, 6.0, 60.0, 60.0, 0.0, 780, 780);
        let markers = labeled_markers(&target, &h_mat, |c| c);

        let res = resolve_origin(&img, &markers, &target, None).expect("must resolve");
        for cell in target.cells() {
            assert_eq!(res.coord_map.apply(cell.coord), cell.coord);
        }
    }

    #[test]
    fn rotated_labeling_is_recovered() {
        let target = small_rect_like();
        // Physically rotate the board 90° in the image; the labeler would then
        // produce canonical (image-axis-aligned) labels that differ from board
        // coords by the inverse rotation.
        let (img, h_mat) = render_plain_target(
            &target,
            6.0,
            720.0,
            60.0,
            std::f64::consts::FRAC_PI_2,
            780,
            780,
        );
        // Canonical relabeling for a 90° board rotation on an 8×8 grid:
        // (u, v) -> (7 - v, u) puts +u back along image +x.
        let markers = labeled_markers(&target, &h_mat, |c| Coord::new(7 - c.v, c.u));

        let res = resolve_origin(&img, &markers, &target, None).expect("must resolve");
        // The resolved map must invert the relabeling.
        for cell in target.cells() {
            let rel = Coord::new(7 - cell.coord.v, cell.coord.u);
            assert_eq!(res.coord_map.apply(rel), cell.coord);
        }
    }

    #[test]
    fn no_fiducials_returns_none() {
        use crate::target::{LatticeGeometry, MarkerCoding, RectGeometry, RingGeometry};
        let target = TargetLayout::new(
            "test_nodots",
            LatticeGeometry::Rect(RectGeometry {
                rows: 8,
                cols: 8,
                pitch_mm: 14.0,
            }),
            RingGeometry {
                outer_radius_mm: 5.6,
                inner_radius_mm: 2.8,
            },
            MarkerCoding::Plain,
            None,
        )
        .expect("valid");
        let (img, h_mat) = render_plain_target(&target, 6.0, 60.0, 60.0, 0.0, 780, 780);
        let markers = labeled_markers(&target, &h_mat, |c| c);
        assert!(resolve_origin(&img, &markers, &target, None).is_none());
    }

    #[test]
    fn missing_dots_in_image_yield_unresolved_not_wrong() {
        let target = small_rect_like();
        // Render WITHOUT dots (e.g. occluded); every candidate is then
        // unscorable or scores near zero — the resolver must decline.
        use crate::target::{LatticeGeometry, MarkerCoding, RectGeometry, RingGeometry};
        let no_dots = TargetLayout::new(
            "test_nodots_render",
            LatticeGeometry::Rect(RectGeometry {
                rows: 8,
                cols: 8,
                pitch_mm: 14.0,
            }),
            RingGeometry {
                outer_radius_mm: 5.6,
                inner_radius_mm: 2.8,
            },
            MarkerCoding::Plain,
            None,
        )
        .expect("valid");
        let (img, h_mat) = render_plain_target(&no_dots, 6.0, 60.0, 60.0, 0.0, 780, 780);
        let markers = labeled_markers(&target, &h_mat, |c| c);
        assert!(
            resolve_origin(&img, &markers, &target, None).is_none(),
            "absent dots must leave the origin unresolved"
        );
    }
}
