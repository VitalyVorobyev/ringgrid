//! Origin resolution: anchoring a grid labeling to the board frame.
//!
//! Grid labeling (`pipeline::assign`) produces cell coordinates in a
//! canonical *relative* frame: the true board labeling differs by an element
//! of the lattice symmetry group plus a lattice translation — a rotation
//! whenever canonicalization could measure the handedness. The target decides
//! which evidence fixes that unknown ([`OriginEvidence`]):
//!
//! - **Decoded IDs** (coded targets): every labeled marker whose code band
//!   decoded knows its board cell, so it votes for the one (symmetry,
//!   translation) per symmetry that maps its relative coordinate onto that
//!   cell. A robust majority of agreeing votes wins; see
//!   [`resolve_by_decoded_ids`].
//! - **Dots** (plain targets with origin fiducials): the triad surrounds board
//!   cell `(0, 0)`, so a placement is a symmetry plus the relative-frame cell
//!   that carries the triad. Every placement with that cell at or next to the
//!   labeled patch is scored by projecting its dots through the patch's
//!   homography and measuring their darkness against the local background —
//!   verify-at-predicted-positions, no separate dot detector needed, and a
//!   cost set by the patch, not the board. The winner must clear an absolute
//!   contrast threshold and a margin over the runner-up; see
//!   [`resolve_by_dots`].
//!
//! Without evidence, or when it is ambiguous, the origin stays unresolved and
//! outputs remain in the relative frame.
//!
//! Candidates span the full dihedral symmetry group (rotations and
//! reflections), but every candidate whose board→image homography is
//! orientation-reversing (non-positive Jacobian determinant) is discarded: an
//! opaque planar target always images through an orientation-preserving
//! homography, so reflected labelings are physically impossible. This is also
//! why fiducial validation (`crate::target`) only rejects rotationally
//! symmetric dot arrangements.

use std::collections::HashMap;

use image::GrayImage;
use nalgebra::Matrix3;
use projective_grid::{Coord, expert::lattice::GridTransform};

use crate::detector::MarkerRecord;
use crate::detector::completion::coord_bbox;
use crate::homography::{estimate_homography_dlt, homography_project};
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;
use crate::target::{OriginFiducials, TargetLayout};

/// How far (lattice steps) beyond the labeled patch's coordinate bounds the
/// triad's anchor cell is searched for. The dots are verified through the
/// patch's own homography, which is measured only where rings were labeled:
/// one step lets the anchor ring itself be missing from the labeling (cut by
/// the image border, blurred) while every dot stays within two steps of
/// labeled rings.
const ANCHOR_SEARCH_MARGIN: i32 = 1;
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
/// Chance level the winning placement's agreeing decodes must beat: the
/// expected number of placements that misreads alone would give as many votes
/// (see [`ln_chance_votes`]). Misreads are common (a blurred code band decodes
/// to a valid-looking word), so agreement is judged against chance, not as a
/// majority of all decodes.
const MAX_CHANCE_WINNER: f64 = 1e-3;
/// The runner-up must look like chance (expected count of placements with its
/// votes at least this): a second placement with more votes than misreads
/// explain means the labeling itself is inconsistent (precision harness: a
/// rect board at 60° labeled with a row skipped split its correct decodes
/// 7 / 6 between two placements; accepting the 7 gave 78 wrong IDs).
const MIN_CHANCE_RUNNER_UP: f64 = 0.1;
/// Distance (in nearest-neighbor spacings) within which a placement's board
/// points count as one line (see [`placement_homography`]).
const COLLINEAR_TOL_SPACINGS: f64 = 1e-3;

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
            matrix: rotation.matrix(),
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

/// Evidence that fixes a relative grid labeling to the board frame.
#[derive(Clone, Copy)]
enum OriginEvidence<'a> {
    /// Coded targets: decoded IDs vote for the placement.
    DecodedIds,
    /// Plain targets with origin fiducials: dots verified at predicted
    /// positions.
    Dots(&'a OriginFiducials),
}

impl<'a> OriginEvidence<'a> {
    /// The evidence a target carries; `None` for plain targets without dots.
    fn of(target: &'a TargetLayout) -> Option<Self> {
        if target.is_coded() {
            Some(Self::DecodedIds)
        } else {
            target.fiducials().map(Self::Dots)
        }
    }
}

/// Resolve the board origin for grid-labeled markers.
///
/// `markers` must carry relative-frame `grid_coord`s (from
/// [`assign_plain_grid`](super::assign::assign_plain_grid)); centers are
/// working-frame. Returns `None` — leaving the caller in the relative frame —
/// when the target carries no origin evidence or its evidence is ambiguous
/// (see [`resolve_by_decoded_ids`] and [`resolve_by_dots`]).
pub(crate) fn resolve_origin(
    gray: &GrayImage,
    markers: &[MarkerRecord],
    target: &TargetLayout,
    mapper: Option<&dyn PixelMapper>,
) -> Option<OriginResolution> {
    let evidence = OriginEvidence::of(target)?;

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

    match evidence {
        OriginEvidence::DecodedIds => resolve_by_decoded_ids(markers, &entries, target),
        OriginEvidence::Dots(fiducials) => {
            let sampler = DistortionAwareSampler::new(gray, mapper);
            resolve_by_dots(&entries, target, sampler, fiducials)
        }
    }
}

/// Decoded-ID anchoring. Each labeled marker with a decoded board ID votes,
/// for every lattice symmetry `g`, for the translation `t = cell(id) − g·c`
/// that maps its relative coordinate `c` onto its decoded cell. Correct
/// decodes all vote for the true placement; for any other symmetry their
/// votes scatter (a nonzero difference of two lattice rotations is
/// invertible, so distinct coordinates give distinct translations), and
/// misreads vote for unrelated placements.
///
/// The winner must have an orientation-preserving board→image homography,
/// more votes than chance gives any placement ([`MAX_CHANCE_WINNER`]), more
/// votes than the runner-up, and a runner-up at chance level
/// ([`MIN_CHANCE_RUNNER_UP`]).
fn resolve_by_decoded_ids(
    markers: &[MarkerRecord],
    entries: &[(Coord, [f64; 2])],
    target: &TargetLayout,
) -> Option<OriginResolution> {
    let transforms = target.lattice_kind().symmetry_transforms();
    let mut tally: HashMap<(usize, [i32; 2]), usize> = HashMap::new();
    let mut n_decoded = 0usize;
    for m in markers {
        let (Some(c), Some(cell)) = (m.grid_coord, m.id.and_then(|id| target.coord_of_id(id)))
        else {
            continue;
        };
        n_decoded += 1;
        for (k, g) in transforms.iter().enumerate() {
            let r = g.apply(Coord::new(c[0], c[1]));
            *tally.entry((k, [cell.u - r.u, cell.v - r.v])).or_default() += 1;
        }
    }
    // Orientation-preserving placements (half the dihedral group) per cell.
    let n_rotations = (transforms.len() / 2).max(1);
    let ln_chance = |votes| ln_chance_votes(n_decoded, votes, n_rotations, target.n_cells());
    let min_votes = (2..=n_decoded)
        .find(|&v| ln_chance(v) <= MAX_CHANCE_WINNER.ln())
        .unwrap_or(usize::MAX);
    // Deterministic ranking: votes descending, then the placement key. Only
    // placements with a physically valid (orientation-preserving) homography
    // compete — for the winner and the runner-up alike: decodes along a
    // lattice line vote equally for the true rotation and its mirror image,
    // and that tie is no ambiguity. A single vote is chance level whatever its
    // placement, so it is not fitted.
    let mut ranked: Vec<((usize, [i32; 2]), usize)> = tally.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut valid = ranked.iter().filter_map(|&((k, offset), votes)| {
        if votes < 2 {
            return Some((None, votes));
        }
        let coord_map = CoordMap::new(transforms[k], offset);
        let h = placement_homography(entries, coord_map, target)?;
        Some((Some((coord_map, h)), votes))
    });
    let (best, votes) = valid.next()?;
    let runner_up = valid.next().map_or(0, |(_, v)| v);
    let accepted = votes >= min_votes
        && votes > runner_up
        && ln_chance(runner_up) >= MIN_CHANCE_RUNNER_UP.ln();
    tracing::info!(
        n_decoded,
        votes,
        runner_up,
        min_votes,
        resolved = accepted,
        "origin anchor (decoded ids)"
    );
    let (coord_map, h) = best.filter(|_| accepted)?;
    Some(OriginResolution { coord_map, h })
}

/// ln of the expected number of placements that `votes` of `n_decoded`
/// decodes agree on by chance: `R · C(n_decoded, votes) · n_cells^(1 − votes)`,
/// with `R` orientation-preserving symmetries. A misread names a near-uniform
/// random cell, so two misreads vote for the same placement with probability
/// about `R / n_cells`, and `v` of them with `R / n_cells^(v−1)`. Fewer than
/// two votes are always at chance level (0).
fn ln_chance_votes(n_decoded: usize, votes: usize, n_rotations: usize, n_cells: usize) -> f64 {
    if votes < 2 {
        return 0.0;
    }
    let ln_binom: f64 = (0..votes)
        .map(|i| ((n_decoded - i) as f64).ln() - ((i + 1) as f64).ln())
        .sum();
    ln_binom + (n_rotations as f64).ln() + (1.0 - votes as f64) * (n_cells as f64).ln()
}

/// Dot anchoring. A placement is fixed by a lattice symmetry `g` and the
/// relative-frame cell `a` that carries the triad (board cell `(0, 0)`), so
/// placements are enumerated by anchor cell near the labeled patch — never by
/// translation over the board, whose count grows with the board while the
/// triad can only be verified where the patch is. Each placement is scored by
/// its weakest dot's contrast, measured through one homography of the labeled
/// patch ([`PatchFrame`]).
///
/// Placements must be physically valid: every labeled ring on a board cell
/// and an orientation-preserving board→image map. The winner must clear
/// [`MIN_DOT_CONTRAST`] and lead every other valid placement by
/// [`MIN_MARGIN`]; `None` also when the patch is degenerate or no placement
/// can be scored (dots out of view).
fn resolve_by_dots(
    entries: &[(Coord, [f64; 2])],
    target: &TargetLayout,
    sampler: DistortionAwareSampler<'_>,
    fiducials: &OriginFiducials,
) -> Option<OriginResolution> {
    let Some(patch) = PatchFrame::fit(entries, target) else {
        tracing::info!("origin anchor: degenerate labeled patch");
        return None;
    };
    let triad = triad_offsets(target)?;
    let board_coords = target.cells().iter().map(|cell| cell.coord);
    let (board_min, board_max) = coord_bbox(board_coords)?;
    let (patch_min, patch_max) = coord_bbox(entries.iter().map(|(c, _)| *c))?;
    let dot_radius_mm = f64::from(fiducials.dot_radius_mm);

    let mut scored: Vec<(CoordMap, f32)> = Vec::new();
    for &g in target.lattice_kind().symmetry_transforms() {
        // board→image = patch homography ∘ (board → relative), and the latter
        // reverses orientation exactly when `g` does. This also admits the
        // reflected placements when canonicalization left the relative frame
        // mirrored (`assign::canonicalize_entries`, no `+u`/`+v` neighbour pair).
        if g.determinant() * patch.orientation <= 0 {
            continue;
        }
        let Some(g_inv) = g.inverse() else { continue };
        let m = g.matrix();
        let mi = g_inv.matrix();
        let dots_rel: Vec<[f64; 2]> = triad
            .iter()
            .map(|d| {
                [
                    f64::from(mi[0][0]) * d[0] + f64::from(mi[0][1]) * d[1],
                    f64::from(mi[1][0]) * d[0] + f64::from(mi[1][1]) * d[1],
                ]
            })
            .collect();
        let (mapped_min, mapped_max) = coord_bbox(entries.iter().map(|(c, _)| g.apply(*c)))?;
        for au in (patch_min.u - ANCHOR_SEARCH_MARGIN)..=(patch_max.u + ANCHOR_SEARCH_MARGIN) {
            for av in (patch_min.v - ANCHOR_SEARCH_MARGIN)..=(patch_max.v + ANCHOR_SEARCH_MARGIN) {
                // c ↦ g·c + t with g·a + t = (0, 0): the anchor cell `a` carries
                // the triad.
                let t = [
                    -(m[0][0] * au + m[0][1] * av),
                    -(m[1][0] * au + m[1][1] * av),
                ];
                let fits_board = mapped_min.u + t[0] >= board_min.u
                    && mapped_min.v + t[1] >= board_min.v
                    && mapped_max.u + t[0] <= board_max.u
                    && mapped_max.v + t[1] <= board_max.v;
                if !fits_board {
                    continue;
                }
                let mut score = f32::INFINITY;
                for d in &dots_rel {
                    let rel = [f64::from(au) + d[0], f64::from(av) + d[1]];
                    match dot_contrast(sampler, &patch.h, patch.mm(rel), dot_radius_mm) {
                        Some(contrast) => score = score.min(contrast),
                        None => {
                            score = f32::NAN;
                            break;
                        }
                    }
                }
                if score.is_finite() {
                    scored.push((CoordMap::new(g, t), score));
                }
            }
        }
    }
    if scored.is_empty() {
        tracing::info!("origin anchor: no placement could be scored (dots out of view?)");
        return None;
    }

    // Best first; the stable sort keeps enumeration order among ties. The
    // bounding-box test above is exact on rect boards; the full embedding
    // test runs only on the placements that can still decide.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let embeds = |map: &CoordMap| {
        entries
            .iter()
            .all(|(c, _)| target.cell_xy_mm(map.apply(*c)).is_some())
    };
    let mut valid = scored.iter().filter(|(map, _)| embeds(map));
    let Some(&(coord_map, best_score)) = valid.next() else {
        tracing::info!("origin anchor: no placement admits the labeled patch");
        return None;
    };
    if best_score < MIN_DOT_CONTRAST {
        tracing::info!(
            best_score,
            "origin anchor: best candidate below contrast threshold"
        );
        return None;
    }
    if let Some(&(_, runner_up)) = valid
        .take_while(|(_, s)| best_score - s < MIN_MARGIN)
        .next()
    {
        tracing::info!(
            best_score,
            runner_up,
            "origin anchor: ambiguous — margin over runner-up too small"
        );
        return None;
    }
    let h = placement_homography(entries, coord_map, target)?;
    tracing::info!(
        best_score,
        n_scored = scored.len(),
        "origin anchor: resolved board origin"
    );
    Some(OriginResolution { coord_map, h })
}

/// The labeled patch's homography from relative-frame lattice millimetres
/// (`basis · coord`) to working pixels, fitted once and shared by every
/// placement: a placement re-expresses board millimetres by a lattice isometry,
/// under which the normalized DLT is equivariant, so the per-placement fit
/// ([`placement_homography`]) is needed only for the winner.
struct PatchFrame {
    h: Matrix3<f64>,
    /// Lattice steps along `u` and `v`, board mm (`LatticeGeometry::basis_mm`).
    basis: [[f64; 2]; 2],
    /// Sign of the homography's Jacobian over the patch (±1).
    orientation: i32,
}

impl PatchFrame {
    /// `None` for a patch whose rings lie on one lattice line (homography
    /// undetermined, see [`placement_homography`]).
    fn fit(entries: &[(Coord, [f64; 2])], target: &TargetLayout) -> Option<Self> {
        let basis = target.lattice().basis_mm();
        let mm = |c: Coord| lattice_mm(basis, [f64::from(c.u), f64::from(c.v)]);
        let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) =
            entries.iter().map(|(c, center)| (mm(*c), *center)).unzip();
        let collinear_tol_mm = COLLINEAR_TOL_SPACINGS * f64::from(target.min_center_spacing_mm());
        if src.is_empty() || all_collinear(&src, collinear_tol_mm) {
            return None;
        }
        let h = estimate_homography_dlt(&src, &dst).ok()?;
        let n = src.len() as f64;
        let centroid = [
            src.iter().map(|p| p[0]).sum::<f64>() / n,
            src.iter().map(|p| p[1]).sum::<f64>() / n,
        ];
        let det = jacobian_det(&h, centroid);
        if !det.is_normal() {
            return None;
        }
        Some(Self {
            h,
            basis,
            orientation: if det > 0.0 { 1 } else { -1 },
        })
    }

    /// Relative-frame lattice millimetres of a (fractional) lattice coordinate.
    fn mm(&self, coord: [f64; 2]) -> [f64; 2] {
        lattice_mm(self.basis, coord)
    }
}

/// `coord[0] · step_u + coord[1] · step_v` for a basis `[step_u, step_v]`.
fn lattice_mm(basis: [[f64; 2]; 2], coord: [f64; 2]) -> [f64; 2] {
    [
        coord[0] * basis[0][0] + coord[1] * basis[1][0],
        coord[0] * basis[0][1] + coord[1] * basis[1][1],
    ]
}

/// The origin dots as fractional lattice offsets from board cell `(0, 0)`, the
/// cell they surround (`target::fiducials::origin_dot_positions_mm`).
fn triad_offsets(target: &TargetLayout) -> Option<Vec<[f64; 2]>> {
    let origin = target.cell_xy_mm(Coord::new(0, 0))?;
    let [bu, bv] = target.lattice().basis_mm();
    let det = bu[0] * bv[1] - bv[0] * bu[1];
    let offsets = target
        .fiducial_dots_mm()
        .iter()
        .map(|d| {
            let dx = f64::from(d[0]) - f64::from(origin[0]);
            let dy = f64::from(d[1]) - f64::from(origin[1]);
            [
                (bv[1] * dx - bv[0] * dy) / det,
                (bu[0] * dy - bu[1] * dx) / det,
            ]
        })
        .collect();
    Some(offsets)
}

/// Board-mm → working-px homography of a placement, fitted by DLT over the
/// labeled entries it maps onto the board. `None` for degenerate or
/// orientation-reversing placements — the physical filter that rejects every
/// reflected candidate.
///
/// Degenerate includes entries that all land on one board line: a homography
/// is undetermined by collinear points, and the DLT's solution — its
/// orientation included — is then set by centre noise alone. A reflection
/// that maps a single lattice line of the labeling onto a board edge line is
/// such a placement (its votes are that line's correct decodes counted a
/// second time); on rect_coded synthetic views the noise sign let it pass as
/// a runner-up and blocked the true placement's anchor.
fn placement_homography(
    entries: &[(Coord, [f64; 2])],
    coord_map: CoordMap,
    target: &TargetLayout,
) -> Option<Matrix3<f64>> {
    let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = entries
        .iter()
        .filter_map(|(coord, center)| {
            let xy = target.cell_xy_mm(coord_map.apply(*coord))?;
            Some(([f64::from(xy[0]), f64::from(xy[1])], *center))
        })
        .unzip();
    let collinear_tol_mm = COLLINEAR_TOL_SPACINGS * f64::from(target.min_center_spacing_mm());
    if src.is_empty() || all_collinear(&src, collinear_tol_mm) {
        return None;
    }
    let h = estimate_homography_dlt(&src, &dst).ok()?;
    let center_mm = [
        src.iter().map(|p| p[0]).sum::<f64>() / src.len() as f64,
        src.iter().map(|p| p[1]).sum::<f64>() / src.len() as f64,
    ];
    (jacobian_det(&h, center_mm) > 0.0).then_some(h)
}

/// Whether every point lies within `tol` of one line (the points' principal
/// axis through their centroid). Board cell positions are exact lattice
/// points: a non-collinear set is off any line by a sizable fraction of the
/// spacing, a collinear one only by `f32` rounding.
fn all_collinear(points: &[[f64; 2]], tol: f64) -> bool {
    let n = points.len() as f64;
    let c = [
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in points {
        let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let normal = [-theta.sin(), theta.cos()];
    points
        .iter()
        .all(|p| (normal[0] * (p[0] - c[0]) + normal[1] * (p[1] - c[1])).abs() <= tol)
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
    use crate::target::LatticeGeometry;
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
            for dot in target.fiducial_dots_mm() {
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
            // Derived triad for an 8x8 board: [49,49], [35,49], [49,63].
            Some(OriginFiducials { dot_radius_mm: 1.4 }),
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

    /// A placement that lands only one lattice line of the labeling on the
    /// board has no determinable homography, whatever the centre noise makes
    /// of its DLT's orientation: it must never compete (as winner or
    /// runner-up) in the decoded-ID vote.
    #[test]
    fn single_line_placements_are_degenerate() {
        let target = small_rect_like();
        let h_mat = Matrix3::new(6.0, 0.4, 60.0, -0.3, 6.2, 70.0, 1e-4, -2e-4, 1.0);
        let transforms = target.lattice_kind().symmetry_transforms();
        let mut n_single_line = 0usize;
        for seed in 0..8u64 {
            // Deterministic sub-0.1 px centre noise, different per seed.
            let mut state = seed.wrapping_mul(6_364_136_223_846_793_005) | 1;
            let mut noise = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.1
            };
            let entries: Vec<(Coord, [f64; 2])> = target
                .cells()
                .iter()
                .map(|cell| {
                    let p = homography_project(
                        &h_mat,
                        f64::from(cell.xy_mm[0]),
                        f64::from(cell.xy_mm[1]),
                    );
                    (cell.coord, [p[0] + noise(), p[1] + noise()])
                })
                .collect();
            for &g in transforms {
                for du in -16..=16 {
                    for dv in -16..=16 {
                        let map = CoordMap::new(g, [du, dv]);
                        let on_board: Vec<Coord> = entries
                            .iter()
                            .map(|(c, _)| map.apply(*c))
                            .filter(|c| target.cell_xy_mm(*c).is_some())
                            .collect();
                        let one_line = on_board.iter().all(|c| c.u == on_board[0].u)
                            || on_board.iter().all(|c| c.v == on_board[0].v);
                        if on_board.len() < 4 || !one_line {
                            continue;
                        }
                        n_single_line += 1;
                        assert!(
                            placement_homography(&entries, map, &target).is_none(),
                            "single-line placement {g:?} {:?} accepted (seed {seed})",
                            [du, dv]
                        );
                    }
                }
            }
        }
        assert!(
            n_single_line > 0,
            "the sweep must reach single-line placements"
        );
        // The true placement is not degenerate.
        let identity = CoordMap::new(transforms[0], [0, 0]);
        let entries: Vec<(Coord, [f64; 2])> = target
            .cells()
            .iter()
            .map(|cell| {
                let xy = [f64::from(cell.xy_mm[0]), f64::from(cell.xy_mm[1])];
                (cell.coord, homography_project(&h_mat, xy[0], xy[1]))
            })
            .collect();
        assert!(placement_homography(&entries, identity, &target).is_some());
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

    /// A 58×58 plain rect board (2 mm pitch) and a 1,640-cell plain hex
    /// board, both with auto dots: windows of them have far more placements
    /// than any fixed cap.
    fn large_boards() -> [TargetLayout; 2] {
        use crate::target::OriginDots;
        [
            TargetLayout::plain_rect(2.0, 58, 58, 0.8, 0.4, OriginDots::Auto).expect("rect"),
            TargetLayout::plain_hex(2.0, 41, 40, 1.2, 0.6, OriginDots::Auto).expect("hex"),
        ]
    }

    /// Paint a filled dark disc at a board point (a dust speck).
    fn paint_dot(img: &mut GrayImage, h_mat: &Matrix3<f64>, xy_mm: [f64; 2], r_px: f64) {
        let c = homography_project(h_mat, xy_mm[0], xy_mm[1]);
        for y in (c[1] - r_px).floor() as u32..=(c[1] + r_px).ceil() as u32 {
            for x in (c[0] - r_px).floor() as u32..=(c[0] + r_px).ceil() as u32 {
                let d2 = (f64::from(x) - c[0]).powi(2) + (f64::from(y) - c[1]).powi(2);
                if d2 <= r_px * r_px {
                    img.put_pixel(x, y, image::Luma([20u8]));
                }
            }
        }
    }

    /// View scale: a 50 px marker spacing.
    fn view_scale(target: &TargetLayout) -> f64 {
        50.0 / f64::from(target.min_center_spacing_mm())
    }

    /// A 700×550 view at [`view_scale`], rotated by `rot_rad`, with board point
    /// `look_mm` at the image centre; markers of the cells fully in view,
    /// labeled `g·coord + offset`.
    fn window_view(
        target: &TargetLayout,
        look_mm: [f64; 2],
        rot_rad: f64,
        g: GridTransform,
        offset: [i32; 2],
    ) -> (GrayImage, Matrix3<f64>, Vec<MarkerRecord>) {
        let (w, h, scale) = (700u32, 550u32, view_scale(target));
        let (sin, cos) = rot_rad.sin_cos();
        let tx = 350.0 - scale * (cos * look_mm[0] - sin * look_mm[1]);
        let ty = 275.0 - scale * (sin * look_mm[0] + cos * look_mm[1]);
        let (img, h_mat) = render_plain_target(target, scale, tx, ty, rot_rad, w, h);
        let r_px = f64::from(target.ring().outer_radius_mm) * scale;
        let relabel = CoordMap::new(g, offset);
        let markers = labeled_markers(target, &h_mat, |c| relabel.apply(c))
            .into_iter()
            .filter(|m| {
                let [x, y] = m.center;
                x >= r_px && y >= r_px && x <= f64::from(w) - r_px && y <= f64::from(h) - r_px
            })
            .collect();
        (img, h_mat, markers)
    }

    fn origin_mm(target: &TargetLayout) -> [f64; 2] {
        let xy = target.cell_xy_mm(Coord::new(0, 0)).expect("origin cell");
        [f64::from(xy[0]), f64::from(xy[1])]
    }

    /// A small window of a large board anchors, whatever the relabeling, at a
    /// cost set by the window: the window's ~150 rings admit ~10⁴ placements on
    /// the board.
    #[test]
    fn small_window_of_a_large_board_resolves() {
        for target in large_boards() {
            let o = origin_mm(&target);
            let rotations: Vec<GridTransform> = target
                .lattice_kind()
                .symmetry_transforms()
                .iter()
                .copied()
                .filter(|g| g.determinant() > 0)
                .collect();
            for (i, &g) in rotations.iter().enumerate() {
                let look = [o[0] + 3.0 - i as f64, o[1] - 2.0 + 0.5 * i as f64];
                let rot = 0.3 * i as f64;
                let (img, _, markers) = window_view(&target, look, rot, g, [5, -3]);
                assert!(markers.len() > 100, "{} in view", markers.len());
                let res = resolve_origin(&img, &markers, &target, None)
                    .unwrap_or_else(|| panic!("{} {g:?}: unresolved", target.name()));
                for cell in target.cells() {
                    let rel = CoordMap::new(g, [5, -3]).apply(cell.coord);
                    assert_eq!(res.coord_map.apply(rel), cell.coord, "{}", target.name());
                }
            }
        }
    }

    /// Without the triad in view, nothing resolves.
    #[test]
    fn window_without_the_triad_is_unresolved() {
        for target in large_boards() {
            let o = origin_mm(&target);
            let g = target.lattice_kind().symmetry_transforms()[0];
            // 12 × 9 spacings away: the triad is well out of the 14 × 11 view.
            let s = f64::from(target.min_center_spacing_mm());
            let look = [o[0] + 12.0 * s, o[1] - 9.0 * s];
            let (img, _, markers) = window_view(&target, look, 0.2, g, [0, 0]);
            assert!(markers.len() > 100);
            assert!(resolve_origin(&img, &markers, &target, None).is_none());
        }
    }

    /// A speck in a far gap changes nothing; a speck that makes a second
    /// triad as dark as the true one (rect: closing the square of gaps around
    /// the triad's corner cell; hex: the next hole around the origin cell)
    /// makes the placement ambiguous: unresolved rather than a guess.
    #[test]
    fn dust_in_a_gap_never_resolves_wrong() {
        for target in large_boards() {
            let o = origin_mm(&target);
            let g = target.lattice_kind().symmetry_transforms()[0];
            let r_dot =
                f64::from(target.fiducials().expect("dots").dot_radius_mm) * view_scale(&target);
            let pitch = f64::from(target.pitch_mm());
            let (img, h_mat, markers) = window_view(&target, [o[0] + 1.0, o[1]], 0.1, g, [0, 0]);
            let (far, square) = match target.lattice() {
                LatticeGeometry::Rect(_) => (
                    [o[0] - 3.5 * pitch, o[1] - 2.5 * pitch],
                    [o[0] - 0.5 * pitch, o[1] + 1.5 * pitch],
                ),
                LatticeGeometry::Hex(_) => {
                    // Holes around the origin cell sit at 30° + k·60°, one pitch out;
                    // the triad holds 30°, 90°, 150°.
                    let at = |deg: f64| {
                        let (s, c) = deg.to_radians().sin_cos();
                        [o[0] + pitch * c, o[1] + pitch * s]
                    };
                    let [bu, bv] = target.lattice().basis_mm();
                    let hole = at(270.0);
                    let far = [
                        hole[0] - 3.0 * bu[0] - 2.0 * bv[0],
                        hole[1] - 3.0 * bu[1] - 2.0 * bv[1],
                    ];
                    (far, at(210.0))
                }
            };
            let mut dusty = img.clone();
            paint_dot(&mut dusty, &h_mat, far, r_dot);
            let res =
                resolve_origin(&dusty, &markers, &target, None).expect("far dust is harmless");
            for m in &markers {
                let c = m.grid_coord.expect("labeled");
                let c = Coord::new(c[0], c[1]);
                assert_eq!(res.coord_map.apply(c), c, "{}", target.name());
            }
            let mut ambiguous = img.clone();
            paint_dot(&mut ambiguous, &h_mat, square, r_dot);
            assert!(
                resolve_origin(&ambiguous, &markers, &target, None).is_none(),
                "{}: a second, equally dark triad must leave the origin unresolved",
                target.name()
            );
        }
    }
}
