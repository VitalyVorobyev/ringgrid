/// Neighbor count for the completion-seed expected radius: wider than the
/// hex valence (6) so the median stays robust when the immediate ring around
/// a missing cell is itself sparse or mis-fitted.
const EXPECTED_RADIUS_K_NEIGHBORS: usize = 12;

use image::GrayImage;
use projective_grid::{Coord, expert::lattice::predict_grid_position};

use crate::pipeline::{GridMap, marker_cell_coord};
use crate::target::TargetLayout;
use std::collections::HashSet;

use crate::conic::Ellipse;
use crate::detector::MarkerRecord;
use crate::detector::id_correction::{affine_to_image, fit_local_affine};
use crate::detector::marker_build::DetectionSource;
use crate::homography::homography_project as project;
use crate::ring::edge_sample::EdgeSampleResult;

use super::{
    CompletionConfig, DetectConfig, OuterFitCandidate,
    fit_outer_candidate_from_prior_for_completion,
    marker_build::{fit_support_score, marker_from_outer_fit},
    median_neighbor_outer_radius_px,
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
    /// Cells filled by adopting an already-fitted unlabeled ring instead of
    /// refitting (see [`complete_with_h`]'s `unlabeled` pool).
    pub n_adopted: usize,
}

struct CandidateQuality {
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

/// The board cell a completion attempt targets: its lattice coordinate, plus
/// its codebook ID on coded targets.
#[derive(Debug, Clone, Copy)]
struct CompletionCell {
    coord: Coord,
    id: Option<usize>,
}

impl std::fmt::Display for CompletionCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.id {
            Some(id) => write!(f, "id={id}"),
            None => write!(f, "cell=({},{})", self.coord.u, self.coord.v),
        }
    }
}

/// Upper bound on unanchored patch-growth rounds; each round can extend the
/// labeled patch by one lattice ring.
const MAX_GROWTH_ROUNDS: usize = 32;

/// Which cells completion visits and in which frame it places them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompletionScope {
    /// Coded target: every coded cell, board frame, one pass.
    Coded,
    /// Coded target anchored from a labeled patch (grid-first): coded cells in
    /// the identified patch's bounding box expanded by one lattice ring per
    /// round, board frame. Growing outward keeps every seed next to identified
    /// neighbors, which a one-pass sweep in board order cannot.
    CodedPatch,
    /// Plain target with a resolved origin: every cell, board frame, one pass.
    PlainAnchored,
    /// Plain target without a resolved origin: grow the labeled patch in the
    /// relative assignment frame, one lattice ring per round. The grid labeler
    /// may recover only an interior core (hex boundary cells in particular),
    /// and the physical board is the only ring source in view, so the quality
    /// gates plus the downstream geometric verify keep growth precise.
    PlainRelative,
}

impl CompletionScope {
    /// Plain-target scope for an anchored / unanchored labeling.
    pub(crate) fn plain(anchored: bool) -> Self {
        if anchored {
            Self::PlainAnchored
        } else {
            Self::PlainRelative
        }
    }

    /// Position (mm) of a cell in this scope's frame: board millimeters, or the
    /// relative assignment frame. `None` for a coordinate off the board.
    fn cell_xy(self, target: &TargetLayout, coord: Coord) -> Option<[f64; 2]> {
        match self {
            Self::Coded | Self::CodedPatch | Self::PlainAnchored => target
                .cell_xy_mm(coord)
                .map(|xy| [f64::from(xy[0]), f64::from(xy[1])]),
            Self::PlainRelative => Some(crate::pipeline::frame_xy_mm(target, coord)),
        }
    }

    /// Cells to try in one round: the board's (coded) cells, or the labeled
    /// patch's bounding box expanded by one lattice ring. `None` when there is
    /// nothing to grow from.
    fn candidates(
        self,
        target: &TargetLayout,
        present: &HashSet<Coord>,
    ) -> Option<Vec<CompletionCell>> {
        let board_cells = |coded_only: bool| {
            target
                .cells()
                .iter()
                .filter(|cell| !coded_only || cell.id.is_some())
                .map(|cell| CompletionCell {
                    coord: cell.coord,
                    id: cell.id,
                })
                .collect()
        };
        match self {
            Self::Coded => Some(board_cells(true)),
            Self::PlainAnchored => Some(board_cells(false)),
            Self::CodedPatch => Some(
                grown_bbox(present)?
                    .filter_map(|coord| {
                        let id = target.id_of(coord)?;
                        Some(CompletionCell {
                            coord,
                            id: Some(id),
                        })
                    })
                    .collect(),
            ),
            Self::PlainRelative => Some(
                grown_bbox(present)?
                    .map(|coord| CompletionCell { coord, id: None })
                    .collect(),
            ),
        }
    }

    /// Board-cell scopes have a fixed candidate set, so one pass covers them.
    fn max_rounds(self) -> usize {
        match self {
            Self::Coded | Self::PlainAnchored => 1,
            Self::CodedPatch | Self::PlainRelative => MAX_GROWTH_ROUNDS,
        }
    }
}

/// Coordinates of the present set's bounding box expanded by one lattice ring.
fn grown_bbox(present: &HashSet<Coord>) -> Option<impl Iterator<Item = Coord>> {
    let (min, max) = coord_bbox(present.iter().copied())?;
    Some(
        ((min.v - 1)..=(max.v + 1))
            .flat_map(move |v| ((min.u - 1)..=(max.u + 1)).map(move |u| Coord::new(u, v))),
    )
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

/// Local-affine seed from the nearest labeled markers, positioned in the
/// scope's frame via their lattice cell.
fn local_affine_seed(
    target_xy: [f64; 2],
    markers: &[MarkerRecord],
    target: &TargetLayout,
    scope: CompletionScope,
) -> Option<[f64; 2]> {
    local_affine_seed_from_neighbors(
        target_xy,
        markers.iter().filter_map(|marker| {
            let coord = marker_cell_coord(marker, target)?;
            Some((scope.cell_xy(target, coord)?, marker.center))
        }),
    )
}

/// Local-affine completion seed from the nearest labeled neighbors.
fn local_affine_seed_from_neighbors(
    target_board_xy: [f64; 2],
    neighbor_source: impl Iterator<Item = ([f64; 2], [f64; 2])>,
) -> Option<[f64; 2]> {
    const MAX_NEIGHBORS: usize = 4;

    let mut neighbors: Vec<([f64; 2], [f64; 2], f64)> = neighbor_source
        .filter_map(|(board_xy, center)| {
            if !(center[0].is_finite() && center[1].is_finite()) {
                return None;
            }
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

/// Image-space seed for a missing cell: lattice prediction from labeled
/// neighbors, else a local-affine fit, else the homography projection.
fn completion_seed(
    coord: Coord,
    cell_xy: [f64; 2],
    h: &nalgebra::Matrix3<f64>,
    markers: &[MarkerRecord],
    grid: &GridMap,
    target: &TargetLayout,
    scope: CompletionScope,
) -> [f64; 2] {
    predict_grid_position(grid, coord, target.lattice_kind())
        .map(|p| [f64::from(p.position.x), f64::from(p.position.y)])
        .or_else(|| local_affine_seed(cell_xy, markers, target, scope))
        .unwrap_or_else(|| project(h, cell_xy[0], cell_xy[1]))
}

/// Labeled rings within this many nearest-neighbour spacings of a cell
/// count as its local support.
const LOCAL_SUPPORT_SPACINGS: f64 = 3.05;
/// Fewest labeled rings near a cell for a completion refit there.
const MIN_LOCAL_SUPPORT: usize = 3;

/// Whether at least [`MIN_LOCAL_SUPPORT`] labeled rings of the detection
/// itself (not earlier refits: a chain of refits extrapolates as surely as a
/// homography) lie within [`LOCAL_SUPPORT_SPACINGS`] lattice spacings of the
/// cell (board frame of `scope`), so its predicted position interpolates.
fn has_local_support(
    cell_xy: [f64; 2],
    markers: &[MarkerRecord],
    target: &TargetLayout,
    scope: CompletionScope,
) -> bool {
    let reach = LOCAL_SUPPORT_SPACINGS * f64::from(target.min_center_spacing_mm());
    markers
        .iter()
        .filter(|m| m.source != DetectionSource::Completion)
        .filter_map(|m| scope.cell_xy(target, marker_cell_coord(m, target)?))
        .filter(|xy| (xy[0] - cell_xy[0]).hypot(xy[1] - cell_xy[1]) <= reach)
        .take(MIN_LOCAL_SUPPORT)
        .count()
        >= MIN_LOCAL_SUPPORT
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
    if let Some(d) = decode_result
        && d.id != expected_id
    {
        return Some(CompletionDecodeNotice {
            reason: CompletionDecodeNoticeReason::DecodeMismatchAccepted,
            expected_id,
            observed_id: d.id,
        });
    }
    None
}

fn check_quality_gates(
    quality: &CandidateQuality,
    params: &CompletionConfig,
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
    target_cell: CompletionCell,
    cand: &OuterFitCandidate,
    projected_center: [f64; 2],
    r_expected: f32,
    config: &DetectConfig,
    stats: &mut CompletionStats,
) -> Result<CandidateQuality, ()> {
    let params = &config.advanced.completion;
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
        config.advanced.outer_fit.max_angular_gap_rad,
    ) {
        tracing::trace!(
            "Completion {} gate_reject={} context={:?}",
            target_cell,
            reject.reason,
            reject.context
        );
        stats.n_failed_gate += 1;
        return Err(());
    }

    // Plain cells have no code band, so the decode gate cannot apply. Every
    // accepted decode is an exact codeword match.
    if params.require_perfect_decode && config.target.is_coded() && cand.decode_result.is_none() {
        tracing::trace!(
            "Completion {} gate_reject={}",
            target_cell,
            CompletionGateRejectReason::PerfectDecodeRequired.code(),
        );
        stats.n_failed_gate += 1;
        return Err(());
    }

    Ok(quality)
}

/// Build the final `MarkerRecord` for an accepted completion candidate.
///
/// Only a decode that agrees with the target cell's ID is attributed to the
/// marker.
fn assemble_completion_marker(
    gray: &GrayImage,
    target_cell: CompletionCell,
    mut cand: OuterFitCandidate,
    config: &DetectConfig,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
) -> MarkerRecord {
    let marker_id = target_cell.id;
    cand.decode_result = cand.decode_result.filter(|d| marker_id == Some(d.id));
    let marker = marker_from_outer_fit(gray, cand, config, mapper);
    MarkerRecord {
        id: marker_id,
        grid_coord: Some([target_cell.coord.u, target_cell.coord.v]),
        source: DetectionSource::Completion,
        ..marker
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
    target_cell: CompletionCell,
    projected_center: [f64; 2],
    markers: &[MarkerRecord],
    config: &DetectConfig,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
    stats: &mut CompletionStats,
) -> Option<MarkerRecord> {
    let r_expected = median_neighbor_outer_radius_px(
        projected_center,
        markers,
        EXPECTED_RADIUS_K_NEIGHBORS,
        None,
        1,
    )
    .map(|r| r as f32)
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
        target_cell,
        &cand,
        projected_center,
        r_expected,
        config,
        stats,
    ) {
        Ok(q) => q,
        Err(()) => return None,
    };

    if let Some(id) = target_cell.id
        && let Some(notice) = check_decode_gate(cand.decode_result.as_ref(), id)
    {
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
        "Completion added {} reproj_err={:.2}px",
        target_cell,
        quality.reproj_err
    );

    Some(assemble_completion_marker(
        gray,
        target_cell,
        cand,
        config,
        mapper,
    ))
}

/// H-guided completion at missing cells.
///
/// Visits the cells of `scope` that no marker is labeled with yet, seeds each
/// from its labeled neighbors (or `h`), and keeps conservative fits that pass
/// the quality gates. For [`CompletionScope::PlainRelative`] the visited set
/// grows round by round with the labeled patch. Each cell is attempted at most
/// once; completed markers carry their cell's `grid_coord`, frame
/// `board_xy_mm`, and — on coded targets — its ID.
///
/// `unlabeled` holds already-fitted rings the labeling stage could not place.
/// Before refitting a cell, the pool ring nearest the cell's seed is adopted
/// instead when within `reproj_gate_px` — the gate a refit center must pass.
pub(crate) fn complete_with_h(
    gray: &GrayImage,
    h: &nalgebra::Matrix3<f64>,
    markers: &mut Vec<MarkerRecord>,
    unlabeled: &mut Vec<MarkerRecord>,
    config: &DetectConfig,
    scope: CompletionScope,
    mapper: Option<&dyn crate::pixelmap::PixelMapper>,
) -> CompletionStats {
    let params = &config.advanced.completion;
    if !params.enable {
        return CompletionStats::default();
    }
    let target = &config.target;

    let mut present: HashSet<Coord> = markers
        .iter()
        .filter_map(|m| marker_cell_coord(m, target))
        .collect();
    if present.is_empty() {
        return CompletionStats::default();
    }

    let (w, h_img) = gray.dimensions();
    let roi_radius = params.roi_radius_px.clamp(8.0, 200.0) as f64;
    let safe_margin = roi_radius + params.image_margin_px.max(0.0) as f64;

    let mut stats = CompletionStats::default();
    let mut tried: HashSet<Coord> = HashSet::new();
    let mut capped = false;

    for _round in 0..scope.max_rounds() {
        let Some(candidates) = scope.candidates(target, &present) else {
            break;
        };
        // Predictions improve as the patch grows; rebuild the map per round.
        let grid = crate::pipeline::build_grid_map(markers, target);
        let mut added_this_round = 0usize;

        for cell in candidates {
            if present.contains(&cell.coord) || tried.contains(&cell.coord) {
                continue;
            }
            stats.n_candidates_total += 1;
            let Some(cell_xy) = scope.cell_xy(target, cell.coord) else {
                continue;
            };
            let seed = completion_seed(cell.coord, cell_xy, h, markers, &grid, target, scope);
            if let Some(marker) = adopt_nearest(unlabeled, seed, params.reproj_gate_px, |m| {
                !decode_contradicts_cell(m, cell.id, target)
            }) {
                markers.push(adopt_into(marker, cell, cell_xy, &mut stats));
                present.insert(cell.coord);
                added_this_round += 1;
                continue;
            }
            if !is_center_in_bounds(seed, f64::from(w), f64::from(h_img), safe_margin) {
                continue;
            }
            // A refit takes the ID of the cell its seed predicts; with no
            // labeled ring nearby that prediction is an extrapolation, which
            // uncorrected lens distortion shifts by a whole cell at the
            // periphery (precision harness, hex at lambda 0.20 / 0.25: refits
            // sitting exactly on the neighbouring ring, labeled with this
            // cell's ID). Plain targets' labels come from this growth itself
            // and are verified by the anchor and geometric verification; the
            // gate there costs a third of their distorted-view recall.
            if target.is_coded() && !has_local_support(cell_xy, markers, target, scope) {
                continue;
            }
            stats.n_in_image += 1;

            if params
                .max_attempts
                .is_some_and(|max| stats.n_attempted >= max)
            {
                capped = true;
                break;
            }
            stats.n_attempted += 1;
            tried.insert(cell.coord);

            let Some(mut marker) =
                try_complete_marker(gray, cell, seed, markers, config, mapper, &mut stats)
            else {
                continue;
            };
            marker.board_xy_mm = Some(cell_xy);
            markers.push(marker);
            present.insert(cell.coord);
            stats.n_added += 1;
            added_this_round += 1;
        }

        if capped || added_this_round == 0 {
            break;
        }
    }

    tracing::info!(
        ?scope,
        n_adopted = stats.n_adopted,
        n_added = stats.n_added,
        n_attempted = stats.n_attempted,
        n_in_image = stats.n_in_image,
        "completion summary"
    );

    stats
}

/// Label an adopted pool ring with its cell (and the cell's ID, counting a
/// contradicting decode in `n_decode_mismatch`). The ring keeps its fit,
/// source and center correction.
fn adopt_into(
    marker: MarkerRecord,
    cell: CompletionCell,
    cell_xy: [f64; 2],
    stats: &mut CompletionStats,
) -> MarkerRecord {
    if marker.id.is_some() && marker.id != cell.id {
        stats.n_decode_mismatch += 1;
    }
    stats.n_adopted += 1;
    MarkerRecord {
        id: cell.id,
        grid_coord: Some([cell.coord.u, cell.coord.v]),
        board_xy_mm: Some(cell_xy),
        ..marker
    }
}

/// Whether a marker's own decode names a board cell other than the one with
/// `cell_id`: then either the decode is a misread or the marker is not this
/// cell's ring (a lattice slip, a ring of a second board in view), and the
/// marker must not take the cell's ID. A decode naming no board cell is a
/// misread of this target and contradicts nothing.
pub(crate) fn decode_contradicts_cell(
    marker: &MarkerRecord,
    cell_id: Option<usize>,
    target: &TargetLayout,
) -> bool {
    marker
        .id
        .is_some_and(|id| Some(id) != cell_id && target.coord_of_id(id).is_some())
}

/// Remove and return the pool ring nearest `seed` when it lies within
/// `gate_px` or within half its own outer radius of it, and `admissible`
/// accepts it. Rings never overlap, so a neighboring cell's center is at least
/// two outer radii away: a ring within half a radius is unambiguously this
/// cell's, at any marker scale (a fixed pixel gate is far tighter than the
/// lattice prediction error on large markers).
fn adopt_nearest(
    pool: &mut Vec<MarkerRecord>,
    seed: [f64; 2],
    gate_px: f32,
    admissible: impl Fn(&MarkerRecord) -> bool,
) -> Option<MarkerRecord> {
    let (i, d) = pool
        .iter()
        .enumerate()
        .map(|(i, m)| (i, (m.center[0] - seed[0]).hypot(m.center[1] - seed[1])))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    let half_radius = pool[i]
        .ellipse_outer
        .as_ref()
        .map_or(0.0, |e| 0.25 * (e.a + e.b));
    (d <= f64::from(gate_px).max(half_radius) && admissible(&pool[i])).then(|| pool.swap_remove(i))
}

/// Inclusive coordinate bounding box of a coordinate set.
pub(crate) fn coord_bbox(coords: impl Iterator<Item = Coord>) -> Option<(Coord, Coord)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::Ellipse;
    use crate::detector::marker_build::FitMetrics;

    #[test]
    fn radii_cv_is_zero_for_degenerate_or_uniform_inputs() {
        // Fewer than 2 samples → 0.
        assert_eq!(radii_coefficient_of_variation(&[5.0]), 0.0);
        // Uniform radii → zero scatter.
        assert_eq!(radii_coefficient_of_variation(&[10.0, 10.0, 10.0]), 0.0);
        // Sub-pixel mean is treated as degenerate → 0.
        assert_eq!(radii_coefficient_of_variation(&[0.5, 0.5]), 0.0);
    }

    #[test]
    fn radii_cv_matches_std_over_mean() {
        // mean = 10, variance = 4, std = 2 → cv = 0.2.
        let cv = radii_coefficient_of_variation(&[8.0, 12.0]);
        assert!((cv - 0.2).abs() < 1e-6, "cv = {cv}");
    }

    #[test]
    fn coord_bbox_spans_all_coordinates() {
        let coords = [Coord::new(1, 2), Coord::new(3, 1), Coord::new(0, 5)];
        let (min, max) = coord_bbox(coords.into_iter()).expect("non-empty");
        assert_eq!((min.u, min.v), (0, 1));
        assert_eq!((max.u, max.v), (3, 5));
    }

    #[test]
    fn coord_bbox_of_empty_is_none() {
        assert!(coord_bbox(std::iter::empty()).is_none());
    }

    #[test]
    fn center_in_bounds_respects_safe_margin() {
        // Comfortably inside a 100x100 image with a 10 px margin.
        assert!(is_center_in_bounds([50.0, 50.0], 100.0, 100.0, 10.0));
        // Too close to the left / bottom edges.
        assert!(!is_center_in_bounds([5.0, 50.0], 100.0, 100.0, 10.0));
        assert!(!is_center_in_bounds([50.0, 95.0], 100.0, 100.0, 10.0));
        // Non-finite centers are always rejected.
        assert!(!is_center_in_bounds([f64::NAN, 50.0], 100.0, 100.0, 10.0));
    }

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
            arc_cov: 0.2,
            fit_confidence: 0.9,
            mean_axis: 20.0,
            scale_ok: true,
            reproj_err: 0.5,
            max_angular_gap_outer: 0.1,
            radii_cv: 0.0,
        };
        let params = CompletionConfig::default();
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

    fn marker_with_id(id: usize, center: [f64; 2]) -> MarkerRecord {
        MarkerRecord {
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
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn local_affine_completion_seed_uses_nearest_decoded_neighbors() {
        let board = crate::TargetLayout::default_hex();
        let target_id = 16usize;
        let neighbor_ids = [0usize, 1usize, 14usize, 15usize];
        let affine = [[2.0, 0.1, 5.0], [-0.2, 1.5, 7.0]];
        let markers: Vec<MarkerRecord> = neighbor_ids
            .iter()
            .map(|&id| {
                let board_xy = board.xy_mm_of_id(id).expect("board xy");
                let center =
                    affine_to_image(&affine, [f64::from(board_xy[0]), f64::from(board_xy[1])]);
                marker_with_id(id, center)
            })
            .collect();

        let target_board_xy = board.xy_mm_of_id(target_id).expect("target board xy");
        let seed = local_affine_seed(
            [f64::from(target_board_xy[0]), f64::from(target_board_xy[1])],
            &markers,
            &board,
            CompletionScope::Coded,
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
    fn completion_seed_falls_back_to_h_with_fewer_than_three_neighbors() {
        let board = crate::TargetLayout::default_hex();
        let target_id = 16usize;
        let markers = vec![
            marker_with_id(0, [11.0, 7.0]),
            marker_with_id(1, [19.0, 7.5]),
        ];
        let h = nalgebra::Matrix3::new(1.0, 0.0, 3.0, 0.0, 1.0, -4.0, 0.0, 0.0, 1.0);
        let target_board_xy = board.xy_mm_of_id(target_id).expect("target board xy");
        let hex_grid = crate::pipeline::build_grid_map(&markers, &board);
        let target_xy = [f64::from(target_board_xy[0]), f64::from(target_board_xy[1])];
        let coord = board.coord_of_id(target_id).expect("target coord");
        let seed = completion_seed(
            coord,
            target_xy,
            &h,
            &markers,
            &hex_grid,
            &board,
            CompletionScope::Coded,
        );
        let expected = project(&h, target_xy[0], target_xy[1]);
        assert!((seed[0] - expected[0]).abs() < 1e-9);
        assert!((seed[1] - expected[1]).abs() < 1e-9);
    }
}
