//! `TargetLayout`: the compositional target model.

use std::collections::HashMap;

use projective_grid::{Coord, LatticeKind};

use super::error::TargetValidationError;
use super::fiducials::{OriginFiducials, origin_dot_positions_mm};
use super::lattice::{HexGeometry, LatticeGeometry, RectGeometry};
use super::ring::{CodedRingSpec, MarkerCoding, RingGeometry};

const DEFAULT_HEX_NAME: &str = "ringgrid_200mm_hex";
const DEFAULT_HEX_PITCH_MM: f32 = 8.0;
const DEFAULT_HEX_ROWS: usize = 15;
const DEFAULT_HEX_LONG_ROW_COLS: usize = 14;
const DEFAULT_OUTER_RADIUS_MM: f32 = 4.8;
const DEFAULT_INNER_RADIUS_MM: f32 = 3.2;
const DEFAULT_RING_WIDTH_MM: f32 = 1.152;

/// One marker cell of a target: lattice coordinate, board position, and
/// (for coded targets) the assigned codebook ID.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetCell {
    /// Lattice coordinate (axial `(q, r)` for hex, `(col, row)` for rect).
    pub coord: Coord,
    /// Cell center in board-frame millimeters.
    pub xy_mm: [f32; 2],
    /// Codebook ID for coded targets; `None` for plain targets.
    pub id: Option<usize>,
}

/// Whether a plain target carries origin fiducial dots.
///
/// Plain markers encode no identity, so a plain board is anchored either by an
/// origin-dot triad ([`Auto`](Self::Auto)) or by detecting the complete board
/// ([`None`](Self::None)). Selector for
/// [`TargetLayout::plain_rect`] / [`TargetLayout::plain_hex`]; coded targets
/// take no dots at all, so this type does not appear in their constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginDots {
    /// Auto-place a rotation-asymmetric dot triad in the lattice gaps near the
    /// board center (see [`OriginFiducials::auto`]). Detection resolves an
    /// absolute board frame.
    Auto,
    /// No dots. Labeling is only known up to the lattice symmetry, so results
    /// stay in a relative canonical frame and success is gated on the complete
    /// board being detected.
    None,
}

/// Compositional target layout: lattice × ring geometry × coding × fiducials.
///
/// This is the runtime description of a printed calibration target. It
/// composes four orthogonal aspects:
///
/// - [`LatticeGeometry`] — how marker cells are arranged (hex or rect grid),
/// - [`RingGeometry`] — the ring radii shared by every marker,
/// - [`MarkerCoding`] — whether markers carry a 16-sector identity code or
///   are plain annuli,
/// - [`OriginFiducials`] — optional filled dots that anchor origin and
///   orientation for targets whose markers don't encode identity.
///
/// Construct via [`TargetLayout::new`], a preset
/// ([`TargetLayout::default_hex`], [`TargetLayout::rect_24x24`]), or the
/// JSON loaders. All construction paths validate; geometry cannot be mutated
/// in place because the derived cell cache would silently desync.
#[derive(Debug, Clone)]
pub struct TargetLayout {
    name: String,
    lattice: LatticeGeometry,
    marker: RingGeometry,
    coding: MarkerCoding,
    fiducials: Option<OriginFiducials>,

    /// Derived cell cache in generation order.
    cells: Vec<TargetCell>,
    /// Derived origin-dot positions in board mm; empty when there are no
    /// fiducials. Cached alongside `cells` because detection projects these
    /// once per anchoring candidate.
    fiducial_dots_mm: Vec<[f32; 2]>,
    /// Fast lookup: codebook ID -> index into `cells` (empty for plain).
    id_to_idx: HashMap<usize, usize>,
    /// Fast lookup: lattice coordinate -> index into `cells`.
    coord_to_idx: HashMap<Coord, usize>,
}

impl TargetLayout {
    /// Construct and validate a target layout.
    pub fn new(
        name: impl Into<String>,
        lattice: LatticeGeometry,
        marker: RingGeometry,
        coding: MarkerCoding,
        fiducials: Option<OriginFiducials>,
    ) -> Result<Self, TargetValidationError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(TargetValidationError::EmptyName);
        }
        lattice.validate()?;
        marker.validate()?;

        let raw_cells = lattice.generate_cells()?;
        coding.validate(&marker, raw_cells.len())?;

        let min_center_spacing = lattice.min_center_spacing_mm();
        if marker.outer_radius_mm * 2.0 >= min_center_spacing {
            return Err(
                TargetValidationError::OuterDiameterExceedsMinCenterSpacing {
                    marker_outer_diameter_mm: marker.outer_radius_mm * 2.0,
                    min_center_spacing_mm: min_center_spacing,
                },
            );
        }
        let draw_diameter_mm = 2.0 * coding.outer_draw_radius_mm(&marker);
        if draw_diameter_mm >= min_center_spacing {
            return Err(
                TargetValidationError::MarkerDrawDiameterExceedsMinCenterSpacing {
                    marker_draw_diameter_mm: draw_diameter_mm,
                    min_center_spacing_mm: min_center_spacing,
                },
            );
        }

        let mut fiducial_dots_mm = Vec::new();
        if let Some(fiducials) = &fiducials {
            // The one excluded combination of the target matrix: coded markers
            // already anchor the board through their decoded IDs, so dots are
            // redundant. Enforced here so every construction path — `new`,
            // `with_auto_fiducials`, the JSON loaders, and the CLI recipe —
            // shares one rule.
            if coding.is_coded() {
                return Err(TargetValidationError::CodedWithFiducials);
            }
            let positions: Vec<[f32; 2]> = raw_cells.iter().map(|(_, xy)| *xy).collect();
            fiducials.validate(&lattice, &positions, coding.outer_draw_radius_mm(&marker))?;
            fiducial_dots_mm = origin_dot_positions_mm(&lattice)?;
        }

        let assignment = match &coding {
            MarkerCoding::Coded16(spec) => spec.id_assignment.as_deref(),
            MarkerCoding::Plain => None,
        };
        let cells: Vec<TargetCell> = raw_cells
            .into_iter()
            .enumerate()
            .map(|(idx, (coord, xy_mm))| TargetCell {
                coord,
                xy_mm,
                id: coding.is_coded().then(|| match assignment {
                    Some(ids) => ids[idx],
                    None => idx,
                }),
            })
            .collect();

        let id_to_idx = cells
            .iter()
            .enumerate()
            .filter_map(|(idx, cell)| cell.id.map(|id| (id, idx)))
            .collect();
        let coord_to_idx = cells
            .iter()
            .enumerate()
            .map(|(idx, cell)| (cell.coord, idx))
            .collect();

        Ok(Self {
            name,
            lattice,
            marker,
            coding,
            fiducials,
            cells,
            fiducial_dots_mm,
            id_to_idx,
            coord_to_idx,
        })
    }

    /// Construct and validate a target layout with automatically-placed origin
    /// fiducials.
    ///
    /// The generic escape hatch for custom lattice geometry: convenience over
    /// [`new`](Self::new) that computes an asymmetric dot triad in the lattice
    /// gaps near the board center via [`OriginFiducials::auto`], then
    /// constructs. The recipe's `fiducials = "auto"` option routes here.
    ///
    /// For the standard shapes prefer the scalar constructors
    /// [`plain_rect`](Self::plain_rect) / [`plain_hex`](Self::plain_hex), which
    /// take this same path without requiring the geometry types to be imported.
    ///
    /// # Errors
    ///
    /// Returns [`TargetValidationError::CodedWithFiducials`] if `coding` is
    /// coded — decoded IDs already anchor the board, so dots are redundant.
    pub fn with_auto_fiducials(
        name: impl Into<String>,
        lattice: LatticeGeometry,
        marker: RingGeometry,
        coding: MarkerCoding,
    ) -> Result<Self, TargetValidationError> {
        // Check before placing, so a tightly-packed coded target reports the
        // real problem (coded + dots) rather than a gap-fitting failure.
        if coding.is_coded() {
            return Err(TargetValidationError::CodedWithFiducials);
        }
        let fiducials = OriginFiducials::auto(&lattice, marker, &coding)?;
        Self::new(name, lattice, marker, coding, Some(fiducials))
    }

    /// Construct a 16-sector coded hex target from direct geometry arguments.
    ///
    /// Uses a deterministic geometry-derived name (the same scheme legacy v4
    /// boards used) so the layout round-trips through JSON without the caller
    /// supplying a name up front.
    pub fn coded_hex(
        pitch_mm: f32,
        rows: usize,
        long_row_cols: usize,
        outer_radius_mm: f32,
        inner_radius_mm: f32,
        ring_width_mm: f32,
    ) -> Result<Self, TargetValidationError> {
        Self::new(
            hex_generated_name(
                pitch_mm,
                rows,
                long_row_cols,
                outer_radius_mm,
                inner_radius_mm,
                ring_width_mm,
            ),
            LatticeGeometry::Hex(HexGeometry {
                rows,
                long_row_cols,
                pitch_mm,
            }),
            RingGeometry {
                outer_radius_mm,
                inner_radius_mm,
            },
            MarkerCoding::Coded16(CodedRingSpec {
                ring_width_mm,
                id_assignment: None,
            }),
            None,
        )
    }

    /// Construct a 16-sector coded rect target from direct geometry arguments.
    ///
    /// The rect counterpart of [`coded_hex`](Self::coded_hex). Coded targets
    /// never carry origin fiducials — decoded IDs anchor the board directly.
    ///
    /// ```
    /// # use ringgrid::TargetLayout;
    /// let target = TargetLayout::coded_rect(14.0, 20, 20, 4.8, 3.2, 1.152)?;
    /// assert_eq!(target.n_cells(), 400);
    /// # Ok::<(), ringgrid::TargetValidationError>(())
    /// ```
    pub fn coded_rect(
        pitch_mm: f32,
        rows: usize,
        cols: usize,
        outer_radius_mm: f32,
        inner_radius_mm: f32,
        ring_width_mm: f32,
    ) -> Result<Self, TargetValidationError> {
        let lattice = LatticeGeometry::Rect(RectGeometry {
            rows,
            cols,
            pitch_mm,
        });
        let marker = RingGeometry {
            outer_radius_mm,
            inner_radius_mm,
        };
        let coding = MarkerCoding::Coded16(CodedRingSpec {
            ring_width_mm,
            id_assignment: None,
        });
        Self::new(
            generated_name(&lattice, marker, &coding, None),
            lattice,
            marker,
            coding,
            None,
        )
    }

    /// Construct a plain (uncoded) hex target from direct geometry arguments.
    ///
    /// `dots` selects how the board is anchored: [`OriginDots::Auto`] places a
    /// rotation-asymmetric dot triad in the lattice gaps near the board center
    /// (absolute board frame), [`OriginDots::None`] omits them (the board is
    /// then labeled only up to lattice symmetry — see
    /// [`DetectionResult::board_complete`](crate::DetectionResult)).
    ///
    /// ```
    /// # use ringgrid::{OriginDots, TargetLayout};
    /// let target = TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::Auto)?;
    /// assert_eq!(target.fiducial_dots_mm().len(), 3);
    /// # Ok::<(), ringgrid::TargetValidationError>(())
    /// ```
    pub fn plain_hex(
        pitch_mm: f32,
        rows: usize,
        long_row_cols: usize,
        outer_radius_mm: f32,
        inner_radius_mm: f32,
        dots: OriginDots,
    ) -> Result<Self, TargetValidationError> {
        Self::plain(
            LatticeGeometry::Hex(HexGeometry {
                rows,
                long_row_cols,
                pitch_mm,
            }),
            RingGeometry {
                outer_radius_mm,
                inner_radius_mm,
            },
            dots,
        )
    }

    /// Construct a plain (uncoded) rect target from direct geometry arguments.
    ///
    /// The rect counterpart of [`plain_hex`](Self::plain_hex); see there for
    /// how `dots` anchors the board.
    ///
    /// ```
    /// # use ringgrid::{OriginDots, TargetLayout};
    /// let target = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)?;
    /// assert_eq!(target.n_cells(), 576);
    /// assert!(target.fiducials().is_some());
    /// # Ok::<(), ringgrid::TargetValidationError>(())
    /// ```
    pub fn plain_rect(
        pitch_mm: f32,
        rows: usize,
        cols: usize,
        outer_radius_mm: f32,
        inner_radius_mm: f32,
        dots: OriginDots,
    ) -> Result<Self, TargetValidationError> {
        Self::plain(
            LatticeGeometry::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm,
            }),
            RingGeometry {
                outer_radius_mm,
                inner_radius_mm,
            },
            dots,
        )
    }

    /// Shared body of [`plain_hex`](Self::plain_hex) / [`plain_rect`](Self::plain_rect):
    /// resolve `dots` to fiducials and construct under a derived name.
    fn plain(
        lattice: LatticeGeometry,
        marker: RingGeometry,
        dots: OriginDots,
    ) -> Result<Self, TargetValidationError> {
        let coding = MarkerCoding::Plain;
        let fiducials = match dots {
            OriginDots::Auto => Some(OriginFiducials::auto(&lattice, marker, &coding)?),
            OriginDots::None => None,
        };
        Self::new(
            generated_name(&lattice, marker, &coding, Some(dots)),
            lattice,
            marker,
            coding,
            fiducials,
        )
    }

    /// Rename this layout, re-validating the new name.
    ///
    /// The scalar constructors ([`coded_hex`](Self::coded_hex),
    /// [`plain_rect`](Self::plain_rect), …) derive a deterministic
    /// geometry-based name so identical geometry always yields an identical
    /// spec. Use this to attach a human-readable one without falling back to
    /// [`new`](Self::new).
    ///
    /// ```
    /// # use ringgrid::{OriginDots, TargetLayout};
    /// let target = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)?
    ///     .with_name("lab_bench_rect")?;
    /// assert_eq!(target.name(), "lab_bench_rect");
    /// # Ok::<(), ringgrid::TargetValidationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TargetValidationError::EmptyName`] if `name` is blank.
    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self, TargetValidationError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(TargetValidationError::EmptyName);
        }
        self.name = name;
        Ok(self)
    }

    /// The classic ringgrid target: 15-row hex lattice of 16-sector coded
    /// rings at 8 mm pitch (203 markers on a 200 mm board).
    pub fn default_hex() -> Self {
        Self::new(
            DEFAULT_HEX_NAME,
            LatticeGeometry::Hex(HexGeometry {
                rows: DEFAULT_HEX_ROWS,
                long_row_cols: DEFAULT_HEX_LONG_ROW_COLS,
                pitch_mm: DEFAULT_HEX_PITCH_MM,
            }),
            RingGeometry {
                outer_radius_mm: DEFAULT_OUTER_RADIUS_MM,
                inner_radius_mm: DEFAULT_INNER_RADIUS_MM,
            },
            MarkerCoding::Coded16(CodedRingSpec {
                ring_width_mm: DEFAULT_RING_WIDTH_MM,
                id_assignment: None,
            }),
            None,
        )
        .expect("default hex target spec must be valid")
    }

    /// A 24×24 plain rect target: a square lattice of plain rings at 14 mm
    /// pitch (outer Ø11.2 / inner Ø5.6 mm) with an origin-dot triad in the cell
    /// gaps near the board center.
    ///
    /// The dot geometry is **frozen**: physical boards printed from this named
    /// preset in earlier releases depend on these exact positions to anchor.
    /// Its dot geometry — Ø2.8 mm dots at `[161, 161]`, `[147, 161]` and
    /// `[161, 175]` mm — is what physical boards printed from this preset
    /// carry. Automatic placement reproduces it exactly (locked by
    /// `rect_24x24_derived_dots_match_the_printed_board`), so the preset is
    /// simply `plain_rect` with a stable name.
    pub fn rect_24x24() -> Self {
        Self::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)
            .and_then(|target| target.with_name("rect_24x24"))
            .expect("rect_24x24 preset must be valid")
    }

    /// Human-readable name of the target layout.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Lattice arrangement of marker cells.
    pub fn lattice(&self) -> &LatticeGeometry {
        &self.lattice
    }

    /// The projective-grid lattice kind matching this target.
    pub fn lattice_kind(&self) -> LatticeKind {
        self.lattice.kind()
    }

    /// Ring radii shared by every marker.
    pub fn ring(&self) -> RingGeometry {
        self.marker
    }

    /// Marker coding style.
    pub fn coding(&self) -> &MarkerCoding {
        &self.coding
    }

    /// Origin-dot centers in board-frame millimeters; empty when the target has
    /// no fiducials.
    ///
    /// Derived from the lattice at construction time and cached, so positions
    /// always track the geometry instead of being stored beside it. See
    /// [`OriginFiducials`] for the placement rule.
    ///
    /// ```
    /// # use ringgrid::TargetLayout;
    /// let dots = TargetLayout::rect_24x24().fiducial_dots_mm().to_vec();
    /// assert_eq!(dots, vec![[161.0, 161.0], [147.0, 161.0], [161.0, 175.0]]);
    /// ```
    pub fn fiducial_dots_mm(&self) -> &[[f32; 2]] {
        &self.fiducial_dots_mm
    }

    /// Origin fiducials, when the target defines them.
    pub fn fiducials(&self) -> Option<&OriginFiducials> {
        self.fiducials.as_ref()
    }

    /// Whether markers encode a decodable identity.
    pub fn is_coded(&self) -> bool {
        self.coding.is_coded()
    }

    /// Lattice pitch in millimeters.
    pub fn pitch_mm(&self) -> f32 {
        self.lattice.pitch_mm()
    }

    /// Minimum center-to-center spacing between adjacent markers (mm).
    pub fn min_center_spacing_mm(&self) -> f32 {
        self.lattice.min_center_spacing_mm()
    }

    /// Outermost drawn radius of a marker (mm), including ring stroke.
    pub fn outer_draw_radius_mm(&self) -> f32 {
        self.coding.outer_draw_radius_mm(&self.marker)
    }

    /// Code band bounds `(inner_mm, outer_mm)` for coded targets.
    pub fn code_band_bounds_mm(&self) -> Option<(f32, f32)> {
        self.coding.code_band_bounds_mm(&self.marker)
    }

    /// All marker cells in generation order.
    pub fn cells(&self) -> &[TargetCell] {
        &self.cells
    }

    /// Total number of marker cells.
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Cell center in board millimeters for a lattice coordinate.
    pub fn cell_xy_mm(&self, coord: Coord) -> Option<[f32; 2]> {
        self.coord_to_idx.get(&coord).map(|&i| self.cells[i].xy_mm)
    }

    /// Codebook ID assigned to a lattice coordinate (coded targets only).
    pub fn id_of(&self, coord: Coord) -> Option<usize> {
        self.coord_to_idx
            .get(&coord)
            .and_then(|&i| self.cells[i].id)
    }

    /// Lattice coordinate of a codebook ID (coded targets only).
    pub fn coord_of_id(&self, id: usize) -> Option<Coord> {
        self.id_to_idx.get(&id).map(|&i| self.cells[i].coord)
    }

    /// Cell center in board millimeters for a codebook ID (coded targets only).
    pub fn xy_mm_of_id(&self, id: usize) -> Option<[f32; 2]> {
        self.id_to_idx.get(&id).map(|&i| self.cells[i].xy_mm)
    }

    /// Cell carrying a codebook ID (coded targets only).
    pub fn cell_of_id(&self, id: usize) -> Option<&TargetCell> {
        self.id_to_idx.get(&id).map(|&i| &self.cells[i])
    }

    /// Iterator over all codebook IDs present on the target.
    pub fn marker_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.cells.iter().filter_map(|cell| cell.id)
    }

    /// Maximum codebook ID present on the target (0 for plain targets).
    pub fn max_marker_id(&self) -> usize {
        self.marker_ids().max().unwrap_or(0)
    }

    /// Axis-aligned cell-center bounds in board mm: `(min_xy, max_xy)`.
    pub fn marker_bounds_mm(&self) -> Option<([f32; 2], [f32; 2])> {
        let first = self.cells.first()?;
        let mut min_xy = first.xy_mm;
        let mut max_xy = first.xy_mm;
        for cell in &self.cells[1..] {
            min_xy[0] = min_xy[0].min(cell.xy_mm[0]);
            min_xy[1] = min_xy[1].min(cell.xy_mm[1]);
            max_xy[0] = max_xy[0].max(cell.xy_mm[0]);
            max_xy[1] = max_xy[1].max(cell.xy_mm[1]);
        }
        Some((min_xy, max_xy))
    }

    /// Axis-aligned cell-center span in board mm (`[width, height]`).
    pub fn marker_span_mm(&self) -> Option<[f32; 2]> {
        self.marker_bounds_mm()
            .map(|(min_xy, max_xy)| [max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]])
    }
}

impl Default for TargetLayout {
    fn default() -> Self {
        Self::default_hex()
    }
}

/// Deterministic geometry-derived name for coded hex targets, matching the
/// scheme legacy v4 boards used so v4 and v5 flows produce identical names for
/// identical geometry.
pub(crate) fn hex_generated_name(
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    outer_radius_mm: f32,
    inner_radius_mm: f32,
    ring_width_mm: f32,
) -> String {
    format!(
        "ringgrid_hex_r{rows}_c{long_row_cols}_p{pitch_mm:.3}_o{outer_radius_mm:.3}_i{inner_radius_mm:.3}_w{ring_width_mm:.3}"
    )
}

/// Deterministic geometry-derived name for the scalar constructors added in
/// 0.11 (`coded_rect`, `plain_hex`, `plain_rect`).
///
/// Kept separate from [`hex_generated_name`], which must stay bit-identical
/// because it reproduces the legacy v4 naming scheme that `coded_hex` boards
/// round-trip through.
fn generated_name(
    lattice: &LatticeGeometry,
    marker: RingGeometry,
    coding: &MarkerCoding,
    dots: Option<OriginDots>,
) -> String {
    let (kind, a, b) = match lattice {
        LatticeGeometry::Hex(hex) => ("hex", hex.rows, hex.long_row_cols),
        LatticeGeometry::Rect(rect) => ("rect", rect.rows, rect.cols),
    };
    let pitch_mm = lattice.pitch_mm();
    let RingGeometry {
        outer_radius_mm,
        inner_radius_mm,
    } = marker;
    let mut name = format!(
        "ringgrid_{kind}_r{a}_c{b}_p{pitch_mm:.3}_o{outer_radius_mm:.3}_i{inner_radius_mm:.3}"
    );
    match coding {
        MarkerCoding::Coded16(spec) => {
            let ring_width_mm = spec.ring_width_mm;
            name.push_str(&format!("_w{ring_width_mm:.3}_coded"));
        }
        MarkerCoding::Plain => name.push_str("_plain"),
    }
    if let Some(OriginDots::Auto) = dots {
        name.push_str("_dots");
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_lattice(rows: usize, long_row_cols: usize, pitch_mm: f32) -> LatticeGeometry {
        LatticeGeometry::Hex(HexGeometry {
            rows,
            long_row_cols,
            pitch_mm,
        })
    }

    fn coded(ring_width_mm: f32) -> MarkerCoding {
        MarkerCoding::Coded16(CodedRingSpec {
            ring_width_mm,
            id_assignment: None,
        })
    }

    /// The scalar constructors cover the whole target matrix. Locks the shape
    /// each one produces so the docs table stays a truthful API index.
    #[test]
    fn scalar_constructors_cover_the_target_matrix() {
        let coded_hex = TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).expect("coded hex");
        assert!(coded_hex.is_coded());
        assert!(coded_hex.fiducials().is_none());
        assert_eq!(coded_hex.n_cells(), 203);

        let coded_rect =
            TargetLayout::coded_rect(14.0, 20, 20, 4.8, 3.2, 1.152).expect("coded rect");
        assert!(coded_rect.is_coded());
        assert!(coded_rect.fiducials().is_none());
        assert_eq!(coded_rect.n_cells(), 400);

        for dots in [OriginDots::Auto, OriginDots::None] {
            let hex = TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, dots).expect("plain hex");
            let rect = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, dots).expect("plain rect");
            for target in [&hex, &rect] {
                assert!(!target.is_coded());
                assert_eq!(target.fiducials().is_some(), dots == OriginDots::Auto);
            }
            assert_eq!(hex.n_cells(), 203);
            assert_eq!(rect.n_cells(), 576);
        }
    }

    /// `OriginDots::Auto` must yield dots that survive the same validation the
    /// detector's anchoring relies on — three of them, clear of every marker
    /// and breaking every lattice rotation.
    #[test]
    fn auto_dots_are_a_valid_anchoring_triad() {
        let target = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)
            .expect("plain rect with dots");
        let fiducials = target.fiducials().expect("auto placed dots");
        assert_eq!(target.fiducial_dots_mm().len(), 3);

        let cells: Vec<[f32; 2]> = target.cells().iter().map(|cell| cell.xy_mm).collect();
        fiducials
            .validate(target.lattice(), &cells, target.outer_draw_radius_mm())
            .expect("auto dots must pass fiducial validation");
    }

    /// Coded markers already anchor the board through decoded IDs, so dots are
    /// the one excluded combination — rejected on *every* construction path.
    #[test]
    fn coded_targets_reject_fiducials_on_every_path() {
        let lattice = hex_lattice(5, 5, 8.0);
        let marker = RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 3.2,
        };

        assert!(matches!(
            TargetLayout::with_auto_fiducials("coded_dots", lattice, marker, coded(1.152)),
            Err(TargetValidationError::CodedWithFiducials)
        ));

        assert!(matches!(
            TargetLayout::new(
                "coded_dots",
                lattice,
                marker,
                coded(1.152),
                Some(OriginFiducials { dot_radius_mm: 0.8 }),
            ),
            Err(TargetValidationError::CodedWithFiducials)
        ));
    }

    /// `coded_hex` reproduces the legacy v4 naming scheme; boards round-trip
    /// through that name, so it must never drift.
    #[test]
    fn generated_names_are_stable() {
        assert_eq!(TargetLayout::default_hex().name(), "ringgrid_200mm_hex");
        assert_eq!(
            TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152)
                .expect("coded hex")
                .name(),
            "ringgrid_hex_r15_c14_p8.000_o4.800_i3.200_w1.152"
        );
        // The 0.11 constructors use their own scheme, tagged by coding and dots
        // so distinct matrix rows never collide on a name.
        assert_eq!(
            TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)
                .expect("plain rect")
                .name(),
            "ringgrid_rect_r24_c24_p14.000_o5.600_i2.800_plain_dots"
        );
        assert_eq!(
            TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::None)
                .expect("plain rect")
                .name(),
            "ringgrid_rect_r24_c24_p14.000_o5.600_i2.800_plain"
        );
    }

    #[test]
    fn with_name_overrides_the_derived_name() {
        let target = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)
            .expect("plain rect")
            .with_name("lab_bench_rect")
            .expect("non-empty name");
        assert_eq!(target.name(), "lab_bench_rect");
        // Renaming must not disturb the derived geometry.
        assert_eq!(target.n_cells(), 576);
        assert!(target.fiducials().is_some());

        assert!(matches!(
            target.with_name("  "),
            Err(TargetValidationError::EmptyName)
        ));
    }

    #[test]
    fn rect_preset_has_expected_shape() {
        let target = TargetLayout::rect_24x24();
        assert_eq!(target.n_cells(), 576);
        assert!(!target.is_coded());
        assert_eq!(target.pitch_mm(), 14.0);
        assert_eq!(target.min_center_spacing_mm(), 14.0);
        assert_eq!(target.outer_draw_radius_mm(), 5.6);
        assert_eq!(target.code_band_bounds_mm(), None);

        // 23 pitches of 14 mm per side.
        let span = target.marker_span_mm().expect("span");
        assert_eq!(span, [322.0, 322.0]);

        // Coordinates are centered: 24 cells run -11..=12 per axis, so the
        // corners are (-11,-11) and (12,12) and cell (0,0) sits mid-board.
        assert_eq!(
            target.cell_xy_mm(projective_grid::Coord::new(-11, -11)),
            Some([0.0, 0.0])
        );
        assert_eq!(
            target.cell_xy_mm(projective_grid::Coord::new(12, 12)),
            Some([322.0, 322.0])
        );
        assert_eq!(
            target.cell_xy_mm(projective_grid::Coord::new(0, 0)),
            Some([154.0, 154.0])
        );

        let fiducials = target.fiducials().expect("preset has dots");
        assert_eq!(target.fiducial_dots_mm().len(), 3);
        assert_eq!(fiducials.dot_radius_mm, 1.4);

        // Plain targets expose no IDs.
        assert_eq!(target.marker_ids().count(), 0);
        assert_eq!(target.id_of(projective_grid::Coord::new(0, 0)), None);
    }

    #[test]
    fn coded_ids_are_sequential_without_assignment() {
        let target = TargetLayout::new(
            "t",
            hex_lattice(3, 4, 8.0),
            RingGeometry {
                outer_radius_mm: 4.8,
                inner_radius_mm: 3.2,
            },
            coded(1.152),
            None,
        )
        .expect("valid");
        for (idx, cell) in target.cells().iter().enumerate() {
            assert_eq!(cell.id, Some(idx));
        }
        assert_eq!(target.coord_of_id(0), Some(target.cells()[0].coord));
        assert_eq!(target.xy_mm_of_id(0), Some([0.0, 0.0]));
    }

    #[test]
    fn rejects_codebook_capacity_overflow() {
        // 40×40 rect = 1600 cells > 893 codewords.
        let err = TargetLayout::new(
            "t",
            LatticeGeometry::Rect(RectGeometry {
                rows: 40,
                cols: 40,
                pitch_mm: 14.0,
            }),
            RingGeometry {
                outer_radius_mm: 5.6,
                inner_radius_mm: 2.8,
            },
            coded(0.8),
            None,
        )
        .expect_err("capacity exceeded");
        assert!(matches!(
            err,
            TargetValidationError::CodebookCapacityExceeded { n_cells: 1600, .. }
        ));

        // The same board as plain is fine: no identity coding needed.
        TargetLayout::new(
            "t",
            LatticeGeometry::Rect(RectGeometry {
                rows: 40,
                cols: 40,
                pitch_mm: 14.0,
            }),
            RingGeometry {
                outer_radius_mm: 5.6,
                inner_radius_mm: 2.8,
            },
            MarkerCoding::Plain,
            None,
        )
        .expect("plain has no capacity limit");
    }

    /// With positions derived from the lattice, the remaining way to specify a
    /// bad dot is its size: too large and it collides with the rings it is
    /// meant to sit between.
    #[test]
    fn rejects_dot_radius_that_overlaps_a_marker() {
        let make = |dot_radius_mm: f32| {
            TargetLayout::new(
                "t",
                LatticeGeometry::Rect(RectGeometry {
                    rows: 4,
                    cols: 4,
                    pitch_mm: 14.0,
                }),
                RingGeometry {
                    outer_radius_mm: 5.6,
                    inner_radius_mm: 2.8,
                },
                MarkerCoding::Plain,
                Some(OriginFiducials { dot_radius_mm }),
            )
        };
        // Gap clearance is pitch/sqrt(2) ~= 9.9 mm and rings draw out to 5.6 mm,
        // so anything past ~4.3 mm touches a marker.
        assert!(matches!(
            make(5.0),
            Err(TargetValidationError::DotOverlapsMarker { .. })
        ));
        assert!(matches!(
            make(0.0),
            Err(TargetValidationError::InvalidDotRadius { .. })
        ));
        assert!(matches!(
            make(f32::NAN),
            Err(TargetValidationError::InvalidDotRadius { .. })
        ));
        make(1.4).expect("a legible dot fits the gap");
    }

    /// Boards too small to hold the triad inside the marker field are rejected
    /// rather than placing dots the anchoring homography would extrapolate to.
    #[test]
    fn rejects_boards_too_small_for_origin_dots() {
        assert!(matches!(
            TargetLayout::plain_rect(14.0, 3, 3, 5.6, 2.8, OriginDots::Auto),
            Err(TargetValidationError::OriginDotsOutsideBoard)
        ));
        // Without dots the same board is fine — it just cannot be anchored.
        TargetLayout::plain_rect(14.0, 3, 3, 5.6, 2.8, OriginDots::None)
            .expect("small board is valid without dots");
    }

    #[test]
    fn validation_rejects_degenerate_geometry() {
        let ring = |outer, inner| RingGeometry {
            outer_radius_mm: outer,
            inner_radius_mm: inner,
        };
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(0, 4, 8.0),
                ring(4.8, 3.2),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::InvalidRows { rows: 0 })
        ));
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(3, 1, 8.0),
                ring(4.8, 3.2),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::InvalidLongRowColsForRows {
                rows: 3,
                long_row_cols: 1,
            })
        ));
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(3, 4, 8.0),
                ring(4.8, 4.8),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::InnerRadiusNotSmallerThanOuter { .. })
        ));
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(3, 4, 8.0),
                ring(4.8, 4.1),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::NonPositiveCodeBandGap { .. })
        ));
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(3, 4, f32::NAN),
                ring(4.8, 3.2),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::InvalidPitch { .. })
        ));
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(3, 4, 5.0),
                ring(4.0, 2.0),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::MarkerDrawDiameterExceedsMinCenterSpacing { .. })
        ));
        assert!(matches!(
            TargetLayout::new(
                "t",
                hex_lattice(3, 4, 8.0),
                ring(4.8, 3.2),
                coded(0.0),
                None
            ),
            Err(TargetValidationError::InvalidRingWidth { .. })
        ));
        assert!(matches!(
            TargetLayout::new(
                "",
                hex_lattice(3, 4, 8.0),
                ring(4.8, 3.2),
                coded(1.152),
                None
            ),
            Err(TargetValidationError::EmptyName)
        ));
    }
}
