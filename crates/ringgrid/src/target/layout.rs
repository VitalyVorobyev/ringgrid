//! `TargetLayout`: the compositional target model.

use std::collections::HashMap;

use projective_grid::{Coord, LatticeKind};

use super::error::TargetValidationError;
use super::fiducials::OriginFiducials;
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

        if let Some(fiducials) = &fiducials {
            let positions: Vec<[f32; 2]> = raw_cells.iter().map(|(_, xy)| *xy).collect();
            fiducials.validate(
                lattice.kind(),
                &positions,
                coding.outer_draw_radius_mm(&marker),
            )?;
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
            id_to_idx,
            coord_to_idx,
        })
    }

    /// Construct and validate a target layout with automatically-placed origin
    /// fiducials.
    ///
    /// Convenience over [`new`](Self::new): computes an asymmetric dot triad in
    /// the lattice gaps near the origin corner via [`OriginFiducials::auto`],
    /// then constructs. Works for any lattice/coding; the `dots: auto` config
    /// option routes here. (Origin dots are only meaningful for plain targets —
    /// coded markers already encode identity — but this method does not forbid
    /// coded targets: the dots are simply inert during coded detection.)
    pub fn with_auto_fiducials(
        name: impl Into<String>,
        lattice: LatticeGeometry,
        marker: RingGeometry,
        coding: MarkerCoding,
    ) -> Result<Self, TargetValidationError> {
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
    /// pitch (outer Ø11.2 / inner Ø5.6 mm) with an auto-placed origin-dot triad
    /// in the cell gaps at the origin corner.
    pub fn rect_24x24() -> Self {
        Self::with_auto_fiducials(
            "rect_24x24",
            LatticeGeometry::Rect(RectGeometry {
                rows: 24,
                cols: 24,
                pitch_mm: 14.0,
            }),
            RingGeometry {
                outer_radius_mm: 5.6,
                inner_radius_mm: 2.8,
            },
            MarkerCoding::Plain,
        )
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

        // Corner cells at lattice coordinates.
        assert_eq!(
            target.cell_xy_mm(projective_grid::Coord::new(0, 0)),
            Some([0.0, 0.0])
        );
        assert_eq!(
            target.cell_xy_mm(projective_grid::Coord::new(23, 23)),
            Some([322.0, 322.0])
        );

        let fiducials = target.fiducials().expect("preset has dots");
        assert_eq!(fiducials.dots_mm.len(), 3);
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

    #[test]
    fn rejects_rotationally_symmetric_fiducials() {
        // A single dot at the exact lattice center is invariant under every
        // rotation.
        let err = TargetLayout::new(
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
            Some(OriginFiducials {
                dot_radius_mm: 1.4,
                dots_mm: vec![[21.0, 21.0]],
            }),
        )
        .expect_err("symmetric dots");
        assert!(matches!(
            err,
            TargetValidationError::FiducialsRotationallySymmetric { .. }
        ));

        // Adding one off-center dot breaks all rotations.
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
            Some(OriginFiducials {
                dot_radius_mm: 1.4,
                dots_mm: vec![[21.0, 21.0], [7.0, 21.0]],
            }),
        )
        .expect("asymmetric dots are valid");
    }

    #[test]
    fn rejects_dot_overlapping_marker() {
        let err = TargetLayout::new(
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
            Some(OriginFiducials {
                dot_radius_mm: 1.4,
                // Directly on a ring center.
                dots_mm: vec![[14.0, 14.0]],
            }),
        )
        .expect_err("dot on marker");
        assert!(matches!(
            err,
            TargetValidationError::DotOverlapsMarker { index: 0 }
        ));
    }

    #[test]
    fn rejects_empty_fiducial_dots_and_bad_radius() {
        let make = |dot_radius_mm: f32, dots_mm: Vec<[f32; 2]>| {
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
                Some(OriginFiducials {
                    dot_radius_mm,
                    dots_mm,
                }),
            )
        };
        assert!(matches!(
            make(1.4, vec![]),
            Err(TargetValidationError::EmptyFiducialDots)
        ));
        assert!(matches!(
            make(0.0, vec![[21.0, 21.0]]),
            Err(TargetValidationError::InvalidDotRadius { .. })
        ));
        assert!(matches!(
            make(1.4, vec![[f32::NAN, 21.0]]),
            Err(TargetValidationError::NonFiniteDot { index: 0 })
        ));
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
