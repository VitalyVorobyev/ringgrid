//! Friendly TOML/JSON target *recipe*: the high-level authoring schema the
//! `ringgrid` CLI lowers into a canonical [`TargetLayout`].
//!
//! The recipe owns *authoring* defaults (dpi, formats, `dots: auto`); the
//! library owns *geometry* defaults and validation. A recipe is the source, the
//! v5 `target_spec.json` is the emitted canonical form.

use serde::{Deserialize, Serialize};

use crate::target::{
    CodedRingSpec, HexGeometry, LatticeGeometry, MarkerCoding, OriginFiducials, RectGeometry,
    RingGeometry, TargetLayout, TargetValidationError,
};

/// A hand-authored target description, lowered to a [`TargetLayout`] via
/// [`TargetRecipe::to_target`]. CLI flags override fields before lowering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRecipe {
    /// Human-readable target name.
    pub name: String,
    /// Lattice arrangement.
    pub lattice: LatticeRecipe,
    /// Ring geometry shared by every marker.
    pub marker: MarkerRecipe,
    /// Marker coding style (default: plain).
    #[serde(default)]
    pub coding: CodingRecipe,
    /// Origin fiducial dots (default: none).
    #[serde(default)]
    pub fiducials: FiducialsRecipe,
    /// Rendering options.
    #[serde(default)]
    pub render: RenderRecipe,
}

/// Lattice arrangement in a recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LatticeRecipe {
    /// Hexagonal lattice.
    Hex {
        /// Number of rows.
        rows: usize,
        /// Columns in the longest rows.
        long_row_cols: usize,
        /// Axial pitch (mm).
        pitch_mm: f32,
    },
    /// Rectangular (square) lattice.
    Rect {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
        /// Center-to-center pitch (mm).
        pitch_mm: f32,
    },
}

/// Ring geometry in a recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkerRecipe {
    /// Outer ring radius (mm).
    pub outer_radius_mm: f32,
    /// Inner ring radius (mm).
    pub inner_radius_mm: f32,
    /// Ring stroke width (mm). Required for coded markers, ignored for plain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ring_width_mm: Option<f32>,
}

/// Coding style: coded 16-sector rings, or plain annuli.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingRecipe {
    /// 16-sector coded rings.
    Coded,
    /// Plain (uncoded) annuli.
    #[default]
    Plain,
}

/// Origin-dot placement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FiducialMode {
    /// No origin dots.
    None,
    /// Auto-place a valid dot triad in the gaps near the board center.
    Auto,
}

/// Origin fiducials in a recipe: `"none"`, `"auto"`, or an explicit dot table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FiducialsRecipe {
    /// `"none"` or `"auto"`.
    Mode(FiducialMode),
    /// Explicit `{ dot_radius_mm, dots_mm }`.
    Explicit(OriginFiducials),
}

impl Default for FiducialsRecipe {
    fn default() -> Self {
        Self::Mode(FiducialMode::None)
    }
}

/// Rendering options for the emitted artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderRecipe {
    /// PNG raster resolution (dots per inch).
    #[serde(default = "default_dpi")]
    pub dpi: f32,
    /// White margin around the target (mm).
    #[serde(default)]
    pub margin_mm: f32,
    /// Draw a printed scale bar.
    #[serde(default = "default_true")]
    pub scale_bar: bool,
    /// Artifact formats to emit.
    #[serde(default = "default_formats")]
    pub formats: Vec<Format>,
}

impl Default for RenderRecipe {
    fn default() -> Self {
        Self {
            dpi: default_dpi(),
            margin_mm: 0.0,
            scale_bar: true,
            formats: default_formats(),
        }
    }
}

fn default_dpi() -> f32 {
    300.0
}
fn default_true() -> bool {
    true
}
fn default_formats() -> Vec<Format> {
    vec![Format::Json, Format::Svg, Format::Png, Format::Dxf]
}

/// An output artifact format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// Canonical v5 `target_spec.json`.
    Json,
    /// Printable SVG.
    Svg,
    /// Printable PNG raster.
    Png,
    /// Fabrication DXF (2D CAD, millimeters).
    Dxf,
}

/// Failure lowering a [`TargetRecipe`] to a [`TargetLayout`].
#[derive(Debug)]
pub enum RecipeError {
    /// Coded markers require `marker.ring_width_mm`.
    MissingRingWidth,
    /// Coded markers cannot use origin dots (redundant with decoded IDs); this
    /// is the one excluded combination of the target matrix.
    CodedWithFiducials,
    /// The lowered geometry failed target validation.
    Validation(TargetValidationError),
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRingWidth => {
                f.write_str("coded markers require marker.ring_width_mm in the recipe")
            }
            Self::CodedWithFiducials => f.write_str(
                "coded markers cannot use origin dots — set coding = \"plain\" or fiducials = \"none\"",
            ),
            Self::Validation(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RecipeError {}

impl From<TargetValidationError> for RecipeError {
    fn from(value: TargetValidationError) -> Self {
        Self::Validation(value)
    }
}

impl TargetRecipe {
    /// Parse a recipe from TOML text.
    pub fn parse_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Parse a recipe from JSON text.
    pub fn parse_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Lower this recipe into a validated [`TargetLayout`].
    pub fn to_target(&self) -> Result<TargetLayout, RecipeError> {
        let lattice = match self.lattice {
            LatticeRecipe::Hex {
                rows,
                long_row_cols,
                pitch_mm,
            } => LatticeGeometry::Hex(HexGeometry {
                rows,
                long_row_cols,
                pitch_mm,
            }),
            LatticeRecipe::Rect {
                rows,
                cols,
                pitch_mm,
            } => LatticeGeometry::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm,
            }),
        };
        let marker = RingGeometry {
            outer_radius_mm: self.marker.outer_radius_mm,
            inner_radius_mm: self.marker.inner_radius_mm,
        };
        let coded = self.coding == CodingRecipe::Coded;
        let coding = if coded {
            let ring_width_mm = self
                .marker
                .ring_width_mm
                .ok_or(RecipeError::MissingRingWidth)?;
            MarkerCoding::Coded16(CodedRingSpec {
                ring_width_mm,
                id_assignment: None,
            })
        } else {
            MarkerCoding::Plain
        };

        match &self.fiducials {
            FiducialsRecipe::Mode(FiducialMode::None) => Ok(TargetLayout::new(
                &self.name, lattice, marker, coding, None,
            )?),
            FiducialsRecipe::Mode(FiducialMode::Auto) => {
                if coded {
                    return Err(RecipeError::CodedWithFiducials);
                }
                Ok(TargetLayout::with_auto_fiducials(
                    &self.name, lattice, marker, coding,
                )?)
            }
            FiducialsRecipe::Explicit(dots) => {
                if coded {
                    return Err(RecipeError::CodedWithFiducials);
                }
                Ok(TargetLayout::new(
                    &self.name,
                    lattice,
                    marker,
                    coding,
                    Some(dots.clone()),
                )?)
            }
        }
    }

    /// Override the lattice pitch (mm) in place (CLI `--pitch-mm`).
    pub fn set_pitch_mm(&mut self, pitch_mm: f32) {
        match &mut self.lattice {
            LatticeRecipe::Hex { pitch_mm: p, .. } | LatticeRecipe::Rect { pitch_mm: p, .. } => {
                *p = pitch_mm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_lowers_every_matrix_combo() {
        for (name, text) in super::super::EXAMPLE_RECIPES {
            let recipe =
                TargetRecipe::parse_toml(text).unwrap_or_else(|e| panic!("{name} parse: {e}"));
            recipe
                .to_target()
                .unwrap_or_else(|e| panic!("{name} lower: {e}"));
        }
    }

    #[test]
    fn rect_plain_dots_matches_preset_shape() {
        let recipe = TargetRecipe::parse_toml(
            super::super::example_recipe("rect_plain_dots").expect("example present"),
        )
        .expect("parse");
        let target = recipe.to_target().expect("lower");
        assert!(!target.is_coded());
        assert!(target.fiducials().is_some());
        assert_eq!(target.n_cells(), 24 * 24);
    }

    #[test]
    fn coded_with_auto_dots_is_rejected() {
        let text = r#"
            name = "bad"
            coding = "coded"
            fiducials = "auto"
            [lattice]
            kind = "hex"
            rows = 5
            long_row_cols = 5
            pitch_mm = 8.0
            [marker]
            outer_radius_mm = 4.8
            inner_radius_mm = 3.2
            ring_width_mm = 1.152
        "#;
        let recipe = TargetRecipe::parse_toml(text).expect("parse");
        assert!(matches!(
            recipe.to_target(),
            Err(RecipeError::CodedWithFiducials)
        ));
    }

    #[test]
    fn coded_without_ring_width_is_rejected() {
        let text = r#"
            name = "bad"
            coding = "coded"
            [lattice]
            kind = "rect"
            rows = 5
            cols = 5
            pitch_mm = 14.0
            [marker]
            outer_radius_mm = 4.8
            inner_radius_mm = 3.2
        "#;
        let recipe = TargetRecipe::parse_toml(text).expect("parse");
        assert!(matches!(
            recipe.to_target(),
            Err(RecipeError::MissingRingWidth)
        ));
    }
}
