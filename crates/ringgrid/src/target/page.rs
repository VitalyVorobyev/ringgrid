//! Printed-page description: paper size, orientation, and margin.
//!
//! Rendering a target answers two independent questions: what the target looks
//! like (geometry, owned by [`TargetLayout`](crate::TargetLayout)) and what
//! sheet it is printed on. This module owns the second one.
//!
//! The JSON shape (`{ "size": { "kind": "a4" }, "orientation": "portrait",
//! "margin_mm": 10.0 }`) deliberately matches the `calib-targets` page model so
//! an application driving both target families handles one contract.

use serde::{Deserialize, Serialize};

/// Millimeters per inch, for imperial paper sizes.
const MM_PER_INCH: f64 = 25.4;

/// Page orientation applied on top of a [`PageSize`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageOrientation {
    /// Tall: the page's longer side runs vertically.
    #[default]
    Portrait,
    /// Wide: the page's longer side runs horizontally.
    Landscape,
}

/// Physical page size for a printed target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PageSize {
    /// A square page sized to the drawn content plus the margin.
    ///
    /// The default, and the only size whose extent depends on the target: the
    /// content is fitted into a square box so a printed sheet can be rotated
    /// without changing which markers fit. [`PageOrientation`] does not apply.
    #[default]
    FitContent,
    /// ISO A4: 210 × 297 mm in portrait.
    A4,
    /// US Letter: 8.5 × 11 inches in portrait.
    Letter,
    /// An explicit size in millimeters, given in portrait.
    Custom {
        /// Page width in millimeters.
        width_mm: f64,
        /// Page height in millimeters.
        height_mm: f64,
    },
}

impl PageSize {
    /// Portrait `(width_mm, height_mm)` for the fixed sizes.
    ///
    /// Returns `None` for [`PageSize::FitContent`], whose extent is only known
    /// once the target's drawn content has been measured.
    ///
    /// A [`PageSize::Custom`] whose dimensions are not finite and positive also
    /// yields `None`; the renderer reports that as an error.
    pub fn portrait_dimensions_mm(&self) -> Option<(f64, f64)> {
        match *self {
            Self::FitContent => None,
            Self::A4 => Some((210.0, 297.0)),
            Self::Letter => Some((8.5 * MM_PER_INCH, 11.0 * MM_PER_INCH)),
            Self::Custom {
                width_mm,
                height_mm,
            } => {
                (width_mm.is_finite() && height_mm.is_finite() && width_mm > 0.0 && height_mm > 0.0)
                    .then_some((width_mm, height_mm))
            }
        }
    }
}

/// Page size, orientation, and margin for a printed target.
///
/// The default is [`PageSize::FitContent`] with a zero margin — a square page
/// sized exactly to the target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PageSpec {
    /// Physical page size.
    #[serde(default)]
    pub size: PageSize,
    /// Orientation applied to [`PageSpec::size`]. Ignored by
    /// [`PageSize::FitContent`], whose page is square.
    #[serde(default)]
    pub orientation: PageOrientation,
    /// Uniform white margin in millimeters on every edge.
    #[serde(default)]
    pub margin_mm: f32,
}

impl Default for PageSpec {
    fn default() -> Self {
        Self {
            size: PageSize::default(),
            orientation: PageOrientation::default(),
            margin_mm: 0.0,
        }
    }
}

impl PageSpec {
    /// A page of the given size at the default orientation and margin.
    #[must_use]
    pub fn new(size: PageSize) -> Self {
        Self {
            size,
            ..Self::default()
        }
    }

    /// Set the orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: PageOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set the uniform margin in millimeters.
    #[must_use]
    pub fn with_margin_mm(mut self, margin_mm: f32) -> Self {
        self.margin_mm = margin_mm;
        self
    }

    /// Oriented `(width_mm, height_mm)` for the fixed sizes, or `None` for
    /// [`PageSize::FitContent`] and an invalid [`PageSize::Custom`].
    pub fn fixed_dimensions_mm(&self) -> Option<(f64, f64)> {
        let (w, h) = self.size.portrait_dimensions_mm()?;
        Some(match self.orientation {
            PageOrientation::Portrait => (w, h),
            PageOrientation::Landscape => (h, w),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_fit_content_with_no_margin() {
        let page = PageSpec::default();
        assert_eq!(page.size, PageSize::FitContent);
        assert_eq!(page.margin_mm, 0.0);
        assert_eq!(page.fixed_dimensions_mm(), None);
    }

    #[test]
    fn landscape_swaps_the_portrait_dimensions() {
        let portrait = PageSpec::new(PageSize::A4);
        let landscape = portrait.with_orientation(PageOrientation::Landscape);
        assert_eq!(portrait.fixed_dimensions_mm(), Some((210.0, 297.0)));
        assert_eq!(landscape.fixed_dimensions_mm(), Some((297.0, 210.0)));
    }

    #[test]
    fn letter_is_exact_inches() {
        let (w, h) = PageSpec::new(PageSize::Letter)
            .fixed_dimensions_mm()
            .expect("letter is a fixed size");
        assert!((w - 215.9).abs() < 1e-9, "width {w}");
        assert!((h - 279.4).abs() < 1e-9, "height {h}");
    }

    #[test]
    fn non_finite_custom_dimensions_are_rejected() {
        for (width_mm, height_mm) in [(f64::NAN, 100.0), (100.0, 0.0), (-1.0, 100.0)] {
            let page = PageSpec::new(PageSize::Custom {
                width_mm,
                height_mm,
            });
            assert_eq!(page.fixed_dimensions_mm(), None, "{width_mm}x{height_mm}");
        }
    }

    #[test]
    fn fit_content_ignores_orientation() {
        let page = PageSpec::default().with_orientation(PageOrientation::Landscape);
        assert_eq!(page.fixed_dimensions_mm(), None);
    }

    #[test]
    fn json_shape_matches_the_documented_contract() {
        let page = PageSpec::new(PageSize::A4)
            .with_orientation(PageOrientation::Landscape)
            .with_margin_mm(10.0);
        let json = serde_json::to_string(&page).expect("serialize");
        assert_eq!(
            json,
            r#"{"size":{"kind":"a4"},"orientation":"landscape","margin_mm":10.0}"#
        );
        let back: PageSpec = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back, page);
    }

    #[test]
    fn custom_size_round_trips_its_dimensions() {
        let page = PageSpec::new(PageSize::Custom {
            width_mm: 300.0,
            height_mm: 400.0,
        });
        let json = serde_json::to_string(&page).expect("serialize");
        let back: PageSpec = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back, page);
        assert_eq!(back.fixed_dimensions_mm(), Some((300.0, 400.0)));
    }

    #[test]
    fn every_field_defaults_when_absent() {
        let page: PageSpec = serde_json::from_str("{}").expect("all fields default");
        assert_eq!(page, PageSpec::default());
    }
}
