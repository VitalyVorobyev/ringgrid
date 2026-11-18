//! Screen fitting and coverage-sampled rasterization in native pixels.

use super::{
    Codebook, codeword_of, content_bounds_mm, render_geometry, sector_is_white, target_codebook,
};
use crate::{HexGeometry, LatticeGeometry, MarkerCoding, RectGeometry, RingGeometry, TargetLayout};
use image::GrayImage;
use png::{BitDepth, ColorType, Encoder};
use serde::{Deserialize, Serialize};
use std::ops::Range;

// Keep resource use bounded on native and WASM callers.
const MAX_SCREEN_PIXELS: u64 = 32 * 1024 * 1024;
const MAX_CELLS: usize = 16_384;
const SAMPLES: i32 = 8;
const WHITE: u8 = (SAMPLES * SAMPLES) as u8;

fn default_margin() -> u32 {
    8
}

/// Native grayscale screen dimensions and the physical marker size to fit.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenTargetOptions {
    /// Native screen width in the intended orientation, in pixels.
    pub width_px: u32,
    /// Native screen height in the intended orientation, in pixels.
    pub height_px: u32,
    /// Physical side length of a square display pixel, in millimeters.
    pub pixel_pitch_mm: f64,
    /// Requested outermost drawn marker diameter, including coded ring stroke.
    pub marker_diameter_mm: f64,
    /// Minimum white margin on each side, in pixels. Defaults to eight.
    #[serde(default = "default_margin")]
    pub margin_px: u32,
}

impl ScreenTargetOptions {
    /// Set the native resolution, physical pixel pitch, and drawn diameter.
    pub fn new(
        width_px: u32,
        height_px: u32,
        pixel_pitch_mm: f64,
        marker_diameter_mm: f64,
    ) -> Self {
        Self {
            width_px,
            height_px,
            pixel_pitch_mm,
            marker_diameter_mm,
            margin_px: default_margin(),
        }
    }

    /// Set the minimum margin on each edge, in pixels.
    #[must_use]
    pub fn with_margin_px(mut self, margin_px: u32) -> Self {
        self.margin_px = margin_px;
        self
    }

    fn available_mm(&self) -> Result<[f64; 2], ScreenTargetError> {
        if self.width_px == 0
            || self.height_px == 0
            || u64::from(self.width_px) * u64::from(self.height_px) > MAX_SCREEN_PIXELS
        {
            return Err(ScreenTargetError::Resolution);
        }
        if !self.pixel_pitch_mm.is_finite()
            || self.pixel_pitch_mm <= 0.0
            || !(f64::from(self.width_px.max(self.height_px)) * self.pixel_pitch_mm).is_finite()
        {
            return Err(ScreenTargetError::PixelPitch);
        }
        if !self.marker_diameter_mm.is_finite() || self.marker_diameter_mm <= 0.0 {
            return Err(ScreenTargetError::MarkerDiameter);
        }
        let margin = self.margin_px.saturating_mul(2);
        if margin >= self.width_px.min(self.height_px) {
            return Err(ScreenTargetError::Margin);
        }
        Ok([self.width_px - margin, self.height_px - margin]
            .map(|size| f64::from(size) * self.pixel_pitch_mm))
    }
}

/// Fitted screen geometry. The canonical target spec contains the marker layout.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenTargetMetadata {
    /// Requested screen and marker dimensions.
    pub screen: ScreenTargetOptions,
    /// Board origin in native screen pixels. Pixel centers are integers.
    pub board_origin_px: [f64; 2],
    /// Fitted row count.
    pub rows: usize,
    /// Fitted column count for the longest row.
    pub columns: usize,
    /// Number of fitted markers.
    pub marker_count: usize,
    /// Actual outermost drawn diameter, in millimeters.
    pub drawn_marker_diameter_mm: f64,
    /// Actual outermost drawn diameter, in native screen pixels.
    pub drawn_marker_diameter_px: f64,
    /// Coded band width in native pixels, when the template is coded.
    pub code_band_width_px: Option<f64>,
}

/// Native PNG, detector-ready geometry, and screen placement.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenTargetArtifacts {
    /// Native-resolution 8-bit grayscale PNG.
    pub png_bytes: Vec<u8>,
    /// Canonical `ringgrid.target.v6` spec, newline-terminated.
    pub json_text: String,
    /// Requested and realized placement in the display frame.
    pub metadata: ScreenTargetMetadata,
}

/// Errors specific to screen fitting and export.
#[non_exhaustive]
#[derive(Debug)]
pub enum ScreenTargetError {
    /// The resolution is empty or exceeds the raster budget.
    Resolution,
    /// The physical pixel pitch is invalid.
    PixelPitch,
    /// The physical marker size is invalid or outside the target model's range.
    MarkerDiameter,
    /// Margins consume the entire screen.
    Margin,
    /// A two-dimensional detection grid cannot fit with the requested geometry.
    NoFit,
    /// An explicit ID assignment cannot move to a different lattice.
    ExplicitIds,
    /// Native PNG encoding failed.
    Png(png::EncodingError),
}

impl std::fmt::Display for ScreenTargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolution => f.write_str(
                "screen dimensions must be positive and contain at most 33554432 pixels",
            ),
            Self::PixelPitch => {
                f.write_str("pixel_pitch_mm must be positive and give a finite screen extent")
            }
            Self::MarkerDiameter => f.write_str(
                "marker_diameter_mm must be positive, finite, and representable in target geometry",
            ),
            Self::Margin => f.write_str("pixel margins leave no usable screen area"),
            Self::NoFit => f.write_str(
                "no valid two-dimensional target fits at that marker diameter and margin",
            ),
            Self::ExplicitIds => f.write_str(
                "screen fitting changed the grid dimensions of a target with explicit codebook IDs",
            ),
            Self::Png(error) => write!(f, "failed to encode screen PNG: {error}"),
        }
    }
}

impl std::error::Error for ScreenTargetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Png(error) => Some(error),
            _ => None,
        }
    }
}

impl From<png::EncodingError> for ScreenTargetError {
    fn from(error: png::EncodingError) -> Self {
        Self::Png(error)
    }
}

impl TargetLayout {
    /// Fit a target to a native grayscale display and render its exact geometry.
    ///
    /// Uses this layout as a template for lattice kind, spacing relative to
    /// diameter, ring proportions, codebook, and origin dots. Maximizes the
    /// cell count at the requested drawn diameter. A screen pixel center is
    /// at its integer index. Board millimeters map to screen pixels by dividing
    /// by the physical pitch and adding `metadata.board_origin_px`.
    pub fn render_fitted_screen_target(
        &self,
        options: &ScreenTargetOptions,
    ) -> Result<ScreenTargetArtifacts, ScreenTargetError> {
        let fit = fit(self, options)?;
        let target = fit.target;
        let radius = f64::from(target.outer_draw_radius_mm());
        let [xmin, ymin, xmax, ymax] = fit.bounds;
        let ppmm = 1.0 / options.pixel_pitch_mm;
        let origin = [
            (f64::from(options.width_px) - 1.0 - (xmin + xmax) * ppmm) * 0.5,
            (f64::from(options.height_px) - 1.0 - (ymin + ymax) * ppmm) * 0.5,
        ];
        let image = rasterize(&target, options, origin);
        let mut png_bytes = Vec::new();
        let mut encoder = Encoder::new(&mut png_bytes, image.width(), image.height());
        encoder.set_color(ColorType::Grayscale);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(image.as_raw())?;
        writer.finish()?;
        Ok(ScreenTargetArtifacts {
            png_bytes,
            json_text: format!("{}\n", target.to_json_string()),
            metadata: ScreenTargetMetadata {
                screen: *options,
                board_origin_px: origin,
                rows: fit.rows,
                columns: fit.cols,
                marker_count: target.n_cells(),
                drawn_marker_diameter_mm: 2.0 * radius,
                drawn_marker_diameter_px: 2.0 * radius * ppmm,
                code_band_width_px: target
                    .code_band_bounds_mm()
                    .map(|(inner, outer)| (f64::from(outer) - f64::from(inner)) * ppmm),
            },
        })
    }
}

struct Fit {
    target: TargetLayout,
    bounds: [f64; 4],
    fill: f64,
    rows: usize,
    cols: usize,
}

fn dimensions(lattice: &LatticeGeometry) -> (usize, usize) {
    match lattice {
        LatticeGeometry::Hex(h) => (h.rows, h.long_row_cols),
        LatticeGeometry::Rect(r) => (r.rows, r.cols),
    }
}

struct FitGeometry {
    pitch: f32,
    ring: RingGeometry,
    coding: MarkerCoding,
    dots: Option<crate::OriginFiducials>,
    hex: bool,
    cell_limit: usize,
}

impl FitGeometry {
    fn scaled(template: &TargetLayout, diameter: f64) -> Result<Self, ScreenTargetError> {
        let scale = diameter / (2.0 * f64::from(template.outer_draw_radius_mm()));
        let scaled = |size| (f64::from(size) * scale) as f32;
        let pitch = scaled(template.pitch_mm());
        let ring = template.ring();
        let ring = RingGeometry {
            outer_radius_mm: scaled(ring.outer_radius_mm),
            inner_radius_mm: scaled(ring.inner_radius_mm),
        };
        if !pitch.is_finite()
            || !ring.outer_radius_mm.is_finite()
            || ring.inner_radius_mm <= 0.0
            || ring.outer_radius_mm <= ring.inner_radius_mm
            || pitch <= 0.0
        {
            return Err(ScreenTargetError::MarkerDiameter);
        }
        let mut coding = template.coding().clone();
        let mut cell_limit = MAX_CELLS;
        if let MarkerCoding::Coded16(spec) = &mut coding {
            spec.ring_width_mm = scaled(spec.ring_width_mm);
            if !spec.ring_width_mm.is_finite() || spec.ring_width_mm <= 0.0 {
                return Err(ScreenTargetError::MarkerDiameter);
            }
            cell_limit = Codebook::from_profile(spec.codebook_profile).len();
            spec.id_assignment = None;
        }
        let dots = template.fiducials().map(|dot| crate::OriginFiducials {
            dot_radius_mm: scaled(dot.dot_radius_mm),
        });
        Ok(Self {
            pitch,
            ring,
            coding,
            dots,
            hex: matches!(template.lattice(), LatticeGeometry::Hex(_)),
            cell_limit,
        })
    }

    fn lattice(&self, rows: usize, cols: usize) -> LatticeGeometry {
        if self.hex {
            LatticeGeometry::Hex(HexGeometry {
                rows,
                long_row_cols: cols,
                pitch_mm: self.pitch,
            })
        } else {
            LatticeGeometry::Rect(RectGeometry {
                rows,
                cols,
                pitch_mm: self.pitch,
            })
        }
    }

    fn count(&self, rows: usize, cols: usize) -> usize {
        // At most one unpaired row. Its length follows the canonical hex row parity.
        if self.hex {
            let first_is_short = (cols - 1) % 2 != (rows / 2) % 2;
            rows * cols - rows / 2 - usize::from(rows % 2 == 1 && first_is_short)
        } else {
            rows * cols
        }
    }

    fn candidate(&self, name: &str, available: [f64; 2], rows: usize, cols: usize) -> Option<Fit> {
        let target = TargetLayout::new(
            name,
            self.lattice(rows, cols),
            self.ring,
            self.coding.clone(),
            self.dots,
        )
        .ok()?;
        let (x0, y0, x1, y1) = content_bounds_mm(&target, f64::from(target.outer_draw_radius_mm()));
        let [x, y] = [(x1 - x0) / available[0], (y1 - y0) / available[1]];
        (x <= 1.0 && y <= 1.0).then_some(Fit {
            target,
            bounds: [x0, y0, x1, y1],
            fill: x.min(y),
            rows,
            cols,
        })
    }
}

fn fit(template: &TargetLayout, options: &ScreenTargetOptions) -> Result<Fit, ScreenTargetError> {
    let available = options.available_mm()?;
    let geometry = FitGeometry::scaled(template, options.marker_diameter_mm)?;
    let pitch = f64::from(geometry.pitch);
    let dx = if geometry.hex {
        pitch * 3.0f64.sqrt()
    } else {
        pitch
    };
    let dy = if geometry.hex { pitch * 1.5 } else { pitch };
    let diameter = 2.0 * f64::from(geometry.coding.outer_draw_radius_mm(&geometry.ring));
    // A one-cell cushion leaves the final boundary decision to the canonical
    // layout after its f32 positions have been generated.
    let max_axis =
        |extent: f64, step: f64| (((extent - diameter) / step).floor() + 2.0).max(0.0) as usize;
    let max_rows = max_axis(available[1], dy).min(geometry.cell_limit);
    let max_cols = max_axis(available[0], dx).min(geometry.cell_limit);
    let within_axis = |count: usize, extent: f64, step: f64| {
        let span = (count - 1) as f64 * step + diameter;
        // RingGrid stores normalized positions in f32. Only the canonical
        // layout decides fit when an edge is within a few coordinate ULPs.
        span <= extent * (1.0 + 4.0 * f64::from(f32::EPSILON))
    };
    let mut best: Option<Fit> = None;
    for rows in (2..=max_rows).rev() {
        if !within_axis(rows, available[1], dy) {
            continue;
        }
        // Rect rows use every slot, hex rows lose at most one slot each.
        let columns_within_budget = geometry.cell_limit / rows + usize::from(geometry.hex);
        for cols in (2..=max_cols.min(columns_within_budget)).rev() {
            if !within_axis(cols, available[0], dx) {
                continue;
            }
            let count = geometry.count(rows, cols);
            if count > geometry.cell_limit {
                continue;
            }
            if best
                .as_ref()
                .is_some_and(|old| count < old.target.n_cells())
            {
                break;
            }
            if let Some(candidate) = geometry.candidate(template.name(), available, rows, cols) {
                if best.as_ref().is_none_or(|old| {
                    (candidate.target.n_cells(), candidate.fill, rows)
                        > (old.target.n_cells(), old.fill, old.rows)
                }) {
                    best = Some(candidate);
                }
                break;
            }
        }
    }
    let mut best = best.ok_or(ScreenTargetError::NoFit)?;
    if let MarkerCoding::Coded16(original) = template.coding()
        && original.id_assignment.is_some()
    {
        if (best.rows, best.cols) != dimensions(template.lattice()) {
            return Err(ScreenTargetError::ExplicitIds);
        }
        let MarkerCoding::Coded16(mut fitted) = best.target.coding().clone() else {
            unreachable!()
        };
        fitted.id_assignment.clone_from(&original.id_assignment);
        best.target = TargetLayout::new(
            template.name(),
            *best.target.lattice(),
            best.target.ring(),
            MarkerCoding::Coded16(fitted),
            None,
        )
        .map_err(|_| ScreenTargetError::NoFit)?;
    }
    Ok(best)
}

struct RingInk {
    // Squared radii in ascending order. Plain: hole and outside.
    // Coded: hole, start of code band, end of code band, outside.
    radii: [f64; 4],
    coded: bool,
}

impl RingInk {
    fn interval(&self, dist: f64) -> usize {
        let count = if self.coded { 4 } else { 2 };
        self.radii[..count].partition_point(|&radius| radius <= dist)
    }

    fn is_ink(&self, dx: f64, dy: f64, word: u32) -> bool {
        let interval = self.interval(dx * dx + dy * dy);
        interval % 2 == 1 || (self.coded && interval == 2 && !sector_is_white(word, dx, dy))
    }

    fn coverage(&self, dx: f64, dy: f64, word: u32) -> u8 {
        let (near, far) = distance_bounds(dx, dy);
        let interval = self.interval(near);
        if interval == self.interval(far) && !(self.coded && interval == 2) {
            return if interval % 2 == 1 { WHITE } else { 0 };
        }
        sample_coverage(dx, dy, |x, y| self.is_ink(x, y, word))
    }
}

fn rasterize(target: &TargetLayout, options: &ScreenTargetOptions, origin: [f64; 2]) -> GrayImage {
    let ppmm = 1.0 / options.pixel_pitch_mm;
    let g = render_geometry(target);
    let radius = g.outer_draw_extent_mm * ppmm;
    let hole = (g.inner_radius_mm - g.ring_half_thickness_mm) * ppmm;
    let radii = if let Some((inner, outer)) = g.code_band_mm {
        [hole, inner * ppmm, outer * ppmm, radius]
    } else {
        [hole, radius, 0.0, 0.0]
    };
    let ink = RingInk {
        radii: radii.map(|r| r * r),
        coded: target.is_coded(),
    };
    let codebook = target_codebook(target);
    let [width, height] = [options.width_px, options.height_px];
    let mut image = GrayImage::new(width, height);
    let center = |xy: [f32; 2]| {
        [
            f64::from(xy[0]) * ppmm + origin[0],
            f64::from(xy[1]) * ppmm + origin[1],
        ]
    };
    for cell in target.cells() {
        let word = cell
            .id
            .map_or(0, |id| u32::from(codeword_of(&codebook, id)));
        paint(&mut image, center(cell.xy_mm), radius, |dx, dy| {
            ink.coverage(dx, dy, word)
        });
    }
    if let Some(dots) = target.fiducials() {
        let radius = f64::from(dots.dot_radius_mm) * ppmm;
        let squared = radius * radius;
        let centers: Vec<_> = target
            .fiducial_dots_mm()
            .iter()
            .copied()
            .map(center)
            .collect();
        for (index, &[cx, cy]) in centers.iter().enumerate() {
            let overlaps: Vec<_> = centers[..index]
                .iter()
                .map(|&[x, y]| [x - cx, y - cy])
                .filter(|&[dx, dy]| dx.hypot(dy) < 2.0 * radius)
                .collect();
            paint(&mut image, [cx, cy], radius, |dx, dy| {
                let (near, far) = distance_bounds(dx, dy);
                if near >= squared {
                    0
                } else if far <= squared && overlaps.is_empty() {
                    WHITE
                } else {
                    // The layout checks clearance to rings. Explicit origin
                    // dots can overlap each other, so count shared ink once.
                    sample_coverage(dx, dy, |x, y| {
                        x * x + y * y < squared
                            && overlaps
                                .iter()
                                .all(|&[ox, oy]| (x - ox).powi(2) + (y - oy).powi(2) >= squared)
                    })
                }
            });
        }
    }
    for sample in image.as_mut() {
        *sample =
            255 - ((u16::from(*sample) * 255 + u16::from(WHITE) / 2) / u16::from(WHITE)) as u8;
    }
    image
}

fn paint(
    image: &mut GrayImage,
    [cx, cy]: [f64; 2],
    radius: f64,
    coverage: impl Fn(f64, f64) -> u8,
) {
    let bounds = |c: f64, limit: u32| -> Range<u32> {
        (c - radius - 0.5).ceil().max(0.0) as u32
            ..((c + radius + 0.5).floor() + 1.0).min(f64::from(limit)) as u32
    };
    let x_range = bounds(cx, image.width());
    for y in bounds(cy, image.height()) {
        for x in x_range.clone() {
            let pixel = &mut image.get_pixel_mut(x, y).0[0];
            // Each call contributes new ink samples. Multiple shapes can
            // contribute disjoint samples within one pixel footprint.
            *pixel += coverage(f64::from(x) - cx, f64::from(y) - cy);
        }
    }
}

fn distance_bounds(x: f64, y: f64) -> (f64, f64) {
    let (x, y) = (x.abs(), y.abs());
    (
        (x - 0.5).max(0.0).powi(2) + (y - 0.5).max(0.0).powi(2),
        (x + 0.5).powi(2) + (y + 0.5).powi(2),
    )
}

fn sample_coverage(dx: f64, dy: f64, is_ink: impl Fn(f64, f64) -> bool) -> u8 {
    let offset = |i| (f64::from(i) + 0.5) / f64::from(SAMPLES) - 0.5;
    let mut count = 0;
    for y in 0..SAMPLES {
        for x in 0..SAMPLES {
            count += u8::from(is_ink(dx + offset(x), dy + offset(y)));
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_tracks_high_resolution_shape_for_fractional_centers() {
        let ink = RingInk {
            radii: [11.3, 14.2, 16.4, 19.6].map(|r| r * r),
            coded: true,
        };
        let word = 0xa682;
        for phase in [0.0, 0.21, 0.5, 0.77] {
            let mut worst = 0.0_f64;
            for (dx, dy) in [
                (0.1, 11.1),
                (13.9, 3.0),
                (15.0, 0.1),
                (-0.3, 16.5),
                (18.9, 1.0),
            ] {
                let (dx, dy) = (dx + phase, dy - phase);
                let mut count = 0;
                let zoom = 128;
                for y in 0..zoom {
                    for x in 0..zoom {
                        let dx = dx + (f64::from(x) + 0.5) / f64::from(zoom) - 0.5;
                        let dy = dy + (f64::from(y) + 0.5) / f64::from(zoom) - 0.5;
                        let r = dx.hypot(dy);
                        let theta = dy.atan2(dx) / std::f64::consts::TAU + 0.5;
                        let sector = ((theta * 16.0).floor() as u32) % 16;
                        let is_ink = (11.3..14.2).contains(&r)
                            || (16.4..19.6).contains(&r)
                            || ((14.2..16.4).contains(&r) && (word & (1 << sector)) == 0);
                        count += i32::from(is_ink);
                    }
                }
                let reference = f64::from(count) / f64::from(zoom * zoom);
                let got = f64::from(ink.coverage(dx, dy, word)) / f64::from(WHITE);
                worst = worst.max((got - reference).abs());
            }
            assert!(worst < 0.065, "phase {phase}: worst coverage error {worst}");
        }
        // The two sides of the zero-angle boundary carry different bits.
        assert!(!ink.is_ink(15.0, 0.001, 1 << 8));
        assert!(ink.is_ink(15.0, -0.001, 1 << 8));
    }
}
