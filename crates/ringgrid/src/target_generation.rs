//! Printable target rendering (SVG and PNG) for any [`TargetLayout`].

use crate::BoardLayout;
use crate::marker::codebook::CODEBOOK;
use crate::target::{MarkerCoding, TargetLayout};
use image::{GrayImage, Luma};
#[cfg(feature = "std")]
use png::{BitDepth, ColorType, Encoder as PngEncoder, EncodingError, PixelDimensions, Unit};
use std::f64::consts::PI;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::BufWriter;
#[cfg(feature = "std")]
use std::path::Path;

const CODE_SECTORS: usize = 16;
const MM_PER_INCH: f64 = 25.4;
const DEFAULT_PNG_DPI: f32 = 300.0;

/// SVG target-generation options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgTargetOptions {
    /// Extra white border around the generated square page, in millimeters.
    pub margin_mm: f32,
    /// Include the default scale bar in the lower-left corner.
    pub include_scale_bar: bool,
}

impl Default for SvgTargetOptions {
    fn default() -> Self {
        Self {
            margin_mm: 0.0,
            include_scale_bar: true,
        }
    }
}

/// PNG target-generation options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PngTargetOptions {
    /// Raster density used to convert millimeters into output pixels.
    ///
    /// When written via [`TargetLayout::write_target_png`], this is also
    /// embedded as PNG physical pixel dimensions (`pHYs`) so the file retains
    /// the intended print scale.
    pub dpi: f32,
    /// Extra white border around the generated square page, in millimeters.
    pub margin_mm: f32,
    /// Include the default scale bar in the lower-left corner.
    pub include_scale_bar: bool,
}

impl Default for PngTargetOptions {
    fn default() -> Self {
        Self {
            dpi: DEFAULT_PNG_DPI,
            margin_mm: 0.0,
            include_scale_bar: true,
        }
    }
}

/// Target-generation failures.
#[derive(Debug)]
pub enum TargetGenerationError {
    /// Margin value is negative or non-finite.
    InvalidMargin {
        /// The invalid margin value.
        margin_mm: f32,
    },
    /// DPI value is non-positive or non-finite.
    InvalidDpi {
        /// The invalid DPI value.
        dpi: f32,
    },
    /// File I/O error during target writing.
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// Image encoding error.
    Image(image::ImageError),
    /// PNG-specific encoding error.
    #[cfg(feature = "std")]
    PngEncoding(EncodingError),
}

impl std::fmt::Display for TargetGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMargin { margin_mm } => {
                write!(f, "margin_mm must be finite and >= 0 (got {margin_mm})")
            }
            Self::InvalidDpi { dpi } => {
                write!(f, "dpi must be finite and > 0 (got {dpi})")
            }
            #[cfg(feature = "std")]
            Self::Io(err) => write!(f, "failed to write target output: {err}"),
            Self::Image(err) => write!(f, "failed to encode target image: {err}"),
            #[cfg(feature = "std")]
            Self::PngEncoding(err) => write!(f, "failed to encode PNG target: {err}"),
        }
    }
}

impl std::error::Error for TargetGenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "std")]
            Self::Io(err) => Some(err),
            Self::Image(err) => Some(err),
            #[cfg(feature = "std")]
            Self::PngEncoding(err) => Some(err),
            Self::InvalidMargin { .. } | Self::InvalidDpi { .. } => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for TargetGenerationError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<image::ImageError> for TargetGenerationError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}

#[cfg(feature = "std")]
impl From<EncodingError> for TargetGenerationError {
    fn from(value: EncodingError) -> Self {
        Self::PngEncoding(value)
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderGeometry {
    outer_radius_mm: f64,
    inner_radius_mm: f64,
    /// Code band bounds; `None` for plain (uncoded) markers.
    code_band_mm: Option<(f64, f64)>,
    /// Half of the ring stroke width; 0 for plain markers.
    ring_half_thickness_mm: f64,
    outer_draw_extent_mm: f64,
}

#[derive(Debug, Clone, Copy)]
struct CanvasLayout {
    side_mm: f64,
    canvas_side_mm: f64,
    offset_x_mm: f64,
    offset_y_mm: f64,
}

#[derive(Debug, Clone, Copy)]
struct ScaleBarParams {
    x0_mm: f64,
    y0_mm: f64,
    bar_len_mm: i32,
    bar_h_mm: f64,
    tick_step_mm: i32,
    tick_w_mm: f64,
    font_size_mm: f64,
}

impl ScaleBarParams {
    fn label(self) -> String {
        format!("{} mm", self.bar_len_mm)
    }
}

impl TargetLayout {
    /// Render a printable SVG target.
    ///
    /// Input marker centers stay in normalized board millimeters with the
    /// first cell anchored at `[0, 0]`. The returned SVG translates those
    /// markers into a square page in millimeters with top-left origin and
    /// `+y` increasing downward.
    pub fn render_target_svg(
        &self,
        options: &SvgTargetOptions,
    ) -> Result<String, TargetGenerationError> {
        let margin_mm = validated_margin(options.margin_mm)?;
        let geometry = render_geometry(self);
        let canvas = canvas_layout(self, geometry.outer_draw_extent_mm, margin_mm);
        let mut lines = Vec::new();

        lines.push("<?xml version=\"1.0\" encoding=\"UTF-8\"?>".to_string());
        lines.push(format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}mm\" height=\"{}mm\" viewBox=\"0 0 {} {}\">",
            svg_fmt(canvas.canvas_side_mm),
            svg_fmt(canvas.canvas_side_mm),
            svg_fmt(canvas.canvas_side_mm),
            svg_fmt(canvas.canvas_side_mm)
        ));
        lines.push(
            "<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"white\"/>".to_string(),
        );

        for (idx, cell) in self.cells().iter().enumerate() {
            let cx = f64::from(cell.xy_mm[0]) + canvas.offset_x_mm;
            let cy = f64::from(cell.xy_mm[1]) + canvas.offset_y_mm;

            match cell.id {
                Some(id) => {
                    lines.push(format!("<g id=\"m{idx}\" data-id=\"{id}\">"));
                    append_coded_marker_svg(&mut lines, cx, cy, geometry, CODEBOOK[id].into());
                    lines.push("</g>".to_string());
                }
                None => {
                    lines.push(format!("<g id=\"m{idx}\">"));
                    lines.push(format!(
                        "<path d=\"{}\" fill=\"black\" fill-rule=\"evenodd\"/>",
                        svg_annulus_path(
                            cx,
                            cy,
                            geometry.outer_radius_mm,
                            geometry.inner_radius_mm
                        )
                    ));
                    lines.push("</g>".to_string());
                }
            }
        }

        if let Some(fiducials) = self.fiducials() {
            lines.push("<g id=\"fiducials\">".to_string());
            for dot in &fiducials.dots_mm {
                lines.push(format!(
                    "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"black\"/>",
                    svg_fmt(f64::from(dot[0]) + canvas.offset_x_mm),
                    svg_fmt(f64::from(dot[1]) + canvas.offset_y_mm),
                    svg_fmt(f64::from(fiducials.dot_radius_mm))
                ));
            }
            lines.push("</g>".to_string());
        }

        if options.include_scale_bar {
            append_scale_bar_svg(&mut lines, self, canvas, geometry);
        }

        lines.push("</svg>".to_string());
        Ok(lines.join("\n") + "\n")
    }

    /// Write a printable SVG target to disk.
    #[cfg(feature = "std")]
    pub fn write_target_svg(
        &self,
        path: &Path,
        options: &SvgTargetOptions,
    ) -> Result<(), TargetGenerationError> {
        let svg = self.render_target_svg(options)?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, svg)?;
        Ok(())
    }

    /// Render a printable PNG target.
    ///
    /// The raster uses integer pixel centers, matching the existing Python
    /// target generator's sampling convention.
    pub fn render_target_png(
        &self,
        options: &PngTargetOptions,
    ) -> Result<GrayImage, TargetGenerationError> {
        let margin_mm = validated_margin(options.margin_mm)?;
        let dpi = validated_dpi(options.dpi)?;
        let geometry = render_geometry(self);
        let canvas = canvas_layout(self, geometry.outer_draw_extent_mm, margin_mm);

        let pixels_per_mm = dpi / MM_PER_INCH;
        let width_px = (canvas.canvas_side_mm * pixels_per_mm).round().max(1.0) as u32;
        let height_px = width_px;
        let mut image = GrayImage::from_pixel(width_px, height_px, Luma([255]));

        let ring_half_thickness_px = geometry.ring_half_thickness_mm * pixels_per_mm;
        let outer_radius_px = geometry.outer_radius_mm * pixels_per_mm;
        let inner_radius_px = geometry.inner_radius_mm * pixels_per_mm;
        let outer_draw_extent_px = geometry.outer_draw_extent_mm * pixels_per_mm;

        let outer_min_sq = square(outer_radius_px - ring_half_thickness_px);
        let outer_max_sq = square(outer_radius_px + ring_half_thickness_px);
        let inner_min_sq = square(inner_radius_px - ring_half_thickness_px);
        let inner_max_sq = square(inner_radius_px + ring_half_thickness_px);
        let code_band_sq = geometry.code_band_mm.map(|(inner_mm, outer_mm)| {
            (
                square(inner_mm * pixels_per_mm),
                square(outer_mm * pixels_per_mm),
            )
        });
        let two_pi = 2.0 * PI;
        let bound = outer_draw_extent_px + 2.0;

        for cell in self.cells() {
            let codeword = cell.id.map(|id| u32::from(CODEBOOK[id]));
            let cx = (f64::from(cell.xy_mm[0]) + canvas.offset_x_mm) * pixels_per_mm;
            let cy = (f64::from(cell.xy_mm[1]) + canvas.offset_y_mm) * pixels_per_mm;

            let x0 = (cx - bound).floor().max(0.0) as u32;
            let x1 = ((cx + bound).ceil() + 1.0).min(f64::from(width_px)) as u32;
            let y0 = (cy - bound).floor().max(0.0) as u32;
            let y1 = ((cy + bound).ceil() + 1.0).min(f64::from(height_px)) as u32;
            if x0 >= x1 || y0 >= y1 {
                continue;
            }

            for y in y0..y1 {
                for x in x0..x1 {
                    let dx = f64::from(x) - cx;
                    let dy = f64::from(y) - cy;
                    let dist_sq = dx * dx + dy * dy;
                    let pixel = image.get_pixel_mut(x, y);

                    match codeword {
                        Some(codeword) => {
                            if (outer_min_sq..=outer_max_sq).contains(&dist_sq) {
                                pixel.0[0] = 0;
                            }
                            if (inner_min_sq..=inner_max_sq).contains(&dist_sq) {
                                pixel.0[0] = 0;
                            }
                            if let Some((code_min_sq, code_max_sq)) = code_band_sq
                                && (code_min_sq..=code_max_sq).contains(&dist_sq)
                            {
                                let angle = dy.atan2(dx);
                                let sector =
                                    (((angle / two_pi) + 0.5) * CODE_SECTORS as f64).floor() as i32;
                                let sector = sector.rem_euclid(CODE_SECTORS as i32) as u32;
                                pixel.0[0] = if ((codeword >> sector) & 1) == 1 {
                                    255
                                } else {
                                    0
                                };
                            }
                        }
                        None => {
                            // Plain marker: filled annulus between the radii.
                            if (inner_min_sq..=outer_max_sq).contains(&dist_sq) {
                                pixel.0[0] = 0;
                            }
                        }
                    }
                }
            }
        }

        if let Some(fiducials) = self.fiducials() {
            let dot_radius_px = f64::from(fiducials.dot_radius_mm) * pixels_per_mm;
            for dot in &fiducials.dots_mm {
                let cx = (f64::from(dot[0]) + canvas.offset_x_mm) * pixels_per_mm;
                let cy = (f64::from(dot[1]) + canvas.offset_y_mm) * pixels_per_mm;
                draw_disk(&mut image, cx, cy, dot_radius_px, 0);
            }
        }

        if options.include_scale_bar {
            draw_scale_bar_raster(
                &mut image,
                self,
                canvas,
                geometry.outer_draw_extent_mm,
                pixels_per_mm,
            );
        }

        Ok(image)
    }

    /// Write a printable PNG target to disk.
    ///
    /// This always writes PNG bytes and embeds the requested `dpi` as PNG
    /// physical pixel dimensions (`pHYs`) so the output preserves print scale.
    #[cfg(feature = "std")]
    pub fn write_target_png(
        &self,
        path: &Path,
        options: &PngTargetOptions,
    ) -> Result<(), TargetGenerationError> {
        let image = self.render_target_png(options)?;
        let dpi = validated_dpi(options.dpi)?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let writer = BufWriter::new(File::create(path)?);
        encode_png(writer, &image, dpi)?;
        Ok(())
    }
}

impl BoardLayout {
    /// Render a printable SVG target (delegates to [`TargetLayout`]).
    pub fn render_target_svg(
        &self,
        options: &SvgTargetOptions,
    ) -> Result<String, TargetGenerationError> {
        TargetLayout::from(self).render_target_svg(options)
    }

    /// Write a printable SVG target to disk (delegates to [`TargetLayout`]).
    #[cfg(feature = "std")]
    pub fn write_target_svg(
        &self,
        path: &Path,
        options: &SvgTargetOptions,
    ) -> Result<(), TargetGenerationError> {
        TargetLayout::from(self).write_target_svg(path, options)
    }

    /// Render a printable PNG target (delegates to [`TargetLayout`]).
    pub fn render_target_png(
        &self,
        options: &PngTargetOptions,
    ) -> Result<GrayImage, TargetGenerationError> {
        TargetLayout::from(self).render_target_png(options)
    }

    /// Write a printable PNG target to disk (delegates to [`TargetLayout`]).
    #[cfg(feature = "std")]
    pub fn write_target_png(
        &self,
        path: &Path,
        options: &PngTargetOptions,
    ) -> Result<(), TargetGenerationError> {
        TargetLayout::from(self).write_target_png(path, options)
    }
}

fn append_coded_marker_svg(
    lines: &mut Vec<String>,
    cx: f64,
    cy: f64,
    geometry: RenderGeometry,
    codeword: u32,
) {
    lines.push(format!(
        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"{}\"/>",
        svg_fmt(cx),
        svg_fmt(cy),
        svg_fmt(geometry.outer_radius_mm),
        svg_fmt(2.0 * geometry.ring_half_thickness_mm)
    ));
    lines.push(format!(
        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"{}\"/>",
        svg_fmt(cx),
        svg_fmt(cy),
        svg_fmt(geometry.inner_radius_mm),
        svg_fmt(2.0 * geometry.ring_half_thickness_mm)
    ));

    let (code_band_inner_mm, code_band_outer_mm) = geometry
        .code_band_mm
        .expect("coded markers always have a code band");
    lines.push(format!(
        "<path d=\"{}\" fill=\"white\" fill-rule=\"evenodd\"/>",
        svg_annulus_path(cx, cy, code_band_outer_mm, code_band_inner_mm)
    ));

    let dtheta = 2.0 * PI / CODE_SECTORS as f64;
    for sector in 0..CODE_SECTORS {
        if ((codeword >> sector) & 1) == 1 {
            continue;
        }
        let angle0 = -PI + sector as f64 * dtheta;
        let angle1 = angle0 + dtheta;
        lines.push(format!(
            "<path d=\"{}\" fill=\"black\"/>",
            svg_annular_sector_path(
                cx,
                cy,
                code_band_outer_mm,
                code_band_inner_mm,
                angle0,
                angle1,
            )
        ));
    }
}

fn draw_disk(image: &mut GrayImage, cx: f64, cy: f64, radius: f64, value: u8) {
    let width = f64::from(image.width());
    let height = f64::from(image.height());
    let x0 = (cx - radius - 1.0).floor().max(0.0) as u32;
    let x1 = ((cx + radius).ceil() + 1.0).min(width) as u32;
    let y0 = (cy - radius - 1.0).floor().max(0.0) as u32;
    let y1 = ((cy + radius).ceil() + 1.0).min(height) as u32;
    let radius_sq = radius * radius;
    for y in y0..y1 {
        for x in x0..x1 {
            let dx = f64::from(x) - cx;
            let dy = f64::from(y) - cy;
            if dx * dx + dy * dy <= radius_sq {
                image.put_pixel(x, y, Luma([value]));
            }
        }
    }
}

fn validated_margin(margin_mm: f32) -> Result<f64, TargetGenerationError> {
    if margin_mm.is_finite() && margin_mm >= 0.0 {
        Ok(f64::from(margin_mm))
    } else {
        Err(TargetGenerationError::InvalidMargin { margin_mm })
    }
}

fn validated_dpi(dpi: f32) -> Result<f64, TargetGenerationError> {
    if dpi.is_finite() && dpi > 0.0 {
        Ok(f64::from(dpi))
    } else {
        Err(TargetGenerationError::InvalidDpi { dpi })
    }
}

#[cfg(feature = "std")]
fn encode_png<W: std::io::Write>(
    writer: W,
    image: &GrayImage,
    dpi: f64,
) -> Result<(), TargetGenerationError> {
    let mut encoder = PngEncoder::new(writer, image.width(), image.height());
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_pixel_dims(Some(PixelDimensions {
        xppu: dpi_to_pixels_per_meter(dpi),
        yppu: dpi_to_pixels_per_meter(dpi),
        unit: Unit::Meter,
    }));

    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.as_raw())?;
    Ok(())
}

fn render_geometry(target: &TargetLayout) -> RenderGeometry {
    let ring = target.ring();
    let is_coded = matches!(target.coding(), MarkerCoding::Coded16(_));
    let ring_half_thickness_mm = if is_coded {
        f64::from(target.outer_draw_radius_mm() - ring.outer_radius_mm)
    } else {
        0.0
    };

    RenderGeometry {
        outer_radius_mm: f64::from(ring.outer_radius_mm),
        inner_radius_mm: f64::from(ring.inner_radius_mm),
        code_band_mm: target
            .code_band_bounds_mm()
            .map(|(inner, outer)| (f64::from(inner), f64::from(outer))),
        ring_half_thickness_mm,
        outer_draw_extent_mm: f64::from(target.outer_draw_radius_mm()),
    }
}

fn canvas_layout(target: &TargetLayout, outer_draw_extent_mm: f64, margin_mm: f64) -> CanvasLayout {
    let (min_xy, max_xy) = target
        .marker_bounds_mm()
        .expect("target layouts are never empty");
    let span_x = f64::from(max_xy[0] - min_xy[0]);
    let span_y = f64::from(max_xy[1] - min_xy[1]);
    let max_span_mm = span_x.max(span_y);
    let side_mm = max_span_mm + 2.0 * outer_draw_extent_mm;
    let offset_x_mm =
        margin_mm + outer_draw_extent_mm + 0.5 * (max_span_mm - span_x) - f64::from(min_xy[0]);
    let offset_y_mm =
        margin_mm + outer_draw_extent_mm + 0.5 * (max_span_mm - span_y) - f64::from(min_xy[1]);

    CanvasLayout {
        side_mm,
        canvas_side_mm: side_mm + 2.0 * margin_mm,
        offset_x_mm,
        offset_y_mm,
    }
}

fn scale_bar_params(
    target: &TargetLayout,
    canvas: CanvasLayout,
    outer_draw_extent_mm: f64,
) -> ScaleBarParams {
    let inset_x_mm = (0.5 * f64::from(target.pitch_mm())).max(2.0);
    let inset_y_mm = (0.25 * f64::from(target.pitch_mm())).max(1.0);

    let marker_bottom_mm = target
        .cells()
        .iter()
        .map(|cell| f64::from(cell.xy_mm[1]) + canvas.offset_y_mm)
        .fold(f64::NEG_INFINITY, f64::max)
        + outer_draw_extent_mm;

    let mut bar_h_mm = (0.4 * f64::from(target.pitch_mm())).clamp(2.0, 4.0);
    let clearance_mm = (0.2 * bar_h_mm).max(0.5);
    let available_mm =
        canvas.canvas_side_mm - inset_y_mm - bar_h_mm - (marker_bottom_mm + clearance_mm);
    if available_mm < 0.0 {
        bar_h_mm = (canvas.canvas_side_mm - inset_y_mm - (marker_bottom_mm + clearance_mm))
            .clamp(1.0, 4.0);
    }

    let usable_w_mm = (canvas.side_mm - 2.0 * inset_x_mm).max(1.0);
    let target_len_mm = (0.5 * usable_w_mm).min(100.0);
    let mut bar_len_mm = (target_len_mm / 10.0).round() as i32 * 10;
    bar_len_mm = bar_len_mm.max(10);
    while f64::from(bar_len_mm) > usable_w_mm && bar_len_mm >= 10 {
        bar_len_mm -= 10;
    }
    bar_len_mm = bar_len_mm.max(10);

    let tick_step_mm = if bar_len_mm >= 50 {
        10
    } else if bar_len_mm >= 20 {
        5
    } else {
        1
    };

    ScaleBarParams {
        x0_mm: 0.5 * (canvas.canvas_side_mm - canvas.side_mm) + inset_x_mm,
        y0_mm: canvas.canvas_side_mm - inset_y_mm - bar_h_mm,
        bar_len_mm,
        bar_h_mm,
        tick_step_mm,
        tick_w_mm: (0.08 * bar_h_mm).max(0.2),
        font_size_mm: (0.7 * bar_h_mm).max(1.2),
    }
}

fn append_scale_bar_svg(
    lines: &mut Vec<String>,
    target: &TargetLayout,
    canvas: CanvasLayout,
    geometry: RenderGeometry,
) {
    let params = scale_bar_params(target, canvas, geometry.outer_draw_extent_mm);
    let label = params.label();

    lines.push("<g id=\"scale_bar\">".to_string());
    lines.push(format!(
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"black\"/>",
        svg_fmt(params.x0_mm),
        svg_fmt(params.y0_mm),
        svg_fmt(f64::from(params.bar_len_mm)),
        svg_fmt(params.bar_h_mm)
    ));

    let n_ticks = ((f64::from(params.bar_len_mm) / f64::from(params.tick_step_mm)).round()) as i32;
    for i in 0..=n_ticks {
        let tx = params.x0_mm + f64::from(i * params.tick_step_mm);
        lines.push(format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"white\" stroke-width=\"{}\"/>",
            svg_fmt(tx),
            svg_fmt(params.y0_mm),
            svg_fmt(tx),
            svg_fmt(params.y0_mm + params.bar_h_mm),
            svg_fmt(params.tick_w_mm)
        ));
    }

    lines.push(format!(
        "<text x=\"{}\" y=\"{}\" fill=\"white\" font-size=\"{}\" font-family=\"monospace\" text-anchor=\"middle\">{}</text>",
        svg_fmt(params.x0_mm + f64::from(params.bar_len_mm) / 2.0),
        svg_fmt(params.y0_mm + 0.75 * params.bar_h_mm),
        svg_fmt(params.font_size_mm),
        label
    ));
    lines.push("</g>".to_string());
}

fn draw_scale_bar_raster(
    image: &mut GrayImage,
    target: &TargetLayout,
    canvas: CanvasLayout,
    outer_draw_extent_mm: f64,
    pixels_per_mm: f64,
) {
    let params = scale_bar_params(target, canvas, outer_draw_extent_mm);
    let x0 = (params.x0_mm * pixels_per_mm).round() as i32;
    let y0 = (params.y0_mm * pixels_per_mm).round() as i32;
    let bar_w = (f64::from(params.bar_len_mm) * pixels_per_mm).round() as i32;
    let bar_h = (params.bar_h_mm * pixels_per_mm).round() as i32;
    let tick_step = (f64::from(params.tick_step_mm) * pixels_per_mm).round() as i32;
    let tick_w = ((params.tick_w_mm * pixels_per_mm).round() as i32).max(1);
    let label = params.label();
    let height = image.height() as i32;
    let width = image.width() as i32;

    if bar_w <= 0 || bar_h <= 0 {
        return;
    }

    let x1 = (x0 + bar_w).min(width);
    let y1 = (y0 + bar_h).min(height);
    let x0c = x0.max(0);
    let y0c = y0.max(0);
    if x0c >= x1 || y0c >= y1 {
        return;
    }

    for y in y0c..y1 {
        for x in x0c..x1 {
            image.put_pixel(x as u32, y as u32, Luma([0]));
        }
    }

    if tick_step > 0 {
        let mut tx = x0;
        while tx <= x0 + bar_w {
            for dx in (-(tick_w / 2))..(tick_w - (tick_w / 2)) {
                let col = tx + dx;
                if !(0..width).contains(&col) {
                    continue;
                }
                for y in y0c..y1 {
                    image.put_pixel(col as u32, y as u32, Luma([255]));
                }
            }
            tx += tick_step;
        }
    }

    let desired_text_h = (((bar_h as f64) * 0.7).round() as i32).max(7);
    let scale = (desired_text_h / 7).max(1);
    let text_h = 7 * scale;
    let text_w = (label.chars().count() as i32) * (5 * scale + scale) - scale;
    let text_x = x0 + (bar_w - text_w) / 2;
    let text_y = y0 + (bar_h - text_h) / 2;
    draw_text_5x7_u8(image, text_x, text_y, &label, scale, 255);
}

fn draw_text_5x7_u8(image: &mut GrayImage, x0: i32, y0: i32, text: &str, scale: i32, value: u8) {
    if scale <= 0 {
        return;
    }

    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut cursor_x = x0;

    for ch in text.chars() {
        let glyph = font_5x7(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                if ((bits >> (4 - col)) & 1) == 0 {
                    continue;
                }

                let px0 = cursor_x + col * scale;
                let py0 = y0 + row as i32 * scale;
                let px1 = px0 + scale;
                let py1 = py0 + scale;
                if px1 <= 0 || py1 <= 0 || px0 >= width || py0 >= height {
                    continue;
                }

                let sx0 = px0.max(0);
                let sy0 = py0.max(0);
                let sx1 = px1.min(width);
                let sy1 = py1.min(height);
                for y in sy0..sy1 {
                    for x in sx0..sx1 {
                        image.put_pixel(x as u32, y as u32, Luma([value]));
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

fn font_5x7(ch: char) -> &'static [u8; 7] {
    const ZERO: [u8; 7] = [
        0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
    ];
    const ONE: [u8; 7] = [
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ];
    const TWO: [u8; 7] = [
        0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
    ];
    const THREE: [u8; 7] = [
        0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
    ];
    const FOUR: [u8; 7] = [
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
    ];
    const FIVE: [u8; 7] = [
        0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
    ];
    const SIX: [u8; 7] = [
        0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
    ];
    const SEVEN: [u8; 7] = [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
    ];
    const EIGHT: [u8; 7] = [
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
    ];
    const NINE: [u8; 7] = [
        0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
    ];
    const LOWER_M: [u8; 7] = [
        0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101,
    ];
    const SPACE: [u8; 7] = [0; 7];

    match ch {
        '0' => &ZERO,
        '1' => &ONE,
        '2' => &TWO,
        '3' => &THREE,
        '4' => &FOUR,
        '5' => &FIVE,
        '6' => &SIX,
        '7' => &SEVEN,
        '8' => &EIGHT,
        '9' => &NINE,
        'm' => &LOWER_M,
        ' ' => &SPACE,
        _ => &SPACE,
    }
}

fn svg_fmt(x: f64) -> String {
    let trimmed = format!("{x:.4}");
    trimmed
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn svg_annulus_path(cx: f64, cy: f64, r_outer: f64, r_inner: f64) -> String {
    let ox = cx + r_outer;
    let ix = cx + r_inner;
    format!(
        "M {} {} A {} {} 0 1 1 {} {} A {} {} 0 1 1 {} {} Z M {} {} A {} {} 0 1 0 {} {} A {} {} 0 1 0 {} {} Z",
        svg_fmt(ox),
        svg_fmt(cy),
        svg_fmt(r_outer),
        svg_fmt(r_outer),
        svg_fmt(cx - r_outer),
        svg_fmt(cy),
        svg_fmt(r_outer),
        svg_fmt(r_outer),
        svg_fmt(ox),
        svg_fmt(cy),
        svg_fmt(ix),
        svg_fmt(cy),
        svg_fmt(r_inner),
        svg_fmt(r_inner),
        svg_fmt(cx - r_inner),
        svg_fmt(cy),
        svg_fmt(r_inner),
        svg_fmt(r_inner),
        svg_fmt(ix),
        svg_fmt(cy)
    )
}

fn svg_annular_sector_path(
    cx: f64,
    cy: f64,
    r_outer: f64,
    r_inner: f64,
    angle0_rad: f64,
    angle1_rad: f64,
) -> String {
    let x0o = cx + r_outer * angle0_rad.cos();
    let y0o = cy + r_outer * angle0_rad.sin();
    let x1o = cx + r_outer * angle1_rad.cos();
    let y1o = cy + r_outer * angle1_rad.sin();
    let x1i = cx + r_inner * angle1_rad.cos();
    let y1i = cy + r_inner * angle1_rad.sin();
    let x0i = cx + r_inner * angle0_rad.cos();
    let y0i = cy + r_inner * angle0_rad.sin();

    format!(
        "M {} {} A {} {} 0 0 1 {} {} L {} {} A {} {} 0 0 0 {} {} Z",
        svg_fmt(x0o),
        svg_fmt(y0o),
        svg_fmt(r_outer),
        svg_fmt(r_outer),
        svg_fmt(x1o),
        svg_fmt(y1o),
        svg_fmt(x1i),
        svg_fmt(y1i),
        svg_fmt(r_inner),
        svg_fmt(r_inner),
        svg_fmt(x0i),
        svg_fmt(y0i)
    )
}

fn square(x: f64) -> f64 {
    x * x
}

#[cfg(feature = "std")]
fn dpi_to_pixels_per_meter(dpi: f64) -> u32 {
    (dpi * 1000.0 / MM_PER_INCH).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_svg_margin() {
        let target = TargetLayout::default_hex();
        let err = target
            .render_target_svg(&SvgTargetOptions {
                margin_mm: -1.0,
                ..SvgTargetOptions::default()
            })
            .expect_err("negative margin must fail");
        assert!(matches!(err, TargetGenerationError::InvalidMargin { .. }));
    }

    #[test]
    fn rejects_invalid_png_dpi() {
        let target = TargetLayout::default_hex();
        let err = target
            .render_target_png(&PngTargetOptions {
                dpi: 0.0,
                ..PngTargetOptions::default()
            })
            .expect_err("non-positive dpi must fail");
        assert!(matches!(err, TargetGenerationError::InvalidDpi { .. }));
    }

    #[test]
    fn render_geometry_uses_ring_edges_for_code_band() {
        let board = BoardLayout::with_name("fixture_gap_free", 7.0, 3, 4, 4.8, 2.8, 1.152)
            .expect("valid geometry");
        let target = TargetLayout::from(&board);
        let geometry = render_geometry(&target);

        let half = 0.5 * f64::from(board.marker_ring_width_mm());
        let (code_inner, code_outer) = geometry.code_band_mm.expect("coded band");
        assert!((code_inner - (f64::from(board.marker_inner_radius_mm()) + half)).abs() < 1e-6);
        assert!((code_outer - (f64::from(board.marker_outer_radius_mm()) - half)).abs() < 1e-6);
        assert!(
            (geometry.outer_draw_extent_mm - (f64::from(board.marker_outer_radius_mm()) + half))
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn svg_codewords_follow_id_assignment() {
        // A board whose first marker carries a non-sequential ID must render
        // that ID's codeword (regression: rendering used the cell index).
        let board = BoardLayout::with_name("fixture_compact_hex", 8.0, 3, 4, 4.8, 3.2, 1.152)
            .expect("valid geometry");
        let mut val: serde_json::Value =
            serde_json::from_str(&board.to_json_string()).expect("json");
        let n = board.n_markers();
        // Reverse assignment: cell i gets ID n-1-i.
        let ids: Vec<usize> = (0..n).rev().collect();
        val["id_assignment"] = serde_json::json!(ids);
        let board = BoardLayout::from_json_str(&serde_json::to_string(&val).expect("json"))
            .expect("valid board");

        let svg = board
            .render_target_svg(&SvgTargetOptions::default())
            .expect("render");
        assert!(
            svg.contains(&format!("<g id=\"m0\" data-id=\"{}\">", n - 1)),
            "first cell must carry the assigned (reversed) ID"
        );
    }

    #[test]
    fn plain_target_renders_annuli_and_dots() {
        let target = TargetLayout::isra_rect_24x24();

        let svg = target
            .render_target_svg(&SvgTargetOptions::default())
            .expect("render svg");
        assert!(svg.contains("<g id=\"fiducials\">"));
        assert!(!svg.contains("data-id"), "plain markers carry no IDs");
        assert!(
            !svg.contains("stroke=\"black\""),
            "plain rings are filled, not stroked"
        );

        // Low-DPI raster for speed: check center-line intensity profile.
        let png = target
            .render_target_png(&PngTargetOptions {
                dpi: 25.4, // 1 px per mm
                margin_mm: 0.0,
                include_scale_bar: false,
            })
            .expect("render png");

        // First ring center is at (5.6, 5.6) mm from the page edge.
        let cx = 5.6f64;
        let probe = |dx: f64, dy: f64| {
            png.get_pixel((cx + dx).round() as u32, (cx + dy).round() as u32)
                .0[0]
        };
        assert_eq!(probe(0.0, 0.0), 255, "ring center is white");
        assert_eq!(probe(4.0, 0.0), 0, "annulus interior is black");
        assert_eq!(probe(6.5, 0.0), 255, "outside the ring is white");

        // Fiducial dot at (161, 161) board mm -> page (166.6, 166.6).
        let dot = png.get_pixel(167, 167).0[0];
        assert_eq!(dot, 0, "fiducial dot is black");
    }
}
