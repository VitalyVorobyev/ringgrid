//! Shared test utilities for image-based unit tests.
//!
//! Consolidated here to avoid the three identical copies of `blur_gray` (in
//! `ring/inner_estimate.rs`, `ring/outer_estimate.rs`, `ring/radial_estimator.rs`)
//! and the five near-identical copies of `draw_ring_image` across the codebase.

use image::{GrayImage, Luma};

/// Render a synthetic annular ring image.
///
/// Pixels at distance `d` from `center` satisfy:
/// - `ring_pix`  if `inner_radius <= d <= outer_radius`
/// - `bg_pix`    otherwise
pub(crate) fn draw_ring_image(
    w: u32,
    h: u32,
    center: [f32; 2],
    outer_radius: f32,
    inner_radius: f32,
    ring_pix: u8,
    bg_pix: u8,
) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - center[0];
            let dy = y as f32 - center[1];
            let d = (dx * dx + dy * dy).sqrt();
            let pix = if d >= inner_radius && d <= outer_radius {
                ring_pix
            } else {
                bg_pix
            };
            img.put_pixel(x, y, Luma([pix]));
        }
    }
    img
}

/// Gaussian-blur a `GrayImage` via `imageproc`.
pub(crate) fn blur_gray(img: &GrayImage, sigma: f32) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut f = image::ImageBuffer::<Luma<f32>, Vec<f32>>::new(w, h);
    for y in 0..h {
        for x in 0..w {
            f.put_pixel(x, y, Luma([img.get_pixel(x, y)[0] as f32 / 255.0]));
        }
    }
    let blurred = imageproc::filter::gaussian_blur_f32(&f, sigma);
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = blurred.get_pixel(x, y)[0].clamp(0.0, 1.0);
            out.put_pixel(x, y, Luma([(v * 255.0).round() as u8]));
        }
    }
    out
}
