//! Gradient-voting radial symmetry for candidate center detection.
//!
//! For each pixel with a strong gradient, votes are cast along the gradient
//! direction at distances in [r_min, r_max]. Ring markers produce peaks in
//! the accumulator at their centers because gradient vectors from the ring
//! boundaries converge radially.

use image::GrayImage;

/// Configuration for center proposal detection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProposalConfig {
    /// Minimum voting radius (pixels).
    pub r_min: f32,
    /// Maximum voting radius (pixels).
    pub r_max: f32,
    /// Gradient magnitude threshold (fraction of max gradient).
    pub grad_threshold: f32,
    /// NMS radius for peak extraction (pixels).
    pub nms_radius: f32,
    /// Minimum accumulator value for a proposal (fraction of max).
    pub min_vote_frac: f32,
    /// Gaussian sigma for accumulator smoothing.
    pub accum_sigma: f32,
    /// Optional cap on number of proposals returned (after score sorting).
    #[serde(default)]
    pub max_candidates: Option<usize>,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            r_min: 3.0,
            r_max: 12.0,
            grad_threshold: 0.05,
            nms_radius: 7.0,
            min_vote_frac: 0.1,
            accum_sigma: 2.0,
            max_candidates: None,
        }
    }
}

/// A proposed marker center with its vote score.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Proposal {
    /// X coordinate (pixels).
    pub x: f32,
    /// Y coordinate (pixels).
    pub y: f32,
    /// Accumulator peak score.
    pub score: f32,
}

/// Deposit a weighted vote into the accumulator using bilinear interpolation.
#[inline]
fn bilinear_add_in_bounds(accum: &mut [f32], stride: usize, x: f32, y: f32, weight: f32) {
    let x0 = x as usize;
    let y0 = y as usize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let base = y0 * stride + x0;
    accum[base] += weight * (1.0 - fx) * (1.0 - fy);
    accum[base + 1] += weight * fx * (1.0 - fy);
    accum[base + stride] += weight * (1.0 - fx) * fy;
    accum[base + stride + 1] += weight * fx * fy;
}

/// Detect candidate marker centers via gradient-based radial symmetry voting.
///
/// Returns proposals sorted by score (highest first).
pub fn find_proposals(gray: &GrayImage, config: &ProposalConfig) -> Vec<Proposal> {
    let (w, h) = gray.dimensions();
    if w < 4 || h < 4 {
        return Vec::new();
    }
    if config.r_max < config.r_min {
        return Vec::new();
    }

    // Compute Scharr gradients (i16 output)
    let gx = imageproc::gradients::horizontal_scharr(gray);
    let gy = imageproc::gradients::vertical_scharr(gray);
    let gx_raw = gx.as_raw();
    let gy_raw = gy.as_raw();

    // Find max gradient magnitude for thresholding
    let mut max_mag_sq: f32 = 0.0;
    for (&gxv, &gyv) in gx_raw.iter().zip(gy_raw.iter()) {
        let gxv = gxv as f32;
        let gyv = gyv as f32;
        let mag_sq = gxv * gxv + gyv * gyv;
        if mag_sq > max_mag_sq {
            max_mag_sq = mag_sq;
        }
    }
    let max_mag = max_mag_sq.sqrt();
    if max_mag < 1e-6 {
        return Vec::new();
    }
    let threshold = config.grad_threshold * max_mag;
    let threshold_sq = threshold * threshold;

    // Vote accumulation
    let stride = w as usize;
    let h_usize = h as usize;
    let n = stride * h_usize;
    let mut accum = vec![0.0f32; n];
    let mut radii = Vec::new();
    let mut r = config.r_min;
    while r <= config.r_max {
        radii.push(r);
        r += 1.0;
    }
    if radii.is_empty() {
        return Vec::new();
    }
    let x_limit = (w - 1) as f32;
    let y_limit = (h - 1) as f32;

    for y in 0..h_usize {
        let y_base = y * stride;
        let yf = y as f32;
        for x in 0..stride {
            let idx = y_base + x;
            let gxv = gx_raw[idx] as f32;
            let gyv = gy_raw[idx] as f32;
            let mag_sq = gxv * gxv + gyv * gyv;
            if mag_sq < threshold_sq {
                continue;
            }

            let mag = mag_sq.sqrt();
            // Normalized gradient direction
            let inv_mag = 1.0 / mag;
            let dx = gxv * inv_mag;
            let dy = gyv * inv_mag;
            let xf = x as f32;

            // Vote along +gradient and -gradient directions
            for &r in &radii {
                let vx_pos = xf + dx * r;
                let vy_pos = yf + dy * r;
                if vx_pos >= 0.0 && vx_pos < x_limit && vy_pos >= 0.0 && vy_pos < y_limit {
                    bilinear_add_in_bounds(&mut accum, stride, vx_pos, vy_pos, mag);
                }

                let vx_neg = xf - dx * r;
                let vy_neg = yf - dy * r;
                if vx_neg >= 0.0 && vx_neg < x_limit && vy_neg >= 0.0 && vy_neg < y_limit {
                    bilinear_add_in_bounds(&mut accum, stride, vx_neg, vy_neg, mag);
                }
            }
        }
    }

    // Smooth accumulator with Gaussian blur
    // Convert to ImageBuffer<Luma<f32>> for imageproc
    let accum_img = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(w, h, accum)
        .expect("accumulator dimensions match");
    let smoothed = imageproc::filter::gaussian_blur_f32(&accum_img, config.accum_sigma);

    // Find max in smoothed accumulator
    let smoothed_data = smoothed.as_raw();
    let max_val = smoothed_data.iter().cloned().fold(0.0f32, f32::max);
    if max_val < 1e-6 {
        return Vec::new();
    }
    let vote_threshold = config.min_vote_frac * max_val;
    let nms_r = config.nms_radius.ceil() as i32;
    let nms_r_sq = config.nms_radius * config.nms_radius;
    let mut nms_offsets = Vec::new();
    for dy in -nms_r..=nms_r {
        for dx in -nms_r..=nms_r {
            if dx == 0 && dy == 0 {
                continue;
            }
            if (dx * dx + dy * dy) as f32 > nms_r_sq {
                continue;
            }
            nms_offsets.push(dy as isize * stride as isize + dx as isize);
        }
    }

    // Non-maximum suppression
    let mut proposals = Vec::new();
    for y in nms_r..(h as i32 - nms_r) {
        for x in nms_r..(w as i32 - nms_r) {
            let idx = y as usize * stride + x as usize;
            let val = smoothed_data[idx];
            if val < vote_threshold {
                continue;
            }
            // Check if local maximum within nms_radius
            let mut is_max = true;
            for &off in &nms_offsets {
                let nidx = idx.wrapping_add_signed(off);
                if smoothed_data[nidx] > val || (smoothed_data[nidx] == val && nidx < idx) {
                    is_max = false;
                    break;
                }
            }
            if is_max {
                proposals.push(Proposal {
                    x: x as f32,
                    y: y as f32,
                    score: val,
                });
            }
        }
    }

    // Sort by score descending
    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    if let Some(max_candidates) = config.max_candidates {
        proposals.truncate(max_candidates.min(proposals.len()));
    }
    proposals
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a small synthetic image with a dark ring on a bright background.
    fn make_ring_image(
        w: u32,
        h: u32,
        cx: f32,
        cy: f32,
        radius: f32,
        ring_width: f32,
    ) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = if (dist - radius).abs() < ring_width {
                    30u8 // dark ring
                } else {
                    200u8 // bright background
                };
                img.put_pixel(x, y, image::Luma([val]));
            }
        }
        img
    }

    #[test]
    fn test_ring_proposal_finds_center() {
        let cx = 40.0f32;
        let cy = 40.0f32;
        let img = make_ring_image(80, 80, cx, cy, 10.0, 3.0);

        let config = ProposalConfig {
            r_min: 5.0,
            r_max: 15.0,
            grad_threshold: 0.03,
            nms_radius: 5.0,
            min_vote_frac: 0.05,
            accum_sigma: 1.5,
            max_candidates: None,
        };

        let proposals = find_proposals(&img, &config);
        assert!(!proposals.is_empty(), "should find at least one proposal");

        // Best proposal should be near the true center
        let best = &proposals[0];
        let err = ((best.x - cx).powi(2) + (best.y - cy).powi(2)).sqrt();
        assert!(
            err < 5.0,
            "best proposal ({}, {}) should be within 5 px of true center ({}, {}), error = {}",
            best.x,
            best.y,
            cx,
            cy,
            err
        );
    }
}
