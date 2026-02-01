//! Gradient-voting radial symmetry for candidate center detection.
//!
//! For each pixel with a strong gradient, votes are cast along the gradient
//! direction at distances in [r_min, r_max]. Ring markers produce peaks in
//! the accumulator at their centers because gradient vectors from the ring
//! boundaries converge radially.

use image::GrayImage;

/// Configuration for center proposal detection.
#[derive(Debug, Clone)]
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
        }
    }
}

/// A proposed marker center with its vote score.
#[derive(Debug, Clone, Copy)]
pub struct Proposal {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// Deposit a weighted vote into the accumulator using bilinear interpolation.
#[inline]
fn bilinear_add(accum: &mut [f32], w: u32, x: f32, y: f32, weight: f32) {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    if x0 + 1 >= w {
        return;
    }
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let stride = w as usize;
    let base = y0 as usize * stride + x0 as usize;
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

    // Compute Scharr gradients (i16 output)
    let gx = imageproc::gradients::horizontal_scharr(gray);
    let gy = imageproc::gradients::vertical_scharr(gray);

    // Find max gradient magnitude for thresholding
    let mut max_mag_sq: f32 = 0.0;
    for y in 0..h {
        for x in 0..w {
            let gxv = gx.get_pixel(x, y)[0] as f32;
            let gyv = gy.get_pixel(x, y)[0] as f32;
            let mag_sq = gxv * gxv + gyv * gyv;
            if mag_sq > max_mag_sq {
                max_mag_sq = mag_sq;
            }
        }
    }
    let max_mag = max_mag_sq.sqrt();
    if max_mag < 1e-6 {
        return Vec::new();
    }
    let threshold = config.grad_threshold * max_mag;

    // Vote accumulation
    let n = (w * h) as usize;
    let mut accum = vec![0.0f32; n];

    for y in 0..h {
        for x in 0..w {
            let gxv = gx.get_pixel(x, y)[0] as f32;
            let gyv = gy.get_pixel(x, y)[0] as f32;
            let mag = (gxv * gxv + gyv * gyv).sqrt();
            if mag < threshold {
                continue;
            }

            // Normalized gradient direction
            let dx = gxv / mag;
            let dy = gyv / mag;

            // Vote along +gradient and -gradient directions
            for &sign in &[-1.0f32, 1.0] {
                let mut r = config.r_min;
                while r <= config.r_max {
                    let vx = x as f32 + sign * dx * r;
                    let vy = y as f32 + sign * dy * r;
                    if vx >= 0.0 && vx < (w - 1) as f32 && vy >= 0.0 && vy < (h - 1) as f32 {
                        bilinear_add(&mut accum, w, vx, vy, mag);
                    }
                    r += 1.0;
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

    // Non-maximum suppression
    let mut proposals = Vec::new();
    for y in nms_r..(h as i32 - nms_r) {
        for x in nms_r..(w as i32 - nms_r) {
            let idx = y as usize * w as usize + x as usize;
            let val = smoothed_data[idx];
            if val < vote_threshold {
                continue;
            }
            // Check if local maximum within nms_radius
            let mut is_max = true;
            'outer: for dy in -nms_r..=nms_r {
                for dx in -nms_r..=nms_r {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if (dx * dx + dy * dy) as f32 > config.nms_radius * config.nms_radius {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    let nidx = ny as usize * w as usize + nx as usize;
                    if smoothed_data[nidx] > val || (smoothed_data[nidx] == val && nidx < idx) {
                        is_max = false;
                        break 'outer;
                    }
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
