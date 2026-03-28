//! Scharr gradient computation and edge thinning.

use image::GrayImage;

/// Compute Scharr gradients and track the maximum squared magnitude.
///
/// Returns `(gx, gy, max_mag_sq)` where gradients are stored in row-major
/// order with the same stride as the image. Border pixels (1px frame) are
/// left as zero.
pub(crate) fn build_scharr_gradients(gray: &GrayImage) -> (Vec<i16>, Vec<i16>, f32) {
    let (w, h) = gray.dimensions();
    let stride = w as usize;
    let height = h as usize;
    let src = gray.as_raw();
    let mut gx = vec![0i16; stride * height];
    let mut gy = vec![0i16; stride * height];
    let mut max_mag_sq = 0.0f32;

    for y in 1..height.saturating_sub(1) {
        let row_above = (y - 1) * stride;
        let row = y * stride;
        let row_below = (y + 1) * stride;

        for x in 1..stride.saturating_sub(1) {
            let idx = row + x;

            let p00 = src[row_above + x - 1] as i32;
            let p01 = src[row_above + x] as i32;
            let p02 = src[row_above + x + 1] as i32;
            let p10 = src[row + x - 1] as i32;
            let p12 = src[row + x + 1] as i32;
            let p20 = src[row_below + x - 1] as i32;
            let p21 = src[row_below + x] as i32;
            let p22 = src[row_below + x + 1] as i32;

            let gxv = (3 * (p02 - p00) + 10 * (p12 - p10) + 3 * (p22 - p20)) as i16;
            let gyv = (3 * (p20 - p00) + 10 * (p21 - p01) + 3 * (p22 - p02)) as i16;

            gx[idx] = gxv;
            gy[idx] = gyv;

            let gxv = gxv as f32;
            let gyv = gyv as f32;
            let mag_sq = gxv * gxv + gyv * gyv;
            if mag_sq > max_mag_sq {
                max_mag_sq = mag_sq;
            }
        }
    }

    (gx, gy, max_mag_sq)
}

/// Apply Canny-style non-maximum suppression along the gradient direction.
///
/// For each interior pixel, checks whether its gradient magnitude is a local
/// maximum along the gradient direction. Non-maximal pixels have their
/// gradients zeroed out, thinning multi-pixel edge bands to single-pixel
/// ridges.
///
/// Uses 4-direction quantization (0/45/90/135 degrees) with integer
/// arithmetic to avoid `atan2` in the hot loop.
///
/// The `max_mag_sq` is updated to reflect the thinned gradients.
pub(crate) fn thin_edges_along_gradient(
    gx: &mut [i16],
    gy: &mut [i16],
    stride: usize,
    height: usize,
) -> f32 {
    // We need a separate magnitude-squared buffer because we suppress
    // in-place, but neighbors of a pixel may not yet have been processed.
    // To avoid allocation, we use a two-row rolling buffer approach.
    // Actually simpler: compute mag_sq on the fly from the original gx/gy
    // (read before any writes in this row). But since we iterate row by row
    // and only look at neighbors, and we modify the current pixel after
    // checking, we need to be careful. The neighbors in the same row at
    // x-1 may already be zeroed. So we need a read-only mag_sq buffer.

    let n = stride * height;
    let mut mag_sq = vec![0i32; n];

    // Build mag_sq in integer space (gx^2 + gy^2 as i32)
    for y in 1..height.saturating_sub(1) {
        let row = y * stride;
        for x in 1..stride.saturating_sub(1) {
            let idx = row + x;
            let gxv = gx[idx] as i32;
            let gyv = gy[idx] as i32;
            mag_sq[idx] = gxv * gxv + gyv * gyv;
        }
    }

    let mut new_max_mag_sq = 0.0f32;

    for y in 2..height.saturating_sub(2) {
        let row = y * stride;
        for x in 2..stride.saturating_sub(2) {
            let idx = row + x;
            let m = mag_sq[idx];
            if m == 0 {
                continue;
            }

            let gxv = gx[idx] as i32;
            let gyv = gy[idx] as i32;
            let ax = gxv.unsigned_abs();
            let ay = gyv.unsigned_abs();

            // 4-direction quantization without atan2:
            //   tan(22.5°) ≈ 0.414, so |gy| < 0.414*|gx| means ~horizontal
            //   Multiply through: 5*|gy| < 2*|gx| (approx 0.4 threshold)
            let (n1, n2) = if 5 * ay < 2 * ax {
                // ~Horizontal: compare left/right
                (mag_sq[idx - 1], mag_sq[idx + 1])
            } else if 5 * ax < 2 * ay {
                // ~Vertical: compare up/down
                (mag_sq[idx - stride], mag_sq[idx + stride])
            } else if (gxv > 0) == (gyv > 0) {
                // ~45° diagonal (top-left to bottom-right)
                (mag_sq[idx - stride - 1], mag_sq[idx + stride + 1])
            } else {
                // ~135° diagonal (top-right to bottom-left)
                (mag_sq[idx - stride + 1], mag_sq[idx + stride - 1])
            };

            if m < n1 || m < n2 {
                // Not a local maximum along gradient direction — suppress
                gx[idx] = 0;
                gy[idx] = 0;
            } else {
                let gxf = gx[idx] as f32;
                let gyf = gy[idx] as f32;
                let ms = gxf * gxf + gyf * gyf;
                if ms > new_max_mag_sq {
                    new_max_mag_sq = ms;
                }
            }
        }
    }

    // Zero out the 2-pixel border that we couldn't process
    // (row 1 and row height-2 may have non-thinned edges)
    for y in [1, height.saturating_sub(2)] {
        let row = y * stride;
        for x in 1..stride.saturating_sub(1) {
            gx[row + x] = 0;
            gy[row + x] = 0;
        }
    }
    for y in 2..height.saturating_sub(2) {
        let row = y * stride;
        for x in [1, stride.saturating_sub(2)] {
            gx[row + x] = 0;
            gy[row + x] = 0;
        }
    }

    new_max_mag_sq
}

/// Intermediate representation of an edge pixel with strong gradient.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StrongEdge {
    pub x: f32,
    pub y: f32,
    pub mag: f32,
    pub dx: f32,
    pub dy: f32,
}

/// Collect interior pixels whose gradient magnitude exceeds `threshold_sq`,
/// returning normalized direction and magnitude for each.
pub(crate) fn collect_strong_edges(
    gx: &[i16],
    gy: &[i16],
    stride: usize,
    height: usize,
    threshold_sq: f32,
) -> Vec<StrongEdge> {
    let mut strong_edges = Vec::new();
    for y in 1..height.saturating_sub(1) {
        let row = y * stride;
        let yf = y as f32;
        for x in 1..stride.saturating_sub(1) {
            let idx = row + x;
            let gxv = gx[idx] as f32;
            let gyv = gy[idx] as f32;
            let mag_sq = gxv * gxv + gyv * gyv;
            if mag_sq < threshold_sq {
                continue;
            }

            let mag = mag_sq.sqrt();
            let inv_mag = 1.0 / mag;
            strong_edges.push(StrongEdge {
                x: x as f32,
                y: yf,
                mag,
                dx: gxv * inv_mag,
                dy: gyv * inv_mag,
            });
        }
    }
    strong_edges
}
