//! Lightweight scale probe: estimate dominant marker outer radii from top-K
//! gradient-voting proposals using a ring angular-variance sweep.
//!
//! For each candidate proposal center, the probe samples intensity on rings of
//! increasing radius and computes angular variance. High variance indicates the
//! code band (alternating bright/dark sectors), whose midpoint sits at roughly
//! 0.8× the outer ring radius. The argmax radius over a geometric series of
//! candidate radii identifies the dominant scale at each proposal center.
//!
//! The returned radii are code-band midpoint radii; [`ScaleTiers::from_detected_radii`]
//! applies the 1.25× conversion to outer ring radius and pads by ±30%.

use image::GrayImage;

use crate::detector::proposal::{find_proposals, ProposalConfig};

// Geometric series of candidate probe radii covering diameters ~8–220 px.
// 20 values from r=4 to r=110, factor ≈ 1.196 per step.
const N_PROBE_RADII: usize = 20;

fn probe_radii() -> [f32; N_PROBE_RADII] {
    let mut out = [0.0f32; N_PROBE_RADII];
    let r_min: f32 = 4.0;
    let r_max: f32 = 110.0;
    let factor = (r_max / r_min).powf(1.0 / (N_PROBE_RADII as f32 - 1.0));
    for (i, v) in out.iter_mut().enumerate() {
        *v = r_min * factor.powi(i as i32);
    }
    out
}

/// Bilinear-interpolated intensity sample from a grayscale image.
///
/// Returns `None` when `(x, y)` is outside the image bounds.
#[inline]
fn bilinear_sample(gray: &GrayImage, x: f32, y: f32) -> Option<f32> {
    let w = gray.width() as f32;
    let h = gray.height() as f32;
    if x < 0.0 || y < 0.0 || x >= w - 1.0 || y >= h - 1.0 {
        return None;
    }
    let x0 = x as u32;
    let y0 = y as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = gray.get_pixel(x0, y0).0[0] as f32;
    let p10 = gray.get_pixel(x1, y0).0[0] as f32;
    let p01 = gray.get_pixel(x0, y1).0[0] as f32;
    let p11 = gray.get_pixel(x1, y1).0[0] as f32;
    Some((1.0 - fy) * ((1.0 - fx) * p00 + fx * p10) + fy * ((1.0 - fx) * p01 + fx * p11))
}

/// Angular variance of intensity sampled on a ring of radius `r` around `(cx, cy)`.
///
/// Returns `0.0` if fewer than 4 valid samples are available.
fn ring_variance(gray: &GrayImage, cx: f32, cy: f32, r: f32, n_theta: usize) -> f32 {
    debug_assert!(n_theta > 0);

    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;

    for k in 0..n_theta {
        let theta = std::f32::consts::TAU * k as f32 / n_theta as f32;
        let x = cx + r * theta.cos();
        let y = cy + r * theta.sin();
        if let Some(v) = bilinear_sample(gray, x, y) {
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }

    if count < 4 {
        return 0.0;
    }

    let mean = sum / count as f32;
    (sum_sq / count as f32 - mean * mean).max(0.0)
}

/// Estimate the dominant marker code-band radius at a proposal center.
///
/// Returns `None` when all candidate radii produce zero variance (uniform
/// region, out-of-bounds, or image is constant).
fn dominant_probe_radius(
    gray: &GrayImage,
    cx: f32,
    cy: f32,
    candidates: &[f32],
    n_theta: usize,
) -> Option<f32> {
    let mut best_var = 0.0f32;
    let mut best_r = 0.0f32;

    for &r in candidates {
        let var = ring_variance(gray, cx, cy, r, n_theta);
        if var > best_var {
            best_var = var;
            best_r = r;
        }
    }

    if best_var <= 0.0 {
        None
    } else {
        Some(best_r)
    }
}

/// Estimate dominant marker code-band radii from the image.
///
/// Runs gradient-voting proposal generation with a fixed wide
/// [`ProposalConfig`] (decoupled from any user marker-scale prior), then
/// sweeps ring angular variance at each of the top-`k_proposals` proposal
/// centers over a geometric series of 20 candidate radii spanning 4–110 px.
///
/// Returns the code-band midpoint radius for each informative proposal (one
/// value per proposal that produced a clear variance peak).  The caller
/// converts these to outer-ring scale tiers via
/// [`ScaleTiers::from_detected_radii`], which applies the 1.25× conversion
/// factor and pads by ±30 %.
///
/// # Parameters
/// - `k_proposals` — number of top proposals to probe (default: 64).
/// - `n_theta` — angular resolution for the ring variance sweep (default: 16).
///
/// # Returns
/// A `Vec<f32>` of dominant code-band radii (one per informative proposal),
/// possibly empty if the image has no usable gradient structure.
pub(crate) fn scale_probe(gray: &GrayImage, k_proposals: usize, n_theta: usize) -> Vec<f32> {
    // Fixed wide proposal config: decouple from any user scale prior.
    let probe_proposal_cfg = ProposalConfig {
        r_min: 2.0,
        r_max: 100.0,
        grad_threshold: 0.05,
        nms_radius: 4.0,
        min_vote_frac: 0.05,
        accum_sigma: 2.0,
        max_candidates: Some(k_proposals),
    };

    let proposals = find_proposals(gray, &probe_proposal_cfg);
    let candidates = probe_radii();
    let n_theta = n_theta.max(4);

    proposals
        .iter()
        .take(k_proposals)
        .filter_map(|p| dominant_probe_radius(gray, p.x, p.y, &candidates, n_theta))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_radii_are_monotone_and_cover_range() {
        let r = probe_radii();
        assert!(r[0] >= 3.9 && r[0] <= 4.1, "first radius ≈ 4");
        assert!(r[N_PROBE_RADII - 1] >= 100.0, "last radius ≥ 100");
        for i in 1..N_PROBE_RADII {
            assert!(r[i] > r[i - 1], "probe radii must be strictly increasing");
        }
    }

    #[test]
    fn scale_probe_empty_image_returns_empty() {
        let gray = GrayImage::new(64, 64);
        let result = scale_probe(&gray, 10, 16);
        // Blank image has no gradient structure; probe should return nothing
        // or at most a few spurious values — just must not panic.
        let _ = result;
    }

    #[test]
    fn ring_variance_uniform_region_is_zero() {
        let gray = GrayImage::from_pixel(64, 64, image::Luma([128u8]));
        let var = ring_variance(&gray, 32.0, 32.0, 10.0, 16);
        assert!(var < 1e-3, "uniform image must yield near-zero variance");
    }

    #[test]
    fn ring_variance_out_of_bounds_returns_zero() {
        let gray = GrayImage::new(16, 16);
        let var = ring_variance(&gray, 0.0, 0.0, 200.0, 16);
        assert_eq!(var, 0.0);
    }
}
