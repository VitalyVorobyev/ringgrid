//! Radial symmetry center detection for ellipse/circle proposals.
//!
//! This module detects candidate ellipse/circle centers in grayscale images
//! using radsym's Radial Symmetry Detector (RSD). For each pixel with a strong
//! gradient, magnitude-weighted votes are cast along the gradient direction at
//! discrete radii. Elliptical features produce peaks in the accumulator at
//! their centers because gradient vectors from the boundary converge radially.
//!
//! # Standalone usage
//!
//! The module is self-contained and has no dependencies on ringgrid-specific
//! types (board layouts, marker specs, etc.).
//!
//! ```no_run
//! use ringgrid::proposal::{find_ellipse_centers, ProposalConfig};
//!
//! let image = image::open("photo.png").unwrap().to_luma8();
//! let config = ProposalConfig {
//!     r_min: 5.0,
//!     r_max: 25.0,
//!     min_distance: 10.0,
//!     ..ProposalConfig::default()
//! };
//! let centers = find_ellipse_centers(&image, &config);
//! for c in &centers {
//!     println!("center at ({:.1}, {:.1}), score={:.1}", c.x, c.y, c.score);
//! }
//! ```

mod config;

pub use config::ProposalConfig;

use image::GrayImage;

// ── Public types ─────────────────────────────────────────────────────────

/// A proposed ellipse center with its vote score.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Proposal {
    /// X coordinate (pixels).
    pub x: f32,
    /// Y coordinate (pixels).
    pub y: f32,
    /// Accumulator peak score.
    pub score: f32,
}

/// Proposals together with the vote heatmap for visualization or custom
/// processing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProposalResult {
    /// Image dimensions `[width, height]`.
    pub image_size: [u32; 2],
    /// Post-smoothing vote accumulator, row-major `[height × width]`.
    pub heatmap: Vec<f32>,
    /// Proposals sorted by score (highest first).
    pub proposals: Vec<Proposal>,
}

// ── Public entry points ──────────────────────────────────────────────────

/// Detect candidate ellipse centers via radial symmetry voting.
///
/// Returns proposals sorted by score (highest first), with all pairs
/// separated by at least [`ProposalConfig::min_distance`] pixels.
pub fn find_ellipse_centers(gray: &GrayImage, config: &ProposalConfig) -> Vec<Proposal> {
    compute_via_radsym(gray, config, false).0
}

/// Detect candidate ellipse centers and return the vote heatmap.
///
/// The heatmap is the post-smoothed RSD accumulator in row-major order
/// with shape `[image_height, image_width]`.
pub fn find_ellipse_centers_with_heatmap(
    gray: &GrayImage,
    config: &ProposalConfig,
) -> ProposalResult {
    let (w, h) = gray.dimensions();
    let n = w as usize * h as usize;
    let (proposals, heatmap) = compute_via_radsym(gray, config, true);
    ProposalResult {
        image_size: [w, h],
        heatmap: heatmap.unwrap_or_else(|| vec![0.0; n]),
        proposals,
    }
}

// ── Internal: radsym RSD adapter ─────────────────────────────────────────

/// Build RSD radii from the continuous `[r_min, r_max]` range.
///
/// Uses every integer radius in the range. RSD is fast enough per-radius
/// that we don't need to cap — the pipeline's `ProposalDownscale` already
/// handles reducing the image and radii for large images.
fn build_radii(r_min: f32, r_max: f32) -> Vec<u32> {
    let lo = (r_min.ceil() as u32).max(1);
    let hi = r_max.floor() as u32;
    if hi < lo {
        return Vec::new();
    }
    (lo..=hi).collect()
}

/// Core implementation: run radsym RSD and convert results.
fn compute_via_radsym(
    gray: &GrayImage,
    config: &ProposalConfig,
    keep_heatmap: bool,
) -> (Vec<Proposal>, Option<Vec<f32>>) {
    let (w, h) = gray.dimensions();
    if w < 4 || h < 4 || config.r_max < config.r_min {
        let heatmap = keep_heatmap.then(|| vec![0.0; w as usize * h as usize]);
        return (Vec::new(), heatmap);
    }

    let radii = build_radii(config.r_min, config.r_max);
    if radii.is_empty() {
        let heatmap = keep_heatmap.then(|| vec![0.0; w as usize * h as usize]);
        return (Vec::new(), heatmap);
    }

    // Convert GrayImage → radsym::ImageView (zero-copy)
    let view = radsym::ImageView::from_slice(gray.as_raw(), w as usize, h as usize)
        .expect("GrayImage dimensions always valid");

    // Compute gradient
    let gradient = match radsym::sobel_gradient(&view) {
        Ok(g) => g,
        Err(_) => {
            let heatmap = keep_heatmap.then(|| vec![0.0; w as usize * h as usize]);
            return (Vec::new(), heatmap);
        }
    };

    // Translate relative gradient threshold to absolute.
    // Scale down by 0.4 to compensate for Sobel vs Scharr magnitude distribution.
    let max_mag = gradient.max_magnitude();
    if max_mag < 1e-6 {
        let heatmap = keep_heatmap.then(|| vec![0.0; w as usize * h as usize]);
        return (Vec::new(), heatmap);
    }
    let abs_threshold = config.grad_threshold * max_mag * 0.4;

    let rsd_config = radsym::RsdConfig {
        radii,
        gradient_threshold: abs_threshold,
        polarity: radsym::Polarity::Both,
        smoothing_factor: 0.5,
    };

    let response = match radsym::rsd_response_fused(&gradient, &rsd_config) {
        Ok(r) => r,
        Err(_) => {
            let heatmap = keep_heatmap.then(|| vec![0.0; w as usize * h as usize]);
            return (Vec::new(), heatmap);
        }
    };

    // Translate min_vote_frac to absolute NMS threshold
    let response_max = response
        .response()
        .data()
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let nms_threshold = config.min_vote_frac * response_max;

    // NMS radius: cap at 10 for efficiency
    let nms_radius = (config.min_distance as usize).clamp(1, 10);

    // Initial budget: generous to allow distance suppression to work properly
    let initial_budget = config.max_candidates.unwrap_or(4096).max(512);

    let nms_config = radsym::NmsConfig {
        radius: nms_radius,
        threshold: nms_threshold,
        max_detections: initial_budget,
    };

    let radsym_proposals =
        radsym::extract_proposals(&response, &nms_config, radsym::Polarity::Both);

    // Apply greedy distance suppression
    let max_detections = config.max_candidates.unwrap_or(initial_budget);
    let suppressed = radsym::suppress_proposals_by_distance(
        &radsym_proposals,
        config.min_distance,
        max_detections,
    );

    // Convert radsym::Proposal → ringgrid::Proposal
    let proposals: Vec<Proposal> = suppressed
        .iter()
        .map(|p| Proposal {
            x: p.seed.position.x,
            y: p.seed.position.y,
            score: p.seed.score,
        })
        .collect();

    // Extract heatmap if requested
    let heatmap = if keep_heatmap {
        Some(response.into_response().into_data())
    } else {
        drop(response);
        None
    };

    (proposals, heatmap)
}

#[cfg(test)]
mod tests;
