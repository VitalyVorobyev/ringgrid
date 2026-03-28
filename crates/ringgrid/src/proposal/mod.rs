//! Gradient-voting radial symmetry for ellipse center detection.
//!
//! This module detects candidate ellipse/circle centers in grayscale images
//! using a radial symmetry voting scheme. For each pixel with a strong
//! gradient, votes are cast along the gradient direction at distances in
//! `[r_min, r_max]`. Elliptical features produce peaks in the accumulator
//! at their centers because gradient vectors from the boundary converge
//! radially.
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
pub(crate) mod gradient;
mod nms;
mod voting;

pub use config::ProposalConfig;

use image::{GrayImage, ImageBuffer, Luma};
use std::time::{Duration, Instant};

use gradient::{build_scharr_gradients, collect_strong_edges};
use nms::{extract_proposals_from_smoothed, suppress_proposals_by_distance, truncate_proposals};
use voting::{accumulate_votes_parallel, accumulate_votes_scalar, parallel_vote_chunk_count};

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

/// Detect candidate ellipse centers via gradient-based radial symmetry
/// voting.
///
/// Returns proposals sorted by score (highest first), with all pairs
/// separated by at least [`ProposalConfig::min_distance`] pixels.
pub fn find_ellipse_centers(gray: &GrayImage, config: &ProposalConfig) -> Vec<Proposal> {
    compute_proposals(gray, config, OutputMode::ProposalsOnly, VotingMode::Auto).into_proposals()
}

/// Detect candidate ellipse centers and return the vote heatmap.
///
/// The heatmap is the post-Gaussian-smoothed accumulator in row-major
/// order with shape `[image_height, image_width]`.
pub fn find_ellipse_centers_with_heatmap(
    gray: &GrayImage,
    config: &ProposalConfig,
) -> ProposalResult {
    compute_proposals(gray, config, OutputMode::WithHeatmap, VotingMode::Auto).into_result()
}

// ── Internal ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    ProposalsOnly,
    WithHeatmap,
}

impl OutputMode {
    fn keeps_accumulator(self) -> bool {
        matches!(self, Self::WithHeatmap)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VotingMode {
    Auto,
    #[cfg(test)]
    Scalar,
    #[cfg(test)]
    Parallel,
}

struct CoreOutput {
    image_size: [u32; 2],
    proposals: Vec<Proposal>,
    heatmap: Option<Vec<f32>>,
}

impl CoreOutput {
    fn empty(image_size: [u32; 2], output_mode: OutputMode) -> Self {
        let n = image_size[0] as usize * image_size[1] as usize;
        let heatmap = output_mode.keeps_accumulator().then(|| vec![0.0; n]);
        Self {
            image_size,
            proposals: Vec::new(),
            heatmap,
        }
    }

    fn into_proposals(self) -> Vec<Proposal> {
        self.proposals
    }

    fn into_result(self) -> ProposalResult {
        let n = self.image_size[0] as usize * self.image_size[1] as usize;
        ProposalResult {
            image_size: self.image_size,
            heatmap: self.heatmap.unwrap_or_else(|| vec![0.0; n]),
            proposals: self.proposals,
        }
    }
}

#[derive(Debug, Default)]
struct TimingStats {
    gradient: Duration,
    #[allow(dead_code)]
    max_scan: Duration,
    compact: Duration,
    vote: Duration,
    blur: Duration,
    nms: Duration,
    strong_edge_count: usize,
    radius_steps: usize,
    estimated_vote_work: u64,
}

#[inline]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn radius_steps(config: &ProposalConfig) -> usize {
    if !config.r_min.is_finite() || !config.r_max.is_finite() {
        return 0;
    }

    let mut count = 0usize;
    let mut r = config.r_min;
    while r <= config.r_max {
        count += 1;
        r += 1.0;
    }
    count
}

fn compute_proposals(
    gray: &GrayImage,
    config: &ProposalConfig,
    output_mode: OutputMode,
    voting_mode: VotingMode,
) -> CoreOutput {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    if w < 4 || h < 4 || config.r_max < config.r_min {
        return CoreOutput::empty(image_size, output_mode);
    }

    let mut timing = TimingStats::default();
    let gradient_start = Instant::now();
    let (mut gx, mut gy, max_mag_sq) = build_scharr_gradients(gray);
    timing.gradient = gradient_start.elapsed();

    let mut voting_mode_label = "scalar";
    let stride = w as usize;
    let height = h as usize;

    // Optional Canny-style edge thinning: suppress non-maximal gradient
    // pixels along the gradient direction, reducing multi-pixel edge bands
    // to single-pixel ridges. This typically cuts strong-edge count by
    // 60-80% and proportionally reduces the voting workload.
    let max_mag_sq = if config.edge_thinning {
        let thinning_start = Instant::now();
        let thinned_max = gradient::thin_edges_along_gradient(&mut gx, &mut gy, stride, height);
        let thinning_elapsed = thinning_start.elapsed();
        tracing::debug!(
            thinning_ms = duration_ms(thinning_elapsed),
            "edge thinning applied"
        );
        thinned_max
    } else {
        max_mag_sq
    };

    let max_scan_start = Instant::now();
    if max_mag_sq < 1e-6 {
        return CoreOutput::empty(image_size, output_mode);
    }
    let max_mag = max_mag_sq.sqrt();
    let threshold = config.grad_threshold * max_mag;
    let threshold_sq = threshold * threshold;
    timing.max_scan = max_scan_start.elapsed();

    let compact_start = Instant::now();
    let strong_edges = collect_strong_edges(&gx, &gy, stride, height, threshold_sq);
    timing.compact = compact_start.elapsed();
    timing.strong_edge_count = strong_edges.len();

    let radius_steps = radius_steps(config);
    timing.radius_steps = radius_steps;
    let estimated_vote_work = strong_edges.len() as u64 * radius_steps as u64 * 2;
    timing.estimated_vote_work = estimated_vote_work;

    let chunk_count = parallel_vote_chunk_count(estimated_vote_work, strong_edges.len());
    let mut parallel_chunks = chunk_count;

    let vote_start = Instant::now();
    let accum = match voting_mode {
        VotingMode::Auto => {
            if chunk_count > 1 {
                voting_mode_label = "parallel";
                parallel_chunks = chunk_count;
                accumulate_votes_parallel(
                    &strong_edges,
                    stride,
                    height,
                    config.r_min,
                    radius_steps,
                    chunk_count,
                )
            } else {
                accumulate_votes_scalar(&strong_edges, stride, height, config.r_min, radius_steps)
            }
        }
        #[cfg(test)]
        VotingMode::Scalar => {
            accumulate_votes_scalar(&strong_edges, stride, height, config.r_min, radius_steps)
        }
        #[cfg(test)]
        VotingMode::Parallel => {
            let forced_chunks = chunk_count.max(2);
            voting_mode_label = "parallel";
            parallel_chunks = forced_chunks;
            accumulate_votes_parallel(
                &strong_edges,
                stride,
                height,
                config.r_min,
                radius_steps,
                forced_chunks,
            )
        }
    };
    timing.vote = vote_start.elapsed();

    let blur_start = Instant::now();
    let accum_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(w, h, accum)
        .expect("accumulator dimensions match");
    let smoothed = imageproc::filter::gaussian_blur_f32(&accum_img, config.accum_sigma);
    timing.blur = blur_start.elapsed();

    let nms_start = Instant::now();
    let mut proposals = extract_proposals_from_smoothed(smoothed.as_raw(), w, h, config);

    // Apply greedy distance suppression if min_distance exceeds the NMS radius
    if config.min_distance > config.nms_radius() {
        proposals = suppress_proposals_by_distance(&proposals, config.min_distance);
    }
    truncate_proposals(&mut proposals, config.max_candidates);

    timing.nms = nms_start.elapsed();
    log_timing(&timing, proposals.len(), voting_mode_label, parallel_chunks);

    CoreOutput {
        image_size,
        proposals,
        heatmap: if output_mode.keeps_accumulator() {
            Some(smoothed.into_raw())
        } else {
            None
        },
    }
}

fn log_timing(
    stats: &TimingStats,
    proposals: usize,
    voting_mode: &'static str,
    parallel_chunks: usize,
) {
    tracing::debug!(
        gradient_mode = "fused_scharr_with_max",
        voting_mode,
        parallel_chunks,
        gradient_ms = duration_ms(stats.gradient),
        compact_ms = duration_ms(stats.compact),
        vote_ms = duration_ms(stats.vote),
        blur_ms = duration_ms(stats.blur),
        nms_ms = duration_ms(stats.nms),
        strong_edge_count = stats.strong_edge_count,
        radius_steps = stats.radius_steps,
        estimated_vote_work = stats.estimated_vote_work,
        proposals,
        "proposal timing summary"
    );
}

// ── Test-only helpers ────────────────────────────────────────────────────

#[cfg(test)]
fn find_proposals_with_mode(
    gray: &GrayImage,
    config: &ProposalConfig,
    voting_mode: VotingMode,
) -> Vec<Proposal> {
    compute_proposals(gray, config, OutputMode::ProposalsOnly, voting_mode).into_proposals()
}

#[cfg(test)]
fn find_proposals_result_with_mode(
    gray: &GrayImage,
    config: &ProposalConfig,
    voting_mode: VotingMode,
) -> ProposalResult {
    compute_proposals(gray, config, OutputMode::WithHeatmap, voting_mode).into_result()
}

#[cfg(test)]
mod tests;
