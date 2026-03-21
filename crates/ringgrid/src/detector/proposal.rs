//! Gradient-voting radial symmetry for candidate center detection.
//!
//! For each pixel with a strong gradient, votes are cast along the gradient
//! direction at distances in [r_min, r_max]. Ring markers produce peaks in
//! the accumulator at their centers because gradient vectors from the ring
//! boundaries converge radially.

use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;
use std::time::{Duration, Instant};

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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Proposal {
    /// X coordinate (pixels).
    pub x: f32,
    /// Y coordinate (pixels).
    pub y: f32,
    /// Accumulator peak score.
    pub score: f32,
}

/// Proposal-stage diagnostics for one image.
///
/// `nms_accumulator` stores the post-Gaussian-smoothed accumulator that is
/// thresholded and scanned during non-maximum suppression. The values are
/// stored in row-major order with shape `[image_size[1], image_size[0]]`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProposalDiagnostics {
    /// Image dimensions `[width, height]`.
    pub image_size: [u32; 2],
    /// Post-smoothing accumulator used for NMS, row-major.
    pub nms_accumulator: Vec<f32>,
    /// Proposals sorted by score (highest first).
    pub proposals: Vec<Proposal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposalOutputMode {
    ProposalsOnly,
    Diagnostics,
}

impl ProposalOutputMode {
    fn keeps_accumulator(self) -> bool {
        matches!(self, Self::Diagnostics)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposalVotingMode {
    Auto,
    #[cfg(test)]
    Scalar,
    #[cfg(test)]
    Parallel,
}

const PARALLEL_VOTE_MIN_WORK: u64 = 8_000_000;
const PARALLEL_VOTE_TARGET_WORK_PER_CHUNK: u64 = 4_000_000;
const PARALLEL_VOTE_MAX_CHUNKS: usize = 4;

#[derive(Debug, Clone, Copy)]
struct StrongEdge {
    x: f32,
    y: f32,
    mag: f32,
    dx: f32,
    dy: f32,
}

#[derive(Debug, Default)]
struct ProposalTimingStats {
    gradient: Duration,
    max_scan: Duration,
    compact: Duration,
    vote: Duration,
    blur: Duration,
    nms: Duration,
    strong_edge_count: usize,
    radius_steps: usize,
    estimated_vote_work: u64,
}

struct ProposalCoreOutput {
    image_size: [u32; 2],
    proposals: Vec<Proposal>,
    nms_accumulator: Option<Vec<f32>>,
}

impl ProposalCoreOutput {
    fn empty(image_size: [u32; 2], output_mode: ProposalOutputMode) -> Self {
        let n = image_size[0] as usize * image_size[1] as usize;
        let nms_accumulator = output_mode.keeps_accumulator().then(|| vec![0.0; n]);
        Self {
            image_size,
            proposals: Vec::new(),
            nms_accumulator,
        }
    }

    fn into_proposals(self) -> Vec<Proposal> {
        self.proposals
    }

    fn into_diagnostics(self) -> ProposalDiagnostics {
        let n = self.image_size[0] as usize * self.image_size[1] as usize;
        ProposalDiagnostics {
            image_size: self.image_size,
            nms_accumulator: self.nms_accumulator.unwrap_or_else(|| vec![0.0; n]),
            proposals: self.proposals,
        }
    }
}

#[inline]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
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

fn build_scharr_gradients(gray: &GrayImage) -> (Vec<i16>, Vec<i16>, f32) {
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

fn collect_strong_edges(
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

fn accumulate_votes_scalar(
    strong_edges: &[StrongEdge],
    stride: usize,
    height: usize,
    r_min: f32,
    radius_steps: usize,
) -> Vec<f32> {
    let mut accum = vec![0.0f32; stride * height];
    let x_limit = (stride - 1) as f32;
    let y_limit = (height - 1) as f32;

    for edge in strong_edges {
        let step_x = edge.dx;
        let step_y = edge.dy;

        let mut vx_pos = edge.x + step_x * r_min;
        let mut vy_pos = edge.y + step_y * r_min;
        let mut vx_neg = edge.x - step_x * r_min;
        let mut vy_neg = edge.y - step_y * r_min;

        for _ in 0..radius_steps {
            if vx_pos >= 0.0 && vx_pos < x_limit && vy_pos >= 0.0 && vy_pos < y_limit {
                bilinear_add_in_bounds(&mut accum, stride, vx_pos, vy_pos, edge.mag);
            }
            if vx_neg >= 0.0 && vx_neg < x_limit && vy_neg >= 0.0 && vy_neg < y_limit {
                bilinear_add_in_bounds(&mut accum, stride, vx_neg, vy_neg, edge.mag);
            }

            vx_pos += step_x;
            vy_pos += step_y;
            vx_neg -= step_x;
            vy_neg -= step_y;
        }
    }

    accum
}

fn parallel_vote_chunk_count(estimated_vote_work: u64, strong_edge_count: usize) -> usize {
    if estimated_vote_work < PARALLEL_VOTE_MIN_WORK || strong_edge_count < 4_096 {
        return 1;
    }

    let chunk_count = estimated_vote_work.div_ceil(PARALLEL_VOTE_TARGET_WORK_PER_CHUNK) as usize;
    chunk_count.clamp(2, PARALLEL_VOTE_MAX_CHUNKS)
}

fn accumulate_votes_parallel(
    strong_edges: &[StrongEdge],
    stride: usize,
    height: usize,
    r_min: f32,
    radius_steps: usize,
    chunk_count: usize,
) -> Vec<f32> {
    if chunk_count <= 1 || strong_edges.is_empty() {
        return accumulate_votes_scalar(strong_edges, stride, height, r_min, radius_steps);
    }

    let edges_per_chunk = strong_edges.len().div_ceil(chunk_count);
    let chunks: Vec<&[StrongEdge]> = strong_edges.chunks(edges_per_chunk).collect();
    let partial_accumulators: Vec<Vec<f32>> = chunks
        .into_par_iter()
        .map(|chunk| accumulate_votes_scalar(chunk, stride, height, r_min, radius_steps))
        .collect();

    let mut accum = vec![0.0f32; stride * height];
    for partial in partial_accumulators {
        for (dst, src) in accum.iter_mut().zip(partial) {
            *dst += src;
        }
    }
    accum
}

fn build_nms_offsets(stride: usize, nms_radius: f32) -> Vec<isize> {
    let nms_r = nms_radius.ceil() as i32;
    let nms_r_sq = nms_radius * nms_radius;
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
    nms_offsets
}

fn extract_proposals_from_smoothed(
    smoothed_data: &[f32],
    w: u32,
    h: u32,
    config: &ProposalConfig,
) -> Vec<Proposal> {
    let stride = w as usize;
    let max_val = smoothed_data.iter().cloned().fold(0.0f32, f32::max);
    if max_val < 1e-6 {
        return Vec::new();
    }

    let vote_threshold = config.min_vote_frac * max_val;
    let nms_r = config.nms_radius.ceil() as i32;
    let nms_offsets = build_nms_offsets(stride, config.nms_radius);

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

    proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    if let Some(max_candidates) = config.max_candidates {
        proposals.truncate(max_candidates.min(proposals.len()));
    }
    proposals
}

fn log_proposal_timing(
    stats: &ProposalTimingStats,
    proposals: usize,
    voting_mode: &'static str,
    parallel_chunks: usize,
) {
    tracing::debug!(
        gradient_mode = "fused_scharr_with_max",
        voting_mode,
        parallel_chunks,
        gradient_ms = duration_ms(stats.gradient),
        max_scan_ms = duration_ms(stats.max_scan),
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

fn compute_proposals(
    gray: &GrayImage,
    config: &ProposalConfig,
    output_mode: ProposalOutputMode,
    voting_mode: ProposalVotingMode,
) -> ProposalCoreOutput {
    let (w, h) = gray.dimensions();
    let image_size = [w, h];
    if w < 4 || h < 4 || config.r_max < config.r_min {
        return ProposalCoreOutput::empty(image_size, output_mode);
    }

    let mut timing = ProposalTimingStats::default();
    let gradient_start = Instant::now();
    let (gx, gy, max_mag_sq) = build_scharr_gradients(gray);
    timing.gradient = gradient_start.elapsed();
    let mut voting_mode_label = "scalar";
    let mut parallel_chunks = 1usize;

    if max_mag_sq < 1e-6 {
        log_proposal_timing(&timing, 0, voting_mode_label, parallel_chunks);
        return ProposalCoreOutput::empty(image_size, output_mode);
    }

    let max_mag = max_mag_sq.sqrt();
    let threshold = config.grad_threshold * max_mag;
    let threshold_sq = threshold * threshold;

    let stride = w as usize;
    let height = h as usize;
    let radius_steps = radius_steps(config);
    if radius_steps == 0 {
        log_proposal_timing(&timing, 0, voting_mode_label, parallel_chunks);
        return ProposalCoreOutput::empty(image_size, output_mode);
    }
    timing.radius_steps = radius_steps;

    let compact_start = Instant::now();
    let strong_edges = collect_strong_edges(&gx, &gy, stride, height, threshold_sq);
    timing.compact = compact_start.elapsed();
    timing.strong_edge_count = strong_edges.len();
    timing.estimated_vote_work = strong_edges.len() as u64 * radius_steps as u64 * 2;

    let vote_start = Instant::now();
    let chunk_count = parallel_vote_chunk_count(timing.estimated_vote_work, strong_edges.len());
    let accum = match voting_mode {
        ProposalVotingMode::Auto => {
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
        ProposalVotingMode::Scalar => {
            accumulate_votes_scalar(&strong_edges, stride, height, config.r_min, radius_steps)
        }
        #[cfg(test)]
        ProposalVotingMode::Parallel => {
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
    let proposals = extract_proposals_from_smoothed(smoothed.as_raw(), w, h, config);
    timing.nms = nms_start.elapsed();
    log_proposal_timing(&timing, proposals.len(), voting_mode_label, parallel_chunks);

    ProposalCoreOutput {
        image_size,
        proposals,
        nms_accumulator: if output_mode.keeps_accumulator() {
            Some(smoothed.into_raw())
        } else {
            None
        },
    }
}

#[cfg(test)]
fn find_proposals_with_mode(
    gray: &GrayImage,
    config: &ProposalConfig,
    voting_mode: ProposalVotingMode,
) -> Vec<Proposal> {
    compute_proposals(gray, config, ProposalOutputMode::ProposalsOnly, voting_mode).into_proposals()
}

/// Detect candidate marker centers via gradient-based radial symmetry voting.
///
/// Returns proposals sorted by score (highest first).
pub(crate) fn find_proposals(gray: &GrayImage, config: &ProposalConfig) -> Vec<Proposal> {
    compute_proposals(
        gray,
        config,
        ProposalOutputMode::ProposalsOnly,
        ProposalVotingMode::Auto,
    )
    .into_proposals()
}

/// Run proposal generation and return the proposals plus the smoothed
/// accumulator used during NMS.
pub(crate) fn find_proposals_diagnostics(
    gray: &GrayImage,
    config: &ProposalConfig,
) -> ProposalDiagnostics {
    compute_proposals(
        gray,
        config,
        ProposalOutputMode::Diagnostics,
        ProposalVotingMode::Auto,
    )
    .into_diagnostics()
}

#[cfg(test)]
fn find_proposals_diagnostics_with_mode(
    gray: &GrayImage,
    config: &ProposalConfig,
    voting_mode: ProposalVotingMode,
) -> ProposalDiagnostics {
    compute_proposals(gray, config, ProposalOutputMode::Diagnostics, voting_mode).into_diagnostics()
}

/// Generate candidate marker centers via gradient-based radial symmetry voting.
///
/// Returns proposals sorted by score (highest first).
pub fn propose(gray: &GrayImage, config: &ProposalConfig) -> Vec<Proposal> {
    find_proposals(gray, config)
}

/// Generate proposal-stage diagnostics, including the smoothed accumulator
/// used by NMS.
pub fn propose_diagnostics(gray: &GrayImage, config: &ProposalConfig) -> ProposalDiagnostics {
    find_proposals_diagnostics(gray, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageReader;
    use std::path::Path;

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

    fn load_fixture_image() -> GrayImage {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/target_3_split_00.png");
        ImageReader::open(path)
            .expect("open fixture image")
            .decode()
            .expect("decode fixture image")
            .to_luma8()
    }

    fn assert_voting_modes_match(img: &GrayImage, config: &ProposalConfig) {
        let scalar = find_proposals_with_mode(img, config, ProposalVotingMode::Scalar);
        let parallel = find_proposals_with_mode(img, config, ProposalVotingMode::Parallel);
        let scalar_diag =
            find_proposals_diagnostics_with_mode(img, config, ProposalVotingMode::Scalar);
        let parallel_diag =
            find_proposals_diagnostics_with_mode(img, config, ProposalVotingMode::Parallel);

        assert_eq!(parallel.len(), scalar.len());
        for (lhs, rhs) in parallel.iter().zip(&scalar) {
            assert_eq!(lhs.x, rhs.x);
            assert_eq!(lhs.y, rhs.y);
            assert!(
                (lhs.score - rhs.score).abs() <= 0.05,
                "proposal score drift too large at ({}, {}): {} vs {}",
                lhs.x,
                lhs.y,
                lhs.score,
                rhs.score
            );
        }

        assert_eq!(scalar_diag.proposals.len(), scalar.len());
        assert_eq!(parallel_diag.proposals.len(), parallel.len());
        for (lhs, rhs) in scalar_diag.proposals.iter().zip(&scalar) {
            assert_eq!(lhs, rhs);
        }
        for (lhs, rhs) in parallel_diag.proposals.iter().zip(&parallel) {
            assert_eq!(lhs, rhs);
        }
        let max_accum_diff = parallel_diag
            .nms_accumulator
            .iter()
            .zip(&scalar_diag.nms_accumulator)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_accum_diff <= 0.05,
            "parallel diagnostics drift too large: max accumulator diff = {max_accum_diff}"
        );
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

    #[test]
    fn proposal_diagnostics_match_proposals_and_accumulator_shape() {
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

        let plain = find_proposals(&img, &config);
        let diagnostics = find_proposals_diagnostics(&img, &config);
        assert_eq!(diagnostics.image_size, [80, 80]);
        assert_eq!(diagnostics.nms_accumulator.len(), 80 * 80);
        assert_eq!(diagnostics.proposals, plain);

        let best = diagnostics.proposals[0];
        let best_idx = best.y as usize * diagnostics.image_size[0] as usize + best.x as usize;
        assert_eq!(diagnostics.nms_accumulator[best_idx], best.score);

        let err = ((best.x - cx).powi(2) + (best.y - cy).powi(2)).sqrt();
        assert!(err < 5.0);
    }

    #[test]
    fn proposal_diagnostics_serde_roundtrip() {
        let img = make_ring_image(64, 64, 31.0, 29.0, 8.0, 2.0);
        let diagnostics = find_proposals_diagnostics(&img, &ProposalConfig::default());
        let json = serde_json::to_string(&diagnostics).expect("serialize");
        let roundtrip: ProposalDiagnostics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip, diagnostics);
    }

    #[test]
    fn fused_scharr_matches_imageproc_on_image_interior() {
        let mut img = GrayImage::new(7, 6);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let value = ((x * 17 + y * 29 + x * y * 3) % 251) as u8;
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let (gx, gy, _) = build_scharr_gradients(&img);
        let gx_ref = imageproc::gradients::horizontal_scharr(&img);
        let gy_ref = imageproc::gradients::vertical_scharr(&img);
        let stride = img.width() as usize;
        let height = img.height() as usize;

        for y in 1..height - 1 {
            let row = y * stride;
            for x in 1..stride - 1 {
                let idx = row + x;
                assert_eq!(gx[idx], gx_ref.as_raw()[idx], "gx mismatch at ({x},{y})");
                assert_eq!(gy[idx], gy_ref.as_raw()[idx], "gy mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn blank_image_yields_empty_proposals_and_zero_diagnostics() {
        let img = GrayImage::new(32, 24);
        let proposals = find_proposals(&img, &ProposalConfig::default());
        let diagnostics = find_proposals_diagnostics(&img, &ProposalConfig::default());

        assert!(proposals.is_empty());
        assert!(diagnostics.proposals.is_empty());
        assert_eq!(diagnostics.image_size, [32, 24]);
        assert!(diagnostics.nms_accumulator.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fixture_proposals_match_diagnostics_in_scalar_mode() {
        let img = load_fixture_image();
        let config = ProposalConfig::default();

        let fast = find_proposals(&img, &config);
        let fast_forced = find_proposals_with_mode(&img, &config, ProposalVotingMode::Scalar);
        let diagnostics =
            find_proposals_diagnostics_with_mode(&img, &config, ProposalVotingMode::Scalar);

        assert!(!fast.is_empty(), "fixture should produce proposals");
        assert_eq!(fast, fast_forced);
        assert_eq!(diagnostics.proposals, fast);
        assert_eq!(
            diagnostics.nms_accumulator.len(),
            img.width() as usize * img.height() as usize
        );

        let best = diagnostics.proposals[0];
        let best_idx = best.y as usize * img.width() as usize + best.x as usize;
        assert_eq!(diagnostics.nms_accumulator[best_idx], best.score);
    }

    #[test]
    fn ring_image_parallel_voting_matches_scalar() {
        let img = make_ring_image(160, 160, 79.0, 81.0, 20.0, 3.5);
        let config = ProposalConfig {
            r_min: 8.0,
            r_max: 28.0,
            grad_threshold: 0.03,
            nms_radius: 6.0,
            min_vote_frac: 0.05,
            accum_sigma: 1.5,
            max_candidates: Some(32),
        };

        assert_voting_modes_match(&img, &config);
    }

    #[test]
    fn fixture_parallel_voting_matches_scalar() {
        let img = load_fixture_image();
        let config = ProposalConfig::default();
        assert_voting_modes_match(&img, &config);
    }
}
