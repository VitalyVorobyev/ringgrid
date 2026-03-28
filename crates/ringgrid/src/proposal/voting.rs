//! Radial symmetry vote accumulation (scalar and parallel).

use rayon::prelude::*;

use super::gradient::StrongEdge;

const PARALLEL_VOTE_MIN_WORK: u64 = 8_000_000;
const PARALLEL_VOTE_TARGET_WORK_PER_CHUNK: u64 = 4_000_000;
const PARALLEL_VOTE_MAX_CHUNKS: usize = 4;

/// Deposit a weighted vote using bilinear interpolation.
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

/// Cast bidirectional radial votes for all strong edges (single-threaded).
pub(crate) fn accumulate_votes_scalar(
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

/// Decide how many parallel chunks to use for vote accumulation.
pub(crate) fn parallel_vote_chunk_count(
    estimated_vote_work: u64,
    strong_edge_count: usize,
) -> usize {
    if estimated_vote_work < PARALLEL_VOTE_MIN_WORK || strong_edge_count < 4_096 {
        return 1;
    }

    let chunk_count = estimated_vote_work.div_ceil(PARALLEL_VOTE_TARGET_WORK_PER_CHUNK) as usize;
    chunk_count.clamp(2, PARALLEL_VOTE_MAX_CHUNKS)
}

/// Cast bidirectional radial votes in parallel chunks and merge.
pub(crate) fn accumulate_votes_parallel(
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
