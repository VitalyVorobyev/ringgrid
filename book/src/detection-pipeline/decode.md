# Code Decoding

After the outer ellipse is fitted, the detector samples the code band — the annular region between the inner and outer rings — to read the marker's 16-sector binary code and match it against the 893-codeword codebook.

## Code Band Sampling

The code band sits at a radius that is a configurable fraction of the outer ellipse size. The `code_band_ratio` parameter (default derived from marker geometry, typically ~0.74) defines where the sampling circle lies relative to the outer ellipse.

For each of the 16 angular sectors, the decoder:

1. Computes the sector center angle: `θ_k = k × 2π/16` for `k = 0..15`
2. Samples pixel intensities at multiple points within the sector (oversampled in both angular and radial directions)
3. Aggregates samples to produce a single intensity value per sector

This multi-sample approach provides robustness against noise, blur, and slight geometric inaccuracies in the ellipse fit.

## Binarization

The 16 sector intensities are converted to binary using an iterative 2-means threshold:

1. Initialize threshold at the mean of all sector intensities
2. Split sectors into two groups (above/below threshold)
3. Recompute threshold as the mean of the group means
4. Repeat until convergence

This local thresholding adapts to the actual contrast of each marker, handling varying illumination across the image.

## Cyclic Codebook Matching

The 16-bit binary word is matched against the embedded codebook (893 codewords) with **cyclic rotation search**:

- For each of the 16 possible rotational offsets, compute the Hamming distance between the observed word and each codebook entry
- Also check the **inverted** (bitwise NOT) word at each rotation, handling both dark-on-light and light-on-dark contrast
- Select the best match: the (codeword, rotation, polarity) triple with minimum Hamming distance

The best match is accepted based on:

- **Hamming distance** (`best_dist`): number of bit disagreements with the closest codeword
- **Margin** (`margin`): gap between the best and second-best Hamming distances
- **Decode confidence**: `clamp(1 - dist/6) × clamp(margin/3)`, a heuristic combining closeness and uniqueness

## DecodeMetrics

The decoding stage produces a `DecodeMetrics` struct:

| Field | Type | Meaning |
|---|---|---|
| `observed_word` | `u16` | The raw 16-bit word before matching |
| `best_id` | `usize` | Matched codebook entry ID |
| `best_rotation` | `u8` | Rotation offset (0–15 sectors) |
| `best_dist` | `u8` | Hamming distance to best match |
| `margin` | `u8` | Gap to second-best match |
| `decode_confidence` | `f32` | Combined confidence score in [0, 1] |

A `best_dist` of 0 means a perfect match. The minimum cyclic Hamming distance in the codebook is 2, so a distance of 1 is still an unambiguous match.

**Source**: `marker/decode.rs`, `marker/codec.rs`, `marker/codebook.rs`
