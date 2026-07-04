# Code Decoding

After the outer ellipse is fitted, the detector samples the code band — the annular region between the inner and outer rings — to read the marker's 16-sector binary code and match it against the active embedded codebook profile. The shipped default is the 893-codeword `base` profile; `extended` is explicit opt-in.

## Elliptical Code Band Sampling

The decoder samples the code band **along the fitted outer ellipse**, scaled inward to the code-band radius — not along a circle at the ellipse's mean radius. The `code_band_ratio` parameter (default `0.76`, auto-derived from the target's inner/outer radius ratio) sets how far in the code band sits relative to the outer ellipse.

For each of the 16 angular sectors, the decoder:

1. Steps the **parametric angle** `θ` across the sector: for sample `j` of `samples_per_sector`, `θ = (s + (j + 0.5)/samples_per_sector) · 2π/16`.
2. Evaluates the ellipse point at that parametric angle — `(a·cos θ, b·sin θ)` — rotated by the ellipse angle and scaled by the code-band ratio (with `n_radial_rings` spanning ≈ 0.90–1.10 × the ratio for radial oversampling).
3. Bilinearly samples image intensity (distortion-aware when a mapper is active) and averages all `samples_per_sector × n_radial_rings` samples into one intensity per sector.

### Why parametric angle, not a circle

The board's 16 sectors are equal-angle **in board space**. Under the affine approximation of the fitted ellipse, equal board angle corresponds to uniform *parametric* angle on the ellipse — so stepping `θ` uniformly keeps every sector's angular support equal regardless of eccentricity. A circle of the mean radius, by contrast, drifts off the elliptical code band on tilted views: near the minor axis its samples land on the inner ring or background, corrupting those sectors. The unknown constant offset between parametric angle and true board rotation is harmless — cyclic codebook matching (below) absorbs it, exactly as it absorbs image rotation.

This equal-support sampling is measurably equal-or-better across the whole benchmark suite and materially better on strongly tilted markers.

## Binarization

The 16 sector intensities are converted to binary using an iterative 2-means threshold:

1. Initialize threshold at the mean of all sector intensities
2. Split sectors into two groups (above/below threshold)
3. Recompute threshold as the mean of the group means
4. Repeat until convergence

This local thresholding adapts to the actual contrast of each marker, handling varying illumination across the image.

## Cyclic Codebook Matching

The 16-bit binary word is matched against the selected embedded codebook profile with **cyclic rotation search**:

- For each of the 16 possible rotational offsets, compute the Hamming distance between the observed word and each codebook entry
- Also check the **inverted** (bitwise NOT) word at each rotation, handling both dark-on-light and light-on-dark contrast
- Select the best match: the (codeword, rotation, polarity) triple with minimum Hamming distance

The best match is accepted based on:

- **Hamming distance** (`best_dist`): number of bit disagreements with the closest codeword
- **Margin** (`margin`): gap between the best and second-best Hamming distances
- **Decode confidence**: `clamp(1 - dist/6) × clamp(margin / active_profile_min_cyclic_dist)`, a heuristic combining closeness and uniqueness. For the shipped `base` profile, the minimum cyclic Hamming distance is `2`; for the opt-in `extended` profile it is `1`.

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

A `best_dist` of 0 means a perfect match. In the shipped `base` profile, minimum cyclic Hamming distance is `2`, so a distance of `1` is still unambiguous. The opt-in `extended` profile weakens that minimum distance to `1` in exchange for more available IDs.

**Source**: `marker/decode.rs`, `marker/codec.rs`, `marker/codebook.rs`
