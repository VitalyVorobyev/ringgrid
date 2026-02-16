# Proposal Generation

The proposal stage identifies candidate marker center positions in the image using gradient-based radial symmetry voting. Ring markers produce strong radially-symmetric gradient patterns at their centers, making gradient voting an effective detector that does not require template matching or multi-scale search.

## Algorithm

### Scharr Gradient Computation

The first step computes horizontal and vertical image derivatives using the 3x3 Scharr kernels, which provide better rotational symmetry than Sobel kernels:

```
        [ -3   0   3 ]            [ -3  -10  -3 ]
Kx =    [ -10  0  10 ]    Ky =    [  0    0   0 ]
        [ -3   0   3 ]            [  3   10   3 ]
```

The implementation uses `imageproc::gradients::horizontal_scharr` and `vertical_scharr` to produce i16 gradient images `gx` and `gy`.

### Gradient Magnitude Thresholding

The maximum gradient magnitude across the image is computed, and a threshold is set as a fraction of this maximum:

```
threshold = grad_threshold * max(sqrt(gx^2 + gy^2))
```

Pixels with gradient magnitude below this threshold are ignored, suppressing noise in flat regions. The default `grad_threshold` is 0.05 (5% of max gradient).

### Radial Symmetry Voting

For each pixel with a sufficiently strong gradient, the algorithm casts votes into an accumulator image along both the positive and negative gradient directions. The key insight is that gradient vectors on a ring boundary point radially toward (or away from) the ring center.

For each qualifying pixel at position `(x, y)` with gradient `(gx, gy)` and magnitude `mag`:

1. Compute the unit gradient direction: `(dx, dy) = (gx/mag, gy/mag)`
2. For each sign in `{-1, +1}`:
   - Walk along the direction `sign * (dx, dy)` at integer radius steps from `r_min` to `r_max`
   - At each voted position, deposit `mag` into the accumulator using bilinear interpolation

Bilinear interpolation ensures sub-pixel accuracy in the accumulator. The vote weight is the gradient magnitude, so stronger edges contribute more to the accumulator peak.

Voting in both directions (positive and negative gradient) ensures that both the inner-to-outer and outer-to-inner transitions of a ring contribute to the same center peak, regardless of contrast polarity.

### Accumulator Smoothing

The raw accumulator is smoothed with a Gaussian blur (sigma controlled by `accum_sigma`, default: 2.0 px). This merges nearby votes that are slightly misaligned due to discretization, producing cleaner peaks.

### Non-Maximum Suppression (NMS)

Peaks are extracted from the smoothed accumulator via local NMS:

1. Scan all pixels outside a border margin of `nms_radius` pixels.
2. Skip pixels below `min_vote_frac * max_accumulator_value` (default: 10% of max).
3. For each candidate pixel, check all neighbors within a circular region of radius `nms_radius`. A pixel is a local maximum if no neighbor has a strictly higher value (ties are broken by pixel index for determinism).
4. Accepted peaks become `Proposal` structs with `(x, y, score)`.

Proposals are sorted by score in descending order. If `max_candidates` is set, the list is truncated.

## Seed Injection in Two-Pass Modes

In two-pass detection modes (`detect_with_mapper`, `detect_with_self_undistort`), the pass-1 detection centers become seed proposals for pass-2. Seeds are assigned a very high score (`seed_score = 1e12` by default) to ensure they are evaluated before gradient-detected proposals.

This mechanism serves two purposes:

1. **Re-detection with improved geometry:** Pass-2 runs with a pixel mapper that corrects for lens distortion, so re-fitting at known centers produces more accurate ellipses.
2. **Recovery of weak detections:** Markers that were detected in pass-1 but might be below threshold in the working frame still get a chance to be evaluated.

The `SeedProposalParams` configuration controls seed injection:

- `merge_radius_px` (default: 3.0): Prevents duplicate proposals when a seed and a gradient-detected proposal coincide.
- `max_seeds` (default: 512): Caps the number of seeds to prevent excessive computation.

## Configuration

The `ProposalConfig` struct controls all proposal parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_min` | 3.0 | Minimum voting radius in pixels |
| `r_max` | 12.0 | Maximum voting radius in pixels |
| `grad_threshold` | 0.05 | Gradient magnitude threshold (fraction of max) |
| `nms_radius` | 7.0 | NMS radius for peak extraction in pixels |
| `min_vote_frac` | 0.1 | Minimum accumulator value (fraction of max) |
| `accum_sigma` | 2.0 | Gaussian sigma for accumulator smoothing |
| `max_candidates` | None | Optional cap on proposals returned |

These defaults are overridden by `DetectConfig` when a `MarkerScalePrior` is set. The scale prior drives:

- `r_min = max(diameter_min * 0.2, 2.0)`
- `r_max = diameter_max * 0.85`
- `nms_radius = max(diameter_min * 0.4, 2.0)`

Additionally, `max_candidates` in `ProposalConfig` limits the total proposals emitted, while `max_candidates` in `fit_decode.rs` separately caps how many proposals enter the fit-decode loop (sorted by score, highest first).

## Connection to Next Stage

Each accepted proposal provides a candidate center position `(x, y)` and a score. In the fit-decode phase, each proposal is passed through the [outer radius estimation](outer-estimate.md) stage to determine the expected ring size before edge sampling and ellipse fitting.

**Source:** `detector/proposal.rs`
