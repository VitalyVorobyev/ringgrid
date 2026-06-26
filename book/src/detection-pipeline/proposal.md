# Proposal Generation

The proposal stage identifies candidate marker center positions in the image using gradient-based radial symmetry voting. Ring markers produce strong radially-symmetric gradient patterns at their centers, making gradient voting an effective detector that does not require template matching or multi-scale search.

The proposal module lives in `crates/ringgrid/src/proposal/` and has a standalone API with no ringgrid-specific dependencies in its core types. For the public proposal-only API and heatmap workflow, see [Proposal Diagnostics](../detection-modes/proposal-diagnostics.md).

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
   - Walk along the direction `sign * (dx, dy)` at radii from `r_min` to `r_max`, stepping by `radius_step` (default 1 = every integer radius). The top of the range is always voted. Setting `radius_step` above 1 subsamples the radius set to cut voting cost roughly proportionally, at the expense of accumulator sensitivity (lower recall on blurry/low-contrast scenes) — it is opt-in.
   - At each voted position, deposit `mag` into the accumulator using bilinear interpolation

Bilinear interpolation ensures sub-pixel accuracy in the accumulator. The vote weight is the gradient magnitude, so stronger edges contribute more to the accumulator peak.

Voting in both directions (positive and negative gradient) ensures that both the inner-to-outer and outer-to-inner transitions of a ring contribute to the same center peak, regardless of contrast polarity.

### Accumulator Smoothing

The raw accumulator is smoothed with a Gaussian blur (radius-relative smoothing applied by the radsym backend). This merges nearby votes that are slightly misaligned due to discretization, producing cleaner peaks.

### Two-Step Non-Maximum Suppression

Peaks are extracted from the smoothed accumulator in two steps, controlled by a single user-facing parameter `min_distance`:

**Step 1 — Local NMS peak extraction:**

1. Use an internal NMS radius of `min(min_distance, 10.0)` pixels, capped for efficiency (offset count scales as pi * r^2).
2. Scan all pixels outside a border margin. Skip pixels below `min_vote_frac * max_accumulator_value` (default: 10% of max).
3. A pixel is a local maximum if no neighbor within the NMS radius has a strictly higher value (ties broken by pixel index for determinism).

**Step 2 — Greedy distance suppression:**

4. Sort NMS survivors by score (descending).
5. Greedily accept proposals, rejecting any that fall within `min_distance` pixels of an already-accepted proposal.
6. Accepted peaks become `Proposal` structs with `(x, y, score)`.

If `max_candidates` is set, the list is truncated after greedy suppression.

## Optional Downscaling

When the ringgrid pipeline uses a wide marker diameter prior, the proposal stage can optionally downscale the image before voting to reduce cost. This is controlled by `ProposalDownscale` on `DetectConfig`:

| Variant | Behavior |
|---------|----------|
| `Auto` | Factor from `floor(d_min / 20.0)` clamped to `[1, 4]` |
| `Off` (default) | No downscaling |
| `Factor(n)` | Explicit integer factor (1–4) |

When active, the image is resized with bilinear interpolation, proposal config parameters (`r_min`, `r_max`, `min_distance`) are scaled down proportionally, and resulting proposal coordinates are scaled back to full resolution. All downstream stages (fit, decode) operate at full resolution.

CLI: `--proposal-downscale auto|off|2|4`

## Seed Injection in Two-Pass Modes

In two-pass detection modes (`detect_with_mapper`, `detect_with_self_undistort`), the pass-1 detection centers become seed proposals for pass-2. Seeds are assigned a very high score (`seed_score = 1e12` by default) to ensure they are evaluated before gradient-detected proposals.

This mechanism serves two purposes:

1. **Re-detection with improved geometry:** Pass-2 runs with a pixel mapper that corrects for lens distortion, so re-fitting at known centers produces more accurate ellipses.
2. **Recovery of weak detections:** Markers that were detected in pass-1 but might be below threshold in the working frame still get a chance to be evaluated.

The `SeedProposalConfig` configuration controls seed injection:

- `merge_radius_px` (default: 3.0): Prevents duplicate proposals when a seed and a gradient-detected proposal coincide.
- `max_seeds` (default: 512): Caps the number of seeds to prevent excessive computation.

## Configuration

The `ProposalConfig` struct controls all proposal parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_min` | 3.0 | Minimum voting radius in pixels |
| `r_max` | 12.0 | Maximum voting radius in pixels |
| `min_distance` | 7.0 | Minimum distance between output proposals (pixels) |
| `grad_threshold` | 0.05 | Gradient magnitude threshold (fraction of max) |
| `min_vote_frac` | 0.1 | Minimum accumulator value (fraction of max) |
| `radius_step` | 1 | Stride between voting radii (`1` = every integer radius; `2`+ subsamples for speed at the cost of recall; max radius always included) |
| `max_candidates` | None | Optional cap on proposals returned |

These defaults are overridden by `DetectConfig` when a `MarkerScalePrior` is set. The scale prior drives:

- `r_min = max(0.15 * spacing_min_px, 2.0)` where `spacing_min_px = spacing_ratio * d_min`
- `r_max = min(0.45 * spacing_max_px, 1.35 * outer_radius_max_px)`
- `min_distance = max(0.16 * d_min, 0.85 * spacing_min_px)`

The `spacing_ratio` is derived from board geometry: `min_center_spacing_mm / (2 * outer_radius_mm)`. This ensures proposal search radii adapt to both marker scale and board density.

Additionally, `max_candidates` in `ProposalConfig` limits the total proposals emitted, while `max_candidates` in `fit_decode.rs` separately caps how many proposals enter the fit-decode loop (sorted by score, highest first).

## Standalone API

The proposal module exposes a standalone API for general-purpose ellipse/circle center detection, independent of ringgrid's marker-specific pipeline:

```rust
use ringgrid::{find_ellipse_centers, find_ellipse_centers_with_heatmap, ProposalConfig};

let config = ProposalConfig { r_min: 5.0, r_max: 30.0, min_distance: 15.0, ..Default::default() };
let proposals = find_ellipse_centers(&gray_image, &config);
let result = find_ellipse_centers_with_heatmap(&gray_image, &config);  // includes heatmap
```

## Connection to Next Stage

Each accepted proposal provides a candidate center position `(x, y)` and a score. In the fit-decode phase, each proposal is passed through the [outer radius estimation](outer-estimate.md) stage to determine the expected ring size before edge sampling and ellipse fitting.

**Source:** `proposal/` module (`mod.rs`, `config.rs`, `gradient.rs`, `voting.rs`, `nms.rs`)
