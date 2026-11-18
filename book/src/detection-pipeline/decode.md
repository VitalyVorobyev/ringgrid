# Code Decoding

After the outer ellipse is fitted, the detector samples the code band — the annular region between the inner and outer rings — to read the marker's 16-sector binary code and match it against **the target's own codewords**: the IDs its cells carry, drawn from the target's codebook profile (`base`, 893 codewords, or the opt-in `extended`).

## Elliptical Code Band Sampling

The decoder samples the code band **along the fitted outer ellipse**, scaled inward to the code-band radius — not along a circle at the ellipse's mean radius. The `code_band_ratio` parameter (auto-derived from the target's ring radii and stroke width) sets how far in the code band sits relative to the outer ellipse.

### Why parametric angle, not a circle

The board's 16 sectors are equal-angle **in board space**. Under the affine approximation of the fitted ellipse, equal board angle corresponds to uniform *parametric* angle on the ellipse — so sampling at uniform parametric angle keeps every sector's angular support equal regardless of eccentricity. A circle of the mean radius, by contrast, drifts off the elliptical code band on tilted views: near the minor axis its samples land on the inner ring or background, corrupting those sectors.

### Sub-sector phase

Where the printed sector boundaries land in parametric angle is a *continuous* unknown — board rotation plus the ellipse's own angle origin. Cyclic codebook matching absorbs whole sectors only. A sampling grid that is off by a fraction of a sector makes every sampled sector straddle two printed ones; at half a sector the sampled word is mostly noise (ADR-027: on the rtv3d captures 89 % of the decodes accepted at an offset of 0.4–0.5 sector were wrong).

The decoder therefore samples the band densely — `samples_per_sector` samples per sector (default 8), each averaged over `n_radial_rings` radii (≈ 0.90–1.10 × the code-band ratio) — and tries every sub-sector phase of the sector grid. For each phase it averages the central half of every sector; the phase whose 16 sector means vary the most (smoothed over neighbouring phases, so the middle of the plateau wins) is the one whose sectors sit inside the printed ones.

## Binarization

The 16 sector means are converted to bits with an iterative 2-means threshold:

1. Initialize the threshold at the midpoint of the intensity range
2. Split sectors into two groups (above/below threshold)
3. Recompute the threshold as the mean of the group means
4. Repeat until convergence

This local thresholding adapts to the actual contrast of each marker. Targets are printed dark on light, so a bright sector is a `1`; the complement is never tried.

## Matching

The 16-bit word is matched against every rotation of every codeword the target's cells carry, and accepted **only on an exact match**. A Hamming-distance tolerance would admit noise: with minimum cyclic distance 2 (base profile), 99.9 % of all 16-bit words lie within one bit of some base codeword, so being one bit off a codeword is no evidence for it. Restricting matching to the target's own codewords (203 of the 893 on the default board) also cuts the share of random words that happen to match exactly from 22 % to 5 %.

## DecodeMetrics

An accepted decode produces a `DecodeMetrics` struct (diagnostics channel):

| Field | Type | Meaning |
|---|---|---|
| `observed_word` | `u16` | The 16-bit word sampled on the phase-aligned grid (bit `s` = grid sector `s`, 1 = bright) |
| `best_id` | `usize` | Matched codebook ID |
| `rotation` | `f32` | Rotation of the printed code in sectors, `[0, 16)`: printed sector `j` starts at parametric angle `(j + rotation)·2π/16` on the fitted outer ellipse. The whole part is the cyclic shift mapping the codeword onto `observed_word`; the fraction is the estimated sub-sector phase. |

The anchor stage uses `rotation`: under a candidate board placement the code rotation at each ring is predictable, and a decode votes only for placements it agrees with (see the [pipeline overview](overview.md) and ADR-027).

**Source**: `marker/decode.rs`, `marker/codec.rs`, `marker/codebook.rs`
