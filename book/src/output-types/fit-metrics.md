# FitMetrics & DecodeMetrics

These two structs provide detailed quality metrics for each detected marker. `FitMetrics` describes how well the ellipse(s) fit the observed edge points. `DecodeMetrics` describes how confidently the 16-sector code was matched to a codebook entry.

**Source:** `crates/ringgrid/src/detector/marker_build.rs` (FitMetrics), `crates/ringgrid/src/marker/decode.rs` (DecodeMetrics)

## FitMetrics

`FitMetrics` is always present on every `DetectedMarker`. It reports edge sampling coverage and ellipse fit quality.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_angles_total` | `usize` | Total number of radial rays cast from the candidate center. |
| `n_angles_with_both_edges` | `usize` | Number of rays where both inner and outer ring edges were found. |
| `n_points_outer` | `usize` | Number of outer edge points used for the ellipse fit. |
| `n_points_inner` | `usize` | Number of inner edge points used for the inner ellipse fit. 0 if no inner fit was performed. |
| `ransac_inlier_ratio_outer` | `Option<f32>` | Fraction of outer edge points classified as RANSAC inliers. |
| `ransac_inlier_ratio_inner` | `Option<f32>` | Fraction of inner edge points classified as RANSAC inliers. |
| `rms_residual_outer` | `Option<f64>` | RMS Sampson distance of outer edge points to the fitted ellipse (in pixels). |
| `rms_residual_inner` | `Option<f64>` | RMS Sampson distance of inner edge points to the fitted ellipse (in pixels). |

### Interpreting FitMetrics

**RANSAC inlier ratio** measures how consistently the edge points agree with the fitted ellipse:

| `ransac_inlier_ratio_outer` | Interpretation |
|------------------------------|----------------|
| > 0.90 | Excellent -- clean edges with minimal outliers |
| 0.80 -- 0.90 | Good -- some edge noise or partial occlusion |
| < 0.70 | Poor -- significant outliers, possible false detection |

**RMS Sampson residual** measures the geometric precision of the fit:

| `rms_residual_outer` | Interpretation |
|-----------------------|----------------|
| < 0.3 px | Excellent sub-pixel precision |
| 0.3 -- 0.5 px | Good precision |
| 0.5 -- 1.0 px | Acceptable but noisy |
| > 1.0 px | Poor fit, possibly wrong feature |

**Arc coverage** is the ratio `n_angles_with_both_edges / n_angles_total`. It indicates how much of the ring perimeter was successfully sampled:

| Coverage ratio | Interpretation |
|----------------|----------------|
| > 0.85 | Full ring visible, high confidence |
| 0.5 -- 0.85 | Partial occlusion or edge-of-frame |
| < 0.5 | Severely occluded, likely unreliable |

## DecodeMetrics

`DecodeMetrics` is present on a `DetectedMarker` when code decoding was attempted. It reports the raw sampled word and the quality of the codebook match.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `observed_word` | `u16` | Raw 16-bit word sampled from the code band. Each bit corresponds to one sector (bright = 1, dark = 0). |
| `best_id` | `usize` | Index of the best-matching codebook entry (0--892). |
| `best_rotation` | `u8` | Cyclic rotation (0--15) that produced the best match. Each unit is 22.5 degrees. |
| `best_dist` | `u8` | Hamming distance between the observed word (at best rotation) and the codebook entry. |
| `margin` | `u8` | Gap between the best and second-best Hamming distances: `second_best_dist - best_dist`. |
| `decode_confidence` | `f32` | Heuristic confidence score in `[0, 1]`, combining Hamming distance and margin. |

### Interpreting DecodeMetrics

**Hamming distance** (`best_dist`) tells how many of the 16 sectors disagree with the matched codeword:

| `best_dist` | Interpretation |
|-------------|----------------|
| 0 | Exact match -- no bit errors |
| 1 -- 2 | Minor noise, still reliable |
| 3 | At the default acceptance threshold |
| > 3 | Rejected by default (configurable via `DecodeConfig::max_decode_dist`) |

**Margin** (`margin`) measures how unambiguous the match is. It is the difference in Hamming distance between the best and second-best codebook matches:

| `margin` | Interpretation |
|----------|----------------|
| >= 4 | Highly unambiguous |
| 3 | Reliable |
| 2 | Acceptable but less certain |
| 1 | Risky -- two codewords are nearly tied |
| 0 | Ambiguous -- the match could be wrong |

**Decode confidence** (`decode_confidence`) is a composite heuristic in `[0, 1]` that accounts for both Hamming distance and margin. Higher values indicate more reliable decodes. The default minimum threshold is 0.15 (configurable via `DecodeConfig::min_decode_confidence`).

### Polarity handling

The decoder tries both normal and inverted polarity of the sampled word (bitwise NOT) and picks whichever produces the better codebook match. The `observed_word` in `DecodeMetrics` reflects the polarity that was actually used for matching.

## Serialization

Both structs derive `serde::Serialize` and `serde::Deserialize`. Optional fields use `#[serde(skip_serializing_if = "Option::is_none")]` and are omitted from JSON output when absent.
