# FitMetrics, DecodeMetrics & MarkerDiagnostics

These structs provide detailed quality metrics for each detected marker.
`FitMetrics` describes how well the ellipse(s) fit the observed edge points.
`DecodeMetrics` describes how confidently the 16-sector code was matched to a
codebook entry. `MarkerDiagnostics` bundles both — plus the raw edge points and
the producing pipeline stage — into the opt-in per-marker diagnostics channel.

**Source:** `crates/ringgrid/src/detector/marker_build.rs` (FitMetrics), `crates/ringgrid/src/marker/decode.rs` (DecodeMetrics), `crates/ringgrid/src/pipeline/result.rs` (MarkerDiagnostics)

## MarkerDiagnostics

`MarkerDiagnostics` carries the per-marker algorithm internals that were fields
of `DetectedMarker` before v0.6. Obtain it via `Detector::detect_with_diagnostics`;
the `DetectionDiagnostics::markers` vector is positionally aligned 1:1 with
`DetectionResult::detected_markers`.

| Field | Type | Description |
|-------|------|-------------|
| `fit` | `FitMetrics` | Fit quality metrics (always present). |
| `decode` | `Option<DecodeMetrics>` | Decode quality metrics. Present when decoding was attempted. |
| `source` | `DetectionSource` | Pipeline path that produced the marker: `fit_decoded`, `completion`, or `seeded_pass`. |
| `edge_points_outer` | `Option<Vec<[f64; 2]>>` | Raw sub-pixel outer edge inlier points used for ellipse fitting. |
| `edge_points_inner` | `Option<Vec<[f64; 2]>>` | Raw sub-pixel inner edge inlier points used for ellipse fitting. |
| `photometric` | `Option<PhotometricMetrics>` | Photometric fit summary, on every marker when the photometric method is selected: `status` (`accepted`, `insufficient_pixels`, `degenerate`, `not_converged`, `shift_too_large`, `implausible`, `poor_fit`, `no_board_position`), fitted `blur_sigma_px`, `contrast`, `residual_sigma`, `n_pixels`, `masked_fraction` and `center_shift_px`. |

`MarkerDiagnostics` is `#[non_exhaustive]` and derives `serde::Serialize` /
`serde::Deserialize`.

## FitMetrics

`FitMetrics` is present on every `MarkerDiagnostics` entry. It reports edge sampling coverage and ellipse fit quality.

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
| `max_angular_gap_outer` | `Option<f64>` | Largest angular gap between consecutive outer-edge samples (radians). Large gaps indicate missing ring coverage or occlusion. |
| `max_angular_gap_inner` | `Option<f64>` | Largest angular gap between consecutive inner-edge samples (radians). |
| `inner_fit_status` | `Option<InnerFitStatus>` | Outcome of the inner-fit stage: `ok`, `rejected`, or `failed`. |
| `inner_fit_reason` | `Option<InnerFitReason>` | Stable reason code explaining why inner fitting was rejected or failed. |
| `neighbor_radius_ratio` | `Option<f32>` | Ratio of this marker's outer radius to nearby decoded neighbors. Low values can indicate inner-as-outer contamination. |
| `inner_theta_consistency` | `Option<f32>` | Fraction of angular samples that agree on the inner-edge location. |
| `radii_std_outer_px` | `Option<f32>` | Standard deviation of per-ray outer radii. High spread suggests unstable outer-edge sampling. |
| `h_reproj_err_px` | `Option<f32>` | Final homography reprojection error for this marker in pixels. Present when a global homography is available. |

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

**Angular gaps** (`max_angular_gap_outer`, `max_angular_gap_inner`) help detect
partial arcs. Even with a decent point count, a single large missing sector can
make a fit less trustworthy than the residual alone suggests.

**Neighbor radius ratio** (`neighbor_radius_ratio`) is a structural sanity
check added late in the pipeline. Values well below `1.0` compared with nearby
decoded markers often indicate that some rays latched onto the inner ring
instead of the outer one.

**Homography reprojection error** (`h_reproj_err_px`) is the most directly
useful per-marker global consistency metric once a homography exists. Higher
values mean the marker's center disagrees with the board layout even if the
local ellipse fit looked good.

## DecodeMetrics

`DecodeMetrics` is present on a `MarkerDiagnostics` entry when the marker's own code band decoded. A decode is accepted only as an exact match (in some rotation) of one of the target's codewords, so it carries no distance or margin.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `observed_word` | `u16` | 16-bit word sampled from the code band on the phase-aligned sector grid. Each bit corresponds to one sector (bright = 1, dark = 0). |
| `best_id` | `usize` | Codebook ID of the matched codeword. |
| `rotation` | `f32` | Rotation of the printed code in sectors, `[0, 16)`: printed sector `j` starts at parametric angle `(j + rotation)·2π/16` on the fitted outer ellipse. The whole part is the cyclic shift that maps the codeword onto `observed_word`; the fraction is the estimated sub-sector phase. Each unit is 22.5°. |

### Polarity

Targets are printed dark on light; the decoder matches the sampled word as is and never tries its complement.

## Serialization

Both structs derive `serde::Serialize` and `serde::Deserialize`. Optional fields use `#[serde(skip_serializing_if = "Option::is_none")]` and are omitted from JSON output when absent.
