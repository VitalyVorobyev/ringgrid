# Sub-Configurations

`DetectConfig` aggregates several focused sub-configuration structs. This chapter documents each one with its fields, defaults, and role in the detection pipeline.

---

## SeedProposalParams

Controls seed injection for multi-pass detection. In the two-pass pipeline, pass-1 detections are injected as high-priority seed proposals for pass-2.

| Field | Type | Default | Description |
|---|---|---|---|
| `merge_radius_px` | `f32` | 3.0 | Radius (px) for merging seed centers with detector proposals. Seeds within this distance of an existing proposal are merged rather than duplicated. |
| `seed_score` | `f32` | 1e12 | Score assigned to injected seed proposals. The high default ensures seeds survive NMS against weaker gradient-based proposals. |
| `max_seeds` | `Option<usize>` | `Some(512)` | Maximum number of seeds consumed per run. `None` removes the cap. |

**Source:** `crates/ringgrid/src/detector/config.rs`

---

## CompletionParams

Controls the homography-guided completion stage. After the global homography filter identifies inliers, the pipeline projects all board marker positions into the image and attempts local fits at positions where no marker was detected. Completion only runs when a valid homography is available.

| Field | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `true` | Master switch for the completion stage. |
| `roi_radius_px` | `f32` | 24.0 | Radial sampling extent (px) for edge sampling around the projected center. Auto-derived from scale prior as `clamp(0.75 * d_nom, 24, 80)`. |
| `reproj_gate_px` | `f32` | 3.0 | Maximum reprojection error (px) between the fitted center and the H-projected board center. Fits exceeding this gate are rejected. |
| `min_fit_confidence` | `f32` | 0.45 | Minimum fit confidence score in [0, 1] for accepting a completion fit. |
| `min_arc_coverage` | `f32` | 0.35 | Minimum arc coverage (fraction of rays with both edges found). Low coverage indicates the marker is partially occluded or near the image boundary. |
| `max_attempts` | `Option<usize>` | `None` | Optional cap on the number of completion fits attempted, in ID order. `None` means try all missing positions. |
| `image_margin_px` | `f32` | 10.0 | Skip completion attempts whose projected center is closer than this to the image boundary. |

**Source:** `crates/ringgrid/src/detector/config.rs`

---

## ProjectiveCenterParams

Controls projective center recovery from the inner/outer conic pencil. When enabled, the detector computes an unbiased center estimate from the intersection geometry of the inner and outer fitted ellipses, correcting for perspective bias in the naive ellipse-center estimate.

Center correction is applied once per marker during the pipeline: before the global filter for fit-decode markers, and after completion for newly added markers.

| Field | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `true` | Enable projective center estimation. |
| `use_expected_ratio` | `bool` | `true` | Use `marker_spec.r_inner_expected` as an eigenvalue prior when selecting among candidate centers. |
| `ratio_penalty_weight` | `f64` | 1.0 | Weight of the eigenvalue-vs-expected-ratio penalty term in candidate selection. Higher values prefer candidates whose conic-pencil eigenvalue ratio matches the expected inner/outer ratio. |
| `max_center_shift_px` | `Option<f64>` | `None` | Maximum allowed shift (px) from the pre-correction center. Large jumps are rejected and the original center is kept. Auto-derived from scale prior as `2 * r_nom`. |
| `max_selected_residual` | `Option<f64>` | `Some(0.25)` | Maximum accepted projective-selection residual. Higher values are less strict. `None` disables this gate. |
| `min_eig_separation` | `Option<f64>` | `Some(1e-6)` | Minimum eigenvalue separation for a stable conic-pencil eigenpair. Low separation indicates numerical instability. `None` disables this gate. |

**Source:** `crates/ringgrid/src/detector/config.rs`

---

## InnerFitConfig

Controls robust inner ellipse fitting. After the outer ellipse is fitted and the code is decoded, the detector fits an inner ellipse to edge points sampled at the expected inner ring radius. The inner ellipse is required for projective center recovery.

| Field | Type | Default | Description |
|---|---|---|---|
| `min_points` | `usize` | 20 | Minimum number of sampled edge points required to attempt a fit. |
| `min_inlier_ratio` | `f32` | 0.5 | Minimum RANSAC inlier ratio for accepting the inner fit. |
| `max_rms_residual` | `f64` | 1.0 | Maximum accepted RMS Sampson residual (px) of the fitted inner ellipse. |
| `max_center_shift_px` | `f64` | 12.0 | Maximum allowed center shift (px) from the outer ellipse center to the inner ellipse center. |
| `max_ratio_abs_error` | `f64` | 0.15 | Maximum absolute error between the recovered inner/outer scale ratio and the radial hint. |
| `local_peak_halfwidth_idx` | `usize` | 3 | Half-width (in radius-sample indices) of the local search window around the radial hint peak. |

### Inner fit RANSAC sub-config

The `ransac` field is a `RansacConfig` struct embedded within `InnerFitConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `ransac.max_iters` | `usize` | 200 | Maximum RANSAC iterations for inner ellipse fitting. |
| `ransac.inlier_threshold` | `f64` | 1.5 | Inlier threshold (Sampson distance in px). |
| `ransac.min_inliers` | `usize` | 8 | Minimum inlier count for a valid inner ellipse model. |
| `ransac.seed` | `u64` | 43 | Random seed for reproducibility. |

**Source:** `crates/ringgrid/src/detector/config.rs`

---

## CircleRefinementMethod

Enum selector for the center correction strategy applied after local ellipse fits.

| Variant | Description |
|---|---|
| `None` | Disable center correction. The naive ellipse center is used as-is. |
| `ProjectiveCenter` | **(default)** Run projective center recovery from the inner/outer conic pencil. Requires both inner and outer ellipses to be successfully fitted. |

```rust
use ringgrid::CircleRefinementMethod;

// Check if projective center is active
let method = CircleRefinementMethod::ProjectiveCenter;
assert!(method.uses_projective_center());
```

**Source:** `crates/ringgrid/src/detector/config.rs`

---

## RansacHomographyConfig

Controls the RANSAC homography estimation used for global filtering and completion. The homography maps board-space marker positions (mm) to image-space pixel coordinates.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_iters` | `usize` | 2000 | Maximum RANSAC iterations. |
| `inlier_threshold` | `f64` | 5.0 | Inlier threshold: maximum reprojection error (px) for a correspondence to be counted as an inlier. |
| `min_inliers` | `usize` | 6 | Minimum number of inliers for the homography to be accepted. The pipeline requires at least 4 decoded markers to attempt RANSAC. |
| `seed` | `u64` | 0 | Random seed for reproducibility. |

**Source:** `crates/ringgrid/src/homography/core.rs`

---

## SelfUndistortConfig

Controls intrinsics-free distortion estimation from ring marker conic consistency. When enabled, the detector estimates a one-parameter division model that maps distorted image coordinates to undistorted working coordinates. The optimization minimizes the RMS Sampson residual of inner/outer ellipse fits across all detected markers using golden-section search over the lambda parameter.

| Field | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `false` | Master switch. When `false`, no self-undistort estimation runs. |
| `lambda_range` | `[f64; 2]` | `[-8e-7, 8e-7]` | Search range for the division model parameter lambda. |
| `max_evals` | `usize` | 40 | Maximum function evaluations for the golden-section 1D optimizer. |
| `min_markers` | `usize` | 6 | Minimum number of markers with both inner and outer edge points required to attempt estimation. |
| `improvement_threshold` | `f64` | 0.01 | Relative improvement threshold: the model is applied only if `(baseline - optimum) / baseline` exceeds this value. |
| `min_abs_improvement` | `f64` | 1e-4 | Minimum absolute objective improvement required. Prevents applying corrections when the objective is near the numerical noise floor. |
| `trim_fraction` | `f64` | 0.1 | Trim fraction for robust aggregation: drop this fraction of scores from both tails before averaging per-marker objectives. |
| `min_lambda_abs` | `f64` | 5e-9 | Minimum absolute value of lambda required. Very small lambda values are treated as "no correction". |
| `reject_range_edge` | `bool` | `true` | Reject solutions that land near the lambda search range boundaries, which may indicate the true optimum lies outside the range. |
| `range_edge_margin_frac` | `f64` | 0.02 | Fraction of the lambda range treated as an unstable boundary zone. |
| `validation_min_markers` | `usize` | 24 | Minimum decoded-ID correspondences needed for homography-based validation of the estimated model. |
| `validation_abs_improvement_px` | `f64` | 0.05 | Minimum absolute homography self-error improvement (px) required during validation. |
| `validation_rel_improvement` | `f64` | 0.03 | Minimum relative homography self-error improvement required during validation. |

When the number of decoded markers exceeds `validation_min_markers`, the estimator uses a homography-based objective (reprojection error) instead of the conic-consistency fallback. The final model must pass both the conic-consistency improvement check and the homography validation check (if enough markers are available) to be applied.

**Source:** `crates/ringgrid/src/pixelmap/self_undistort.rs`
