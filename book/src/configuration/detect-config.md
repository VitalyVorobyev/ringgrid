# DetectConfig

`DetectConfig` is the top-level configuration struct for the ringgrid detection pipeline. It aggregates all sub-configurations -- from proposal generation and edge sampling through homography RANSAC and self-undistort -- into a single value that drives every stage of detection.

## Construction

`DetectConfig` is designed to be built from a [`TargetLayout`](../targets/target-model.md) (target geometry) and an optional scale prior. Three recommended constructors cover the common cases:

```rust
use ringgrid::{TargetLayout, DetectConfig, MarkerScalePrior};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target_spec.json")).unwrap();

// 1. Default scale prior (14--66 px diameter range)
let cfg = DetectConfig::from_target(target.clone());

// 2. Explicit scale range
let scale = MarkerScalePrior::new(24.0, 48.0);
let cfg = DetectConfig::from_target_and_scale_prior(target.clone(), scale);

// 3. Fixed marker diameter hint (min == max)
let cfg = DetectConfig::from_target_and_marker_diameter(target.clone(), 32.0);
```

All three constructors call two internal derivation functions:

- **`apply_target_geometry_priors`** -- derives `marker_spec.r_inner_expected` and `decode.code_band_ratio` from the target's inner/outer radius ratio (accounting for the ring stroke on coded targets; plain annuli expose the radii directly).
- **`apply_marker_scale_prior`** -- derives proposal radii, edge sampling range, ellipse validation bounds, and completion ROI from the scale prior. See [MarkerScalePrior](marker-scale-prior.md) for the full derivation rules.

## The Detector wrapper

Most users interact with `DetectConfig` through the `Detector` struct, which wraps a config and exposes detection methods:

```rust
use ringgrid::{TargetLayout, Detector, MarkerScalePrior};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target_spec.json")).unwrap();

// Convenience constructors mirror DetectConfig
let det = Detector::new(target.clone());                           // default scale
let det = Detector::with_marker_scale(target.clone(),
              MarkerScalePrior::new(24.0, 48.0));                  // explicit range
let det = Detector::with_marker_diameter_hint(target.clone(), 32.0); // fixed size

// One-step from JSON file
let det = Detector::from_target_json_file(Path::new("target_spec.json")).unwrap();

// Full config control
let cfg = DetectConfig::from_target(target);
let det = Detector::with_config(cfg);

// Detect (0.8: detect* return Result<_, DetectError>)
let result = det.detect(&image).unwrap();

// Adaptive multi-scale APIs (wide size variation scenes)
let result = det.detect_adaptive(&image).unwrap();
let result = det.detect_adaptive_with_hint(&image, Some(32.0)).unwrap();
```

## Stable vs advanced fields

`DetectConfig` keeps only the durable user choices at the top level:

- `target` — target geometry (`TargetLayout`)
- `marker_scale` — expected marker diameter range
- `circle_refinement` — post-fit center-correction method
- `self_undistort` — division-model self-undistort policy
- `advanced` — an `AdvancedDetectConfig` holding every per-stage tuning knob

All stage tuning (proposal, edge sampling, fitting, decode, completion,
homography RANSAC, ID correction, global-filter toggles, …) lives under the
nested `advanced` field. The config dump/overlay JSON nests these under an
`"advanced"` object.

## Post-construction tuning

After building a `Detector`, use `config_mut()` to override individual fields.
Top-level fields are set directly; per-stage knobs go through `advanced`:

```rust
let mut det = Detector::new(target);
det.config_mut().self_undistort.enable = true;
det.config_mut().advanced.completion.enable = false;
det.config_mut().advanced.use_global_filter = false;
```

Calling `set_marker_scale_prior()` or `set_marker_diameter_hint_px()` on `DetectConfig` re-derives all scale-coupled parameters automatically.

## Field reference

The `Field` column shows the access path from a `DetectConfig` value. Per-stage
knobs are reached through the `advanced` sub-config.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `target` | `TargetLayout` | default hex | Target layout defining marker positions and geometry. Not serialized — re-attach via `with_target` when loading a config from JSON. |
| `marker_scale` | `MarkerScalePrior` | 14.0--66.0 px | Expected marker diameter range in pixels. Drives derivation of many downstream parameters. |
| `circle_refinement` | `CircleRefinementMethod` | `ProjectiveCenter` | Center correction strategy selector: `None` or `ProjectiveCenter`. |
| `self_undistort` | `SelfUndistortConfig` | disabled | Self-undistort estimation from conic consistency of detected ring edges. |
| `advanced.outer_estimation` | `OuterEstimationConfig` | (see sub-configs) | Outer-edge radius hypothesis generation from radial profile peaks. |
| `advanced.proposal` | `ProposalConfig` | (derived from scale) | Scharr gradient voting and NMS proposal generation. `r_min`, `r_max`, `nms_radius` are auto-derived. |
| `advanced.seed_proposals` | `SeedProposalConfig` | merge=3.0, score=1e12, max=512 | Controls seed injection for multi-pass detection. |
| `advanced.edge_sample` | `EdgeSampleConfig` | (derived from scale) | Radial edge sampling range and ray count. `r_min`, `r_max` are auto-derived. |
| `advanced.decode` | `DecodeConfig` | (derived from board) | 16-sector code sampling. `code_band_ratio` is auto-derived from board geometry; `codebook_profile` defaults to `base`. |
| `advanced.marker_spec` | `MarkerSpecConfig` | (derived from board) | Marker geometry specification. `r_inner_expected` is auto-derived from board inner/outer radius ratio. |
| `advanced.inner_fit` | `InnerFitConfig` | (see sub-configs) | Robust inner ellipse fitting: RANSAC params, validation gates. |
| `advanced.outer_fit` | `OuterFitConfig` | (see sub-configs) | Robust outer ellipse fitting: RANSAC params, scoring weights. |
| `advanced.projective_center` | `ProjectiveCenterConfig` | (see sub-configs) | Projective center recovery gates and tuning. `max_correction_shift_px` (renamed from `max_center_shift_px`) defaults to `None` = "auto" (nominal marker diameter at the point of use); explicit values survive target re-derivation. |
| `advanced.completion` | `CompletionConfig` | (see sub-configs) | Completion at missing H-projected board positions. `roi_radius_px` is auto-derived from scale. |
| `advanced.max_aspect_ratio` | `f64` | 3.0 | Maximum aspect ratio (a/b) for a valid ellipse. |
| `advanced.dedup_radius` | `f64` | 6.0 | NMS deduplication radius (px) for final markers. |
| `advanced.use_global_filter` | `bool` | `true` | Enable RANSAC homography global filter (requires board layout with marker positions). |
| `advanced.geometric_verify` | `bool` | `true` | Final precision-first geometric verification gate. After the final homography, each labeled marker is checked against its lattice-neighbor midpoint prediction (locally affine, so distortion-robust) and its final-H reprojection residual, with thresholds that adapt to the observed inlier-residual distribution. Markers the lattice judges inconsistent are removed, so only trusted board correspondences reach the output. Lattice-generic — applies to coded and plain targets. Set `false` to keep every marker and filter yourself. See [Detection Quality & Rejection](../detection-quality.md). |
| `advanced.ransac_homography` | `RansacConfig` | iters=2000, thresh=5.0 | RANSAC parameters for homography estimation. |
| `advanced.id_correction` | `IdCorrectionConfig` | enabled | Structural consistency verification/recovery of decoded IDs before global filter. |
| `advanced.inner_as_outer_recovery` | `InnerAsOuterRecoveryConfig` | enabled | Recovery for markers whose outer fit locked onto the inner ring. |

Fields marked "auto-derived" are overwritten by the constructors. If you modify `marker_scale` after construction, call `set_marker_scale_prior()` to re-derive them.

## Source

`crates/ringgrid/src/detector/config/`, `crates/ringgrid/src/api.rs`
