# DetectConfig

`DetectConfig` is the top-level configuration struct for the ringgrid detection pipeline. It aggregates all sub-configurations -- from proposal generation and edge sampling through homography RANSAC and self-undistort -- into a single value that drives every stage of detection.

## Construction

`DetectConfig` is designed to be built from a `BoardLayout` (target geometry) and an optional scale prior. Three recommended constructors cover the common cases:

```rust
use ringgrid::{BoardLayout, DetectConfig, MarkerScalePrior};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("board_spec.json")).unwrap();

// 1. Default scale prior (20--56 px diameter range)
let cfg = DetectConfig::from_target(board.clone());

// 2. Explicit scale range
let scale = MarkerScalePrior::new(24.0, 48.0);
let cfg = DetectConfig::from_target_and_scale_prior(board.clone(), scale);

// 3. Fixed marker diameter hint (min == max)
let cfg = DetectConfig::from_target_and_marker_diameter(board.clone(), 32.0);
```

All three constructors call two internal derivation functions:

- **`apply_target_geometry_priors`** -- derives `marker_spec.r_inner_expected` and `decode.code_band_ratio` from the board's inner/outer radius ratio.
- **`apply_marker_scale_prior`** -- derives proposal radii, edge sampling range, ellipse validation bounds, completion ROI, and projective-center shift gate from the scale prior. See [MarkerScalePrior](marker-scale-prior.md) for the full derivation rules.

## The Detector wrapper

Most users interact with `DetectConfig` through the `Detector` struct, which wraps a config and exposes detection methods:

```rust
use ringgrid::{BoardLayout, Detector, MarkerScalePrior};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("board_spec.json")).unwrap();

// Convenience constructors mirror DetectConfig
let det = Detector::new(board.clone());                           // default scale
let det = Detector::with_marker_scale(board.clone(),
              MarkerScalePrior::new(24.0, 48.0));                 // explicit range
let det = Detector::with_marker_diameter_hint(board.clone(), 32.0); // fixed size

// One-step from JSON file
let det = Detector::from_target_json_file(Path::new("board_spec.json")).unwrap();

// Full config control
let cfg = DetectConfig::from_target(board);
let det = Detector::with_config(cfg);

// Detect
let result = det.detect(&image);
```

## Post-construction tuning

After building a `Detector`, use `config_mut()` to override individual fields:

```rust
let mut det = Detector::new(board);
det.config_mut().completion.enable = false;
det.config_mut().use_global_filter = false;
det.config_mut().self_undistort.enable = true;
```

Calling `set_marker_scale_prior()` or `set_marker_diameter_hint_px()` on `DetectConfig` re-derives all scale-coupled parameters automatically.

## Field reference

| Field | Type | Default | Purpose |
|---|---|---|---|
| `marker_scale` | `MarkerScalePrior` | 20.0--56.0 px | Expected marker diameter range in pixels. Drives derivation of many downstream parameters. |
| `outer_estimation` | `OuterEstimationConfig` | (see sub-configs) | Outer-edge radius hypothesis generation from radial profile peaks. |
| `proposal` | `ProposalConfig` | (derived from scale) | Scharr gradient voting and NMS proposal generation. `r_min`, `r_max`, `nms_radius` are auto-derived. |
| `seed_proposals` | `SeedProposalParams` | merge=3.0, score=1e12, max=512 | Controls seed injection for multi-pass detection. |
| `edge_sample` | `EdgeSampleConfig` | (derived from scale) | Radial edge sampling range and ray count. `r_min`, `r_max` are auto-derived. |
| `decode` | `DecodeConfig` | (derived from board) | 16-sector code sampling. `code_band_ratio` is auto-derived from board geometry. |
| `marker_spec` | `MarkerSpec` | (derived from board) | Marker geometry specification. `r_inner_expected` is auto-derived from board inner/outer radius ratio. |
| `inner_fit` | `InnerFitConfig` | (see sub-configs) | Robust inner ellipse fitting: RANSAC params, validation gates. |
| `circle_refinement` | `CircleRefinementMethod` | `ProjectiveCenter` | Center correction strategy selector: `None` or `ProjectiveCenter`. |
| `projective_center` | `ProjectiveCenterParams` | (see sub-configs) | Projective center recovery gates and tuning. `max_center_shift_px` is auto-derived from scale. |
| `completion` | `CompletionParams` | (see sub-configs) | Completion at missing H-projected board positions. `roi_radius_px` is auto-derived from scale. |
| `min_semi_axis` | `f64` | 3.0 | Minimum semi-axis length (px) for a valid outer ellipse. Auto-derived from scale. |
| `max_semi_axis` | `f64` | 15.0 | Maximum semi-axis length (px) for a valid outer ellipse. Auto-derived from scale. |
| `max_aspect_ratio` | `f64` | 3.0 | Maximum aspect ratio (a/b) for a valid ellipse. |
| `dedup_radius` | `f64` | 6.0 | NMS deduplication radius (px) for final markers. |
| `use_global_filter` | `bool` | `true` | Enable RANSAC homography global filter (requires board layout with marker positions). |
| `ransac_homography` | `RansacHomographyConfig` | iters=2000, thresh=5.0 | RANSAC parameters for homography estimation. |
| `board` | `BoardLayout` | empty | Board layout defining marker positions and geometry. |
| `id_correction` | `IdCorrectionConfig` | enabled | Structural consistency verification/recovery of decoded IDs before global filter. |
| `self_undistort` | `SelfUndistortConfig` | disabled | Self-undistort estimation from conic consistency of detected ring edges. |

Fields marked "auto-derived" are overwritten by the constructors. If you modify `marker_scale` after construction, call `set_marker_scale_prior()` to re-derive them.

## Source

`crates/ringgrid/src/detector/config.rs`, `crates/ringgrid/src/api.rs`
