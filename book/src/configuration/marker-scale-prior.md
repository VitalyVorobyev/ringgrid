# MarkerScalePrior

`MarkerScalePrior` tells the detector the expected range of marker outer diameters in working-frame pixels. This single prior drives the derivation of proposal search radii, edge sampling extent, ellipse validation bounds, completion ROI size, and projective-center shift gates.

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `diameter_min_px` | `f32` | 14.0 | Minimum expected marker outer diameter in pixels. |
| `diameter_max_px` | `f32` | 66.0 | Maximum expected marker outer diameter in pixels. |

## Constructors

```rust
use ringgrid::MarkerScalePrior;

// Explicit range
let scale = MarkerScalePrior::new(24.0, 48.0);

// Fixed size (min == max)
let scale = MarkerScalePrior::from_nominal_diameter_px(32.0);
```

**`new(min, max)`** accepts any order; normalization swaps values if `min > max` and enforces a hard floor of 4.0 px on both bounds.

**`from_nominal_diameter_px(d)`** sets both `diameter_min_px` and `diameter_max_px` to `d`, producing a fixed-size prior. The same 4.0 px floor applies.

## Normalization

Every constructor and accessor normalizes the stored range:

1. Non-finite values are replaced with the corresponding default (14.0 or 66.0).
2. If `min > max`, the two are swapped.
3. `min` is clamped to at least 4.0 px.
4. `max` is clamped to at least `min`.

The `normalized()` method returns a normalized copy without mutating the original.

## Methods

| Method | Return | Description |
|---|---|---|
| `diameter_range_px()` | `[f32; 2]` | Normalized `[min, max]` diameter in pixels. |
| `nominal_diameter_px()` | `f32` | Midpoint of the range: `0.5 * (min + max)`. |
| `nominal_outer_radius_px()` | `f32` | Half of nominal diameter: `0.25 * (min + max)`. |

## Scale-dependent derivation

When a `DetectConfig` is constructed (or `set_marker_scale_prior()` is called), the scale prior drives the following parameter derivations. Let `r_min = diameter_min_px / 2`, `r_max = diameter_max_px / 2`, `r_nom = (r_min + r_max) / 2`, and `d_nom = r_min + r_max`:

### Proposal search radii

Proposal parameters depend on both the marker scale and the board geometry. The derivation uses a **spacing ratio** `S = min_center_spacing_mm / (2 * outer_radius_mm)` from the active `BoardLayout`, which captures how densely markers are packed relative to their size. Let `spacing_min_px = S * d_min` and `spacing_max_px = S * d_max`:

| Derived field | Formula |
|---|---|
| `proposal.r_min` | `max(0.15 * spacing_min_px, 2.0)` |
| `proposal.r_max` | `min(0.45 * spacing_max_px, 1.35 * r_max)` |
| `proposal.min_distance` | `max(0.16 * d_min, 0.85 * spacing_min_px)` |

### Edge sampling range

| Derived field | Formula |
|---|---|
| `edge_sample.r_max` | `2.0 * r_max` |
| `edge_sample.r_min` | `1.5` (fixed) |

### Outer estimation

| Derived field | Formula |
|---|---|
| `outer_estimation.search_halfwidth_px` | `max(max((r_max - r_min) * 0.5, 2.0), base_default)` |

Note: the number of angular rays for outer estimation is controlled by `EdgeSampleConfig::n_rays` (default 48), not by `OuterEstimationConfig`.

### Ellipse validation bounds

| Derived field | Formula |
|---|---|
| `min_semi_axis` | `max(0.3 * r_min, 2.0)` |
| `max_semi_axis` | `max(2.5 * r_max, min_semi_axis)` |

### Completion ROI

| Derived field | Formula |
|---|---|
| `completion.roi_radius_px` | `clamp(0.75 * d_nom, 24.0, 80.0)` |

### Projective center shift gate

| Derived field | Formula |
|---|---|
| `projective_center.max_center_shift_px` | `Some(2.0 * r_nom)` |

## Usage guidance

- **All markers roughly the same size**: use `from_nominal_diameter_px(d)`. This sets both bounds equal, producing tight proposal search and validation windows. Measure `d` as the outer ring diameter in pixels at the typical working distance.

- **Markers vary in apparent size** (perspective, varying distance): use `new(min, max)` with the smallest and largest expected diameters. This widens search and validation windows to accommodate the range. A wider range makes detection more permissive but may increase false positives.

- **Unsure about scale**: start with the default (14--66 px) and inspect detection results. Narrow the range once you know the actual marker sizes in your images.

- **Post-construction update**: call `config.set_marker_scale_prior(new_scale)` or `config.set_marker_diameter_hint_px(d)` to re-derive all coupled parameters without rebuilding the full config.

- **Very wide scale variation**: use adaptive multi-scale methods
  (`Detector::detect_adaptive`, `Detector::detect_adaptive_with_hint`, or
  `Detector::detect_multiscale`) instead of forcing one very wide prior.

## Source

`crates/ringgrid/src/detector/config.rs`
