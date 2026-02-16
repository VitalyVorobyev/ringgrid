# Ring Structure

A ringgrid marker consists of two concentric circular rings printed on a planar
calibration target. These rings serve a dual purpose: their edges provide
high-contrast features for sub-pixel ellipse fitting, and the annular region
between them carries a binary code that identifies each marker uniquely.

## Physical geometry

Each marker is defined by two radii measured from the marker center:

| Parameter | Default value | Description |
|-----------|---------------|-------------|
| Outer radius | 4.8 mm | Radius of the outer ring centerline |
| Inner radius | 3.2 mm | Radius of the inner ring centerline |
| Ring width | 0.576 mm (0.12 * outer radius) | Width of each dark ring band |
| Pitch | 8.0 mm | Center-to-center spacing on the hex lattice |

The **outer ring** is a dark annular band centered at the outer radius. Its
outer edge (at `outer_radius + ring_width/2`) forms the outermost visible
boundary of the marker. Similarly, the **inner ring** is a dark annular band
centered at the inner radius.

Between the two dark ring bands lies the **code band** -- the annular region
where binary sector patterns encode the marker's identity. The code band
occupies the gap between the inner edge of the outer ring and the outer edge of
the inner ring.

<!-- TODO: Cross-section diagram showing ring structure -->

## Ring bands and edge detection

The detector identifies markers by locating the sharp intensity transitions at
the boundaries of the dark ring bands. Under image blur, the physical ring
width causes these transitions to broaden, so the detector targets the
*boundary* of the merged dark band rather than the ring centerline. This means
the effective detected edges sit at:

- **Outer edge**: `outer_radius + ring_width` (outside of the outer band)
- **Inner edge**: `inner_radius - ring_width` (inside of the inner band)

For the default geometry:

- Outer edge = 4.8 + 0.576 = 5.376 mm, but in normalized units the detector
  works with `outer_radius * (1 + 0.12)` = pitch * 0.672
- Inner edge = 3.2 - 0.576 = 2.624 mm, or pitch * 0.328

The ratio of these detected edges defines the key geometric invariant used
during inner ring estimation.

## Outer-normalized coordinates

Internally, ringgrid expresses all marker geometry in **outer-normalized
coordinates** where the detected outer edge radius equals 1.0. This
normalization makes the geometry scale-invariant: the same `MarkerSpec`
parameters apply regardless of the marker's apparent size in the image.

In these units, the expected inner edge radius is:

```
r_inner_expected = (inner_radius - ring_width) / (outer_radius + ring_width)
                 = (pitch * 0.328) / (pitch * 0.672)
                 ≈ 0.488
```

The pitch cancels, so this ratio depends only on the relative proportions of
the marker design, not on the physical scale.

## The `MarkerSpec` type

The `MarkerSpec` struct encodes the expected marker geometry and controls how
the inner ring estimator searches for the inner edge:

```rust
pub struct MarkerSpec {
    /// Expected inner radius as fraction of outer radius.
    pub r_inner_expected: f32,
    /// Allowed deviation in normalized radius around `r_inner_expected`.
    pub inner_search_halfwidth: f32,
    /// Expected sign of dI/dr at the inner edge.
    pub inner_grad_polarity: GradPolarity,
    /// Number of radii samples per theta.
    pub radial_samples: usize,
    /// Number of theta samples.
    pub theta_samples: usize,
    /// Aggregator across theta.
    pub aggregator: AngularAggregator,
    /// Minimum fraction of theta samples required for a valid estimate.
    pub min_theta_coverage: f32,
    /// Minimum fraction of theta samples that must agree on
    /// the inner edge location.
    pub min_theta_consistency: f32,
}
```

Key defaults:

| Field | Default | Notes |
|-------|---------|-------|
| `r_inner_expected` | 0.488 | 0.328 / 0.672 |
| `inner_search_halfwidth` | 0.08 | Search window: [0.408, 0.568] |
| `inner_grad_polarity` | `LightToDark` | Light center to dark inner ring |
| `radial_samples` | 64 | Resolution along radial profiles |
| `theta_samples` | 96 | Angular samples around the ring |
| `aggregator` | `Median` | Robust to code-band sector outliers |
| `min_theta_coverage` | 0.6 | At least 60% of angles must be valid |
| `min_theta_consistency` | 0.35 | At least 35% must agree on edge location |

The `search_window()` method returns the normalized radial interval
`[r_inner_expected - halfwidth, r_inner_expected + halfwidth]` where the inner
ring estimator looks for the intensity transition.

## Gradient polarity

The `GradPolarity` enum describes the expected direction of the radial
intensity change at a ring edge:

```rust
pub enum GradPolarity {
    DarkToLight,   // dI/dr > 0: intensity increases outward
    LightToDark,   // dI/dr < 0: intensity decreases outward
    Auto,          // try both, pick the more coherent peak
}
```

For the default marker design (dark rings on a light background), the inner
edge of the inner ring is a `LightToDark` transition when traversing radially
outward from the marker center: you move from the light center region into the
dark inner ring band.

## Design constraints

The marker geometry must satisfy several constraints for reliable detection:

1. **Non-overlapping markers**: The outer diameter (2 * outer_radius) must be
   smaller than the minimum center-to-center distance on the hex lattice
   (`pitch * sqrt(3)`). The default 4.8 mm radius gives a 9.6 mm diameter
   versus a ~13.86 mm nearest-neighbor distance.

2. **Sufficient code band width**: The gap between inner and outer rings must
   be wide enough to sample 16 angular sectors with adequate spatial
   resolution.

3. **Ring width vs. blur**: The ring bands must be wide enough to produce
   detectable gradient peaks after optical blur, but narrow enough not to
   encroach on the code band.

These relationships are baked into the `MarkerSpec` defaults and validated by
the `BoardLayout` loader (see [Hex Lattice Layout](hex-lattice.md)).
