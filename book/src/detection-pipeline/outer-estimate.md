# Outer Radius Estimation

Before fitting an ellipse to the outer ring edge, the pipeline needs a radius estimate to anchor the search. The outer radius estimator samples radial intensity profiles around the proposal center and identifies the outer ring edge as a peak in the aggregated radial derivative.

## Why This Stage Exists

A ring marker has multiple concentric edges (inner ring, code band boundaries, outer ring). Without guidance, an edge sampler might lock onto the wrong edge. This estimator uses the `MarkerScalePrior` to focus the search on a narrow window around the expected outer radius, avoiding confusion with stronger inner or code-band edges.

## Algorithm

### Radial Intensity Sampling

From the proposal center, the estimator casts `theta_samples` (default: 48) radial rays evenly spaced in angle. Along each ray, `radial_samples` (default: 64) intensity values are sampled at uniform radial steps within a search window:

```
window = [r_expected - search_halfwidth, r_expected + search_halfwidth]
```

where `r_expected` is the nominal outer radius from `MarkerScalePrior` and `search_halfwidth_px` (default: 4.0 px) defines the search extent. The window minimum is clamped to at least 1.0 px.

When a `PixelMapper` is active, sampling is distortion-aware: the `DistortionAwareSampler` maps working-frame coordinates to image-frame coordinates for pixel lookup, using bilinear interpolation.

### Radial Derivative Computation

For each ray, the sampled intensity profile is differentiated using central differences to produce a `dI/dr` curve:

```
d[i] = (I[i+1] - I[i-1]) / (2 * r_step)   for interior samples
d[0] = (I[1] - I[0]) / r_step               forward difference at boundary
d[N-1] = (I[N-1] - I[N-2]) / r_step         backward difference at boundary
```

A 3-point moving average smooth is applied to reduce noise.

### Theta Coverage Check

Rays that go out of image bounds are discarded. If the fraction of valid rays falls below `min_theta_coverage` (default: 0.6), the estimate fails. This prevents unstable results when the marker is partially occluded or near the image boundary.

### Polarity Selection and Aggregation

The outer ring edge has a characteristic sign in `dI/dr` depending on contrast polarity:

- **Dark-to-light** (`Polarity::Pos`): Moving outward, intensity increases at the outer edge (dark ring interior to bright background).
- **Light-to-dark** (`Polarity::Neg`): The opposite convention.

The `grad_polarity` setting (default: `DarkToLight`) determines which polarities are tried. In `Auto` mode, both are evaluated and the best is selected.

For each polarity candidate, the per-theta derivative curves are aggregated at each radial sample using the configured `AngularAggregator`:

- **Median** (default): Robust to outlier rays from code-band sectors.
- **TrimmedMean**: Trims a configurable fraction of extreme values before averaging.

### Peak Detection

Local maxima in the aggregated response (or its negation for `Neg` polarity) are identified. Peaks at the search window boundaries are excluded. Each peak is evaluated for **theta consistency**: the fraction of per-theta peaks that fall within a tolerance of the aggregated peak radius. Peaks with theta consistency below `min_theta_consistency` (default: 0.35) are rejected.

### Multiple Hypotheses

When `allow_two_hypotheses` is enabled (default: true), the estimator may return up to two hypotheses if the runner-up peak has at least `second_peak_min_rel` (default: 85%) of the best peak's strength. Multiple hypotheses improve robustness when the expected radius is slightly off: both candidates are evaluated in the [outer fit stage](outer-fit.md) and the better one is selected.

## Output

The `OuterEstimate` struct contains:

- `r_outer_expected_px`: The expected radius from the scale prior.
- `search_window_px`: The `[min, max]` radial search window.
- `polarity`: The selected contrast polarity.
- `hypotheses`: Up to two `OuterHypothesis` structs, sorted best-first, each with `r_outer_px`, `peak_strength`, and `theta_consistency`.
- `status`: `Ok` or `Failed` with a diagnostic reason.

## Configuration

The `OuterEstimationConfig` struct controls this stage:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_halfwidth_px` | 4.0 | Search half-width around expected radius |
| `radial_samples` | 64 | Number of radial samples per ray |
| `theta_samples` | 48 | Number of angular rays |
| `aggregator` | Median | Angular aggregation method |
| `grad_polarity` | DarkToLight | Expected edge polarity |
| `min_theta_coverage` | 0.6 | Minimum fraction of valid rays |
| `min_theta_consistency` | 0.35 | Minimum fraction of rays agreeing with peak |
| `allow_two_hypotheses` | true | Emit runner-up hypothesis if strong enough |
| `second_peak_min_rel` | 0.85 | Runner-up must be this fraction of best peak |
| `refine_halfwidth_px` | 1.0 | Per-theta local refinement half-width |

When `DetectConfig` derives parameters from `MarkerScalePrior`, the search halfwidth is expanded to cover the full diameter range.

## Connection to Adjacent Stages

The outer estimate receives the proposal center from the [proposal stage](proposal.md) and the expected radius from `MarkerScalePrior`. Its hypotheses are consumed by the [outer ellipse fit stage](outer-fit.md), which samples edge points near each hypothesis radius and fits ellipses to evaluate which hypothesis produces the best detection.

**Source:** `ring/outer_estimate.rs`, `ring/radial_profile.rs`
