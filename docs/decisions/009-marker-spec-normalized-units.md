# DEC-009: MarkerSpec Uses Outer-Normalized Radius Units

**Status:** Active
**Date:** 2025

## Decision

`MarkerSpec` expresses all radial geometry in **outer-normalized units**
(outer radius = 1.0). This decouples geometry specification from pixel scale.

### Key fields

| Field | Meaning | Default |
|-------|---------|---------|
| `r_inner_expected` | Inner edge radius / outer edge radius | 0.488 (≈ 0.328/0.672) |
| `inner_search_halfwidth` | ± window around `r_inner_expected` | 0.08 |
| `code_band_ratio` | Midpoint of code sampling band (normalized) | Derived from `r_inner_expected` |

### Derivation from physical geometry

Defaults match `tools/gen_synth.py`:
- `outer_radius = pitch_mm × 0.6`
- `inner_radius = pitch_mm × 0.4`
- `ring_width = outer_radius × 0.12`
- Edge sampler finds merged dark-band boundary, so:
  - `r_inner_edge = inner_radius - ring_width`
  - `r_outer_edge = outer_radius + ring_width`
  - `r_inner_expected = r_inner_edge / r_outer_edge`

### Auto-derivation

When `BoardLayout` provides `marker_outer_radius_mm` and
`marker_inner_radius_mm`, `apply_target_geometry_priors()` recomputes
`r_inner_expected` and `code_band_ratio` with a 12% edge expansion factor.

### Invariant

All radial search windows, edge sampling, and inner/outer scale estimation
operate in these normalized coordinates. Pixel scale enters only through
`MarkerScalePrior` (which converts normalized radii to pixel radii).
