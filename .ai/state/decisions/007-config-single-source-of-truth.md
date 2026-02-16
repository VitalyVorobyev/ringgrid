# DEC-007: Config — Single Source of Truth

**Status:** Active
**Date:** 2025

## Decision

Each configuration value has exactly one authoritative definition. No mirrored
structs, no duplicated defaults.

### Rules

1. **One config struct per domain.** No `*Params` vs `*Config` twins carrying
   the same fields. If a translation layer is unavoidable, document why it
   exists and ensure only one side owns defaults.

2. **Scale-dependent parameters are auto-derived.** When `MarkerScalePrior`
   changes (via `set_marker_scale_prior` or constructors), all coupled
   parameters are recomputed by `apply_marker_scale_prior()`:
   - Proposal search radii (`r_min`, `r_max`, `nms_radius`)
   - Edge sampling range (`edge_sample.r_max`, `edge_sample.r_min`)
   - Outer estimation window (`search_halfwidth_px`, `theta_samples`)
   - Ellipse validation bounds (`min_semi_axis`, `max_semi_axis`)
   - Completion ROI (`roi_radius_px`)
   - Projective center max shift

3. **Board geometry priors are auto-derived.** `apply_target_geometry_priors()`
   computes `r_inner_expected` and `code_band_ratio` from board outer/inner
   radii.

4. **Recommended construction:** use `DetectConfig::from_target_and_scale_prior`,
   `from_target`, or `from_target_and_marker_diameter`. Direct `Default::default()`
   also applies all priors.

### Anti-patterns to avoid

- Adding a threshold in module A that duplicates a field in `DetectConfig`.
- Creating a local `Params` struct that mirrors fields from an existing config.
- Hard-coding magic numbers that should flow from `MarkerScalePrior`.
