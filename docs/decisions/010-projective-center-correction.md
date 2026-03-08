# DEC-010: Projective Center Correction — Once Per Marker

**Status:** Active
**Date:** 2025

## Decision

Projective center correction (conic-pencil eigenpair recovery from inner/outer
ellipses) is applied exactly **once per detected marker** during `finalize.rs`.

Each marker gets corrected at the point it enters the pipeline:

| Marker origin | When corrected |
|---------------|----------------|
| Fit-decode markers | Before global filter (line 139-141 of `finalize.rs`) |
| Completion markers | After completion, applied only to the new slice `[n_before_completion..]` (line 171-173) |

### Why once per marker

- Fit-decode markers have their final ellipses at this point — no subsequent
  stage changes their ellipse fits.
- Completion markers are new detections; they get corrected immediately after
  being added.
- There is no re-correction of already-corrected markers.

### Quality gates

Each correction applies gates from `ProjectiveCenterParams`:
- `max_center_shift_px` — rejects large jumps (default: `2 x nominal_outer_radius`).
- `max_selected_residual` — rejects bad eigenpair selection (default: 0.25).
- `min_eig_separation` — rejects unstable conic pencils (default: 1e-6).

### Requirement

Both `ellipse_inner` and `ellipse_outer` must be present on a marker for
correction to apply. Markers with only an outer ellipse are silently skipped.

### Mathematical foundation

Uses Wang et al. (2019) conic-pencil approach: for concentric circles under
perspective, the true projected center is the intersection of polar lines
from the generalized eigenvectors of `Q_outer x Q_inner^{-1}`.
Scale-invariant (tested).
