# DEC-004: Pipeline Stage Ordering

**Status:** Active
**Date:** 2025

## Decision

The detection pipeline executes in a fixed stage order. Stages must not be
reordered, skipped, or duplicated without understanding downstream dependencies.

### Stage sequence

```
fit_decode.rs (stages 1–6):
  1. Proposal         — Scharr gradient voting + NMS → candidate centers
  2. Outer Estimate   — radius hypotheses via radial profile peaks
  3. Outer Fit        — RANSAC ellipse fitting (Fitzgibbon direct LS)
  4. Decode           — 16-sector code sampling → codebook match
  5. Inner Fit        — inner ring ellipse fit
  6. Dedup            — spatial + ID-based deduplication

finalize.rs (stages 7–10):
  7. Projective Center — correct fit-decode marker centers (once per marker)
  8. Global Filter     — RANSAC homography (≥4 decoded markers)
  9. Completion        — conservative fits at missing H-projected IDs
     + Projective Center for completion-only markers (once per new marker)
 10. Final H Refit     — refit homography from all corrected centers
```

### Critical ordering dependencies

- **Outer Fit before Decode:** decode samples the code band at radii derived
  from the outer ellipse.
- **Dedup before Global Filter:** prevents duplicate IDs from corrupting the
  homography.
- **Projective Center before Global Filter:** corrected centers improve
  homography estimation.
- **Global Filter before Completion:** completion uses the homography to
  project missing board positions.
- **Projective Center on completion markers:** applied only to the new slice,
  not re-applied to already-corrected markers.
- **Final H Refit uses all corrected centers:** must run after all corrections.

### Two-pass detection (separate from stage ordering)

When a `PixelMapper` is active, the entire pipeline runs twice:
- Pass 1: in image space, no mapper.
- Pass 2: in working (undistorted) space, with mapper, seeded by pass-1 results.
Each pass executes the full stage sequence independently. Markers from
different passes are distinct detections.

## Orchestration

- `pipeline/run.rs` — top-level sequencing and two-pass/self-undistort logic.
- `pipeline/fit_decode.rs` — stages 1–6.
- `pipeline/finalize.rs` — stages 7–10.
