# DEC-002: Coordinate Frame Contracts

**Status:** Active
**Date:** 2025

## Decision

Two coordinate frames exist: **Image** (raw distorted pixels) and **Working**
(undistorted pixels, produced by a `PixelMapper`).

### Invariants

1. **`DetectedMarker.center` is always image-space.**
   Regardless of whether a mapper is active, the `center` field holds raw
   image-pixel coordinates. This is the stable anchor for downstream consumers.

2. **`DetectedMarker.center_mapped` holds working-frame coordinates** when a
   mapper is active; `None` otherwise.

3. **Ellipses (`ellipse_outer`, `ellipse_inner`) are in the working frame**
   when a mapper is active; image-space otherwise.

4. **The homography maps board mm → working frame** (or board mm → image when
   no mapper is used). `DetectionResult.homography_frame` records which.

5. **`DetectionResult.center_frame`** is always `Image` (centers are never
   rewritten to working-frame). `homography_frame` is `Working` when a mapper
   was used.

### Edge points

`edge_points_outer` and `edge_points_inner` live in the same frame as the
corresponding ellipse (working frame if mapper active, image otherwise).

## Rationale

Consumers need a single reliable anchor for overlay rendering. Image-pixel
centers serve that role. Working-frame ellipses and homography keep the
geometric model internally consistent.

## Violations to watch for

- Never store working-frame coordinates in `center`.
- Never mix image-frame edge points with working-frame ellipse fits.
- When applying projective center correction, the correction operates on
  `center` (or `center_mapped` in working frame) — both must stay consistent.
