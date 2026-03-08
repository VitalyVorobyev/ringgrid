# DEC-015: Two-Pass Detection Architecture

**Status:** Active
**Date:** 2025

## Decision

When a `PixelMapper` (camera model) is available, detection runs two passes:

### Pass 1 (baseline)

- Runs without mapper, in image-pixel space.
- Produces initial marker detections used as **seeds** for pass 2.

### Pass 2 (mapper-aware)

- Uses pass-1 detections as seed proposals (converted to `Proposal` structs).
- Runs with the mapper active (distortion-aware edge sampling).
- Seeds are merged with gradient-voting proposals via `SeedProposalParams`.

### Self-undistort variant

When `SelfUndistortConfig.enable` is true:
1. Run pass 1 (no mapper).
2. Estimate a division-model distortion from detected ellipse edge points.
3. If estimation succeeds and `applied` is true, run pass 2 with the
   estimated `DivisionModel` as the mapper.

### Seed proposal contract

- Seeds come from `DetectionResult::seed_proposals()`.
- Each seed carries the detection confidence as its score.
- `SeedProposalParams.max_seeds` caps the number consumed (default: 512).
- `merge_radius_px` (default: 3.0) is used to merge seeds with newly
  detected proposals.
