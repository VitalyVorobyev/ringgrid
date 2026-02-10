# Module Structure

This document describes the refactored `ringgrid` module boundaries.

## Goals

- Keep high-level call order visible in one place.
- Keep detection primitives reusable and independent of orchestration.
- Avoid duplicated type definitions.
- Keep internal layering explicit for future refactors.

## Ownership

### `pipeline/` (internal orchestration)

`pipeline` owns the end-to-end stage flow:

1. proposal generation
2. local fit/decode
3. dedup
4. optional homography filter/refine
5. optional completion
6. final result/debug assembly

Public detection entrypoints (`detect_rings*`) are implemented in this module and re-exported from `lib.rs`.

### `detector/` (primitives, no orchestration policy)

`detector` contains reusable detection logic components:

- local outer/inner fitting
- completion logic
- dedup/global filtering
- center correction
- proposal detector
- shared detection config types

`detector` does not define high-level stage ordering.

### `api.rs` (public facade)

`Detector` and `TargetSpec` live in `api.rs` to avoid naming ambiguity with the `detector` primitive module.

### `marker/`

Marker-specific primitives:

- marker geometry spec
- ring-sector decode config/diagnostics
- codebook + codec

CLI-only codebook/codec module exports remain feature-gated via `cli-internal`.

### `ring/`

Shared ring geometry estimators and samplers:

- edge sampling
- inner/outer radial estimators
- projective center solver

### `pixelmap/`

Camera and distortion mapping components:

- pinhole camera model
- radial-tangential distortion
- division-model self-undistort
- `PixelMapper` trait

### `conic/` and `homography/`

Math primitives used by detector/pipeline internals.

## Type consolidation

`EllipseParams` was removed.

`DetectedMarker` now stores `Option<conic::Ellipse>` directly for `ellipse_outer` and `ellipse_inner`.

This removes duplicated ellipse representations and conversion glue.

## Dependency direction

Intended direction:

- `api` -> `pipeline`
- `pipeline` -> `detector`, `ring`, `pixelmap`, math modules
- `detector` -> `ring`, `marker`, `pixelmap`, math modules
- `ring` / `marker` / `pixelmap` -> math + core data types

`detector` should not depend on `pipeline`.
