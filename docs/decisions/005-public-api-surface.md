# DEC-005: Public API Surface

**Status:** Active
**Date:** 2025

## Decision

The public API is defined exclusively through re-exports in `lib.rs`.
All detection goes through the `Detector` struct. No public free functions.

### Exported types

| Category | Types |
|----------|-------|
| Entry point | `Detector` |
| Results | `DetectionResult`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, `RansacStats`, `DetectionFrame` |
| Config | `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod`, `CompletionParams`, `InnerFitConfig`, `ProjectiveCenterParams`, `SeedProposalParams`, `RansacHomographyConfig` |
| Geometry | `BoardLayout`, `BoardMarker`, `Ellipse`, `MarkerSpec` |
| Camera | `CameraModel`, `CameraIntrinsics`, `RadialTangentialDistortion`, `DivisionModel`, `PixelMapper`, `SelfUndistortConfig`, `SelfUndistortResult` |
| Codec | `codebook`, `codec` (modules) |

### Visibility rules

1. **`lib.rs` is purely re-exports.** No type definitions, no logic.
2. **Types are defined at their construction site**, not in `lib.rs`.
3. **Internal modules use `pub(crate)`** for cross-module sharing within the
   crate (e.g., `fit_outer_ellipse_with_reason`, `InnerFitResult`, pipeline
   helpers).
4. **`pub(super)`** for module-private sharing within a `mod.rs` hierarchy.
5. **Private functions** for module-internal helpers (default).

### What must NOT become public

- Pipeline internals (`Proposal`, `EdgeSampleResult`, `OuterHypothesis`).
- Fitting internals (`fit_ellipse_direct`, `try_fit_ellipse_ransac`).
- Homography internals (`compute_homography_dlt`, `homography_ransac`).
- Projective center solver (`ring_center_projective_with_debug`).
- Individual config sub-types for internal stages (`ProposalConfig`,
  `OuterEstimationConfig`, `EdgeSampleConfig`, `DecodeConfig`) — these are
  reachable as fields of `DetectConfig` but not re-exported at crate root.

## Rationale

Small, stable API surface allows aggressive internal refactoring without
breaking consumers. Config types that affect detection quality are public;
internal algorithm types are not.
