# DEC-005: Public API Surface

**Status:** Active
**Date:** 2025

## Decision

The public API is defined exclusively through re-exports in `lib.rs`.
Full detection flows go through the `Detector` struct.

Proposal-only access is provided through the standalone `proposal` module and through
detector-bound methods:
- `ringgrid::proposal::find_ellipse_centers` / `find_ellipse_centers_with_heatmap` are
  the standalone entry points — no `Detector` or ringgrid-specific types needed.
- `Detector::propose` / `Detector::propose_with_heatmap` are the detector-bound
  proposal entry points with scale-prior derivation.

### Exported types

| Category | Types |
|----------|-------|
| Entry point | `Detector` |
| Proposal module | `proposal::find_ellipse_centers`, `proposal::find_ellipse_centers_with_heatmap`, `Proposal`, `ProposalResult`, `ProposalConfig` |
| Results | `DetectionResult`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, `RansacStats`, `DetectionFrame` |
| Config | `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod`, `CompletionParams`, `InnerFitConfig`, `ProjectiveCenterParams`, `SeedProposalParams`, `ProposalConfig`, `ProposalDownscale`, `RansacHomographyConfig`, `DecodeConfig`, `EdgeSampleConfig`, `OuterEstimationConfig` |
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

- Proposal internals beyond the stable surface (gradient computation,
  accumulator-construction helpers, NMS scratch offsets).
- Pipeline internals (`EdgeSampleResult`, `OuterHypothesis`).
- Fitting internals (`fit_ellipse_direct`, `try_fit_ellipse_ransac`).
- Homography internals (`compute_homography_dlt`, `homography_ransac`).
- Projective center solver (`ring_center_projective_with_debug`).
- Stage-local scratch/result structs that would freeze implementation details.

## Rationale

Small, stable API surface allows aggressive internal refactoring without
breaking consumers. Config types that affect detection quality are public, and
proposal-stage diagnostics are public because they are useful for analysis,
tuning, and downstream proposal-only workflows. The free-function exception is
kept intentionally narrow so full detection orchestration remains centered on
`Detector`.
