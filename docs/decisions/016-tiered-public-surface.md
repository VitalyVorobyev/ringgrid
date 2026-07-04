# DEC-016: Tiered Public Surface

**Status:** Active
**Date:** 2026-07-04
**Supersedes:** [DEC-005](005-public-api-surface.md)

## Context

DEC-005 defined a **flat** public surface: every re-exported type lived at
the crate root at a single stability level. That put per-marker diagnostics
(`FitMetrics`, `DecodeMetrics`), homography RANSAC statistics (`RansacStats`),
and codebook inspection helpers on equal footing with the stable primary
output (`DetectionResult`, `DetectedMarker`). DEC-005 justified this on the
grounds that "proposal-stage diagnostics are public because they are useful
for analysis."

Two problems surfaced in the pre-0.8 workspace-revision review:

1. **Every root re-export is a semver contract at the same tier.** Diagnostics
   types describe fit residuals, RANSAC iteration counts, and stage timings —
   fields whose shape tracks the algorithms that produce them. Freezing them at
   root-tier stability blocks exactly the internal refactoring DEC-005 set out
   to protect. A flat surface also conflates three audiences: consumers of
   `Detector::detect`, tuners who want diagnostics, and codebook inspectors.
2. **The v4 `BoardLayout` hex facade was still a root-level peer** of the
   compositional `TargetLayout` model that had replaced it, presenting two
   competing geometry entry points as equally canonical.

## Decision

Re-tier the public surface into three explicit tiers (shipped in PR #54,
commit `09a04ac8`, "refactor(api): tier the public surface; deprecate the v4
board facade"). `lib.rs` remains purely re-exports; only the grouping changed.

### Tier 1 — stable root (semver-stable contract)

The surface a consumer of `Detector::detect` needs:

| Category | Types |
|----------|-------|
| Entry point | `Detector`, `DetectError` |
| Config | `DetectConfig`, `MarkerScalePrior`, `CircleRefinementMethod`, `ScaleTier`, `ScaleTiers` (+ the full detector/decode/ring config family) |
| Results | `DetectionResult`, `DetectedMarker`, `BoardFrame`, `DetectionFrame` |
| Target geometry | `TargetLayout`, `LatticeGeometry`, `RingGeometry`, `MarkerCoding`, `OriginFiducials` (+ `HexGeometry`/`RectGeometry`/`CodedRingSpec`/`TargetCell` and target errors/generation options) |
| Camera / distortion | `CameraModel`, `CameraIntrinsics`, `PixelMapper` (+ `DivisionModel`, `RadialTangentialDistortion`, `SelfUndistortConfig`/`SelfUndistortResult`, `UndistortConfig`) |
| Geometry | `Ellipse` |
| Proposal | `Proposal`, `ProposalConfig`, `ProposalResult` (+ `find_ellipse_centers*`, `propose_with_marker_scale*`) |

### Tier 2 — `ringgrid::diagnostics` (opt-in, faster-moving)

Returned only by `Detector::detect_with_diagnostics`. Deliberately not at the
root so its types may evolve between releases without breaking root consumers:
`DetectionDiagnostics`, `MarkerDiagnostics`, `FitMetrics`, `DecodeMetrics`,
`DetectionSource`, `InnerFitReason`, `InnerFitStatus`, `RansacStats`,
`StageTimings`.

### Tier 3 — `ringgrid::codebook` (utility)

Codebook inspection helpers: `CodebookInfo`, `CodewordMatch`, `codebook_info`,
`decode_word`.

### Deprecated v4 facade

`BoardLayout`, `BoardMarker`, and the error aliases `BoardLayoutLoadError` /
`BoardLayoutValidationError` remain re-exported at the root behind
`#[allow(deprecated)]`, **scheduled for removal after 0.8**. `TargetLayout` and
the compositional target model are the canonical geometry surface.

### Verification against `lib.rs`

The tier *boundaries* — root vs. `diagnostics` vs. `codebook` — and the
membership of the `diagnostics` and `codebook` modules match the code exactly
(`crates/ringgrid/src/lib.rs`). The root tier is **broader** than the
"Key public types" highlight list in the project docs: alongside the
highlighted types it also re-exports the full config sub-type family
(`AdvancedDetectConfig`, `CompletionConfig`, `IdCorrectionConfig`,
`InnerAsOuterRecoveryConfig`, `InnerFitConfig`, `OuterFitConfig`,
`ProjectiveCenterConfig`, `ProposalDownscale`, `SeedProposalConfig`,
`RansacConfig`), decode/ring sub-configs (`AngularAggregator`,
`CodebookProfile`, `DecodeConfig`, `GradPolarity`, `EdgeSampleConfig`,
`OuterEstimationConfig`), the full target-geometry detail types and
`target_generation` options, `MarkerSpecConfig`, and the complete
camera/distortion model set. The highlight list is a curated subset, not the
complete root export set.

## Consequences

**Positive:**
- Semver discipline is now per-tier: the root is the stable contract, while
  `diagnostics` and `codebook` are explicitly lower-stability. Diagnostics
  types can change shape as algorithms evolve without breaking root consumers.
- The three audiences (result consumers, tuners, codebook inspectors) are
  separated at the type-system level.
- `TargetLayout` is unambiguously the canonical geometry entry point.

**Negative:**
- Breaking change: callers that used root-level `FitMetrics`, `DecodeMetrics`,
  or `RansacStats` must import from `ringgrid::diagnostics` instead. Acceptable
  pre-1.0 (see *Decisive cleanup*).
- `BoardLayout` users must migrate to `TargetLayout` before the post-0.8
  removal.

**Neutral:**
- `lib.rs` is still purely re-exports; type definitions remain at their
  construction sites.

### Migration

- `ringgrid::FitMetrics` → `ringgrid::diagnostics::FitMetrics` (and likewise
  `DecodeMetrics`, `RansacStats`, `DetectionSource`, `InnerFitReason`,
  `InnerFitStatus`).
- `ringgrid::BoardLayout` / `BoardMarker` → `ringgrid::TargetLayout` and the
  compositional target model.
