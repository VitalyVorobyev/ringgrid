---
name: metrology-invariants
description: Apply this skill whenever ringgrid work touches geometry, coordinate transforms, center semantics, homography frames, or accuracy metrics. It enforces pixel-center conventions, explicit frame labeling, and subpixel tolerance gates.
metadata:
  short-description: Coordinate-frame and subpixel-accuracy guardrails
---

# Metrology Invariants

Use this skill for any change that can shift geometry semantics or measured accuracy: conic fitting, sampling, center correction, homography, mapping, or scoring.

## Core Invariants

1. Pixel-center convention: integer index `i` maps to center coordinate `i as f32`.
2. Frame explicitness: every center/homography value is in a named frame (`image`, `working`, or mapper frame).
3. Mapping direction clarity: any conversion states source frame, destination frame, and failure behavior.
4. Metric units: center and reprojection errors are reported in pixels in the same frame as the compared points.
5. Numerical safety: reject NaN/Inf and degenerate geometry before publishing results.

## Execution Checklist

1. Build a small frame table before coding.
- For each touched field, record: producer stage, frame, and consumer stage.
2. Preserve frame contracts at API boundaries.
- Public types (`DetectedMarker`, `DetectionResult`) must not rely on implicit frame inference.
3. Validate transform assumptions.
- If using a mapper or homography, verify invertibility and drop/flag points that cannot be mapped.
4. Update assertions and docs.
- Tests and docs must state frame semantics explicitly.
5. Re-check tolerances.
- Unit checks: ~0.1 px.
- Precision checks: ~0.05 px.
- Regression alert: investigate if mean center error worsens by >0.01 px.

## Failure Patterns To Watch

- Mixed-frame subtraction (image minus working coordinates).
- Silent fallback to stale centers after mapping failures.
- Inconsistent frame semantics between Rust API output and Python scorer assumptions.
- Hidden `+0.5` shifts introduced by external formulas or copied code.

## Handoff Requirements

Include:
- Which fields changed frame semantics (if any).
- Why tolerances remain valid.
- How mapping failures are handled.
