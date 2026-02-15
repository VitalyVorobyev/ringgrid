# Detection Pipeline Analysis

## 1. Architecture Overview

The detection pipeline is organized around the `Detector` facade in `api.rs`:

```
Detector (api.rs)
  +-- detect()                  -- config-driven single-pass or self-undistort flow
  +-- detect_with_mapper()      -- two-pass with explicit PixelMapper
        |
        v
pipeline/run.rs
  +-- fit_decode::run()         -- proposals -> fit/decode -> dedup
  +-- finalize::run()           -- projective-center -> filter -> completion -> final H
```

Single-pass entry points build proposals and call `run::run()` directly.
Two-pass/self-undistort reuse the same core `run::run()` stage executor.

## 2. Public Entry Points

| Method | Mapper | Seeds | Two-pass |
|--------|--------|-------|----------|
| `detect` | none or estimated from `self_undistort` | none or auto | config-driven |
| `detect_with_mapper` | explicit `&dyn PixelMapper` | auto | yes |

Notes:
- Mapper is passed explicitly; `DetectConfig` does not store camera parameters.
- Seed injection is controlled by `DetectConfig.seed_proposals`.
- `DetectionResult.self_undistort` is populated only by `detect()` when self-undistort is enabled.

## 3. Pipeline Stages

### Stage 0: Proposal generation
File: `pipeline/run.rs` + `detector/proposal.rs`

- Scharr gradient voting + NMS produces initial proposals.
- In pass-2 modes, pass-1 markers are converted into seed proposals.

### Stage 1: Local fit/decode
File: `pipeline/fit_decode.rs`

For each proposal:
1. map image center to working frame (`DistortionAwareSampler`)
2. estimate outer radius hypotheses
3. sample outer edges and fit outer ellipse
4. decode ring sectors
5. fit inner ellipse
6. build `DetectedMarker`

### Stage 2: Dedup
File: `detector/dedup.rs`

- Sort by confidence (descending)
- Dedup by proximity (`dedup_radius`)
- Dedup by decoded ID (keep highest confidence)

### Stage 3: Global filter
Files: `detector/global_filter.rs`, `pipeline/finalize.rs`

- Build board/image correspondences from decoded markers
- Fit homography via RANSAC
- Keep inliers only

### Stage 4: Completion
File: `detector/completion.rs`

- Requires `completion.enable=true` and valid homography
- For missing board IDs, project centers through H and run local fits
- Apply gates: arc coverage, fit confidence, reprojection error, scale
- Add accepted markers conservatively

### Stage 5: Final H refit
File: `pipeline/finalize.rs`

- Refit homography from final marker set
- Keep refit only when it improves reprojection error
- Emit final `RansacStats`

## 4. Center Correction Order

Projective-center correction is applied once per marker:
1. before global filtering
2. after completion (for completion-added markers only)

## 5. Coordinate Frames

| Frame | Meaning |
|------|---------|
| image frame | raw distorted pixel coordinates |
| working frame | undistorted pixel coordinates (or image frame when no mapper) |

`DistortionAwareSampler` keeps geometry in working coordinates while sampling image intensities in raw image space.

Public result contract:
- `DetectedMarker.center` is always image frame.
- `DetectedMarker.center_mapped` is optional working frame center when mapper-driven passes are active.
- `DetectionResult.center_frame` and `DetectionResult.homography_frame` carry explicit frame metadata.

## 6. Short-circuit Behavior

When `use_global_filter=false`, finalization short-circuits after projective-center correction:
- no homography-based completion
- no final homography refit

This keeps behavior explicit: no homography means no H-driven post-processing.
