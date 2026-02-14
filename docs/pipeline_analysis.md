# Detection Pipeline Analysis Report

## 1. Architecture Overview

The detection pipeline has **two layers of orchestration**:

```
Detector (api.rs)                  -- public API, owns DetectConfig
  |
  +-- detect()                     -- config-driven (single-pass or self-undistort flow)
  +-- detect_with_mapper()         -- two-pass with custom PixelMapper
  +-- detect_with_debug()          -- single-pass with debug dump (feature-gated)
  |
  v
pipeline/mod.rs                    -- entry points: detect_single_pass, detect_two_pass, etc.
  |
  +-- run::run(proposals)          -- glue: fit_decode -> finalize
  |     +-- fit_decode::run()      -- proposals -> fit -> decode -> dedup
  |     +-- finalize::run()        -- PC -> global_filter -> refine_h -> PC(reapply) -> completion -> PC -> final_h
  |
  +-- two_pass.rs                  -- two-pass + self-undistort orchestration
        +-- detect_two_pass()      -- pass-1 (raw) -> pass-2 (mapper + seeds) -> merge
        +-- detect_with_self_undistort() -- baseline -> estimate model -> pass-2
```

Single-pass entry points build proposals and call `run::run()` directly from `mod.rs`. Two-pass and self-undistort orchestration lives in `pipeline/two_pass.rs`. Proposal building (including seed injection) happens before `run::run()` — the core pipeline accepts pre-built `Vec<Proposal>`.

### 1.1 Entry Points

All detection goes through `Detector` methods. No public free functions.

| Method | Mapper | Seeds | Debug | Two-Pass |
|--------|--------|-------|-------|----------|
| `detect` | none or estimated from `self_undistort` config | none or auto | no | config-driven |
| `detect_with_mapper` | explicit `&dyn PixelMapper` | auto | no | yes |
| `detect_with_debug` | optional | none | yes | no |

The mapper is always passed explicitly — `DetectConfig` does not store a camera model. Seed injection is an internal detail of two-pass orchestration (hardcoded `SeedProposalParams::default()`).

Internal `pub(crate)` pipeline functions: `detect_single_pass`, `detect_two_pass`, `detect_with_self_undistort`, `detect_single_pass_with_debug`.

---

## 2. Pipeline Stages In Detail

### Stage 0: Proposal Generation
**File**: `pipeline/fit_decode.rs`, calls `find_proposals_with_seeds` (via `pipeline/two_pass.rs`)

- Scharr gradient voting + NMS generates candidate centers
- Seed centers (from pass-1 in two-pass mode) are merged or injected
- Seeds with score `1e12` are effectively guaranteed to be processed

### Stage 1: Fit + Decode (per candidate)
**File**: `pipeline/fit_decode.rs`, function `process_candidate`

For each proposal:
1. Map image-space proposal to working coordinates via `sampler.image_to_working_xy`
2. **Outer estimate**: radial profile peaks give radius hypotheses (`estimate_outer_from_prior_with_mapper`)
3. **Outer edge sampling**: for each radius hypothesis, sample edge points around the ring
4. **Outer ellipse fit**: RANSAC (>=8 pts) or direct Fitzgibbon LS (>=6 pts)
5. **Decode**: 16-sector code sampling from the fitted outer ellipse
6. **Hypothesis scoring**: `2.0 * decode_score + 0.7 * (arc_cov * inlier_ratio) + 0.3 * size_score - 0.05 * residual`
7. Pick best hypothesis
8. **Inner fit**: fit inner ellipse from outer hint (concentric constraint)
9. Build `DetectedMarker`

**All candidates that produce a valid outer ellipse are kept** -- even those with no decoded ID. There is no confidence gate here.

### Stage 2: Dedup
**File**: `detector/dedup.rs`

Two-phase dedup:
1. Sort by confidence (descending)
2. **Proximity dedup**: within `dedup_radius` px, keep highest-confidence
3. **ID dedup**: if same ID decoded by multiple candidates, keep highest-confidence

### Stage 3: Global Filter (RANSAC Homography)
**File**: `detector/global_filter.rs`, called from `pipeline/finalize.rs::phase_filter_and_refine_h`

- Builds board-to-image correspondences from decoded markers
- Fits homography via RANSAC (DLT with Hartley normalization)
- **Keeps only inliers** (outlier markers are dropped entirely)
- Requires >= 4 decoded markers
- When disabled: all markers pass through, no homography available

### Stage 4: H-guided Refinement
**File**: `detector/refine_h.rs`, called from `pipeline/finalize.rs::phase_filter_and_refine_h`

- Only runs when `refine_with_h == true` AND homography exists AND >= 10 markers
- For each decoded marker: project board position through H, re-run local ring fit at that prior
- **Requires decode match**: `decode_result.filter(|d| d.id == id)` -- if re-fit doesn't decode to the same ID, keeps original marker
- Scale gate: rejects if new mean axis deviates >25%/33% from expected
- Non-debug path avoids debug allocation overhead

### Stage 5: Completion
**File**: `detector/completion.rs`, called from `pipeline/finalize.rs::phase_completion`

- Only runs when `completion.enable == true` AND homography exists
- Iterates over **all board marker IDs** not yet detected
- Projects each missing ID through H to get image location
- Runs local ring fit at projected location
- Quality gates: arc coverage, fit confidence, reprojection error, scale
- Decode mismatch is recorded as a note but doesn't block acceptance
- Adds marker with the board ID (even if decode disagrees or fails)

### Center Correction (three-pass application)
**File**: `detector/center_correction.rs`, called from `pipeline/finalize.rs`

Projective center correction runs in **three passes** throughout finalization:

1. **1st pass** (before global filter): `apply_projective_centers` on fit-decode markers. Skips markers already corrected. This ensures the initial RANSAC homography uses corrected centers for better precision.
2. **2nd pass** (after refine-H): `reapply_projective_centers` on refined markers. Clears and recomputes all corrections because refinement produces new ellipses.
3. **3rd pass** (after completion): `apply_projective_centers` on the full marker set. Only affects newly added completion markers (existing markers are skipped via the `center_projective.is_some()` guard).

Two functions implement this:
- `apply_projective_centers` — skips markers where `center_projective.is_some()`
- `reapply_projective_centers` — clears then recomputes all markers

### Final H Refit
**File**: `pipeline/finalize.rs::phase_final_h`

- Refits homography from final marker set
- Compares new vs current by mean reprojection error, keeps better one
- Recomputes `RansacStats`

### Short Circuit
**File**: `pipeline/finalize.rs`

When `use_global_filter == false` AND debug is off, the pipeline short-circuits: skips completion and final H refit. Projective center correction (1st pass) is still applied inside `phase_filter_and_refine_h`.

---

## 3. Design Notes and Known Limitations

### 3.1 ~~RESOLVED~~ Projective Center Ordering

Previously applied once after completion. Now applied in **three passes**: before global filter (for better H), after refine-H (reapply with new ellipses), and after completion (new markers only). Uses skip-if-corrected semantics (`apply_projective_centers`) and force-reapply semantics (`reapply_projective_centers`).

### 3.2 DESIGN: Global Filter Drops Markers Silently

When the global filter runs, outlier markers are **removed entirely**. This means:
- Undecoded markers (no ID) are never passed to the global filter -- they are dropped
- Decoded markers that disagree with the homography consensus are dropped
- There is no mechanism to retain high-confidence detections that happen to be geometric outliers

This is intentional: the global filter exists to enforce geometric consistency. Undecoded markers cannot participate in homography estimation and would not benefit from completion.

### 3.3 DESIGN: Short Circuit Skips Completion

When `use_global_filter == false` (and no debug), the pipeline short-circuits and **skips completion entirely**. This is because completion requires a homography. The short circuit also skips final H refit.

### 3.4 DESIGN: Inconsistent Marker Acceptance Between Stages

| Stage | Acceptance criterion |
|-------|---------------------|
| Fit-Decode | Any valid outer ellipse (no confidence threshold) |
| Global Filter | Decoded + inlier to H consensus |
| H-Refinement | Must decode to same ID + scale gate |
| Completion | Quality gates (arc, confidence, reproj, scale) -- no decode match required |

The criteria differ across stages by design. Each stage has different goals:
- Fit-decode: maximize recall (cast a wide net)
- Global filter: enforce geometric consistency
- H-refinement: improve precision while preserving identity
- Completion: fill gaps with conservative quality requirements

### 3.5 DESIGN: Two-Pass vs Debug Incompatibility

**Debug collection only works in single-pass mode.** Two-pass detection (via `detect_with_mapper` or `detect` with `self_undistort.enable=true`) cannot produce debug dumps. This is a known limitation.

### 3.6 ~~RESOLVED~~ Refine-H Always Runs Debug Path

Previously `refine_with_homography` always allocated debug records. Now has separate debug/non-debug code paths (`refine_with_homography` vs `refine_with_homography_with_debug`).

### 3.7 MINOR: `h_current` Mutation Pattern in Finalize

The homography matrix flows through multiple stages that can mutate it:
```
PC(1st pass) -> global_filter -> h_current (initial, from corrected centers)
refine_h -> PC(2nd pass, reapply) -> completion -> PC(3rd pass)
phase_final_h -> final refit (optionally updated)
```

The final `phase_final_h` does its own refit from scratch and uses "keep better" comparison.

### 3.8 OBSERVATION: Completion Edge Points Are Collected

Completion markers store `edge_points_outer` and `edge_points_inner`. This allows them to participate in self-undistort estimation. However, completion markers are assigned board IDs unconditionally (even with decode mismatch), so their edge points may be associated with incorrect IDs.

---

## 4. Data Flow Summary

```
proposals: Vec<Proposal>
    |
    | (per-candidate processing)
    v
markers: Vec<DetectedMarker>   [all valid fits, decoded or not]
    |
    | (dedup by proximity + ID)
    v
markers: Vec<DetectedMarker>   [deduped]
    |
    | (projective center correction, 1st pass — skip already-corrected)
    v
markers: Vec<DetectedMarker>   [center-corrected for better H]
    |
    | (global_filter: keeps only decoded inliers)
    v
markers: Vec<DetectedMarker>   [decoded inliers only]
    |
    | (refine_h: re-fit at H-projected priors)
    v
markers: Vec<DetectedMarker>   [refined or original]
    |
    | (projective center correction, 2nd pass — reapply all, new ellipses)
    v
markers: Vec<DetectedMarker>   [re-corrected after refinement]
    |
    | (completion: add missing IDs)
    v
markers: Vec<DetectedMarker>   [+ completion markers]
    |
    | (projective center correction, 3rd pass — only new completion markers)
    v
markers: Vec<DetectedMarker>   [all center-corrected]
    |
    | (final H refit)
    v
DetectionResult
```

---

## 5. Remaining Improvement Opportunities

### 5.1 Unify Acceptance Logic

Create a single `MarkerAcceptanceGate` struct that encodes acceptance criteria consistently across fit-decode, completion, and refine-h stages.

### 5.2 Make Two-Pass a Pipeline Concern

Currently, two-pass logic lives alongside the pipeline (in `pipeline/two_pass.rs`) but calls `pipeline/run.rs::run` twice. Consider making two-pass a first-class pipeline concept so that debug collection and self-undistort can work together.

### 5.3 Separate Debug from Logic

The `collect_debug` flag creates significant code branching throughout `pipeline/finalize.rs`. Many functions exist in pairs (e.g., `global_filter` vs `global_filter_with_debug`). Consider a "debug sink" trait/interface that can be either real or no-op, reducing branching.

---

## 6. Detector API Surface

| Method | When to use |
|--------|------------|
| `detect` | Standard non-camera entrypoint; single-pass or self-undistort based on `DetectConfig` |
| `detect_with_mapper` | Custom distortion adapter via PixelMapper trait (triggers two-pass) |
| `detect_with_debug` | CLI/internal use (behind `cli-internal` feature gate), accepts optional mapper |

**Notes**:
- The mapper is always passed explicitly to the method — `DetectConfig` does not store a camera model.
- `detect` with `self_undistort.enable=true` runs unique orchestration (run once, estimate, optionally re-run). This is the only public path where `DetectionResult.self_undistort` is populated.
- Two-pass is triggered by `detect_with_mapper()` and by `detect()` when self-undistort is enabled.

---

## 7. Distortion / Mapper Analysis

### 7.1 Coordinate Frames

| Frame | Description |
|-------|------------|
| **Image frame** | Raw pixel coordinates in the distorted image |
| **Working frame** | Undistorted pixel coordinates (= image frame when no mapper) |

The `DistortionAwareSampler` handles the bridge:
- `image_to_working_xy`: undistort (image → working)
- `working_to_image_xy`: redistort (working → image)
- `sample_checked(x_working, y_working)`: maps working→image, then bilinear-samples the raw image

### 7.2 How the Mapper Is Used Inside a Single Pass

When `pipeline/run.rs::run` receives a mapper:

1. **Proposals are always in image frame**: `find_proposals()` runs Scharr gradients on the raw `gray` image
2. **Proposal mapped to working frame**: `sampler.image_to_working_xy([proposal.x, proposal.y])`
3. **Outer estimation**: working-frame coordinates, pixel lookups via `sampler.sample_checked` (working→image→bilinear)
4. **Edge sampling**: working-frame coordinates, same pixel lookup pattern
5. **Ellipse fit**: working-frame edge points → working-frame ellipse
6. **Decode**: working-frame ellipse, pixel lookups via mapper
7. **All output coordinates are in working frame**

**Key insight: the mapper defines a coordinate transform. All geometric computations happen in working frame. Pixel values are fetched by mapping working→image→bilinear lookup. The raw image is never undistorted.**

### 7.3 Self-Undistort Flow

```
detect() with self_undistort.enable=true:
  |
  +-- run_single_pass(no mapper)       → pass-1 result in IMAGE frame
  |     edge_points_outer/inner are in image frame
  |
  +-- estimate_self_undistort()         → DivisionModel {lambda, cx, cy}
  |     uses pass-1 edge points to estimate lambda
  |
  +-- run_pass2(model):                → pass-2 in WORKING frame
        |     proposals + seeds from pass-1
        |     → mapped to working frame
        |     → edge sampling in working frame (pixels via mapper)
        |     → all results in working frame
        |
        +-- merge: pass-1 fallback markers remapped to working frame
             (ellipse/edge fields dropped for remapped markers)
```

The self-undistort path runs the image **twice** total:
1. Initial single-pass detection (for lambda estimation + pass-1 seeds)
2. Pass-2 with estimated `DivisionModel` (final result)

The initial detection result is reused as the internal pass-1 (optimized from the original 3x approach).

---

## 8. Summary of Remaining Recommendations

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| Medium | Undecoded markers silently dropped by global filter | Document or add option to retain |
| Medium | Two-pass + debug incompatibility | Document limitation or implement two-pass debug |
| Medium | Inconsistent acceptance criteria | Unify with shared gate struct |
