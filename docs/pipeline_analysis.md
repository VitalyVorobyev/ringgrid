# Detection Pipeline Analysis Report

## 1. Architecture Overview

The detection pipeline has **three layers of orchestration**:

```
Detector (detector.rs)          -- public API, owns DetectConfig
  |
  +-- detect_rings()            -- single-pass entry point
  +-- detect_rings_with_mapper  -- two-pass orchestrator
  +-- detect_with_self_undistort -- self-undistort + re-run
  |
  v
stages::run()  (stages/mod.rs)  -- glue: fit_decode -> finalize
  |
  +-- stage_fit_decode::run()   -- proposals -> fit -> decode -> dedup
  +-- stage_finalize::run()     -- global_filter -> refine_h -> completion -> nl_refine -> projective_center -> final_h
```

### 1.1 Entry Points

There are **6 public/semi-public detection entry points** in `detect.rs`:

| Function | Mapper | Seeds | Debug | Two-Pass |
|----------|--------|-------|-------|----------|
| `detect_rings` | from config.camera | none | no | no |
| `detect_rings_with_mapper` | explicit | none | no | yes if mapper |
| `detect_rings_two_pass_with_mapper` | explicit | pass-1 centers | no | yes |
| `detect_rings_with_debug` | from config.camera | none | yes | no |
| `detect_rings_with_debug_and_mapper` | explicit | none | yes | no |
| (private) `detect_rings_with_mapper_and_seeds` | any | any | no | no |

Plus 5 `Detector` methods: `detect`, `detect_with_camera`, `detect_with_mapper`, `detect_with_self_undistort`, `detect_precision`.

---

## 2. Pipeline Stages In Detail

### Stage 0: Proposal Generation
**File**: `stage_fit_decode.rs` line 342, calls `find_proposals_with_seeds`

- Scharr gradient voting + NMS generates candidate centers
- Seed centers (from pass-1 in two-pass mode) are merged or injected
- Seeds with score `1e12` are effectively guaranteed to be processed

### Stage 1: Fit + Decode (per candidate)
**File**: `stage_fit_decode.rs`, function `process_candidate`

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

### Projective Center (conditional, first application)
**File**: `stage_fit_decode.rs` lines 376-378

If `circle_refinement == ProjectiveCenter`, `apply_projective_centers` runs here, **before dedup**. This means projective-corrected centers are used for dedup proximity decisions.

### Stage 2: Dedup
**File**: `dedup.rs`

Two-phase dedup:
1. Sort by confidence (descending)
2. **Proximity dedup**: within `dedup_radius` px, keep highest-confidence
3. **ID dedup**: if same ID decoded by multiple candidates, keep highest-confidence

### Stage 3: Global Filter (RANSAC Homography)
**File**: `global_filter.rs`, called from `stage_finalize.rs::phase_filter_and_refine_h`

- Builds board-to-image correspondences from decoded markers
- Fits homography via RANSAC (DLT with Hartley normalization)
- **Keeps only inliers** (outlier markers are dropped entirely)
- Requires >= 4 decoded markers
- When disabled: all markers pass through, no homography available

### Stage 4: H-guided Refinement
**File**: `refine_h.rs`, called from `phase_filter_and_refine_h`

- Only runs when `refine_with_h == true` AND homography exists AND >= 10 markers
- For each decoded marker: project board position through H, re-run local ring fit at that prior
- **Requires decode match**: `decode_result.filter(|d| d.id == id)` -- if re-fit doesn't decode to the same ID, keeps original marker
- Scale gate: rejects if new mean axis deviates >25%/33% from expected

### Stage 5: Completion
**File**: `completion.rs`, called from `phase_completion`

- Only runs when `completion.enable == true` AND homography exists
- Iterates over **all board marker IDs** not yet detected
- Projects each missing ID through H to get image location
- Runs local ring fit at projected location
- Quality gates: arc coverage, fit confidence, reprojection error, scale
- Decode mismatch is recorded as a note but doesn't block acceptance
- Adds marker with the board ID (even if decode disagrees or fails)

### Stage 6: NL Board-Plane Refinement
**File**: `refine.rs` + `refine/pipeline.rs`, called from `phase_nl_refine`

- Only runs when `circle_refinement == NlBoard` AND homography exists
- Per-marker: sample outer edge points, project to board plane via H, solve circle-center via LM or IRLS
- Iterative H refit loop (configurable, default disabled)

### Center Correction (second application, conditional)
**File**: `stage_finalize.rs::phase_center_correction`

`apply_projective_centers` runs again if `circle_refinement == ProjectiveCenter`. This is the **second call** to the same function in the pipeline.

### Final H Refit
**File**: `stage_finalize.rs::phase_final_h`

- Refits homography from final marker set
- Compares new vs current by mean reprojection error, keeps better one
- Recomputes `RansacStats`

### Short Circuit
**File**: `stage_finalize.rs` line 486

When `use_global_filter == false` AND debug is off, the pipeline short-circuits: skips completion, NL refine, final H refit. Only applies projective center correction.

---

## 3. Identified Issues

### 3.1 CRITICAL: Projective Center Applied Twice

`apply_projective_centers` is called in **two separate locations**:

1. `stage_fit_decode.rs:376` -- before dedup, inside the fit-decode stage
2. `stage_finalize.rs:516` (via `phase_center_correction`) -- after completion + NL refine

The function begins by clearing all projective fields:
```rust
for m in markers.iter_mut() {
    m.center_projective = None;
    m.vanishing_line = None;
    m.center_projective_residual = None;
}
```

Then applies the correction by overwriting `m.center`.

**Impact**: For markers that survive from stage 1 through all stages, the correction is applied twice. The first application changes `m.center` from the ellipse center to the projective center. The second application then runs the solver again on the same ellipse data, which should produce the same result -- BUT `m.center` has already been moved, so the `max_center_shift_px` gate now measures shift from the *already-corrected* center, not from the original ellipse center. This is benign if the correction is idempotent (which it is, since it reads from `ellipse_outer`/`ellipse_inner`, not from `center`), but it's wasteful and conceptually confusing.

**However**, for **completion markers** (added after stage 1), the first application doesn't apply. Only the second call covers them. This is likely the intent -- but the double execution for non-completion markers is unnecessary.

### 3.2 DESIGN: Global Filter Drops Markers Silently

When the global filter runs, outlier markers are **removed entirely**. This means:
- Undecoded markers (no ID) are never passed to the global filter -- they are dropped
- Decoded markers that disagree with the homography consensus are dropped
- There is no mechanism to retain high-confidence detections that happen to be geometric outliers

The global filter returns only inlier decoded markers. **All undecoded markers are silently lost** after this stage.

### 3.3 DESIGN: Short Circuit Skips Completion

When `use_global_filter == false` (and no debug), the pipeline short-circuits and **skips completion entirely**. This is because completion requires a homography. However, the short circuit also skips:
- NL refinement
- Final H refit

This means `circle_refinement == NlBoard` with `use_global_filter == false` silently does nothing.

### 3.4 DESIGN: Inconsistent Marker Acceptance Between Stages

| Stage | Acceptance criterion |
|-------|---------------------|
| Fit-Decode | Any valid outer ellipse (no confidence threshold) |
| Global Filter | Decoded + inlier to H consensus |
| H-Refinement | Must decode to same ID + scale gate |
| Completion | Quality gates (arc, confidence, reproj, scale) -- no decode match required |

The criteria are very different across stages. Notably:
- Fit-decode accepts undecoded candidates, but global filter drops them
- H-refinement requires decode match, completion doesn't
- No minimum confidence gate at any stage (except completion's `min_fit_confidence`)

### 3.5 DESIGN: Two-Pass vs Single-Pass Divergence

**Single-pass** (default): `detect_rings` -> `stages::run` with mapper from `config.camera`.

**Two-pass** (`detect_rings_with_mapper` / `detect_rings_two_pass_with_mapper`):
- Pass 1: raw (no mapper), produces seed centers
- Pass 2: with mapper, seeds injected
- Merges pass-1 fallback markers into pass-2 results

The two-pass path calls `stages::run` twice (independently) and then does its own merging. There is no shared state between passes. The merge logic (`merge_two_pass_markers`) does proximity dedup and ID dedup, but the dedup parameters are **different** from the intra-pass dedup (uses `config.dedup_radius` vs the built-in radius used in `dedup_markers`).

**Debug collection only works in single-pass mode**. The doc comment says so: "Debug collection currently uses single-pass execution." This means two-pass detection (camera, self-undistort) cannot be debugged.

### 3.6 DESIGN: Projective Center Correction Ordering

The projective center correction happens at two different points depending on the pipeline path:

| Context | When PC runs |
|---------|-------------|
| Fit-decode stage | Before dedup, before global filter |
| Finalize stage | After completion, after NL refine |

For **NL Board** mode: projective center is not used (mutually exclusive via `CircleRefinementMethod`). No conflict.

For **ProjectiveCenter** mode: PC runs in both locations. The first application affects dedup decisions (markers with corrected centers may not merge). The second application re-runs PC on all markers including completion markers.

### 3.7 MINOR: Refine-H Always Runs Debug Path

`refine_with_homography_with_debug` always produces debug records. The non-debug wrapper (`refine_with_homography`) calls the debug version and discards the debug output. This is minor but means the debug allocation cost is always paid.

### 3.8 MINOR: `h_current` Mutation Pattern in Finalize

The homography matrix flows through multiple stages that can mutate it:
```
global_filter -> h_current (initial)
NL refine H-refit -> h_current (optionally updated)
phase_final_h -> final refit (optionally updated)
```

The final `phase_final_h` does its own refit from scratch, which may undo the NL refine's H refit. The "keep better" comparison is sound, but the overall flow is hard to trace.

### 3.9 OBSERVATION: Completion Edge Points Are Collected

Completion markers now store `edge_points_outer` and `edge_points_inner` (added in recent self-undistort work). This is good -- they can participate in self-undistort estimation. However, completion markers are assigned board IDs unconditionally (even with decode mismatch), so their edge points may be associated with incorrect IDs.

---

## 4. Data Flow Summary

```
proposals: Vec<Proposal>
    |
    | (per-candidate processing)
    v
markers: Vec<DetectedMarker>   [all valid fits, decoded or not]
    |
    | (projective center -- 1st time, if enabled)
    |
    | (dedup by proximity + ID)
    v
markers: Vec<DetectedMarker>   [deduped]
    |
    | (global_filter: keeps only decoded inliers)
    v
markers: Vec<DetectedMarker>   [decoded inliers only]
    |
    | (refine_h: re-fit at H-projected priors)
    v
markers: Vec<DetectedMarker>   [refined or original]
    |
    | (completion: add missing IDs)
    v
markers: Vec<DetectedMarker>   [+ completion markers]
    |
    | (NL refine: board-plane circle fit)
    |
    | (projective center -- 2nd time, if enabled)
    v
markers: Vec<DetectedMarker>   [center-corrected]
    |
    | (final H refit)
    v
DetectionResult
```

---

## 5. Simplification Opportunities

### 5.1 Eliminate Double Projective-Center Application

The current design applies projective center in two places because:
1. Pre-dedup: so that corrected centers are used for spatial dedup
2. Post-completion: so that completion markers also get corrected

**Proposed fix**: Apply projective center **once**, after completion, and change dedup to use the ellipse center (not `m.center`) for proximity decisions. This requires a minor change to `dedup_by_proximity` to use `m.ellipse_outer.center_xy` instead of `m.center`.

### 5.2 Unify Acceptance Logic

Create a single `MarkerAcceptanceGate` struct that encodes acceptance criteria:
```rust
struct MarkerAcceptanceGate {
    require_decode: bool,
    min_confidence: Option<f32>,
    min_arc_coverage: Option<f32>,
    max_reproj_err_px: Option<f32>,
    scale_range: Option<(f32, f32)>,
}
```

Use it consistently across fit-decode, completion, and refine-h stages.

### 5.3 Flatten the Orchestration

Currently: `stages::run` -> `stage_fit_decode::run` + `stage_finalize::run`, where `stage_finalize::run` has 5+ sub-phases as separate functions.

The `stages/mod.rs` `run` function is a trivial 8-line glue function that calls two sub-functions. Consider merging `stage_finalize::run` into the top-level `stages::run` to eliminate one level of indirection.

### 5.4 Make Two-Pass a Pipeline Concern

Currently, two-pass logic is outside the pipeline (in `detect.rs`), calling `stages::run` twice and doing its own merge. Consider making two-pass a first-class concept inside the pipeline, so that debug collection and self-undistort can work together naturally.

### 5.5 Separate Debug from Logic

The `collect_debug` flag creates significant code branching throughout `stage_finalize`. Many functions exist in pairs (e.g., `global_filter` vs `global_filter_with_debug`). Consider a "debug sink" trait/interface that can be either real or no-op, reducing branching.

---

## 6. Detector API Surface Assessment

The current `Detector` API exposes:

| Method | When to use |
|--------|------------|
| `detect` | Standard single-pass, possibly with camera in config |
| `detect_with_camera` | Override camera model per-image |
| `detect_with_mapper` | Custom distortion adapter |
| `detect_with_self_undistort` | Estimate + apply lens distortion |
| `detect_precision` | Alias for `detect_with_camera` |
| `detect_with_debug` | CLI/internal use (behind feature gate) |

**Observations**:
- `detect_precision` is a pure alias for `detect_with_camera` with no additional logic. Consider removing.
- `detect_with_camera` clones the entire config to set one field. Could use `config.camera` directly.
- `detect_with_self_undistort` has unique orchestration (run once, estimate, optionally re-run). This is the only place where the result's `self_undistort` field is populated.
- When `config.camera` is set, `detect()` implicitly triggers the two-pass pipeline via `config_mapper()`. This implicit behavior could surprise users.

### 6.1 `detect()` Implicit Two-Pass Behavior

`detect_rings()` calls `config_mapper(config)` which returns `Some(&CameraModel)` when `config.camera` is set. However, looking more carefully:

```rust
fn config_mapper(config: &DetectConfig) -> Option<&dyn PixelMapper> {
    config.camera.as_ref().map(|c| c as &dyn PixelMapper)
}

pub fn detect_rings(gray: &GrayImage, config: &DetectConfig) -> DetectionResult {
    detect_rings_with_mapper_and_seeds(gray, config, config_mapper(config), &[], &SeedProposalParams::default())
}
```

This passes the mapper to `stages::run`, which uses it for coordinate transforms during edge sampling. But this is **NOT** the two-pass path -- it's single-pass with a mapper. The two-pass path requires calling `detect_rings_with_mapper()` explicitly.

So: `detect()` with `config.camera` set does single-pass detection in the undistorted frame. `detect_with_camera()` also does single-pass. The two-pass pipeline is only triggered by `detect_with_mapper()` and `detect_with_self_undistort()`.

This is actually correct but could be clearer in documentation.

---

## 7. Distortion / Mapper Analysis

### 7.1 Coordinate Frames

The pipeline operates in two coordinate frames:

| Frame | Description |
|-------|------------|
| **Image frame** | Raw pixel coordinates in the distorted image |
| **Working frame** | Undistorted pixel coordinates (= image frame when no mapper) |

The `DistortionAwareSampler` handles the bridge:
- `image_to_working_xy`: undistort (image → working)
- `working_to_image_xy`: redistort (working → image)
- `sample_checked(x_working, y_working)`: maps working→image, then bilinear-samples the raw image

### 7.2 How the Mapper Is Used Inside a Single Pass

When `stages::run` receives a mapper:

1. **Proposals are always in image frame**: `find_proposals()` runs Scharr gradients on the raw `gray` image. Returned `Proposal.{x, y}` are raw pixel coordinates.

2. **Proposal mapped to working frame**: In `process_candidate`, the first operation is `sampler.image_to_working_xy([proposal.x, proposal.y])`. The returned `center_prior` is in **working frame**.

3. **Outer estimation**: `estimate_outer_from_prior_with_mapper` receives `center_prior` in working frame. It casts radial rays in working-frame coordinates and uses `sampler.sample_checked(x, y)` which internally maps working→image for pixel lookup. The radius hypotheses and all coordinate computations are in working frame.

4. **Edge sampling**: `sample_outer_edge_points` receives `center_prior` in working frame. It casts rays `cx + dx * r, cy + dy * r` in working frame, samples via `sampler.sample_checked` (which does working→image→bilinear). **Crucially: the returned `outer_points` are in working frame** (line 205-207: `let x = cx + dx * r`).

5. **Ellipse fit**: `fit_outer_ellipse_with_reason` fits an ellipse to working-frame edge points. The fitted `Ellipse` has center/axes in working frame.

6. **Decode**: `decode_marker_with_diagnostics_and_mapper` receives the fitted ellipse (working frame) and creates a `DistortionAwareSampler` internally. It samples at `cx + r*cos(theta), cy + r*sin(theta)` using `sampler.sample_checked`, so sampling coordinates are working frame → mapped to image for pixel lookup.

7. **All output coordinates are in working frame**: `marker.center`, `marker.ellipse_outer`, `marker.edge_points_outer`, etc. are all in working-frame coordinates.

**Key insight: the mapper is NOT used to undistort the image. The raw image is always used for pixel lookups. The mapper defines a coordinate transform: all geometric computations (ray casting, ellipse fitting, center coordinates) happen in working frame, and pixel values are fetched by mapping working→image→bilinear lookup.**

### 7.3 Statement Verification

#### Statement 1: "If mapper is provided, the two-pass pipeline is run"

**PARTIALLY CORRECT, depends on the entry point.**

- `Detector::detect_with_mapper()` → calls `detect_rings_with_mapper(Some(mapper))` → **YES**, triggers two-pass via `detect_rings_two_pass_with_mapper`.
- `Detector::detect()` with `config.camera` set → calls `detect_rings()` → `detect_rings_with_mapper_and_seeds(mapper=config_mapper())` → **NO**, this is single-pass WITH mapper. The mapper is used for coordinate transforms within the single pass, but there is no pass-1-without-mapper.
- `Detector::detect_with_camera()` → clones config with camera, calls `detect_rings()` → same as above, **single-pass with mapper**.
- `Detector::detect_with_self_undistort()` → pass-1 via `detect_rings()` (no mapper since `config.camera` defaults to None for self-undistort use case), then `detect_rings_two_pass_with_mapper` → **YES**, two-pass.

**Summary**: Two-pass is triggered only by `detect_with_mapper()` and `detect_with_self_undistort()`. The `detect()` and `detect_with_camera()` methods run single-pass even when a camera/mapper is present.

#### Statement 2: "Pass 1 detects markers on a raw image (no mapper is used). The mapper is used for the homography fit. Pass 1 also saves all edge elements for all detected markers."

**MOSTLY CORRECT with important nuances.**

- Pass 1 runs via `detect_rings_with_mapper_and_seeds(gray, config, None, &[], &params.seed)` — mapper=None, so everything is in image frame. **Correct: no mapper used in pass 1.**
- **Incorrect about homography**: The homography fit in the global filter uses marker centers (which are in image frame in pass 1). The mapper plays no role in the homography fit itself. The homography maps board-mm → image-px (pass 1) or board-mm → working-px (pass 2).
- **Correct about edge points**: All markers store `edge_points_outer` and `edge_points_inner`. In pass 1, these are in image-frame coordinates.
- **Nuance**: Pass-1 marker centers fed as seeds to pass 2 are in **image frame** (raw pixel coordinates). This is correct because pass-2 proposals also start in image frame before being mapped to working frame.

#### Statement 3: "Second pass uses the mapper to undistort edge elements for the detected markers, then fit markers again in the undistorted space and get the final result."

**INCORRECT in its description of the mechanism, but the effect is approximately as described.**

The second pass does NOT take pass-1 edge points and undistort them. Instead:

1. Pass-1 marker centers are collected as seed proposals (in image frame)
2. Pass 2 runs the **entire pipeline from scratch** with mapper=Some(mapper):
   - New proposals are generated (from raw image + injected seeds)
   - Each proposal is mapped image→working
   - New edge points are sampled in working frame (with pixel lookups via mapper)
   - New ellipses are fitted in working frame
   - New decode, dedup, global filter, completion, etc.
3. The pass-2 result is a completely independent detection in working frame
4. Pass-1 fallback markers (those with IDs not found in pass 2) are remapped to working frame via `map_marker_image_to_working` (which drops ellipse/edge-point fields since they don't transform correctly through nonlinear mapping)

**The pass-1 edge points are NOT reused in pass 2.** Pass 2 re-samples everything from scratch. The only information transferred from pass 1 to pass 2 is the set of marker center locations (used as proposal seeds).

#### Statement 4: "In the case of self-undistortion, the distortion model is fitted after the pass 1 and then is used in the pass 2 the same way as external would be used."

**CORRECT.** The flow in `detect_with_self_undistort`:

1. `detect_rings(image, &self.config)` — standard single-pass, no mapper, returns markers with edge points in image frame
2. `estimate_self_undistort(&result.detected_markers, image_size, su_cfg)` — uses stored edge points to estimate a `DivisionModel` (lambda parameter)
3. If `applied == true`: `detect_rings_two_pass_with_mapper(image, &self.config, &model, &TwoPassParams::default())` — the `DivisionModel` implements `PixelMapper` and is used exactly like any other mapper in the two-pass pipeline

The `DivisionModel` is treated identically to a `CameraModel` in the two-pass pipeline.

### 7.4 Corrected Understanding

```
detect_with_self_undistort:
  |
  +-- detect_rings(no mapper)          → pass-1 result in IMAGE frame
  |     edge_points_outer/inner are in image frame
  |
  +-- estimate_self_undistort()         → DivisionModel {lambda, cx, cy}
  |     uses pass-1 edge points to estimate lambda
  |
  +-- detect_rings_two_pass_with_mapper(model):
        |
        +-- detect_rings(no mapper)     → internal pass-1 in IMAGE frame
        |     produces seed centers
        |
        +-- detect_rings(with model)    → pass-2 in WORKING frame
        |     proposals: image frame (Scharr on raw)
        |     → mapped to working frame
        |     → edge sampling in working frame (pixels via mapper)
        |     → ellipse fit in working frame
        |     → all results in working frame
        |
        +-- merge: pass-1 fallback markers remapped to working frame
             (ellipse/edge fields dropped for remapped markers)
```

**Note**: The self-undistort path actually runs the image THREE times total:
1. Initial `detect_rings` (for lambda estimation)
2. Internal pass-1 of `detect_rings_two_pass_with_mapper` (for seed generation)
3. Pass-2 of `detect_rings_two_pass_with_mapper` (final result)

Passes 1 and 2 from step (1) above produce essentially the same result (both are single-pass without mapper). This is a potential optimization opportunity — the initial detection result could be reused as the internal pass-1.

---

## 8. Summary of Recommended Actions

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| High | Double projective center application | Move to single post-completion application |
| High | Undecoded markers silently dropped by global filter | Document or add option to retain |
| Medium | Short circuit skips NL refine silently | Add warning or remove short circuit |
| Medium | Two-pass + debug incompatibility | Document limitation or implement two-pass debug |
| Medium | Inconsistent acceptance criteria | Unify with shared gate struct |
| Low | `detect_precision` is a pure alias | Consider removing |
| Medium | Self-undistort runs image 3x (could be 2x) | Reuse initial detection as internal pass-1 |
| Low | Refine-H always allocates debug | Add non-debug fast path |
| Low | `stages/mod.rs::run` is trivial glue | Consider flattening |
