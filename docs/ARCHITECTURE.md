# ringgrid Architecture

## Scope and status

This document describes the architecture that is implemented today and the planned evolution needed for precision and maintainability.

`ringgrid` detects coded ring markers on a known planar target. The current production path is:

`proposal -> local ring fit/decode -> dedup -> optional global homography filter -> optional refinement/completion -> optional NL board-plane refine`.

## Current crate layout

```text
crates/
  ringgrid-core/
    src/
      ring/
        proposal.rs       # center proposals (gradient-voting radial symmetry)
        outer_estimate.rs # outer edge radius hypotheses near expected scale
        edge_sample.rs    # low-level image sampling helpers
        detect.rs         # top-level orchestration (delegates into ring/detect/*)
        detect/           # non_debug/debug/completion/refine_h helpers
        pipeline/         # dedup + global-filter shared stages
        inner_estimate.rs # inner radius estimation constrained by outer ellipse
        decode.rs         # 16-sector sampling + codebook matching
      conic.rs            # ellipse/conic fit and utilities
      homography.rs       # DLT + RANSAC homography fitting
      projective_center.rs # projective unbiased center from inner/outer conics
      refine.rs           # public refine API + wrappers
      refine/             # math/sampling/solver/pipeline helpers
      debug_dump.rs       # versioned debug schema
      marker_spec.rs      # marker geometry priors and estimator controls
      board_spec.rs       # generated embedded board constants
      lib.rs              # public result/data structs
  ringgrid-cli/
    src/main.rs           # CLI and config wiring
```

## Implemented pipeline (2026-02)

### 1) Proposal stage (`ring/proposal.rs`)

- Uses Scharr gradients and radial voting to produce candidate centers.
- Applies accumulator smoothing and NMS.

### 2) Local fit/decode stage (`ring/detect.rs` + helpers)

For each proposal:

1. Estimate outer-radius hypotheses near expected marker size (`outer_estimate.rs`).
2. Sample radial edge points around each hypothesis (`detect.rs`).
3. Fit outer ellipse with RANSAC/direct fallback (`conic.rs`).
4. Decode 16-sector code (`decode.rs`).
5. Estimate inner radius from outer ellipse (`inner_estimate.rs`) for geometry/debug.
6. Build `DetectedMarker` with fit/decode metrics.

### 3) Dedup stage (`ring/detect.rs`)

- Proximity dedup by center radius.
- ID dedup (keep best confidence per decoded ID).

### 4) Global board filter (`ring/detect.rs` + `homography.rs`)

- Build board-to-image correspondences from decoded IDs.
- Fit homography with RANSAC.
- Keep inlier decoded markers.

### 5) H-guided local refine (`ring/detect.rs`)

- Reproject board IDs via fitted `H`.
- Refit local ring geometry at projected priors.
- Keep updates only if decode/scale gates pass.

### 6) Homography-guided completion (`ring/detect.rs`)

- For missing IDs, attempt conservative local fits at `H`-projected locations.
- Accept only with strict arc/fit/reprojection gates.

### 7) NL board-plane refine (`refine.rs`)

- Unproject sampled edge points to board plane.
- Solve robust circle-center optimization per marker.
- Optionally refit homography and rerun refine.

### 8) Outputs

- `DetectionResult` (`lib.rs`) for stable detector output.
- Includes optional `center_projective`, `vanishing_line`, and selection residual per marker.
- Optional versioned debug dump (`debug_dump.rs`, schema `ringgrid.debug.v1`).

## Known architecture debt

### Mixed responsibilities

- `ring/detect.rs` is now a slim orchestrator, but `ring/detect/debug_pipeline.rs` (~727 LOC) and `ring/detect/completion.rs` (~634 LOC) are still large mixed-responsibility modules.
- `refine.rs` is now slim, but `refine/pipeline.rs` (~524 LOC) still concentrates many per-marker decision branches.
- `conic.rs` combines model definitions and multiple algorithmic layers (fit + solver internals + RANSAC).
- CLI `run_detect` has broad parameter plumbing and policy coupling.

### Duplicate or near-duplicate logic

- Debug/non-debug branch duplication has been reduced for dedup/global-filter/refine-H core paths, but marker assembly is still repeated across multiple stage modules.
- Marker assembly (`FitMetrics`, `DecodeMetrics`, `DetectedMarker`) is repeated at several call sites.
- `inner_estimate.rs` and `outer_estimate.rs` repeat similar radial aggregation/peak-consistency code.
- Radial edge probing code appears in both `ring/detect/*` and `refine/*`.
- `ring/edge_sample.rs::sample_edges` is currently not used by production pipeline.

## Refactoring roadmap

### Phase R1: Split pipeline orchestration (no behavior change)

Status: completed.

- `detect_rings*` signatures remained stable.
- Logic was extracted into focused modules (`ring/pipeline/*`, `ring/detect/*`, `ring/detect/non_debug/*`).

Exit criteria:
- All current tests pass. Completed.
- Synthetic eval aggregate parity checks are byte-identical on baseline run. Completed.

### Phase R2: Remove duplication and centralize primitives

Status: in progress.

- Add shared marker-build helpers (fit/decode structs + conversion functions).
- Introduce shared radial-profile utility module used by inner/outer/refine.
- Decide fate of `edge_sample::sample_edges`:
  - adopt as canonical local sampling path, or
  - deprecate/remove if superseded.

Work completed under this phase so far:
- `refine_markers_circle_board` flow split into `refine/pipeline.rs` with helper modules (`refine/math.rs`, `refine/sampling.rs`, `refine/solver.rs`).
- Inner-ellipse estimation is now run for every accepted outer fit (not decode-gated), reducing branching differences between decoded/undecoded local fits.
- `Conic2D` was moved into `conic.rs` as a shared primitive, removing duplicate conic-matrix conversion code from `projective_center.rs`.

Exit criteria:
- Duplicate-path functions removed or wrapped by common core path.
- No new behavior regressions in synthetic eval.

### Phase R3: Ellipse center correction (projective bias fix)

Problem:
- Ellipse center is not the projected circle center under perspective.

Status: partially completed (R3A complete, R3B pending).

Implemented in R3A:

- Added `projective_center.rs` with conic-pencil eigen recovery (`A = Q_outer * inv(Q_inner)`), robust candidate selection, and explicit degenerate-input errors.
- Added deterministic tests:
  - exact synthetic homography recovery,
  - conic scale invariance,
  - mild-noise stability.
- Integrated into detection outputs for all accepted markers that have both conics, in both debug and non-debug flows.
- Exposed config knobs in `DetectConfig.projective_center`:
  - `enable`,
  - `use_expected_ratio`,
  - `ratio_penalty_weight`.
- Added `DetectConfig.circle_refinement` selector:
  - `none`,
  - `projective_center_only`,
  - `nl_board_only`,
  - `nl_board_and_projective_center` (default).

Remaining R3B integration decisions:

1. Decide whether downstream stages (dedup/global filter/refine/completion) should consume `center_projective` as primary center.
2. Add synthetic eval reporting that compares legacy center vs projective center against GT.
3. Add stability gates for runtime fallback if future datasets expose new failure modes.

Exit criteria:
- New perspective-stress synthetic tests show center-error improvement.
- Global filter/decode rates do not regress.

### Phase R4: Camera calibration and distortion-aware sampling

Goal:
- Improve subpixel precision by accounting for lens distortion in edge sampling.

Plan:

1. Add camera module with explicit calibration structs.
2. Extend detect API/config with optional camera parameters.
3. Add distortion-aware sampling utility used by local fit and NL refine.
4. Add synthetic-distortion generation/eval support in tools.

Initial scope:
- Radial-tangential model only (policy decision for v1).

Exit criteria:
- Distorted synthetic benchmarks improve center precision.
- No regression when camera params are absent.

### Phase R5: Public API and target-spec stabilization

- Introduce a stable `Detector` API with layered options.
- Reduce default exposed parameter set; move advanced tunables into nested expert config.
- Adopt versioned runtime target specification schema.

Exit criteria:
- API docs cover stable fields and backward-compatibility guarantees.
- CLI maps cleanly onto the stable config surface.

## Proposed final public API direction

Keep two levels:

1. Simple API (stable, small):
   - `Detector::new(target, options)`
   - `Detector::detect(image)`
2. Expert API (advanced tuning):
   - nested option groups for proposal/edge/outer/inner/decode/homography/refine/completion.

Design guidance:

- Keep target/camera/codebook immutable within `Detector`.
- Require runtime target JSON in public API v1 (no embedded-target fallback at API boundary).
- Keep run-time toggles explicit and minimal.
- Keep debug output optional and versioned.

## Proposed stable parameter surface

Candidate stable top-level options:

- expected marker scale (`marker_diameter_px` or equivalent)
- minimum marker distance (`min_marker_separation_px`)
- global filter toggle + reprojection threshold
- completion toggle
- circle refinement method selector
- NL refine toggle (`enable_nl_refine`) as a required stable control
- decode confidence threshold
- optional radial-tangential camera calibration

Initial advanced groups (provisional, can be revised):

- proposal internals (radius range, gradient threshold, NMS knobs)
- outer/inner estimation internals (search windows, polarity, sample counts, consistency gates)
- decode sampling internals (band ratio, angular/radial sampling densities)
- completion internals (ROI radius, reprojection/fit/coverage gates, attempt caps)
- homography internals (RANSAC iters/min-inliers/seed)
- NL refine internals (solver iters, robust loss, min points, reject gates, H-refit loop)

Everything else should be advanced/experimental (not semver-stable initially).

## Target specification format (proposed)

Evolve from generated constants to a versioned runtime JSON schema, still allowing generated embedding.

Recommended schema fields:

- `schema` (for example `ringgrid.target.v1`)
- `name`, `units`
- `board`:
  - explicit marker list (`id`, `xy_mm`, optional `q`, `r`), or
  - parametric hex-lattice descriptor
- `marker_geometry`:
  - outer/inner/code-band radii
  - sector count
- `coding`:
  - codebook bit width and codewords (or referenced artifact)
- optional validation/tolerance metadata

Policy decision for public API v1:
- target specification JSON is mandatory input.

## Testing strategy for roadmap phases

- Keep existing unit tests green.
- Add targeted tests for:
  - conic-pencil/eigen center recovery and degeneracy handling
  - center-correction regression on perspective synthetic data
  - distortion-aware sampling consistency
- Keep end-to-end synthetic eval as release gate.

## Open decisions

- Which advanced parameters should remain publicly configurable vs internal-only?
- Should `min_marker_separation_px` be user-controlled directly or derived from target geometry?
- Should camera calibration remain optional in v1, or required by a strict profile mode?
