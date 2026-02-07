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
        detect.rs         # end-to-end orchestration + debug orchestration
        inner_estimate.rs # inner radius estimation constrained by outer ellipse
        decode.rs         # 16-sector sampling + codebook matching
      conic.rs            # ellipse/conic fit and utilities
      homography.rs       # DLT + RANSAC homography fitting
      refine.rs           # NL marker center refinement in board coordinates
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
- Optional versioned debug dump (`debug_dump.rs`, schema `ringgrid.debug.v1`).

## Known architecture debt

### Mixed responsibilities

- `ring/detect.rs` is a monolith (~2960 LOC) combining orchestration, local geometry logic, global filtering, completion, debug mapping, and many utility functions.
- `refine.rs` combines edge re-sampling policy, solver control, and debug/report assembly.
- `conic.rs` combines model definitions and multiple algorithmic layers (fit + solver internals + RANSAC).
- CLI `run_detect` has broad parameter plumbing and policy coupling.

### Duplicate or near-duplicate logic

- Debug/non-debug branches duplicate key operations in `detect.rs`:
  - dedup
  - global filtering
  - homography-based refinement
- Marker assembly (`FitMetrics`, `DecodeMetrics`, `DetectedMarker`) is repeated at several call sites.
- `inner_estimate.rs` and `outer_estimate.rs` repeat similar radial aggregation/peak-consistency code.
- Radial edge probing code appears in both `detect.rs` and `refine.rs`.
- `ring/edge_sample.rs::sample_edges` is currently not used by production pipeline.

## Refactoring roadmap

### Phase R1: Split pipeline orchestration (no behavior change)

- Extract `detect.rs` internals into focused modules:
  - `ring/pipeline/local_fit.rs`
  - `ring/pipeline/dedup.rs`
  - `ring/pipeline/global_filter.rs`
  - `ring/pipeline/refine_h.rs`
  - `ring/pipeline/completion.rs`
  - `ring/pipeline/debug_map.rs`
- Keep `detect_rings*` signatures stable.

Exit criteria:
- All current tests pass.
- Synthetic eval metrics unchanged within tolerance.

### Phase R2: Remove duplication and centralize primitives

- Add shared marker-build helpers (fit/decode structs + conversion functions).
- Introduce shared radial-profile utility module used by inner/outer/refine.
- Decide fate of `edge_sample::sample_edges`:
  - adopt as canonical local sampling path, or
  - deprecate/remove if superseded.

Exit criteria:
- Duplicate-path functions removed or wrapped by common core path.
- No new behavior regressions in synthetic eval.

### Phase R3: Ellipse center correction (projective bias fix)

Problem:
- Ellipse center is not the projected circle center under perspective.

Planned math:

- Represent observed ellipse as conic matrix `C`.
- Compute board-plane vanishing line `l` from homography `H`:
  - `l ~ H^{-T} [0, 0, 1]^T`.
- Corrected center is pole of `l` w.r.t. `C`:
  - `p ~ C^{-1} l` (or `adj(C) l` for numerical robustness).
- Dehomogenize `p` to image coordinates.

Integration plan:

1. Add conic-matrix conversion + pole helper utility.
2. When `H` is available, compute corrected center for every accepted local fit (policy decision).
3. If inner ellipse exists, compute second corrected center and fuse with quality weighting.
4. Gate by reprojection consistency; fallback to legacy center when unstable.

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
  - conic pole/line math and degeneracy handling
  - center-correction regression on perspective synthetic data
  - distortion-aware sampling consistency
- Keep end-to-end synthetic eval as release gate.

## Open decisions

- Which advanced parameters should remain publicly configurable vs internal-only?
- Should `min_marker_separation_px` be user-controlled directly or derived from target geometry?
- Should camera calibration remain optional in v1, or required by a strict profile mode?
