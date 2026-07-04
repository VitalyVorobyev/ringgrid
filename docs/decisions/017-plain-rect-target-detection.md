# ADR-017: Plain / Rect Target Detection & the Compositional Target Model

- **Status:** accepted
- **Date:** 2026-07-04
- **Author role:** pipeline-architect
- **Supersedes:** none

## Context

ringgrid began as a detector for a single target: a dense hex lattice of
16-sector **coded** rings, where each marker decodes to a globally unique
codebook ID and IDs anchor the board-to-image homography. Real calibration work
also needs **plain** (uncoded) targets — plain annuli on a **rectangular**
lattice — which are cheaper to fabricate and standard in some metrology
pipelines. Plain rings carry no identity, so the coded pipeline's decode →
ID-correction → global-filter back half cannot label them, and there is no
per-marker key to anchor a board frame.

Two questions had to be answered together:

1. **How is the target described?** A `BoardLayout` hard-wired to hex + coded
   could not express rect lattices, plain rings, or origin fiducials without
   accreting parallel flags and mirrored constructors.
2. **How are plain targets detected and reported?** Without IDs there is no
   global correspondence; labeling must come from lattice topology, and a board
   frame (absolute millimeters) can only be recovered from extra fiducials.

## Decision

**Compositional target model.** Replace the monolithic board with
`TargetLayout = lattice × ring geometry × coding × optional fiducials`
(`target/` module), where each aspect is an independent, validated type:
`LatticeGeometry {Hex | Rect}`, `RingGeometry`, `MarkerCoding {Coded16 | Plain}`,
`OriginFiducials`. Illegal combinations fail at construction, not at detection.
The legacy `BoardLayout` becomes a thin, deprecated v4 facade (removed after
0.8). The canonical serialized form is `ringgrid.target.v5`, auto-migrating v4.

**Grid-labeling detection for plain targets.** The plain finalize path
(`pipeline/finalize/plain.rs`) keeps the shared front half (fit + projective
center) but replaces decode/ID-correction/global-filter with:

- **Grid assignment** (`pipeline/assign.rs`): topological labeling via
  `projective_grid::detect_grid`, canonicalized to a stable *relative* frame,
  then a `f64` RANSAC frame homography over the labeled correspondences —
  structurally the same "keep homography inliers" contract as the coded global
  filter.
- **Origin anchoring** (`pipeline/anchor.rs`): when the target carries origin
  fiducials, enumerate determinant-`+1` coordinate maps (rotations only —
  reflections would decode a mirror target), fit each board→image homography,
  and score by projecting the fiducial dots and measuring their darkness. Accept
  only above an absolute contrast floor **and** a margin over the runner-up.

**Honest frame contract.** Anchored detections report `BoardFrame::Absolute`
with `board_xy_mm`; unanchored detections report `BoardFrame::RelativeCanonical`
and clear `board_xy_mm` entirely. A wrong millimeter position is worse than
none.

## Consequences

**Positive:**
- One target model expresses hex/rect × coded/plain × fiducials; new
  combinations need no new match arms.
- Plain rect targets detect, label, and (with dots) anchor to absolute board mm.
- The geometric-verify gate is lattice-generic and coordinate-keyed, so it
  guards coded and plain outputs identically.
- Reflection exclusion and the contrast+margin acceptance make origin resolution
  fail *safe* (unresolved) rather than *wrong*.

**Negative:**
- Plain anchoring depends on visible fiducial dots; without them, outputs are
  relative-frame only.
- Origin enumeration is capped (`MAX_CANDIDATES = 512`); pathologically large
  labeled patches are truncated (logged, not silently).

**Neutral:**
- Rect **coded** targets are supported but skip hex-neighbor ID correction.
- `board_frame` is only meaningful for plain targets; coded targets are always
  absolute.

## Evidence

Covered by `crates/ringgrid/tests/plain_target_e2e.rs`: plain rect anchors under
image rotation, plain rect/hex without dots stay in the relative frame, and
coded rect detects absolute IDs. Origin-resolution unit tests in
`pipeline/anchor.rs` confirm identity/rotated labelings resolve and that missing
dots yield *unresolved* rather than a wrong origin. Rect synthetic benchmark:
`tools/run_rect_benchmark.sh` (baseline in `tools/ci/regression_baseline.json`).

## Affected Modules

- `crates/ringgrid/src/target/` (`layout.rs`, `lattice.rs`, `ring.rs`,
  `fiducials.rs`, `schema.rs`, `error.rs`)
- `crates/ringgrid/src/board_layout.rs` (deprecated v4 facade)
- `crates/ringgrid/src/pipeline/finalize/plain.rs`, `finalize/mod.rs`
- `crates/ringgrid/src/pipeline/assign.rs`, `pipeline/anchor.rs`
- `crates/ringgrid/src/pipeline/result.rs` (`BoardFrame`, `grid_coord`,
  `board_xy_mm`)
