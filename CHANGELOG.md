# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Published `ringgrid` CLI** (built with `--features cli`): `gen`, `detect`,
  `batch`, and `example` subcommands. `gen` is recipe-driven — a small TOML/JSON
  target recipe (with CLI flag overrides) lowers to a `TargetLayout` and renders
  `target_spec.json` + SVG/PNG/DXF. Example recipes for all six valid target
  combinations ship embedded (`ringgrid example --list`) and in
  `crates/ringgrid/examples/targets/`. The feature is off by default so library
  builds keep a clean dependency graph (`cargo install ringgrid --features cli`).
- **Automatic origin-dot placement** — `OriginFiducials::auto` and
  `TargetLayout::with_auto_fiducials` place a valid asymmetric dot triad in the
  gaps near the board center (breaks every lattice rotation while staying inside
  the densely-labeled interior). Single source of truth for automatic placement,
  used by the recipe `dots: auto` option. (The `rect_24x24` preset keeps its own
  frozen dot geometry for print compatibility.)
- **Board-completeness signal** — `DetectionResult::board_complete`,
  `DetectConfig::require_complete_board`, and `DetectError::IncompleteBoard`.
  This is the success criterion for plain, no-dots targets, whose orientation can
  only be resolved from a fully-detected board.
- End-to-end test coverage of all six valid target combinations
  ({hex, rect} × {coded, plain} × {dots, no dots}, excluding coded + dots).

### Fixed

- **Hex origin-dot anchoring** — the plain-target origin anchor now enumerates
  the full dihedral symmetry group (rotations **and** reflections) and lets the
  physical orientation-preserving check select the valid embedding. The hex
  relative-labeling frame is mirrored relative to the board's axial frame, so
  hex + plain + dots boards previously never resolved an `Absolute` frame.

### Changed

- The in-repo dev CLI binary is renamed `ringgrid` → `ringgrid-dev`
  (`cargo run -p ringgrid-cli --bin ringgrid-dev -- …`), unpublished, freeing the
  `ringgrid` name for the published CLI. Its target writing now delegates to the
  shared library helper.
- Pre-1.0 migration guides moved from the mdBook to `docs/migrations/`.
- The interactive WASM demo now covers all six target combinations (one bundled
  sample each) instead of two, is data-driven (each sample carries its own
  inline target spec — no per-target WASM helpers), and surfaces the
  `board_complete` signal in the results.

## [0.9.0] — 2026-07-04

Post-0.8 cleanup: the deprecated v4 `BoardLayout` facade is removed, the private
"ISRA" label is gone from the public surface, targets export DXF for
fabrication, the Python `TargetLayout` API is fully typed and can render, and the
release pipeline is idempotent.

### Added

- **DXF target export** (`TargetLayout::render_target_dxf` /
  `write_target_dxf`, Python `TargetLayout.write_dxf`, and a `.dxf` from every
  CLI `gen-target` run): a pure-Rust R12 DXF in millimeters for laser/CNC
  fabrication of both coded-hex and plain-rect targets — ring boundaries as
  circles, dark code-band sectors as closed polylines, split across
  `rings` / `code` / `fiducials` layers.
- **Typed Python `TargetLayout` API** (mixed Rust/Python package): dataclasses
  mirroring the v5 model (`HexGeometry`/`RectGeometry`, `RingGeometry`,
  `Coded16`/`Plain`, `OriginFiducials`), presets, `from_json`/`from_dict`
  round-trips, and `Detector.from_target(...)`. `TargetLayout` now also exposes
  `write_svg` / `write_png` / `write_dxf` (previously only the deprecated
  `BoardLayout` could render).

### Changed

- **Breaking:** renamed the `isra_rect_24x24` preset to `rect_24x24` across
  every public token — Rust `TargetLayout::rect_24x24()`, the emitted JSON
  `name` value `rect_24x24`, CLI `gen-target preset rect24x24`, and WASM
  `rect_24x24_target_json()`. "ISRA / XG3D / drawing 5256-57-102" references are
  removed from docs and help; the geometry is unchanged.

### Removed

- **Breaking:** the deprecated v4 `BoardLayout` / `BoardMarker` facade (and the
  `BoardLayoutValidationError` / `BoardLayoutLoadError` aliases, the
  `Detector.from_board` / `DetectConfig.board` Python paths, and the Python v4
  `board_spec.json` helpers) is removed from the Rust and Python public API.
  It was a pure shim over `TargetLayout` — use `TargetLayout` instead
  (`TargetLayout::default_hex()` is the geometry-identical replacement). Legacy
  v4 `board_spec.json` files still load unchanged: `TargetLayout::from_json_*`
  (Rust) / `ringgrid.TargetLayout.from_json(...)` (Python) auto-migrate the v4
  schema to v5.

### CI

- Release workflows tolerate an already-published version (manual publish
  before tagging no longer fails the tag job): crates.io checks the sparse
  index and falls back to swallowing "already exists"; PyPI uses
  `skip-existing`; npm guards on `npm view`.

### Docs

- New book pages: consolidated **Plain / Rect Target Detection** algorithm page
  and a **both-targets** generate→detect→interpret tutorial; ADR-017 records the
  compositional-target-model and plain-detection rationale.
- Performance dashboard shows a **varied** scene set (sparse/dense hex + plain
  rect, differing resolutions) each with a **detection overlay**.

## [0.8.0] — 2026-07-04

The compositional target model ships end to end: rect lattices and plain
(uncoded) rings are first-class alongside the classic coded hex board, origin
fiducials resolve absolute board frames for targets with no per-marker IDs,
and the public surface is tiered so diagnostics/codebook internals no longer
crowd the stable root. Outer-radius estimation and code-band sampling both
become ellipse-aware, holding accuracy at tilt without touching the nominal
benchmark suite; several id-correction and conic-degeneracy soundness gaps
found in review are closed. The interactive WASM demo moves into the book and
the 0.8 documentation pass is complete.

### Added

- **Compositional target model** (`TargetLayout`, new `target` module): a
  printed target is now lattice (`hex` | `rect`) × ring geometry × coding
  (`coded16` | `plain`) × optional origin fiducials. Presets:
  `TargetLayout::default_hex()` (the classic board) and
  `TargetLayout::rect_24x24()` (24×24 rect lattice of plain rings at
  14 mm pitch with three Ø2.8 mm origin dots). Fiducial validation enforces
  dot/ring clearance and requires the dot pattern to break every rotational
  symmetry of the lattice (rotations only — an opaque planar target always
  images through an orientation-preserving homography, so reflections cannot
  cause ambiguity).
- **Target JSON schema v5** (`ringgrid.target.v5`): compositional
  lattice/marker/coding/fiducials shape. Loaders (`TargetLayout::from_json_*`,
  CLI `--target`, Python/WASM detector construction) accept v5 and
  auto-migrate legacy v4; writers emit v5.
- **Rect target generation**: SVG/PNG rendering dispatches on coding
  (stroked rings + 16-sector code band vs. filled annulus) and draws origin
  fiducial dots. CLI `gen-target` becomes a subcommand family:
  `hex` (classic flags), `rect` (`--rows --cols --pitch_mm ...` with optional
  `--dot_mm x,y` / `--dot_radius_mm`), `preset` (`rect24x24`, `default-hex`),
  and `from-spec` (render any target JSON). Output spec file is
  `target_spec.json` (v5).
- `DetectError` — `Detector::detect*` now return
  `Result<DetectionResult, DetectError>`, reserving room for future failure
  modes (currently infallible for all built-in targets).
- **Plain-target detection**: every built-in lattice × coding combination now
  detects end-to-end. Plain (uncoded) rings skip decoding entirely — a
  photometric ring-evidence score (annulus + inner-edge contrast) fills the
  decode slot in outer-fit candidate ranking — and are labeled with lattice
  coordinates by `projective_grid::detect_grid` (labeling only; the frame
  homography is refit in `f64` by ringgrid's RANSAC). Completion and
  geometric verification are coordinate-keyed and lattice-generic; plain
  completion grows the labeled patch iteratively, recovering cells the
  labeler drops.
- **Origin resolution** (`pipeline/anchor`): when a plain target carries
  origin fiducials, every (rotation × translation) labeling candidate is
  scored by verifying dot darkness at its predicted image positions through
  the candidate homography; the winner must clear an absolute contrast
  threshold plus a runner-up margin. Orientation-reversing candidates are
  rejected outright (physically impossible for an opaque planar target).
  Without fiducials — or when verification is ambiguous — outputs stay in a
  canonical relative frame and no millimeter positions are emitted.
- **Result surface**: `DetectedMarker.grid_coord: Option<[i32; 2]>` (lattice
  cell coordinate; the only key for plain targets, which have no IDs) and
  `DetectionResult.board_frame: Option<BoardFrame>`
  (`absolute` | `relative_canonical`). Exposed in the Python typed API and in
  the WASM/CLI JSON output.
- **Rect synthetic eval**: `tools/gen_synth_rect.py` (rect-plain
  renderer with in-plane rotation), `tools/score_detect_rect.py`
  (coordinate-keyed scoring with best-over-symmetry matching for unresolved
  frames, origin-resolution metrics), `tools/run_rect_benchmark.sh`
  (dots + no-dots modes), with baselines appended to
  `tools/ci/regression_baseline.json`.
- **Embedded live browser demo** (`book/demo/`, served at `/demo/` and iframe-
  embedded on the book's new Interactive Demo page): runs always-adaptive
  WASM detection over hex-coded and rect-plain sample images directly from
  the documentation, replacing the old standalone
  `crates/ringgrid-wasm/demo/` page (retired).
- **0.8 book documentation pass completed**: every chapter now reflects the
  compositional `TargetLayout` model, the `gen-target` subcommand family, and
  the 0.8 config/result surface — new Targets part, a 0.7→0.8 migration
  guide, and updated pipeline/config-reference/output-schema/CLI pages.

### Changed

- **`DetectConfig.board` → `DetectConfig.target: TargetLayout`** (breaking);
  `with_board` → `with_target`. `DetectConfig::from_target*` and `Detector`
  constructors now take `impl Into<TargetLayout>`, so existing `BoardLayout`
  callers keep compiling via `From<BoardLayout> for TargetLayout`.
- `BoardLayout` is now a thin facade over the target module: validation and
  hex cell generation have exactly one implementation, and the facade's
  geometry is bit-identical to `TargetLayout::default_hex()` (locked by a
  parity test). `BoardLayoutValidationError` / `BoardLayoutLoadError` are
  aliases of `TargetValidationError` / `TargetLoadError`.
- Free proposal helpers (`propose_with_marker_scale`,
  `propose_with_heatmap_and_marker_scale`) take `&TargetLayout`.
- ID correction (hex-neighbor BFS consensus) now runs only for hex coded
  targets — its algorithmic domain; rect coded targets rely on the global
  filter + geometric verification instead.
- **Public surface tiered** (breaking): the opt-in diagnostics types
  (`DetectionDiagnostics`, `MarkerDiagnostics`, `FitMetrics`, `DecodeMetrics`,
  `DetectionSource`, `InnerFitReason`, `InnerFitStatus`, `RansacStats`,
  `StageTimings`) moved from the crate root to `ringgrid::diagnostics`;
  codebook inspection helpers (`CodebookInfo`, `CodewordMatch`,
  `codebook_info`, `decode_word`) moved to `ringgrid::codebook`. The stable
  primary output (`DetectionResult` and its field types) stays at the root.
- `ProjectiveCenterConfig.max_center_shift_px` →
  `max_correction_shift_px` — the old name collided with the unrelated
  inner-fit gate `InnerFitConfig.max_center_shift_px`. A serde alias keeps
  0.7.x JSON configs loading; the Python `ProjectiveCenterConfig` dataclass
  mirrors the rename with the same back-compat.
- New `TargetLayout::coded_hex(pitch_mm, rows, long_row_cols, outer_r,
  inner_r, ring_width)` constructor with the deterministic geometry-derived
  name previously exclusive to `BoardLayout::new`; the name generator now has
  a single implementation shared by both.
- WASM `default_board_json()` now emits the v5 schema (consumers already
  accept v4 and v5 on input).
- New `DetectConfig::with_json_overlay(overlay)` — the canonical way to
  apply a partial config overlay (recursive merge + legacy-key
  normalization + target re-attachment). The CLI `--config`, Python, and
  WASM overlay paths now share this one implementation instead of three
  copies; it also keeps pre-0.8 overlays loading (a legacy
  `max_center_shift_px` key merged onto a serialized base would otherwise
  be rejected as a duplicate of the renamed field).
- Internal float orderings use `total_cmp` instead of
  `partial_cmp().unwrap()` — identical order for finite values, no panic
  path on NaN.
- `detector/config.rs` and `pipeline/finalize.rs` split into focused
  submodules (`config/{fit,scale,stages}`, `finalize/{coded,plain,common}`);
  pure moves, no behavior change.
- **Decode samples the code band along the fitted ellipse** at uniform
  parametric angle instead of a circle at the mean radius. A circle drifts
  off the elliptical code band on tilted views and corrupts sectors near the
  minor axis; parametric sampling keeps every sector's angular support equal
  at any eccentricity (the constant offset is absorbed by cyclic matching,
  exactly like image rotation). Equal-or-better across the whole benchmark
  suite (reference center-mean 0.0849 → 0.0846 px / 0.0600 → 0.0598 px).
- **Outer estimation is eccentricity-aware**: a second-harmonic radius model
  `r(θ) = c0 + c1·cos2θ + c2·sin2θ` (the first-order elliptical signature)
  is fitted to the per-theta edge peaks; when it clears its attach gates
  (same-edge bound, plausible amplitude, 2× SNR over its own residuals, and
  strictly better theta-consistency than the constant radius) it drives both
  the consistency gate and the per-ray refine centers, so strongly tilted
  markers stop losing rays near the major/minor axes. Near-circular and
  noisy (heavily blurred) peak fields keep the constant-radius path — the
  nominal benchmark suite is unchanged.
- `projective_center.max_correction_shift_px = None` now means "auto"
  (nominal marker diameter at the point of use). Previously the value was
  re-derived into the config on every `with_target`, silently clobbering
  explicit settings.
- One canonical neighborhood-radius statistic (`detector/neighbors`) backs
  the dedup, completion, and recovery size gates: f64, proper median,
  self-excluded. Completion seeds previously used an f32 upper median, and
  the recovery ratio included the queried marker's own radius in its own
  neighborhood median.

### Deprecated

- `BoardLayout`, `BoardMarker`, `BoardLayoutValidationError`,
  `BoardLayoutLoadError` (the flat v4 hex facade) — use `TargetLayout` and
  the target error types. The facade stays fully functional for the 0.8
  release cycle and will be removed after it.

### Fixed

- Target rendering drew each marker's codeword from its **cell index** instead
  of its assigned marker ID, so boards with a non-sequential `id_assignment`
  printed wrong codes. SVG/PNG generation now uses the assigned ID
  (regression-tested); sequential boards are unaffected.
- Coded-target validation now rejects boards with more cells than the
  embedded codebook (893) and `id_assignment` entries beyond it — both
  previously wrapped or decoded to unreachable IDs silently.
- Origin fiducial dots placed outside the marker bounding box were clipped
  from rendered SVG/PNG targets; canvas bounds now include fiducial extents.
- The inner-as-outer recovery warn threshold now follows the configured
  `ratio_threshold` instead of a separately hardcoded 0.75.
- ID-correction scale-vote predictions converted one-hop image distances to
  millimeters via the axial pitch, but hex board-adjacent centers sit at
  `pitch·√3` — predictions fell ~42 % short and outside the acceptance
  tolerance, so the scale-vote fallback silently never fired on hex boards.
- ID-correction scale votes now follow the locally-estimated board→image
  rotation (from the same trusted adjacent pairs); the previous axis-aligned
  prediction assumed an unrotated board and, near the hex 60° symmetry,
  voted for exactly the wrong lattice site.
- The ID-correction affine hypothesis casts one vote instead of one per
  neighbor — a single ill-conditioned affine could satisfy `min_votes` by
  itself with zero corroboration.
- Within one ID-correction batch, duplicate claims on the same ID are
  resolved by anchor-homography reprojection error (previously both applied,
  and the later confidence-based conflict pass could evict a correct
  assignment in favor of a confident wrong one).
- `homography_min_trusted` now acts as a ceiling — the effective floor
  scales with the visible marker count (`max(8, n/3)`), so sparse/partial
  views are no longer locked out of the geometric fallback.
- `conic_to_ellipse` degeneracy guards are scale-relative (conic
  coefficients are homogeneous; the old absolute epsilons falsely rejected
  validly-shaped small-scaled conics), and the ellipse-constrained
  eigenvalue solve survives repeated eigenvalues, falling back to nalgebra's
  Schur eigenvalues when the closed-form cubic route fails.
- **Python bindings**: `to_dict()` now serializes
  `projective_center.max_correction_shift_px`, `completion.max_attempts`, and
  `seed_proposals.max_seeds` as explicit JSON `null` when the Python value is
  `None`, instead of omitting the key. Overlays merge rather than replace, so
  omitting the key left a previously-set explicit value stuck forever —
  resetting to auto/unlimited could never round-trip.

- **Dependency migration: `projective-grid` 0.9 → 0.10.1.** The 0.10 release
  rewrote projective-grid's public API; ringgrid's three call sites migrate
  with no intended behavior change:
  - `GridCoords { i, j }` → `Coord { u, v }` (hex axial `q, r` map to `u, v`).
    Grid coordinates are in-memory only — no serialized output changes.
  - The deleted `hex::hex_predict_grid_position` → the lattice-generic
    `predict_grid_position(&grid, coord, LatticeKind::Hex)` introduced in
    projective-grid 0.10.1 (same opposite-pair midpoint math, so the
    geometric-verify local test and completion seeding keep their numerics).
  - Root `estimate_homography` → `geometry::estimate_homography` (move only;
    identical signature and math).
- `radsym` 0.4: no source changes required — ringgrid's proposal adapter uses
  only the RSD path, which the 0.4 API revision left intact.

## [0.7.0] — 2026-06-30

Production-grade precision for sensor calibration. Detection now provably rejects
markers that are geometrically inconsistent with the hex lattice, so only trusted
`{id, board_xy_mm, center}` correspondences reach the output — a single false
correspondence poisons a calibration bundle-adjust, so precision is paramount.
This is a precision-first behavior change: a few geometrically impossible
detections that previously survived (with merely reduced confidence) are now
removed.

### Added

- **Final geometric verification gate** (`AdvancedDetectConfig::geometric_verify`,
  default `true`). After the final homography, every decoded marker is checked
  against the hex lattice via two complementary tests and removed if inconsistent:
  a **local hex-midpoint** prediction (homography-free, locally affine so robust
  to lens distortion) and a **global final-H reprojection** backstop for boundary
  markers. Both thresholds adapt to the observed inlier-residual distribution
  (`max(floor, median + k·MAD)`), so the gate is recall-safe on clean and
  distorted boards alike. Set `geometric_verify = false` to keep every decoded
  marker and apply your own filtering.
- **Confirm-by-consistency** ID recovery
  (`IdCorrectionConfig::confirm_by_consistency`, default `true`): correctly
  decoded but non-exact markers in sparse/partial views that the voting stages
  cannot reach are promoted to trusted when a board-adjacent neighbor supports
  them and none contradicts — precision-first, never confirming an ID a confident
  local vote disputes.
- Opt-in `ProposalConfig::radius_step` to subsample voting radii for a faster
  proposal stage (default `1` = full coverage; subsampling trades recall under
  blur and is therefore opt-in).

### Changed

- **BREAKING (behavior): markers geometrically inconsistent with the hex lattice
  are now removed** rather than kept with a softened confidence. For calibration
  consumers a `Some(id)` marker is now a trusted, lattice-consistent
  correspondence.
- The axis-ratio consistency filter now **removes** outlier markers instead of
  clearing their `id` (which previously left a phantom `id: None` blob that
  scoring counts as a false positive).
- `FitMetrics::h_reproj_err_px` is now computed in the working frame by the
  geometric verification gate (fixing a latent frame-mixing inflation under an
  active distortion mapper); the gate is its sole writer.
- Hardening: non-finite guard in radial edge sampling, NaN-safe `total_cmp`
  ordering in radial-profile aggregation, and a deterministic ID tie-break in
  nearest-neighbor lookup.

### Removed

- **BREAKING: `AdvancedDetectConfig::topology_filter_threshold_px`** — superseded
  by the always-adaptive `geometric_verify` gate. The old fixed-threshold filter
  shipped disabled because no single threshold suited both clean and distorted
  boards; the adaptive gate makes that choice automatically and ships on.
- **BREAKING: `AdvancedDetectConfig::h_reproj_confidence_alpha`** and its soft
  confidence penalty — replaced by the hard geometric gate. The soft penalty also
  wrongly down-weighted true peripheral markers under uncorrected distortion.
- Dead `ProposalConfig` fields `edge_thinning` and `accum_sigma` (never read by
  the radsym-0.2 adapter).

Old JSON configs carrying the removed fields still deserialize (unknown fields are
ignored); only Rust callers that named those fields need updating.

## [0.6.0] — 2026-05-18

Public API revision: a deliberate, batched breaking cleanup of the `ringgrid`
crate surface. The goal is a small, stable contract — primary results stay
compact, opt-in diagnostics are a separate channel, and per-stage tuning is no
longer part of the durable API.

### Removed

- **Hidden `codec` / `codebook` module re-exports.** `ringgrid::codec` and
  `ringgrid::codebook` (previously `#[doc(hidden)]` re-exports of raw decoder
  internals) are gone. Use the new `ringgrid::{codebook_info, decode_word}`
  functions, which return `CodebookInfo` / `CodewordMatch`.
- **`DetectionResult::seed_proposals()`** — the pipeline-internal seed helper is
  no longer on the public primary result.
- **`DetectionResult::ransac`** — homography RANSAC statistics moved to
  `DetectionDiagnostics` (see below). `homography` and `self_undistort` remain on
  `DetectionResult`.
- **`DetectedMarker` fields `fit`, `decode`, `source`, `edge_points_outer`,
  `edge_points_inner`** — moved to the new `MarkerDiagnostics` type.
- **BREAKING: `Ellipse::to_conic` is now crate-private (`pub(crate)`).** It
  returned `ConicCoeffs`, a type in the private `conic` module that is not
  re-exported — callers could invoke the method but never name its return
  type, making it dead public API.
- **BREAKING: `DecodeConfig::DEFAULT_*` associated constants are now
  crate-private (`pub(crate)`).** The nine `DEFAULT_*` constants restated the
  `Default` impl as public API; no other stage config exposed its defaults
  this way. The `Default` impl remains the public contract for defaults.
- **BREAKING: the `proposal` module is no longer a public path.**
  `ringgrid::proposal` is now private — it was `pub` *and* its contents were
  root re-exported, so every item was reachable twice. The five public items
  (`Proposal`, `ProposalConfig`, `ProposalResult`, `find_ellipse_centers`,
  `find_ellipse_centers_with_heatmap`) remain available at the crate root;
  only the duplicate `ringgrid::proposal::*` access path is removed.
- **BREAKING: `propose_with_marker_diameter` and
  `propose_with_heatmap_and_marker_diameter` removed.** Each was exactly its
  `_scale` sibling with `MarkerScalePrior::from_nominal_diameter_px` applied to
  the argument — a public function for a one-call transform. Migration: call
  `propose_with_marker_scale(img, board,
  MarkerScalePrior::from_nominal_diameter_px(d))` (and likewise
  `propose_with_heatmap_and_marker_scale`).

### Changed

- **BREAKING: `DetectConfig` split into stable + advanced.** Per-stage tuning
  knobs moved under a nested `advanced: AdvancedDetectConfig`. The top level now
  holds only the durable user choices: `board`, `marker_scale`,
  `circle_refinement`, `self_undistort`. Migration: `cfg.inner_fit` becomes
  `cfg.advanced.inner_fit`, and likewise for every stage sub-config
  (`outer_fit`, `decode`, `proposal`, `edge_sample`, `completion`,
  `id_correction`, `ransac_homography`, `use_global_filter`, …). The config
  dump/overlay JSON now nests stage tuning under an `"advanced"` object — old
  flat config JSON no longer deserializes.
- **BREAKING: `DetectedMarker` slimmed.** It now carries only `id`, `confidence`,
  `center`, `center_mapped`, `board_xy_mm`, `ellipse_outer`, `ellipse_inner`.
  The detailed fit/decode/source/edge-point fields are on `MarkerDiagnostics`,
  obtained via `Detector::detect_with_diagnostics`.
- **BREAKING: `DetectionResult` slimmed.** `ransac` was removed (now on
  `DetectionDiagnostics`) and the `seed_proposals()` method was removed.
  `homography` and `self_undistort` are unchanged.
- **BREAKING: `BoardLayout` geometry fields are now getter methods.** Use
  `board.name()`, `board.pitch_mm()`, `board.rows()`, and
  `board.long_row_cols()` instead of direct field access. This protects the
  validated marker cache from going out of sync.
- **BREAKING: CLI detection JSON nests diagnostics.** Per-marker
  `fit` / `decode` / `source` / `edge_points_*` and the per-result `ransac`
  block moved under a nested `diagnostics` object in `detect.json`.
- **BREAKING: `#[non_exhaustive]` added to public config, diagnostics, and
  result structs.** `DetectConfig`, `AdvancedDetectConfig`, the 11 stage
  sub-configs, the diagnostics structs (`FitMetrics`, `DecodeMetrics`,
  `MarkerDiagnostics`, `DetectionDiagnostics`), and the result types can no
  longer be built with a struct literal from outside the crate. Construct them
  through their constructors or `Default` and mutate fields afterwards.
- **BREAKING: four stage sub-config types renamed for naming consistency.**
  Every per-stage tuning struct in `AdvancedDetectConfig` now ends in `Config`:
  `CompletionParams` → `CompletionConfig`, `ProjectiveCenterParams` →
  `ProjectiveCenterConfig`, `SeedProposalParams` → `SeedProposalConfig`, and
  `MarkerSpec` → `MarkerSpecConfig`. Pure type renames — field names, the
  `AdvancedDetectConfig` field names (`completion`, `projective_center`,
  `seed_proposals`, `marker_spec`), and all serde JSON keys are unchanged.
- **BREAKING: `RansacHomographyConfig` removed; merged into `RansacConfig`.**
  The two types had the identical four-field shape (`max_iters`,
  `inlier_threshold`, `min_inliers`, `seed`) and differed only in defaults.
  `AdvancedDetectConfig::ransac_homography` keeps its field name but its type
  is now `RansacConfig`; the homography defaults (`2000 / 5.0 / 6 / 0`) are
  preserved by `AdvancedDetectConfig::default()`. `RansacConfig` gained
  `#[non_exhaustive]` and `#[serde(default)]`. The `ransac_homography` serde
  JSON key is unchanged. Python bindings rename the `RansacHomographyConfig`
  dataclass to `RansacConfig` accordingly.
- **BREAKING: `ScaleTiers` tuple field is now private.** `ScaleTiers` exposed
  its `Vec<ScaleTier>` as a `pub` tuple field *and* offered a `tiers()`
  accessor — two read paths for the same data, with the `pub` field letting any
  consumer inject an empty tier list. The field is now private; construct
  custom tier sets with the new `ScaleTiers::new(tiers: Vec<ScaleTier>)`
  constructor and read them via `tiers()`. The preset constructors
  (`four_tier_wide`, `two_tier_standard`, `single`, `from_detected_radii`) are
  unchanged. The Python-level `ScaleTiers([...])` API is unaffected.

### Added

- **`Detector::detect_with_diagnostics`** (and the mapper variant
  `detect_with_mapper_diagnostics`) — returns
  `(DetectionResult, DetectionDiagnostics)`. `detect()` keeps its signature and
  returns the slim result; request diagnostics explicitly when you need
  per-marker fit/decode internals, edge points, or homography RANSAC stats.
- **`DetectionDiagnostics`** and **`MarkerDiagnostics`** — the opt-in
  debugging/tuning channel. `MarkerDiagnostics` is positionally aligned 1:1 with
  `DetectionResult` markers.
- **`codebook_info`** / **`decode_word`** and the **`CodebookInfo`** /
  **`CodewordMatch`** types — a small explicit codebook-inspection API replacing
  the removed raw `codec` / `codebook` re-exports.
- **`AdvancedDetectConfig`** — the advanced per-stage tuning struct now reachable
  from `DetectConfig::advanced`.
- **Root re-exports of `RansacConfig`, `AngularAggregator`, `GradPolarity`, and
  `UndistortConfig`** — types that previously appeared in public signatures or
  config fields without being nameable from the crate root.

### Migration

Before (0.5.x):

```rust
let mut cfg = DetectConfig::from_target(board);
cfg.completion.enable = false;
cfg.inner_fit.require_inner_fit = true;
let result = Detector::with_config(cfg).detect(&image);
for m in &result.detected_markers {
    println!("{:?} dist={}", m.id, m.decode.best_dist);
}
```

After (0.6.0):

```rust
let mut cfg = DetectConfig::from_target(board);
cfg.advanced.completion.enable = false;
cfg.advanced.inner_fit.require_inner_fit = true;
let (result, diag) = Detector::with_config(cfg).detect_with_diagnostics(&image);
for (m, d) in result.detected_markers.iter().zip(&diag.markers) {
    println!("{:?} dist={}", m.id, d.decode.best_dist);
}
```

## [0.5.6] — 2026-04-09

### Added

**WASM bindings: full config and detection API**
- `RinggridDetector.with_config(board_json, config_json)` — constructor with full config control.
- `config_json()` — get current detection config as JSON.
- `update_config(overlay_json)` — apply partial config overlay (only provided fields are updated).
- `detect_adaptive_with_hint` / `detect_adaptive_with_hint_rgba` — adaptive detection with nominal diameter hint.
- `detect_multiscale` / `detect_multiscale_rgba` — explicit multi-scale detection with custom tier sets.
- `default_config_json(board_json)` — free function returning default config for a board.
- `scale_tiers_four_tier_wide_json()` / `scale_tiers_two_tier_standard_json()` — scale tier preset JSON helpers.
- Config dump/overlay types mirror the Python bindings' proven pattern, including 3 fields not yet exposed in Python: `h_reproj_confidence_alpha`, `topology_filter_threshold_px`, `proposal_downscale`.

**Optimized marker ID assignment**
- Optional `id_assignment` field in `ringgrid.target.v4` board spec JSON for non-sequential codebook ID assignment optimized for local dissimilarity.
- Python optimizer (`tools/optimize_id_assignment.py`): greedy initialization + simulated annealing to maximize minimum cyclic Hamming distance between hex-adjacent markers.
- Pre-optimized board specs: `tools/board/board_spec_optimized.json` (base profile, min adjacent distance 5) and `tools/board/board_spec_extended_opt.json` (extended profile).
- Diagnostic script (`tools/board_adjacency_report.py`) for analyzing adjacency distance statistics.
- Results on default 203-marker board: min adjacent distance raised from 2 (sequential) to 5 (optimized base), mean from 4.67 to 6.54.

**WASM demo enhancements**
- Detection config panel with quick controls (marker scale, completion toggle, circle refinement, proposal downscale) and full JSON editor.
- Two new detection modes: `detect_adaptive` with diameter hint and `detect_multiscale` with tier presets.
- Edge point visualization toggle (outer and inner edge points rendered as dots).
- Persistent detector lifecycle: detector is created once and reused across detections.

### Changed

- Demo now requires serving from the repository root for correct test image and WASM module paths.

## [0.5.5] — 2026-04-08

### Fixed

- Fix npm publish 404 by using scoped package name `@vitavision/ringgrid` in the release workflow.
- Guard pixel-count multiplication against overflow in WASM validation (wasm32 `usize` is 32-bit).

## [0.5.4] — 2026-04-07

### Fixed

- Fix WASM runtime panic ("RuntimeError: unreachable") caused by `std::time::Instant` usage in the detection pipeline. The `wasm32-unknown-unknown` target does not implement `Instant::now()`. Replaced with `web-time` crate on `wasm32` targets, which delegates to `performance.now()` in browsers.

### Added

- Comprehensive unit tests for `ringgrid-wasm` crate (20 native parity tests + 6 headless WASM integration tests):
  - Detection parity: verify WASM bindings produce identical results to native Rust for `detect`, `detect_adaptive`, `detect_rgba`, and `propose_with_heatmap`.
  - WASM integration tests run in headless Safari via `wasm-pack test` with embedded fixture image.
- Interactive browser demo (`crates/ringgrid-wasm/demo/index.html`):
  - Three detection modes: `detect`, `detect_adaptive`, `propose_with_heatmap`.
  - Overlay visualization with ellipses, centers, IDs, and confidence color coding.
  - Heatmap canvas, collapsible JSON result panel, timing display.
- `init_panic_hook()` WASM export for better error messages via `console_error_panic_hook`.
- mdBook documentation page for WASM usage and demo instructions.

### Dependencies

- Added `web-time` (1.x) as a conditional dependency for `wasm32` targets in `ringgrid`.
- Added `console_error_panic_hook` (0.1) to `ringgrid-wasm`.

## [0.5.3] — 2026-04-05

### Added

- Integrated `projective-grid` crate as a dependency for hex-lattice geometry.
- Hex-neighbor completion seeds via `hex_predict_grid_position`: completion now
  predicts missing marker positions from axial midpoints of detected hex
  neighbors before falling back to local affine or global homography.
  Validated on rtv3d dataset: +6 decoded markers on strategy A, +5 on strategy B.
- Optional topology-aware outlier filter (`topology_filter_threshold_px` config):
  flags markers whose image positions deviate from hex-neighbor midpoint
  predictions. Disabled by default; available for high-false-positive scenarios.


### Changed

**Proposal stage: replaced internal implementation with `radsym` RSD**
- The proposal module now delegates center detection to radsym's fused Radial
  Symmetry Detector (`rsd_response_fused`) with Scharr gradient, instead of
  the internal Scharr gradient voting implementation.
- Uses radsym 0.1.3's fused multi-radius API: all radii accumulated in a single
  pass with one Gaussian blur, matching the old code's algorithmic structure.
- Deleted internal `gradient.rs`, `nms.rs`, and `voting.rs` (~440 lines); the
  proposal module is now a thin adapter over radsym (~60 lines of glue code).
- Public API unchanged: `Proposal`, `ProposalConfig`, `ProposalResult`,
  `find_ellipse_centers`, and `find_ellipse_centers_with_heatmap` retain their
  signatures and serde format.
- `imageproc` moved from runtime dependency to dev-dependency (only used in
  test utilities).
- Performance at parity: 5.5% faster on the 720×540 fixture benchmark (29.5ms
  vs 30.1ms on main); larger synthetic images ~25% slower due to Gaussian blur
  scaling.
- All 4 regression benchmarks pass (reference, distortion, blur3, rtv3d).

**Workspace versioning**
- Introduced shared `workspace.package` fields (version, edition, license,
  repository, homepage, rust-version) in the root `Cargo.toml`. Crate-level
  manifests now inherit these via `field.workspace = true`.

## [0.5.1] — 2026-03-28

### Added

**Rust CLI target generation**
- Added `ringgrid gen-target` to generate canonical `board_spec.json` plus
  printable SVG/PNG artifacts directly from board geometry arguments.
- The Rust CLI path is now documented as equivalent to the Rust API writers and
  the repo-local `tools/gen_target.py` workflow for the same geometry/print
  settings.
- Added `gentarget.sh` as a repo helper for common target-generation inputs.

**Proposal diagnostics as public API**
- Extracted proposal generation into a standalone public `ringgrid::proposal`
  module with `find_ellipse_centers(...)` and
  `find_ellipse_centers_with_heatmap(...)`.
- Added detector-backed Rust helpers `Detector::propose(...)` and
  `Detector::propose_with_heatmap(...)`, plus size-aware free functions that
  derive proposal tuning from board geometry and marker scale hints.
- Added Python proposal diagnostics surfaces:
  `Detector.propose(...)`, `Detector.propose_with_heatmap(...)`,
  module-level `ringgrid.propose(...)`, `ringgrid.propose_with_heatmap(...)`,
  and the `ProposalResult` payload with proposals plus accumulator heatmap.
- Added `ringgrid.viz.plot_proposal_diagnostics(...)` and the repo-local
  `tools/plot_proposal.py` workflow for overlaying proposal peaks and heatmaps.
- Added CLI output support for proposal diagnostics via
  `ringgrid detect --include-proposals`, which writes top-level
  `proposal_frame`, `proposal_count`, and `proposals` fields.

**Performance instrumentation**
- Added a dedicated `detect_fixture` benchmark and checked-in detection fixture
  data for proposal-stage performance work and reproducible measurements.

### Changed

**Proposal stage architecture**
- Refactored the old monolithic proposal implementation out of
  `detector/proposal.rs` into a dedicated `proposal/` module split across
  `config`, `gradient`, `voting`, `nms`, and focused tests.
- This separation makes proposal generation usable as a standalone stage without
  routing through the full detection pipeline and clarifies the ownership
  boundary between pass-1 center finding and later fit/decode stages.

**Proposal-stage tuning and throughput**
- Optimized the proposal voting hot path and added benchmark coverage for the
  dominant proposal-stage workload.
- Added `ProposalDownscale` to `DetectConfig` so pass-1 proposals can run on a
  downscaled image while all downstream fitting and decoding stays at full
  resolution.
- Surfaced proposal downscale and related proposal diagnostics controls through
  the CLI/config plumbing, making proposal-cost tuning explicit instead of
  implicit.

**Documentation and interface framing**
- Reworked the root `README.md` into a user-oriented entry point that routes
  readers to the CLI guide, Rust crate docs, Python package docs, output-format
  reference, and performance notes.
- Expanded the Rust and Python READMEs so target generation, proposal-only
  diagnostics, and output-contract references are documented consistently across
  interfaces.

### Documentation

- Added a dedicated mdBook page for the detection output schema, covering
  `DetectionResult`, top-level frame metadata, optional homography/RANSAC,
  CLI-only `camera`, and CLI-only proposal diagnostics fields.
- Added a dedicated mdBook page for proposal diagnostics, including standalone
  Rust/Python proposal APIs, the accumulator heatmap contract, and the plotting
  workflow.
- Extended the CLI guide, fast-start docs, and target-generation docs to cover
  `ringgrid gen-target`, printable artifact generation parity, and the
  `--include-proposals` detection-output option.
- Added `docs/proposal-performance-analysis.md` to capture the current proposal
  hotspot profile, the rationale for the existing algorithm, and the most
  credible future alternatives.

### CI

- Modernized GitHub Actions workflows to current stable major versions for core
  checkout, Python setup, artifact upload/download, and GitHub release steps.
- Replaced hand-rolled tool installation in CI with maintained installer
  actions, added tighter default permissions and job timeouts, and added
  Dependabot updates for GitHub Actions dependencies.

## [0.5.0] — 2026-03-12

### Added

**Rust target generation**
- `BoardLayout` now exposes file-oriented target generation for canonical `ringgrid.target.v4`
  JSON plus printable SVG/PNG output from the Rust crate.
- Added deterministic JSON/SVG/PNG fixtures and integration coverage for the new generator
  surface.

**Python target generation**
- `ringgrid-py` now exposes installed-package target-generation helpers on `BoardLayout`:
  `to_spec_json()`, `write_svg()`, and `write_png()`.
- Added dedicated repo CLI `tools/gen_target.py` for print-target generation without going
  through the synth pipeline.

**Codebook profiles**
- Added opt-in `extended` codebook profile while preserving shipped `base` IDs `0..892`.
- Surfaced profile selection through CLI helpers, Rust/Python config, and regenerated
  codebook artifacts/docs.

**CLI calibration loading**
- `ringgrid detect --calibration <file.json>` now loads a Brown-Conrady `CameraModel`
  from JSON as an additive alternative to inline `--cam-*` flags.
- The CLI accepts either the direct serde camera-model shape or detector-output JSON with
  a top-level `camera` wrapper.

**Rust CLI target generation**
- Added `ringgrid gen-target` to generate canonical `board_spec.json` plus
  printable SVG/PNG directly from the Rust CLI.
- The Rust CLI generation command matches the dedicated `tools/gen_target.py`
  geometry/print options and writes the same artifact set.

### Changed

**Dependency upgrades**
- `rand` upgraded to `0.10`.
- `criterion` upgraded to `0.8`.
- `nalgebra` upgraded to `0.34`.

**`rand` API migration**
- Replaced deprecated/removed `Rng::gen_range(...)` calls with
  `RngExt::random_range(...)` across library code, tests, and benchmarks.
- Kept deterministic behavior intact by preserving all explicit `StdRng::seed_from_u64(...)`
  seeds in test and benchmark fixtures.

**`criterion` API migration**
- Replaced deprecated `criterion::black_box(...)` with `std::hint::black_box(...)`
  in benchmark code.
- Bench harness remains compatible with `criterion_group!` / `criterion_main!`.

**Tooling compatibility**
- Updated benchmark/test support code to compile cleanly with `-D warnings`
  under the new dependency set.

**Python config performance**
- `ringgrid-py` `DetectConfig` now caches resolved JSON snapshots for hot getter/setter paths
  while preserving `to_dict()` output and overlay semantics.

**Completion and fit robustness**
- Completion now seeds missing IDs from a local board-mm affine fit when 3-4 nearby decoded
  neighbors are available, falling back to the global homography otherwise.
- Outer-edge sampling now drops rays whose recovered radius differs from the expected outer
  radius by more than 40%, screening inner-ring contamination before ellipse fitting.
- Final marker cleanup now clears IDs whose inner/outer mean-axis ratio deviates strongly from
  the global fit-decoded median, catching residual inner-as-outer failures missed earlier in
  the pipeline.

**Documentation and onboarding**
- Refactored the root `README.md` into a user-first landing page with quickstart and interface
  routing.
- Split maintainer-focused material into `docs/development.md` and scoring/benchmark guidance
  into `docs/performance.md`.
- Expanded `crates/ringgrid-py/README.md` with a full Python-facing `DetectConfig` field guide.
- Reconciled codebook docs and decision records with shipped baseline invariants and the new
  `extended` profile contract.

**Target-generation documentation**
- Reworked README and mdBook target-generation guidance so Rust API, Rust CLI,
  and Python script workflows are documented as equivalent paths over the same
  target-generation engine.
- Canonical target JSON moved to `ringgrid.target.v4` with explicit
  `marker_ring_width_mm`; the runtime loader no longer accepts `v3`.

### Fixed

- `hotpaths` benchmark build now passes with latest `rand`/`criterion` APIs.
- `tools/gen_codebook.py --base_json ...` now preserves source-seed provenance in emitted JSON
  and Rust metadata.
- Legacy Python decode-config payloads missing `codebook_profile` now default to `"base"`.
- The `extended` codebook appendix excludes new complement-collision exact matches under
  inverted polarity.

## [0.4.0] — 2026-03-01

### Added

**Detection provenance**
- `DetectionSource` enum (`FitDecoded | Completion | SeededPass`) on `DetectedMarker`
  — tags the pipeline stage that produced each marker; serialised as `"fit_decoded"`,
  `"completion"`, `"seeded_pass"` in JSON output.

**Fit diagnostics in `FitMetrics`**
- `h_reproj_err_px: Option<f32>` — reprojection error (pixels) between the marker center
  and its board position projected through the final homography; populated for all markers
  with a valid decoded ID after the final H refit.
- `radii_std_outer_px: Option<f32>` — standard deviation of per-ray outer radii; high
  values (> 30 % of the mean) indicate inner/outer edge contamination.

**Completion quality gates**
- `CompletionParams::max_radii_std_ratio` (default `0.35`) — rejects completion candidates
  whose outer-radii coefficient of variation exceeds the threshold; prevents contaminated
  fits where some rays sample the inner ring.
- `CompletionStats::n_decode_mismatch` — counts how many completion attempts produced a
  valid fit that decoded to a different ID than the expected board position.

**Confidence refinement**
- `DetectConfig::h_reproj_confidence_alpha` (default `0.2`) — after annotating
  `h_reproj_err_px`, applies a soft penalty: `confidence *= 1 / (1 + alpha × err)`.
  A 5 px reprojection error reduces confidence by ≈ 50 %. Set to `0.0` to disable.

**Documentation**
- `docs/` — adaptive-scale detection guide and fast-start usage guide.

**Python API**
- Adaptive tier introspection: `ScaleTiers.tiers` property exposes the configured
  `ScaleTier` list from Python.
- Simplified `detect_adaptive` call: keyword-argument defaults mirror the Rust API
  without requiring explicit tier construction for common cases.

---

## [0.3.0] — 2026-03-01

### Added

**Adaptive multi-scale detection**
- `Detector::detect_adaptive(image, config)` — automatically estimates dominant marker
  radius via a ring angular-variance sweep (`scale_probe`), selects the appropriate
  scale tier, and runs the full pipeline.
- `Detector::detect_adaptive_with_hint(image, config, hint)` — same as above with a
  user-supplied radius hint bypassing the probe.
- `Detector::detect_multiscale(image, tiers, config)` — runs one fit-decode pass per
  `ScaleTier` and merges results with size-consistency-aware NMS.
- `ScaleTier` — single-scale configuration (min/max diameter in px).
- `ScaleTiers` — ordered collection of tiers; built-in presets:
  - `four_tier_wide` — `[8–24, 20–60, 50–130, 110–220]` px (widest coverage)
  - `two_tier_standard` — `[14–42, 36–100]` px
  - `single` — single tier matching `MarkerScalePrior`
  - `from_detected_radii` — derives tiers from a pass-1 radius histogram

**Scale probe**
- `pipeline/scale_probe.rs` — ring angular-variance sweep over 20 geometric radius
  candidates (4–110 px) at top-K gradient proposals; code-band midpoint at ≈ 0.8×
  outer radius; results feed `ScaleTiers::from_detected_radii`.

**Multi-scale pipeline split**
- `finalize_premerge` — projective-center correction per tier; returns
  `Vec<DetectedMarker>` without global filter or completion.
- `merge_multiscale_markers` — size-consistency-aware NMS across all tier outputs;
  prefers markers whose outer radius matches the k = 6 hex-lattice neighbor median;
  confidence breaks ties.
- `finalize_postmerge` — global filter + H-guided completion + final H refit, run
  exactly once on the merged pool.

**Python**
- Python 3.14 support.

### Changed

- `MarkerScalePrior` default updated from `[20, 56]` to `[14, 66]` px,
  yielding +11.4 % on the rtv3d benchmark dataset.

---

## [0.2.5] — 2026-02-28

### Fixed

- CI: pinned manylinux Python interpreter for maturin wheel builds.
- CI: removed unsupported `replace()` call from PyPI release workflow.

---

## [0.2.2] — 2026-02-28

### Changed

- Python docstrings polished with structured parameter and return documentation.
- Example image paths updated to reference `testdata/` directory.

---

## [0.2.1] — 2026-02-22

### Fixed

- CI: maturin `develop` command now runs inside a virtual environment.
- CI: miscellaneous Python release pipeline fixes.

---

## [0.2.0] — 2026-02-22

### Added

- **Python bindings** via PyO3 and maturin; published to PyPI as `ringgrid`.
- `Detector`, `DetectConfig`, `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, and
  `DetectionResult` exposed to Python with full docstrings.

---

## [0.1.0] — 2026-02-22

Initial release. Full detection pipeline for dense coded ring calibration targets on a
hex lattice.

### Detection pipeline

- **Proposal** — Scharr gradient voting + NMS for candidate center extraction.
- **Outer estimate** — radial intensity profile peak-finding for radius hypotheses.
- **Outer fit** — Fitzgibbon direct ellipse fitting + RANSAC (Sampson distance inlier
  metric).
- **Decode** — 16-sector ring code sampling → match against 893-codeword codebook with
  cyclic rotation search and Hamming margin.
- **Inner fit** — inner ring ellipse from radial edge samples; configurable quality gates.
- **Dedup** — spatial NMS + per-ID deduplication.
- **Projective center** — unbiased center recovery from inner + outer conic pair.
- **Structural ID correction** — BFS hex-neighborhood consensus with local 2-D affine
  fitting; corrects swapped IDs and clears unverifiable detections.
- **Global filter** — RANSAC homography (DLT + Hartley normalization) from decoded
  markers; outlier rejection gate in pixels.
- **H-guided completion** — conservative fits at board positions projected by H but not
  yet detected; optional perfect-decode gate for high-distortion setups.
- **Final H refit** — homography updated from the full corrected marker pool.

### Result types

- `DetectedMarker` — center (image px), optional inner/outer ellipses, optional edge
  point arrays, fit metrics, decode metrics, board-space coordinates (`board_xy_mm`).
- `FitMetrics` — RANSAC inlier ratios, RMS Sampson residuals, angular gap, inner fit
  status and rejection reason, neighbor radius ratio, inner theta consistency.
- `DecodeMetrics` — observed word, best ID, best rotation, Hamming distance, margin,
  decode confidence.
- `DetectionResult` — detected markers, homography, RANSAC stats, image size.

### Configuration

- `DetectConfig` — unified config struct; all sub-configs (proposal, edge sampling,
  decode, inner/outer fit, projective center, completion, homography RANSAC,
  self-undistort, ID correction) accessible as named fields.
- `MarkerScalePrior` — min/max diameter range in working-frame pixels.
- `BoardLayout` — hex-lattice geometry loader from JSON.
- `MarkerSpec` — inner/outer radius ratio and sector geometry.

### Camera / distortion support

- `CameraModel` + `CameraIntrinsics` — optional radial-tangential distortion model.
- Two-pass detection: pass-1 without mapper; pass-2 with mapper using pass-1 seeds.
- `SelfUndistortConfig` — estimates a single-parameter division-model distortion from
  detected ellipse edge points; re-runs detection with the estimated mapper.

### Infrastructure

- Pure Rust — no OpenCV bindings.
- `tracing`-based structured logging (`RUST_LOG=debug|info|trace`).
- Typed reject-reason enums throughout the pipeline (no stringly-typed errors).
- Baseline-locked maintainability guardrails enforced in CI.
- CLI binary (`ringgrid detect`) with `--image`, `--out`, `--marker-diameter`,
  `--circle-refine-method`, and camera calibration flags.
