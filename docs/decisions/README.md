# Architecture decision records

## The rule

An ADR records one decision in about a page: the context, the decision, its
consequences, and the alternatives rejected, each with a one-line reason.
Measurements, tables and study narrative go in `docs/studies/`, and the ADR
links to them. When the decision changes, amend the ADR in place so that it
states the decision as it stands; when another ADR replaces it, mark it
superseded and link the successor. Start from [`000-template.md`](000-template.md).

## Index

Status as checked against the code and the later ADRs on 2026-11-22.

| ADR | Title | Status | Current decision |
|---|---|---|---|
| [001](001-pure-rust-no-opencv.md) | Pure Rust, no OpenCV | accepted | All image processing and geometry are pure Rust; no OpenCV or other C/C++ vision bindings anywhere in the workspace. |
| [002](002-coordinate-frames.md) | Coordinate frames | accepted | `DetectedMarker.center` is always in image pixels; with a mapper, ellipses, edge points and the homography are in the working frame and `center_mapped` holds the working-frame centre. |
| [003](003-ellipse-canonicalization.md) | Ellipse canonicalization | accepted | `Ellipse` keeps `a ≥ b > 0`, `angle ∈ (−π/2, π/2]` and finite fields; ellipse ↔ conic round-trips losslessly. |
| [004](004-pipeline-stage-ordering.md) | Pipeline stage ordering | partially superseded by ADR-023 | Stages 1–6 (proposal, outer estimate, outer fit, decode, inner fit, dedup) keep their order; finalize is grid-first (ADR-023) and photometric refinement runs last (ADR-022). |
| [005](005-public-api-surface.md) | Public API surface | superseded by ADR-016 | — |
| [006](006-codebook-invariants.md) | Codebook invariants | partially superseded by ADR-027 | Generated codebook: base 893 words (minimum cyclic distance 2), extended 2180 with the base as prefix (distance 1), no rotational symmetry, rotation-unique. Detection decodes per ADR-027. |
| [007](007-config-single-source-of-truth.md) | Config single source of truth | accepted | One authoritative definition per config value; scale- and geometry-dependent parameters are derived from `MarkerScalePrior` and the target, also by `Default`. |
| [008](008-homography-convention.md) | Homography convention | accepted | `H` maps board mm to working-frame (or image) pixels, normalized DLT + RANSAC; with ≥ 10 markers the final refit is kept only if it lowers the mean reprojection error. |
| [009](009-marker-spec-normalized-units.md) | MarkerSpec normalized units | accepted | Radial geometry is in outer-normalized units; pixel scale enters only through `MarkerScalePrior`. |
| [010](010-projective-center-correction.md) | Projective centre once per marker | accepted | The two-conic projective centre is applied exactly once per marker: fit-decode markers before grid labeling, completion markers right after completion. |
| [011](011-workspace-boundaries.md) | Workspace boundaries | accepted | Algorithms live in `ringgrid`, which never depends on a CLI crate; see the drift note below. |
| [012](012-error-handling.md) | Error handling | accepted | `Option`/`Result` inside the algorithms, failed candidates skipped, no panics or printing in library code, `tracing` for logs. |
| [013](013-deterministic-reproducibility.md) | Deterministic reproducibility | accepted | Seeded RNGs only; the same image and config give the same output. |
| [014](014-ellipse-fitting-method.md) | Ellipse fitting | accepted | Fitzgibbon direct least squares inside RANSAC (Sampson distance); direct fit as the fallback and below 8 points, rejection below 6. |
| [015](015-two-pass-detection.md) | Two-pass detection | accepted | With a mapper (given, or estimated by self-undistort) a second, mapper-aware pass runs, seeded by the first. |
| [016](016-tiered-public-surface.md) | Tiered public surface | accepted | Three tiers: the stable root, `ringgrid::diagnostics`, `ringgrid::codebook`. |
| [017](017-plain-rect-target-detection.md) | Plain / rect targets, compositional model | partially superseded by ADR-021, ADR-023, ADR-025 | `TargetLayout` = lattice × ring × coding × fiducials; an origin-anchored plain view reports `BoardFrame::Absolute`, an unanchored one `RelativeCanonical` without board mm. |
| [018](018-c-cpp-vcpkg-api.md) | C / C++ / vcpkg API | accepted | Flat C ABI in `ringgrid-c` (JSON at the boundary, status codes), cbindgen header, header-only C++ wrapper, CMake package, vcpkg port. |
| [019](019-inner-as-outer-recovery-gating.md) | Inner-as-outer recovery gating | accepted | Rings whose outer fit took the inner edge are re-fitted with gates relaxed only on the recovery path, plus a post-fit size gate. |
| [020](020-agent-workflow-adopted.md) | Agent workflow | accepted (in doubt) | Algorithm, pipeline and performance roles, each running its own validation gates before handoff. |
| [021](021-derived-origin-fiducials.md) | Derived origin fiducials | accepted | Rect lattice coordinates are centred; origin-dot positions are derived from the lattice (an L of three gaps at cell `(0, 0)`); only the dot radius is stored. |
| [022](022-photometric-center-refinement.md) | Photometric centre refinement | accepted | Opt-in `Photometric` refinement fits a model of the blurred printed pattern to every marker with a board position, with a covariance that holds under correlated noise. |
| [023](023-grid-first-finalize.md) | Grid-first finalize | accepted; amended by ADR-024, ADR-026, ADR-027 | One back half for every target: label every fitted ring, anchor, identify by cell, join and complete, final H, verify; ID correction only in the fallback. |
| [024](024-fail-closed-finalize.md) | Fail-closed finalize | accepted; amended by ADR-026, ADR-027, ADR-028 | A wrong output is worse than a missing one: no homography means no IDs, an anchor must beat chance, only perspective-corrected centres are reported, `min_confidence` 0.03, one board per image. |
| [025](025-plain-anchor-by-anchor-cell.md) | Plain anchor by anchor cell | accepted | Origin placements are enumerated by the relative cell that carries the triad (no cap), scored with one homography, and re-anchored after completion. |
| [026](026-large-plain-ring-recall.md) | Large plain ring recall | partially superseded by ADR-029 | Labels checked against the ring shapes (fail closed), ellipse-exact outer fit, adaptive scoring by board position, no code-band probe on plain targets, completion of uncorrected rings, local support for coded refits. Decision 3 is replaced by ADR-029. |
| [027](027-phase-aligned-exact-decode.md) | Phase-aligned exact decode | accepted | Phase-aligned sampling, dark-on-light only, exact match against the target's own codewords, anchor votes checked against the predicted code rotation. |
| [028](028-plain-recall-tail.md) | Verification on refined centres | partially superseded by ADR-029 | Local verification on the deviation field, unverifiable markers removed, photometric configurations verified on refined centres, adaptive candidates joined. Decision 4 is replaced by ADR-029. |
| [029](029-photometric-confidence-error-model.md) | Photometric confidence | accepted | A photometric centre's confidence uses its precision `1/(1 + σ_eff / 0.03 px)`, `σ_eff` from the centre covariance and the blur-gradient bias; no sharpness factor. |

### Drift

Details these ADRs state that no longer match the code, without a later ADR
that records the change:

- **009** names `BoardLayout`, removed in 0.9; the priors come from the
  `TargetLayout`.
- **010** places the correction "before the global filter" with `finalize.rs`
  line numbers; it now runs before grid labeling (`pipeline/finalize/`), and
  the config type is `ProjectiveCenterConfig`.
- **011** says the library has no file I/O and no CLI; `ringgrid` has file
  I/O behind the `std` feature and the published CLI behind `cli`, while
  `ringgrid-cli` holds the unpublished `ringgrid-dev`.
- **012** says `Detector::detect` never returns `Result`; it returns
  `Result<DetectionResult, DetectError>` since 0.8.
- **015** names `SeedProposalParams` and `merge_radius_px`; the type is
  `SeedProposalConfig` and the field was removed in 0.16.
- **016** keeps the deprecated v4 `BoardLayout` facade; it was removed in 0.9.
- **017** and **021** describe loading v4 / v5 target files; since 0.16 only
  `ringgrid.target.v6` loads. `finalize/plain.rs` no longer exists.
- **019** adds a recovery-only `min_ring_depth`; it was removed in 0.17, and
  recovery uses the detector's contrast-relative depth gate.
- **020** creates the `.ai/` role files; they were removed in 2026-03, though
  the role names are still used in ADR headers.

### Length

Still longer than the rule, to be converted when next touched: 023, 024, 026,
027 and 028 carry measurement tables and experiment logs inline; 016 and 025
run past a page. ADR-022 and ADR-029 follow the rule; their evidence is in
[`docs/studies/photometric-center.md`](../studies/photometric-center.md).
