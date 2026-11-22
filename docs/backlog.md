# Backlog

## Status Values
- `todo` — not started
- `in-progress` — actively being worked
- `blocked` — waiting on something
- `done` — completed

## Priority Values
- `P0` — blocking release or correctness
- `P1` — next up
- `P2` — planned
- `P3` — someday

## ID Model
- Backlog ids (`INFRA-011`, `ALGO-014`, `DOCS-003`) are the stable planning ids used in this file.

---

## Active Sprint

_None currently assigned._

## Up Next

_None currently assigned._

## Backlog

Context for the blur items below: calibration-rs `docs/notes/scheimpflug-literature.md`
(literature review 2026-11-21) and ADR-022 / ADR-029 here. On a tilted-focus
camera the per-marker blur is also a calibration observation (the plane of
sharp focus, the tilt direction, the depth of field), so its value and its
uncertainty are wanted next to the centre.

- `ALGO-020` · P3 · todo — **Moment-centre study estimator.** The ink
  centroid with the analytic projective correction (Song et al., CVPR 2024)
  is immune to a blur gradient under scatter blur
  (`blur_gradient_ink_centroid`). As a `study`-feature estimator it
  separates the gradient bias from other errors on real captures.
  Caveats: neighbour spill, background shading, radial distortion only.
  In the same study, the centre half of the exposure-bias check (Cledat,
  Rufener & Cucci, Pattern Recognition 2021: saturation with the PSF moves
  the edges of concentric-circle targets with exposure; the σ half belongs
  to ALGO-017). Here it should mostly change the stroke width, which the
  per-image printed-radii fit absorbs, and a symmetric width change does
  not move the centre. Test: whether the fitted radii and centres follow
  exposure or contrast across images, on the `blur_scene` renderer's
  radiance/exposure split and on a real exposure series.

- `ALGO-021` · P2 · todo — **Print radii are a property of the target, not
  of an image.** (a) The per-image consensus radius is biased low when the
  blur is comparable to the ring width (thin ring: 4.48–4.55 vs 4.60 mm;
  plain hex at 8 px outer radius: 2.27–2.33 vs 2.40 mm) and fails outright
  at low contrast (consensus in 6–12 of 24 exposures), which shifts
  `DetectedMarker::blur` σ by −2…−6 % and adds the per-image σ offset its
  covariance does not contain. (b) Fixing the radii across images (known or
  pooled) makes var(σ) honest across images (reported over scatter
  0.96–1.15). Decide where known or pooled print radii enter (target spec
  vs a detector input; coded markers have three); fix the consensus bias
  first.

## API / Interface Tracking

- Rust API backlog direction: add file-oriented JSON/SVG/PNG target-generation API in `ringgrid` crate using direct board geometry args.
- Python API backlog direction: expose the same target-generation capability in `ringgrid-py` package surface (installed-package usable).
- `DetectConfig` backlog direction: internal Python caching/refactor only; no intentional public behavior changes.
- Codebook backlog direction: default profile unchanged; optional extension profile additive and opt-in.
- Proposal module direction: standalone `proposal/` module with no ringgrid-type dependencies in core API; `ProposalConfig` with unified `min_distance`; `find_ellipse_centers()` entry points; `ProposalResult` with `heatmap` field.

## Acceptance Scenarios (Attached to Tasks)

## Locked Defaults

- ID convention: keep existing `ALGO`/`INFRA` streams and add `DOCS-*` for docs-focused backlog work.
- Target-generation outputs in first milestone: JSON + SVG + PNG.
- API style for target generation: file-oriented for both Rust and Python.
- Codebook extension policy: stable base + optional extension, 16-bit only, target extension size = max feasible.
- Root README policy: user-quickstart-first with links to separate developer/performance docs.

## Historical Notes

## Done

| ID | Date | Type | Title | Notes |
|----|------|------|-------|-------|
| ALGO-017 | 2026-11-22 | algo/api | Blur as a stable photometric output | `DetectedMarker::blur`: σ (Gaussian edge width incl. pixel integration), anisotropy and their 3×3 covariance, honest for differences between markers of one image (per-image print-radii offset not included); reported for photometric centres with σ ≤ 0.25 × outer radius; replaces `PhotometricMetrics::blur_sigma_px`. |
| ALGO-018 | 2026-11-22 | algo | Fixed-gradient second photometric pass — rejected | Measured against subtracting the law `1.24·σ·∇σ` from the one-σ centre (no refit) on `blur_gradient_bias_matrix` (133 scatter-form cells, 4 seeds): residual bias 0.46 vs 0.50 of the one-σ fit's, neither removes 80 %; the refit costs 2.1–2.3× photometric time per refitted marker and up to 2.6× centre scatter in the worst cell; with the true gradient the refit leaves 0.16 and the law 0.29, so the neighbour-plane gradient estimate, not the model, is the limit; a calibration absorbs 92 % of the bias anyway. Code not kept. |
| ALGO-019 | 2026-11-22 | algo | A blur prior per marker — rejected | Closed with ALGO-018: a calibrated-focus gradient would lift the refit to ~84 % removal, at twice the fit cost, for a bias a calibration already absorbs. A caller who needs it can subtract the law `1.24·σ·∇σ` using its own focus model's ∇σ and the reported `DetectedMarker::blur.sigma_px`. |
