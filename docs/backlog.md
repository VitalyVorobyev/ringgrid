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

- `ALGO-018` · P2 · todo — **Fixed-gradient second photometric pass.**
  ADR-029 already computes each marker's neighbour σ-plane
  (`attach_blur_gradient_bias`). Refit only the markers whose predicted bias
  `b = 1.24·σ·|∇σ|` exceeds a threshold, with that gradient *fixed* in the
  PSF model (the scatter form of the archived gradient term,
  `archive/blur-gradient-term`), so the bias is removed without the two free
  parameters and their 1.4–1.8× scatter cost on plain rect markers. Opt-in.
  Gate: `blur_gradient_bias_matrix` bias down by ≥ 80 % with centre scatter
  within 10 % of the one-σ fit, **and** a residual bias at most half of
  what the free alternative leaves: subtracting the first-order law
  `1.24·σ·∇σ` from the one-σ centre, with no refit, which a caller can do
  from the reported σ alone once ALGO-017 lands. If the refit does not beat
  that, it is a dominated alternative: close this item and ALGO-019, and
  document the caller-side correction instead. Note: calibration-rs measured where this
  bias goes on six tilted-focus cameras (its `docs/notes/ringgrid-bias.md`):
  92 % is absorbed by the camera and the poses, the residual moves by
  0.003 px, and `fy/fx` moves by +3.1 … +4.6e-4 (0.8–1.9 σ). Its value is
  for aspect-sensitive work and blur observations, not the residual. Keep
  the PSF centred by construction: a PSF whose centroid is free trades
  against the centre (a global translational ambiguity, Shi, "Generic
  Camera Calibration using Blurry Images", arXiv 2603.05159, which anchors
  it to a camera calibrated from sharp images). The same holds for any
  later non-Gaussian PSF model.
- `ALGO-019` · P3 · todo — **A blur prior per marker.** Accept an optional
  predicted σ (and ∇σ) per board position, from a calibrated focus model,
  as a prior in the photometric fit: the detect → calibrate → predict →
  refit loop, and stability for small, strongly blurred rings. Widens the
  public config; only after ALGO-018 shows the fixed-gradient fit pays.
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
