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

- `ALGO-021` · P2 · todo — **Print radii consensus is biased low when the
  blur is comparable to the ring width.** (a) Thin ring: 4.48–4.55 vs
  4.60 mm; plain hex at 8 px outer radius: 2.27–2.33 vs 2.40 mm. At low
  contrast the consensus fails outright (6–12 of 24 exposures). Both shift
  `DetectedMarker::blur` σ by −2…−6 % and add a per-image σ offset that its
  covariance does not contain. Reproduce on `photometric_target_matrix`,
  split the bias into per-marker fit vs aggregation, and fix what the data
  identify. (b) Decided for now: radii stay per-image; pooled or known radii
  are a `study` input only (`PhotometricStudyOptions::with_fixed_radii`),
  because they gave no centre gain
  ([center metrology](studies/center-metrology.md#measurements-and-decision)).
  Reopen (b) when a calibration consumer needs σ comparable across images.

- `ALGO-022` · P3 · todo — **Exposure dependence of the fitted radii and
  centres** (split from ALGO-020; Cledat, Rufener & Cucci, Pattern
  Recognition 2021: saturation with the PSF moves the edges of
  concentric-circle targets with exposure). Expected: the stroke width
  follows exposure and the per-image radii absorb it, while the centres stay
  put. Synthetic half: one `blur_scene` radiance exposed at several gains,
  with clipping and a non-unit response. Real half: needs an exposure series.

- `ALGO-023` · P2 · blocked (tilted rtv3d captures with fractional-centre
  `gen-screen` targets) — **Capture-wide focus-band blur-gradient
  correction.** A per-camera focus-band field predicts σ and ∇σ from the
  rest of the capture and corrects the centre by `1.24·σ·∇σ`. On synthetic
  two-sided bands it recovers 68 % of the oracle RMS gain against 36 % for
  the neighbour plane; on `rtv3d_0` it predicts held-out σ with
  0.33 px RMSE against 0.53 px for neighbours
  ([focus-field study](studies/camera-focus-field.md)). `rtv3d_0` cannot
  decide adoption: its display snaps marker centres to pixels, which floors
  any geometric proxy. Acceptance: on the tilted captures, the field
  correction lowers the held-out planar projection error
  (`metrology_real.py`'s geometry proxy) relative to both the
  neighbour-plane correction and no correction, with whole-capture bootstrap
  intervals excluding zero, and no wrong IDs. Commands, once the images are
  in `data/<dir>` with their target spec:
  `.venv/bin/python tools/run_rtv3d_eval.py --data-dir data/<dir> --target <spec> --strategies adaptive_photometric`,
  then `.venv/bin/python tools/camera_focus_field.py --real tools/out/rtv3d_eval/detections`,
  then `.venv/bin/python tools/metrology_real.py --datasets rtv3d --rtv3d data/<dir> --rtv3d-target <spec>`.
  `camera_focus_field.py` assumes the `rtv3d_0` rig (six 720×540 tiles per
  strip); adjust it if the tilted rig differs.

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
| ALGO-020 | 2026-11-24 | algo/study | Moment-centre study estimator — rejected as an estimator | The projectively corrected ink centroid shows the same gradient-reversal bias as the photometric fit (−0.029 vs −0.028 px over 717 paired observations), drifts more under added blur on the sharp IPGS plate (0.022 vs 0.017 px RMS) and declines clipped or spilled patches. Kept as a `study` comparator (`PhotometricStudyOptions::moment_centers`); see [center metrology](studies/center-metrology.md). The exposure half moved to ALGO-022. |
| ALGO-017 | 2026-11-22 | algo/api | Blur as a stable photometric output | `DetectedMarker::blur`: σ (Gaussian edge width incl. pixel integration), anisotropy and their 3×3 covariance, honest for differences between markers of one image (per-image print-radii offset not included); reported for photometric centres with σ ≤ 0.25 × outer radius; replaces `PhotometricMetrics::blur_sigma_px`. |
| ALGO-018 | 2026-11-22 | algo | Fixed-gradient second photometric pass — rejected | Measured against subtracting the law `1.24·σ·∇σ` from the one-σ centre (no refit) on `blur_gradient_bias_matrix` (133 scatter-form cells, 4 seeds): residual bias 0.46 vs 0.50 of the one-σ fit's, neither removes 80 %; the refit costs 2.1–2.3× photometric time per refitted marker and up to 2.6× centre scatter in the worst cell; with the true gradient the refit leaves 0.16 and the law 0.29, so the neighbour-plane gradient estimate, not the model, is the limit; a calibration absorbs 92 % of the bias anyway. Code not kept. |
| ALGO-019 | 2026-11-22 | algo | A blur prior per marker — rejected | Closed with ALGO-018: a calibrated-focus gradient would lift the refit to ~84 % removal, at twice the fit cost, for a bias a calibration already absorbs. A caller who needs it can subtract the law `1.24·σ·∇σ` using its own focus model's ∇σ and the reported `DetectedMarker::blur.sigma_px`. |
