# ADR-029: Photometric Confidence from the Centre's Predicted Error

- **Status:** accepted
- **Date:** 2026-11-19
- **Author role:** algorithm-engineer
- **Amends:** ADR-028 (decision 4, photometric confidence)

## Context

Users cut imprecise centres with a confidence threshold (the review tool's
slider). For photometric centres (`CircleRefinementMethod::Photometric`) on
the rtv3d captures — a tilted-focus camera, sharp middle rows, defocused top
and bottom — the cut looked wrong: measured with an 8-neighbour homography
leave-one-out (LOO) residual, the top band had the worst centres (median
0.38 px) and the highest confidence (median 0.54), within the top and middle
bands higher confidence went with larger residuals (Spearman ρ +0.23 /
+0.35), and the residual was 7–16× the fit's own σ.

ADR-028 had set an accepted photometric centre's confidence to evidence ×
outer fit × border × `1/(1 + (σ / 0.1 px)²)` (`σ = √(tr Σ / 2)`, `Σ` its
centre covariance) × ring-relative sharpness `1/(1 + (w / max(3 px,
0.1875 r))²)`.

### What was measured

**1. Most of that LOO residual is target and lens geometry, not centre
error** (rtv3d, 102 tiles, aggregates only).

- Per marker ID the LOO offset (in board mm) repeats across views: split-half
  correlation 0.94 / 0.88 (x / y) between disjoint halves of the strips, and
  0.86 / 0.78 between photometric and projective centres of the same rings.
  94 % / 92 % of the x / y offset is explained by the marker's board column /
  row alone — the e-reader draws each column and row of markers off its
  nominal position (display-pixel snapping). No code-band dependence: the
  codeword's dark-mass dipole explains 0.1 % of it.
- The remaining LOO residual is mostly smooth curvature an 8-neighbour
  homography does not absorb: on the same interior markers a local quadratic
  warp over 18 neighbours leaves 0.04 px (photometric) where the homography
  leaves 0.15–0.17 px.
- Both effects are fixed in board millimetres, so in pixels they grow with
  the marker's size, and the top band holds the largest rings (median outer
  radius 29 px against 19): the "worst centres with the highest confidence"
  were the largest rings.

With a geometry-robust proxy — a cubic warp (10 terms per coordinate) over
the 20 nearest labelled neighbours, the marker left out, its prediction noise
propagated from the neighbours' reported covariances — the photometric
covariance is 1.2–1.3× optimistic in the middle and bottom bands (0.9–1.5×
for fitted blur ≥ 1 px; not 4×), 2.3× for the sharpest markers (fitted blur
< 1 px: an error floor of ~0.03 px the fit does not model), and the
confidence ranks errors in the middle and bottom bands (cubic-proxy
residual > 0.1 px: ROC AUC 0.79 / 0.83). The top band has 26–59 markers with
a usable proxy and no predictor — covariance, blur, ring size, confidence —
ranks their residuals.

**2. The confidence did not use the photometric precision**
(`tests/blur_gradient.rs`, `blur_gradient_confidence`: 864 synthetic scenes,
three target kinds × outer radius 10 / 17 / 28 px × uniform blur 0.8–3.5 px
or a focus gradient from 0.6 px growing 0.004 / 0.008 px per row × tilt ×
contrast × 3 seeds; 45,543 accepted photometric centres against exact
truth).

- The covariance is honest under uniform blur (RMS error / σ = 1.01–1.08;
  1.30 at true blur ≥ 3 px on the held-out split) but the centre errors are
  small: median 0.013 px, 90th
  percentile 0.027 px. Against the 0.1 px scale the precision factor is
  0.93–0.99 for nearly every centre: its spread (s.d. of its log) is 0.037,
  against 0.195 for the sharpness factor and 0.172 for evidence × outer fit
  × border. The confidence ranked the edge sampler and the ring size, not
  the photometric centre.
- The covariance alone ranks the errors better than the confidence did
  (ρ −0.59 vs −0.38; AUC for errors beyond 0.05 px 0.88 vs 0.80).
- Under a focus gradient the error grows faster with blur than σ: RMS error
  / σ = 1.08–1.18 below 3 px of true blur, 1.47 (held-out) – 2.08 (training)
  above. That is the blur-gradient bias
  (`blur_gradient_bias_matrix`): a fit with one PSF width lands about
  `1.24 g σ` towards the sharper side (blur σ, gradient `g` px per px),
  common to the whole marker, invisible to the covariance.

## Decision

An accepted photometric centre's centre support is its **precision**
`1/(1 + σ_eff / 0.03 px)`, from its **predicted error**
`σ_eff = √((tr Σ + b²) / 2)` (per axis):

- `Σ` is the fit's centre covariance (noise, residual correlation), as
  reported in `DetectedMarker::center_cov`;
- `b = 1.24 · σ_blur · |∇σ_blur|` is the blur-gradient bias, `σ_blur` the
  fitted PSF width and `∇σ_blur` the image gradient of the least-squares plane
  through the fitted blur of the marker and its 8 nearest accepted
  neighbours (`photometric_refine::attach_blur_gradient_bias`; no bias when
  fewer than 4 accepted fits or a degenerate neighbourhood). The coefficient
  is the bias law's, not refitted.

Confidence = evidence × outer fit × precision × border. Edge sharpness is no
longer a factor of a photometric centre's confidence: the fit models the PSF
width, and the blur-dependent error is now in `σ_eff`. Edge-fitted centres
are unchanged (bit-identical). `center_cov` keeps its meaning; the bias is
internal (`MarkerRecord::center_bias_px`).

**The fall-off and scale.** `1/(1 + σ/s)` with `s = 0.03 px` spans the
errors photometric centres have (0.01 px → 0.75, 0.03 → 0.5, 0.1 → 0.23) and
reaches the default `min_confidence` (0.03) at a 1 px predicted error — where
the edge-fitted confidence drops centres too (ADR-024). Chosen on the
training split (radius 10 / 28 px, coded hex and plain rect, seeds 0–1)
among `1/(1 + (σ/s)²)` and `1/(1 + σ/s)` at `s` = 0.02–0.05 px: the forms
rank errors alike at 0.03 px (held-out AUC > 0.05 px: 0.922 / 0.917), but the
squared one drops 5.3 % of the rtv3d photometric centres (σ ≈ 0.1–0.2 px,
heavily blurred) at the default floor, the linear one 0.24 %.

### Alternatives considered

- **Inflate the covariance by a fitted factor.** Least squares on the
  training split gives 0.74 — the covariance is not optimistic on synthetic
  data — and a constant factor does not change ranking. The real-data floor
  (sharp markers) is not reproducible in synthetic scenes, and fitting it on
  rtv3d would tune the model to one display.
- **Fit κ and the bias coefficient together** (least squares on squared
  errors): κ 0.66, coefficient 1.39, worse held-out ranking (AUC for errors
  beyond 0.05 px 0.912 vs 0.932, each predicted error alone) — dominated by
  a few large errors. The physical coefficient is kept.
- **Correct the bias instead of predicting it.** ADR-022 measured a fitted
  gradient term (two more parameters per fit): it removes ~90 % of the
  synthetic bias but left the real-capture calibration residual unchanged.
  Out of scope here.
- **Model the display / lens geometry in the confidence.** It is a property
  of the target and camera, the same for every centre estimator, not of a
  centre estimate.

## Consequences

- Photometric `confidence` values change on every image; a threshold tuned
  on 0.17-dev photometric confidences needs retuning. Typical values: 0.68 /
  0.56 / 0.48 / 0.39 (medians) for true blur < 1 / 1–2 / 2–3 / ≥ 3 px on the
  synthetic study (was 0.60 / 0.46 / 0.35 / 0.35).
- A confidence cut keeps blurred markers whose centres are as good as sharp
  ones instead of dropping every blurred marker: at a 0.4 cut the study keeps
  6,483 of 8,554 centres at true blur 2–3 px (4 % beyond 0.05 px) where the
  old confidence kept 2,537 (4 %).
- `min_confidence` (0.03) bites at a ~1 px predicted error for a decoded,
  well-fitted ring.
- Recall at the default `min_confidence`: unchanged on the precision harness
  (no accepted photometric centre drops); rtv3d `adaptive_photometric`
  decoded IDs 5,874 → 5,866 (bottom band 2,634 → 2,625; the dropped centres
  are heavily blurred, fitted blur ≈ 4.5 px, σ ≈ 0.11 px from the covariance
  alone). Only reported markers can be removed by the change, so no ID
  becomes wrong. Regression gate: 77 / 77 checks pass.

## Evidence

**Synthetic confidence study** (before → after, same scenes and centres):

| split | ρ(confidence, error) | AUC error > 0.05 px | AUC error > 0.1 px | below 0.03 |
|---|---|---|---|---|
| held-out (radius 17 px, plain hex, seed 2; 31,884) | −0.393 → −0.556 | 0.821 → 0.917 | 0.940 → 0.979 | 0 → 0 |
| training (13,659) | −0.435 → −0.553 | 0.825 → 0.892 | 0.934 → 0.964 | 0 → 0 |

Per band of true blur (all scenes): median confidence, ρ within the band,
and kept / fraction beyond 0.05 px at a 0.4 cut.

| true blur | n | error median | confidence before → after | ρ within band | cut 0.4 before | cut 0.4 after |
|---|---|---|---|---|---|---|
| < 1 px | 14,160 | 0.010 px | 0.60 → 0.68 | −0.15 → −0.42 | 13,811 / 0.002 | 13,839 / 0.002 |
| 1–2 px | 20,952 | 0.016 px | 0.46 → 0.56 | −0.15 → −0.48 | 15,576 / 0.032 | 19,519 / 0.024 |
| 2–3 px | 8,554 | 0.022 px | 0.35 → 0.48 | −0.24 → −0.52 | 2,537 / 0.041 | 6,483 / 0.039 |
| ≥ 3 px | 1,877 | 0.032 px | 0.35 → 0.39 | −0.44 → −0.62 | 352 / 0.091 | 822 / 0.074 |

Predicted error against the true one, held-out: RMS error / σ_eff = 1.05 /
1.01 / 1.02 / 1.08 at true blur < 1 / 1–2 / 2–3 / ≥ 3 px (covariance alone:
1.07 / 1.07 / 1.15 / 1.45).

**rtv3d** (photometric adaptive, `min_confidence = 0` to see every centre;
proxies as above):

Reproduce: `cargo run --release -p ringgrid-cli --example
photometric_confidence_dump -- --out dump.jsonl <strips>`, then
`tools/rtv3d_confidence_report.py dump.jsonl --display`. Per tile-row band,
before → after; `cubic` is the geometry-robust proxy (markers with 20
labelled neighbours within 40 mm), `loo8` the 8-neighbour homography LOO.

| band | centres | confidence median | p10 | below 0.03 | cubic: n, median | AUC cubic > 0.1 px | ρ(conf, cubic) | loo8 median | AUC loo8 > 0.5 px |
|---|---|---|---|---|---|---|---|---|---|
| top | 305 | 0.54 → 0.44 | 0.22 → 0.18 | 0 → 0 | 26, 0.076 px | 0.45 → 0.44 | −0.01 → +0.03 | 0.38 px | 0.43 → 0.43 |
| middle | 2,939 | 0.52 → 0.43 | 0.26 → 0.21 | 0.0003 → 0 | 1,472, 0.046 px | 0.79 → 0.83 | −0.30 → −0.37 | 0.16 px | 0.40 → 0.52 |
| bottom | 2,632 | 0.43 → 0.34 | 0.15 → 0.11 | 0.0015 → 0.0053 | 1,356, 0.048 px | 0.83 → 0.84 | −0.36 → −0.38 | 0.20 px | 0.83 → 0.86 |
| all | 5,876 | 0.48 → 0.40 | 0.20 → 0.15 | 0.0009 → 0.0024 | 2,854, 0.047 px | 0.80 → 0.83 | −0.32 → −0.37 | 0.19 px | 0.60 → 0.67 |

For reference, projective centres on the same captures score AUC 0.75 / 0.78
for a quadratic-proxy residual beyond 0.3 / 0.5 px (median residual
0.22 px against 0.05 px photometric).

The top band stays as confident as the middle one: its fitted blur (1.71 px)
is no larger than the bottom band's (1.70 px), its rings are the largest
(29 px), and its 26 cubic-proxy residuals (median 0.076 px, like the
0.071 px of middle-band rings of the same size) are ranked by no image-based
predictor. Whether they are centre error or board geometry the cubic warp
leaves is not resolved by this data.

**Precision harness, photometric refinement**
(`RINGGRID_PRECISION_REFINE=photometric`, `precision_full_sweep`):

Identical before and after, per board × condition group and in total: of
202,302 visible rings 171,389 reported, 146,367 correct; wrong IDs 81 (all
on scenes with a second board in view), wrong positions 0, wrong labels 5
(the known `plainhex-few-spread-6`), centres beyond 1 px 7. No accepted
photometric centre of the catalogue falls below the default `min_confidence`
under either model, so no reported output changes; only the confidence
values do. Photometric confidence feeds nothing else after the photometric
stage: the output contract's `min_confidence` cut and the final homography
refit over the kept markers (unweighted); adaptive candidate scoring and
pass-2 seeds run before the stage.

## Affected Modules

- `detector/marker_build.rs` — `marker_confidence`, `MarkerRecord::center_bias_px`
- `detector/photometric_refine.rs` — `attach_blur_gradient_bias`
- `pipeline/result.rs` — `DetectedMarker::confidence` docs
- `tests/blur_gradient.rs` — `blur_gradient_confidence` study
- `tests/precision_audit.rs` — `RINGGRID_PRECISION_REFINE`
