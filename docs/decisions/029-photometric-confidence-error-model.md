# ADR-029: Photometric Confidence from the Centre's Predicted Error

- **Status:** accepted
- **Date:** 2026-11-19
- **Author role:** algorithm-engineer
- **Amends:** ADR-028 (decision 4, photometric confidence)
- **Evidence:** [photometric centre study](../studies/photometric-center.md#photometric-confidence-adr-029)

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

What was measured:

1. **Most of that LOO residual is target and lens geometry, not centre
   error**: the display draws markers off their nominal positions, and an
   8-neighbour homography leaves smooth curvature. Against a geometry-robust
   proxy the covariance is 1.2–1.3× optimistic in the middle and bottom
   bands and the confidence ranks their errors
   ([study](../studies/photometric-center.md#the-real-capture-residual)).
2. **The confidence did not use the photometric precision**: centre errors
   are small (median 0.013 px), so against the 0.1 px scale the precision
   factor was 0.93–0.99 for nearly every centre, and the confidence ranked
   the edge sampler and the ring size. Under a focus gradient the error grows
   faster with blur than σ: the blur-gradient bias, common to the whole
   marker and invisible to the covariance
   ([study](../studies/photometric-center.md#synthetic-confidence-study)).

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

The fall-off `1/(1 + σ/s)` with `s = 0.03 px` spans the errors photometric
centres have and reaches the default `min_confidence` (0.03) at a 1 px
predicted error, where the edge-fitted confidence drops centres too
(ADR-024). It ranks errors like the squared form but drops far fewer heavily
blurred real centres at the default floor
([study](../studies/photometric-center.md#fall-off-and-scale)).

### Alternatives rejected

Measurements in the [study](../studies/photometric-center.md#alternatives-measured).

- **Inflate the covariance by a fitted factor** — the fit gives 0.74 on
  synthetic data and a constant factor does not change ranking; fitting the
  real-data floor would tune the model to one display.
- **Fit κ and the bias coefficient together** — worse held-out ranking,
  dominated by a few large errors; the physical coefficient is kept.
- **Correct the bias instead of predicting it** — ADR-022: a fitted gradient
  term left the real-capture calibration residual unchanged.
- **Model the display / lens geometry in the confidence** — a property of the
  target and camera, not of a centre estimate.

## Consequences

- Photometric `confidence` values change on every image; a threshold tuned
  on 0.17-dev photometric confidences needs retuning.
- A confidence cut keeps blurred markers whose centres are as good as sharp
  ones instead of dropping every blurred marker.
- `min_confidence` (0.03) bites at a ~1 px predicted error for a decoded,
  well-fitted ring.
- Recall at the default `min_confidence`: unchanged on the precision harness;
  a few heavily blurred real centres drop. Only reported markers can be
  removed by the change, so no ID becomes wrong.

Typical values and recall figures:
[study](../studies/photometric-center.md#effect-on-reported-outputs).

## Affected Modules

- `detector/marker_build.rs` — `marker_confidence`, `MarkerRecord::center_bias_px`
- `detector/photometric_refine.rs` — `attach_blur_gradient_bias`
- `pipeline/result.rs` — `DetectedMarker::confidence` docs
- `tests/blur_gradient.rs` — `blur_gradient_confidence` study
- `tests/precision_audit.rs` — `RINGGRID_PRECISION_REFINE`
