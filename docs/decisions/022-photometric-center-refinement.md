# ADR-022: Photometric Center Refinement Under Defocus Blur

- **Status:** accepted
- **Date:** 2026-11-14 (amended 2026-11-14: every target kind, correlated
  residual covariance, anisotropic PSF; amended 2026-11-17: plain markers keep
  the homography's rotation; amended 2026-11-22: blur as a reported
  measurement; amended 2026-11-24: fixed print radii and the ink moment
  rejected, unweighted radii consensus, second-order curved edge, saturation
  gate)
- **Author role:** algorithm-engineer
- **Supersedes:** none
- **Related:** ADR-028 and ADR-029 (the confidence of a photometric centre)
- **Evidence:** [photometric centre study](../studies/photometric-center.md),
  [paired centre metrology](../studies/center-metrology.md)

## Context

Edge-based centers degrade badly under defocus blur: on synthetic 33 px
markers the projective center scatters by 0.45–0.74 px RMS at every blur
level from 0.8 to 3.0 px. The dominant mechanism is the code band: the
blurred paper sectors move each ray's edge point in or out depending on the
neighbouring sector's bit, a code- and blur-dependent offset that does not
average out across markers and that every edge-based estimator shares.
Printed boards also do not necessarily follow the nominal stroke width, so a
model of the code band must use the printed radii
([study](../studies/photometric-center.md#why-edge-based-centres-fail-under-blur)).

## Decision

Add `CircleRefinementMethod::Photometric` (opt-in): after the normal
pipeline, every marker with a board position (a decoded id on a coded target,
a lattice label on a plain one; any lattice) in a result with a board
homography is re-fitted from its pixels with a model of the blurred printed
pattern (`ring/photometric_center.rs`, stage in
`detector/photometric_refine.rs`; full specification in the
[study](../studies/photometric-center.md#the-fitted-model)):

- one model for every target kind: ink coverage derived from the target
  definition (the marker's annulus, a coded marker's code sectors, the
  neighbours' outer edges and the origin dots within reach); a plain marker
  is the coded one without a code band;
- the board homography corrected per marker by a free translation `p` (the
  reported center) and a free 2×2 matrix `C`;
- a fitted Gaussian PSF with an anisotropy; affine paper level and contrast,
  with a cubic retry for glare halos; Levenberg–Marquardt with Huber weights,
  saturated pixels masked;
- every edge a Gaussian-blurred step with the circular edge's curvature to
  second order in `σ/R`
  ([study](../studies/photometric-center.md#print-radii-and-the-curved-edge));
- printed radii estimated per image from free-radii fits of ≤ 16 markers, on
  by default, as the unweighted mean after a robust outlier cut (precision
  weights favoured thick-stroke fits where blur and stroke width trade off);
  priors and gates are fractions of the outer radius `r3`;
- gates (convergence, shift, σ, contrast, residual, and saturation: a patch
  more than half masked whose centre σ exceeds 0.08 px,
  [study](../studies/center-metrology.md#exposure-and-saturation-algo-022));
  a rejected fit keeps the previous center;
- every marker reports `center_source`; with the method selected a marker
  the stage cannot place reports `no_board_position`, never a silent
  unrefined centre.

**Covariance under correlated residuals.** Under noise correlated over ~1 px
the inverse Fisher information reports 0.29–0.38× the Monte-Carlo centre
scatter, so the center covariance is a sandwich with a residual noise
autocovariance `C(d) = w δ(d) + c exp(−|d|²/2ℓ²)`, fitted on the residual
part antisymmetric about the fitted centre
([study](../studies/photometric-center.md#covariance-under-correlated-residuals)).

**Plain markers keep the homography's rotation.** A plain annulus looks the
same rotated about its centre; with `C` free, Levenberg–Marquardt crept along
that flat valley and dragged the centre. A plain marker's `C` is fitted in
the three-dimensional complement of the rotation generator (`CBasis`); coded
markers keep all four directions, bit-identical
([study](../studies/photometric-center.md#plain-markers-keep-the-homographys-rotation)).

**Blur is a reported measurement.** Each photometric centre reports the blur
the fit measured (`DetectedMarker::blur`: σ as the Gaussian edge width
including pixel integration, the PSF anisotropy, and their 3×3 covariance
from the same sandwich), for use as a focus observation. The covariance is
stated for differences between markers of one image: with print radii
estimated per image, all markers of an image share one further σ offset it
does not contain. Not reported when σ exceeds a quarter of the outer radius
(measured accuracy and limits: the book's *Photometric Center Refinement →
Blur*). It replaces the diagnostics field `PhotometricMetrics::blur_sigma_px`.

### Alternatives rejected

- **Iterated ray sampling with a sub-pixel edge** — keeps the code-dependent
  edge shift ([study](../studies/photometric-center.md#model-alternatives)).
- **Rendering + convolution** — an order of magnitude slower for a bias
  already below 0.03 px at blur 4 px.
- **Quadratic paper level by default** — costs precision on clean markers.
- **Default on** — ≈6–15 ms per fit in release.
- **Parallel models per target kind** — one derived pattern covers every kind.
- **Isotropic PSF only** — kept as the special case of the anisotropic one.
- **Tapered ACF or cross-product HAC covariance** — ill-conditioned; 0.74 on
  correlated noise.
- **Plain rotation by a larger iteration cap, a looser convergence test or a
  stiff rotation prior** — slower, or accepts centres still moving.
- **A blur gradient across each marker** — removes about 90 % of the
  synthetic bias, but the calibrated residual of six tilted-focus cameras did
  not move: the calibration takes up 92 % of the bias
  ([study](../studies/photometric-center.md#blur-gradient-across-a-marker),
  [why](../studies/photometric-center.md#why-the-residual-did-not-move)).
- **A fixed-gradient second pass (ALGO-018)** — 0.46 of the one-σ bias left
  against 0.50 for the law `1.24·σ·∇σ` alone, at 2.1–2.3× the time.
- **Fixed print radii pooled over training captures (ALGO-021)** — no centre
  gain over the per-image radii (identical gradient-reversal bias, equal or
  larger blur drift on sharp real controls), and per-image radii already
  remove the blur bias of a mis-printed stroke
  ([study](../studies/center-metrology.md#measurements-and-decision)).
- **The projectively corrected ink moment (ALGO-020)** — the same
  gradient-reversal bias as the fit (−0.029 vs −0.028 px), more blur drift on
  the sharp plate and fewer markers measurable; kept as a `study` comparator
  ([study](../studies/center-metrology.md#measurements-and-decision)).

## Consequences

- **+** Center error drops by an order of magnitude on the synthetic
  benchmark ([study](../studies/photometric-center.md#synthetic-accuracy-under-blur));
  detection and decoding are unchanged (post-stage).
- **+** Per-marker center covariance, fitted blur and fit status for
  downstream weighting; detector gross errors (a projective center several px
  off) are repaired when the photometric fit lands within the shift gate with
  a good residual.
- **−** A few ms per decoded marker (two passes when print radii are
  estimated).
- **−** The covariance covers spatially correlated noise and centrally
  symmetric misfit, not a bias common to the marker. Asymmetric misfit — a
  coded marker's sector corners under a non-Gaussian PSF — makes it
  conservative (1.4–2.1× the scatter on coded markers under anisotropic or
  disc PSFs).
- **−** In the relative frame of a plain board without a resolved origin an
  undetected neighbour or a nearby origin dot is not modelled.
- **−** `CircleRefinementMethod` gains a variant (exhaustive matches break).
- The result homography and its RANSAC statistics are refit from the refined
  centers (finalize's final-H step), after the stage. (Originally the
  homography stayed the one fitted to the pre-refinement centers.)

## Affected Modules

`ring/photometric_center.rs`, `detector/photometric_refine.rs` (new);
`detector/config/{scale,stages,mod}.rs`,
`detector/{marker_build,center_correction}.rs`, `pipeline/{result,run}.rs`,
`api.rs`; `tests/photometric_target_matrix.rs`,
`tests/support/blur_scene.rs` (PSF variants, gamma, correlated noise,
radiance/exposure split); `ringgrid-py/python/ringgrid/_api.py`
(`PhotometricMetrics`, `ResidualAutocorrelation`).
