# ADR-022: Photometric Center Refinement Under Defocus Blur

- **Status:** accepted
- **Date:** 2026-11-14
- **Author role:** algorithm-engineer
- **Supersedes:** none

## Context

Edge-based centers degrade badly under defocus blur. On synthetic 33 px
markers (σ = 2 noise, 40–80 grey levels of contrast, 0–30° tilt, illumination
gradient, glare blob) the projective center scatters by 0.45–0.74 px RMS at
every blur level from 0.8 to 3.0 px, and the outer-ellipse center is no
better. The dominant mechanism is the code band: once the PSF is comparable
to the ring and code-band widths, the blurred paper sectors move each ray's
edge point (the argmax of a radial derivative) in or out depending on the
neighbouring sector's bit, so the ellipse center gets a code- and
blur-dependent offset that does not average out across markers. The
two-conic projective center inherits it from both ellipses. Smaller
contributors: the argmax on a 0.5 px grid without sub-sample interpolation,
rays cast from the integer proposal, and an algebraic ellipse fit.

A second finding: printed boards do not necessarily follow the nominal
stroke width. A board printed with 0.84 mm strokes (code band 1.15 mm)
instead of the nominal 1.152 mm (code band 0.45 mm) decodes fine with the
nominal layout, but any model of the code band must use the printed radii.

## Decision

Add `CircleRefinementMethod::Photometric` (opt-in): after the normal
pipeline, every decoded marker of a coded target with a board homography is
re-fitted from its pixels with a model of the blurred printed pattern
(`ring/photometric_center.rs`, stage in `detector/photometric_refine.rs`):

- local projective map = board homography (through the mapper when present),
  corrected per marker by a free translation `p` and a free 2×2 matrix `C`;
  the reported center is `p`, the image of the ring center;
- ink coverage of both rings, the decoded code sectors and the neighbours'
  outer rings, blurred by a fitted isotropic Gaussian σ (normal-CDF edge steps
  of the image-space signed distance with a first-order curvature
  correction; code sectors as separable radial × angular steps);
- affine paper level and affine contrast; markers whose fit residual exceeds
  1.5× the image median are retried with a cubic paper level, adopted only if it lowers the residual σ below 0.85× (glare halos);
- Levenberg–Marquardt with Huber weights, saturated pixels masked;
- center covariance = `p` block of `(JᵀWJ/s² + prior)⁻¹`;
- printed radii estimated per image from free-radii fits of ≤ 16 markers
  (precision-weighted consensus), on by default;
- gates (convergence, shift ≤ half the outer radius, plausible σ and
  contrast, residual ≤ 3× the image median); a rejected fit keeps the
  previous center.

Every marker reports `center_source` (`outer_ellipse`, `projective_center`,
`photometric`), which also removes the silent fallback of the projective stage.

### Alternatives considered

- **Iterated ray sampling with an erf/parabolic sub-pixel edge and a
  geometric ellipse refinement.** Rejected without building it: it fixes the
  grid quantization and the ray origin but not the dominant term — every
  edge-based estimator that ignores the code band sees the code-dependent
  shift of the edge profile under blur.
- **Rendering + convolution instead of analytic steps.** More exact, about an
  order of magnitude slower; the analytic model's noise-free bias is below
  0.03 px at blur 4 px (unit test), well under the noise floor.
- **Quadratic paper level by default.** Tested; it rescues glare-adjacent
  markers but costs precision on clean markers, so it is used only through
  the cubic retry.
- **Default on.** Not yet: it costs a few ms per marker, and on real captures
  the reported covariance underestimates the scatter (below).

## Consequences

**Positive:**
- Center error drops by an order of magnitude on the synthetic benchmark
  (Evidence); detection and decoding are unchanged (post-stage).
- Per-marker center covariance, fitted blur and fit status for downstream
  weighting.
- Detector gross errors (a projective center several px off) are repaired
  when the photometric fit lands within the shift gate with a good residual.

**Negative:**
- A few ms per decoded marker (two passes when print radii are estimated).
- The covariance models pixel noise only. On real captures with a non-Gaussian
  defocus PSF the model leaves systematic residuals at the edges and the
  scatter exceeds the reported σ several-fold; the noise-only covariance is a
  relative weight, not an absolute uncertainty there.
- `CircleRefinementMethod` gains a variant (exhaustive matches break).

**Neutral:**
- The result homography stays the one fitted to the pre-refinement centers.

## Evidence

`crates/ringgrid/tests/blur_center_accuracy.rs`, wide-code print, nominal
detector config, `detect_adaptive`, 6 frames per cell (tilts 0/20/30°),
center RMS in px (decoded markers):

| blur σ | contrast | projective-center | photometric |
|---|---|---|---|
| 0.8 | 40 | 0.485 | 0.013 |
| 1.5 | 40 | 0.449 | 0.022 |
| 0.8 | 80 | 0.614 | 0.008 |
| 1.5 | 80 | 0.715 | 0.013 |
| 2.5 | 80 | 0.600 | 0.028 |
| 3.0 | 80 | 0.736 | 0.040 |

At 40 grey levels and blur ≥ 2.5, and at blur 4 for any contrast, the decoder
loses (or mis-decodes) most markers for every center method. The fitter
itself, seeded near the truth, stays below 0.03 px noise-free bias at blur 4.
Monte-Carlo, reported/empirical center σ: single marker (120 noise seeds,
σ = 2, contrast 40) 0.99 at blur 1, 0.95 at blur 3; whole scenes (30 noise
seeds, `scene_covariance_monte_carlo`) 0.995 / 0.989 / 0.964 at blur
0.8 / 1.5 / 3.0, per-marker ratio p10–p90 0.87–1.16. Against ground truth the
error is 1.1–1.7× the reported σ (model bias, glare halos).

## Affected Modules

- `crates/ringgrid/src/ring/photometric_center.rs` (new)
- `crates/ringgrid/src/detector/photometric_refine.rs` (new)
- `crates/ringgrid/src/detector/config/{scale,stages,mod}.rs`
- `crates/ringgrid/src/detector/{marker_build,center_correction}.rs`
- `crates/ringgrid/src/pipeline/{result,run}.rs`, `crates/ringgrid/src/api.rs`
