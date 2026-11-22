# ADR-022: Photometric Center Refinement Under Defocus Blur

- **Status:** accepted
- **Date:** 2026-11-14 (amended 2026-11-14: every target kind, correlated
  residual covariance, anisotropic PSF; amended 2026-11-17: plain markers keep
  the homography's rotation)
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
pipeline, every marker with a board position (a decoded id on a coded target,
a lattice label on a plain one; any lattice) in a result with a board
homography is re-fitted from its pixels with a model of the blurred printed
pattern (`ring/photometric_center.rs`, stage in
`detector/photometric_refine.rs`):

- local projective map = board homography (through the mapper when present),
  corrected per marker by a free translation `p` and a free 2×2 matrix `C`
  (three free directions for a plain marker, whose rotation about its centre
  stays the homography's: see *Plain markers keep the homography's
  rotation*); the reported center is `p`, the image of the ring center;
- ink coverage derived from the target definition: the marker's ink annulus
  `[r0, r3]` (coded: the two stroked rings and the code band; plain: the
  filled annulus), minus the paper sectors of a coded marker's codeword in
  `[r1, r2]`, plus ink discs for the neighbouring markers' outer edges and
  the origin dots, each within reach of the patch whatever the lattice
  (hex, rect including diagonals). One model; a plain marker is the coded one
  without a code band;
- blurred by a fitted Gaussian PSF of σ with an anisotropy `[a, b]`
  (covariance `σ² [[1+a, b], [b, 1−a]]`, weak prior σ 0.5, |[a,b]| ≤ 0.9):
  every edge is a normal-CDF step of its signed image-space distance, blurred
  by the PSF's width across that edge, with a first-order curvature
  correction; code sectors as separable radial × angular steps;
- affine paper level and affine contrast; markers whose fit residual exceeds
  1.5× the image median are retried with a cubic paper level, adopted only if it lowers the residual σ below 0.85× (glare halos);
- Levenberg–Marquardt with Huber weights, saturated pixels masked;
- center covariance: the sampling covariance of the regularized estimator,
  `A⁻¹ JᵀW^½ C W^½J A⁻¹` with `A = JᵀWJ + s²P` and `C` the residual noise
  autocovariance over pixel lags (see *Covariance under correlated
  residuals*);
- printed radii estimated per image from free-radii fits of ≤ 16 markers
  (precision-weighted consensus), on by default: `r0..r2` for coded markers,
  `r0` for plain ones (`r3` is the scale reference); priors and gates are
  fractions of `r3` (0.1, 0.02), not fixed millimetres;
- gates (convergence, shift ≤ half the outer radius, plausible σ and
  contrast, residual ≤ 3× the image median); a rejected fit keeps the
  previous center;
- where the board frame comes from: coded → `xy_mm_of_id`; plain with a
  resolved origin → `cell_xy_mm(grid_coord)`, every printed cell and origin
  dot modelled; plain in the relative canonical frame → the assignment-frame
  position of the label, only the labelled markers modelled (the frame does
  not know the board's extent or where the dots are).

Every marker reports `center_source` (`outer_ellipse`, `projective_center`,
`photometric`), which also removes the silent fallback of the projective stage.
With the method selected every marker also carries a photometric status; a
marker the stage cannot place reports `no_board_position` (undecoded,
unlabelled, or no board homography), never a silent unrefined centre.

### Covariance under correlated residuals

The inverse Fisher information assumes independent pixel residuals. With
sensor noise correlated over ~1 px (Gaussian-filtered white noise of σ 1 px,
as demosaicing produces) it reports 0.29–0.38× the Monte-Carlo centre scatter
on every target kind. The covariance is therefore a sandwich with a residual
noise autocovariance `C(d) = w δ(d) + c exp(−|d|²/2ℓ²)`:

- `(w, c)` by non-negative least squares on the residual lag statistics at
  lags |d|∞ ≤ 4 px, for `ℓ` on a grid 0.4–4.8 px. The fit absorbs part of the
  noise (`e = (I−H)ε`): exactly `w(p δ − h)` for the white part, and
  `c(g p − G h)` for the correlated part when `H` is smooth over `ℓ`, with
  `h(d) = Σ_x H_{x,x+d}` computed from a Cholesky factor of `A⁻¹`. Without this
  correction the estimate stays ~20 % low (the background and contrast terms
  absorb the low-frequency part of the noise).
- `c` is used only when it exceeds 3 standard errors; otherwise the
  non-negativity constraint turns sampling noise into a positive correlation
  that a long `ℓ` integrates into a large inflation (seen: 1.25–1.35 on white
  noise at blur 3).
- PSD by construction (nugget plus Gaussian kernel), so no lag taper; the
  window is 3ℓ (≤ 16 px, ≤ a third of the patch).
- The statistics come from the part of the residual field antisymmetric
  about the fitted centre (reflection through the nearest half-pixel). A
  misfit symmetric about the centre — a PSF shape or response curve the model
  lacks, on a centrally symmetric marker — cannot move the centre; the
  symmetric-residual version inflated plain markers' σ by 1.2–2× under such
  misfits while their scatter and error stayed at the noise level.
- No prior term in the middle of the sandwich: where a direction the data
  barely constrain (the in-plane rotation of a plain ring under perspective)
  couples into `p`, the posterior `s²A⁻¹` reported up to 2× the scatter (thin
  plain rings, 33 px, 20° tilt) while the MAP estimate does not scatter by
  the prior's width.

Alternatives tried and dropped: a non-parametric tapered ACF (Bartlett,
adaptive bandwidth) with a projection correction solved for the integrated
covariance — ill-conditioned once the window's hat share `Σ h/p` approaches 1
(κ swung to ±100 s²); the classic cross-product HAC — 0.74 on correlated noise
(residual orthogonality to the score cancels it).

The study output (`study` cargo feature, `PhotometricResidualMarker`) reports
`covariance_inflation = sqrt(tr C / tr C_white)` and `residual_autocorrelation`,
the residual ACF at lags 1–6 px per axis. (Until the stage moved into the
pipeline they were fields of `PhotometricMetrics`.)

### Plain markers keep the homography's rotation

A plain annulus looks the same rotated about its centre. In the local map
that rotation is `C -> M R(t) M^-1 C`, `M` the map's Jacobian at the centre:
it changes the pixels only through the projective terms of the map and the
neighbours' ink at the patch edge. With all four entries of `C` free it was
held by the `C` prior (σ 0.15) alone — a long, curved, nearly flat valley
between walls the data make steep. Levenberg–Marquardt creeps along such a
valley: on large tilted rings the step was dominated by `C` after the first
few iterations, alternated between a rejected and an accepted trial, and
lowered the cost by 0.01–0.5 (in units of the noise variance) per
iteration, while `C` turned (median 3.5° at the 30-iteration cap, up to 60°
when run to convergence) and dragged the centre with it.

So a plain marker's `C` is fitted in the three-dimensional complement of the
rotation generator `M Ω M⁻¹` at `C = I` (Frobenius-orthonormal basis,
`CBasis` in `ring/photometric_center.rs`). The identity is in that complement
(the generator is traceless), the prior is the same isotropic one restricted
to it, and the marker keeps the board homography's rotation, which the whole
board determines to a fraction of a degree. Coded markers keep all four
directions (the code band fixes the rotation); their fits are bit-identical.

Alternatives rejected:

- **A larger iteration cap.** On a real image, 46 of 75 fits needed more than
  30 iterations, median 49 and up to 283; the runtime grows accordingly, and
  where the valley ends is set by the weak prior and model misfit.
- **A looser or scale-aware convergence test.** The test is not mis-scaled
  (it bounds the centre and blur steps in pixels and the cost change relative
  to the cost); with the rotation fixed the same test is met in 5–12
  iterations at every marker size. Loosening it would accept centres still
  moving: at the cap the centre was a median 0.13 and up to 1.8 of its
  reported σ away from where the fit converged, and moved 0.10 σ (median) over
  the last ten iterations.
- **A stiff prior on the rotation only.** A soft version of the same
  constraint that keeps a stiff valley for the solver; the exact constraint
  costs one parameter less.

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
- **Default on.** Not yet: it costs a few ms per marker (≈6–15 ms per fit in
  release, two passes when print radii are estimated).
- **Parallel models per target kind.** Rejected: the pattern is the target's
  ink layout, so one model with a derived pattern covers every kind; a plain
  marker is a coded one without a code band.
- **Isotropic PSF only.** Kept as the special case: the anisotropic PSF costs
  ≈2 ms per fit at small blur, none at blur 3, leaves isotropic scenes
  unchanged (identical RMS in every Gaussian cell), and under a PSF of axis
  ratio 1.8 lowers the centre RMS by 10–13 % on coded and 2–8 % on plain
  markers and removes most of the structured residual (old covariance
  over-report 1.42–1.72 → 1.07–1.13 on coded markers). Mixed at the 8 px /
  blur-3 cells (±15 %), which are outside the accepted regime anyway.
- **A blur gradient across each marker (two more parameters).** Built and
  measured, not adopted. On synthetic scenes with a blur width that varies
  across the image (`tests/blur_gradient.rs`) the centre is biased toward the
  sharper side by about gradient × blur width (factor 0.9–1.8 by target
  kind, independent of marker size), and a gradient term removes about 90 %
  of it. It costs about 14 % per fit and, without a gradient, up to 1.4–1.8×
  the centre error on plain rect markers whose blur is a fifth of their
  radius. On real captures of a tilted-focus camera, calibrating six cameras
  with the term on and off left the calibrated residual unchanged, so the
  measured benefit does not cover the cost. (Earlier revisions quoted
  "pooled 0.137 px RMS both ways". No recorded on/off solve backs that
  figure: it equals the pooled training RMS of the calibration library's
  default tilted-sensor route on those cameras, a solve without the term.
  The measurement below replaces it.)
  The intensity centroid of a blurred marker does not move under a blur
  gradient, so this is an estimator bias, not an optical shift.
  A fixed-gradient second pass (the neighbour σ-plane gradient held fixed
  in the PSF, no free parameters) was also measured and rejected
  (ALGO-018): on 133 scatter-form cells of `blur_gradient_bias_matrix` it
  left 0.46 of the one-σ bias against 0.50 for subtracting the law
  `1.24·σ·∇σ` without a refit (0.16 against 0.29 with the true gradient),
  at 2.1–2.3× the photometric time per refitted marker and up to 2.6× the
  centre scatter in the worst cell.

  *Why the residual did not move.* calibration-rs measured where the bias
  goes, on six tilted-focus cameras (4921 observations in 110 views; each
  camera solved from the raw centres, from centres corrected by the law
  `b = −1.24·σ·∇σ`, and from a sign-flipped control): 92 % of the bias
  energy is taken up by the camera intrinsics and the per-view poses, and
  the pooled mean residual moves by 0.003 px (0.109 → 0.106). Under a
  defocus model `σ² = σ0² + κ²(1/z − π·x)²` the bias is, to first order, a
  lateral translation of each view plus a rescaling of `fy`, and a
  calibration estimates both. The measured signature is `fy/fx` moving by
  +3.1 … +4.6e-4 (0.8–1.9 σ, the same sign on all six cameras) and the
  tilt by at most 0.44 σ; a synthetic test in calibration-rs reproduces
  the `fy/fx` shift within 2 % of the first-order prediction. So removing
  the bias cannot lower a calibrated residual. It changes the aspect
  ratio, which matters where the aspect ratio or the blur itself is the
  measurement.

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
- The covariance covers spatially correlated noise and centrally symmetric
  misfit, not a bias common to the marker. Asymmetric misfit — a coded
  marker's sector corners under a non-Gaussian PSF — makes it conservative
  (1.4–2.1× the scatter on coded markers under anisotropic or disc PSFs).
- In the relative frame of a plain board without a resolved origin an
  undetected neighbour or a nearby origin dot is not modelled.
- `CircleRefinementMethod` gains a variant (exhaustive matches break).

**Neutral:**
- The result homography and its RANSAC statistics are refit from the refined
  centers (finalize's final-H step), after the stage. (Originally the
  homography stayed the one fitted to the pre-refinement centers.)

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

### Evidence across target kinds (amendment, 2026-11-14)

`crates/ringgrid/tests/photometric_target_matrix.rs` (`photometric_target_matrix`,
`photometric_misfit_matrix`, both `#[ignore]`d; the default run checks one
medium cell per kind). Kinds: coded hex (8 mm pitch, wide code band), coded
rect (14 mm, band 3.7–5.1 mm), plain hex with and without origin dots
(4.8/2.4 mm annulus), plain rect with and without dots (5.6/2.8 mm), plain rect
with a thin 1 mm annulus; each detected with its own target. 720×540, clean
illumination gradient, σ = 2 noise, no glare; 6 noise seeds per cell; marker
outer radius 8/17/33 px × blur 0.8/1.5/3 × contrast 40/80 × tilt 0/20/30°.
"Before" is 0.14.0 on the same scenes (photometric did not run on plain
targets).

Medium and large markers (17 and 33 px), blur ≤ 3, cells where markers are
detected:

| kind | proj. RMS | photometric RMS before (max) | after median / max | scatter ratio before | after |
|---|---|---|---|---|---|
| coded hex | 0.07–0.87 | 0.045 | 0.012 / 0.045 | 0.89–1.04 | 0.91–1.61 |
| coded rect | 0.06–0.82 | 0.162 | 0.012 / 0.246 | 0.46–2.15 | 0.81–1.59 |
| plain hex, dots | 0.06–0.87 | not run | 0.016 / 0.034 | — | 0.95–1.38 |
| plain hex, no dots | 0.06–1.29 | not run | 0.016 / 0.042 | — | 0.95–1.24 |
| plain rect, dots | 0.05–0.88 | not run | 0.015 / 0.024 | — | 0.94–1.55 |
| plain rect, no dots | 0.05–1.12 | not run | 0.015 / 0.024 | — | 0.94–1.53 |
| plain rect, thin | 0.07–0.59 | not run | 0.014 / 0.028 | — | 0.97–1.43 |

Cells outside the acceptance (RMS ≤ 0.05 px, scatter ratio 0.8–1.25):

- coded rect, 17 px, blur 3, contrast 80, 20°: RMS 0.246 (0.162 before) —
  the decoder finds 48 of 144 markers with one wrong id; a few markers at the
  detection limit carry > 1 px errors.
- coded, blur 3 (17 and 33 px): scatter ratio 1.27–1.61; the model's own
  approximation of the blurred sector corners leaves an asymmetric residual
  (bias 0.008–0.039 px), which the covariance treats as noise —
  conservative against the scatter; against the truth the ratio is
  0.68–1.45.
- plain, blur 3 and thin rings at 33 px with tilt: scatter ratio up to
  1.38–1.55 (conservative); independent of the board raster density (600 vs
  2400 dpi identical).
- Small markers (8 px): RMS 0.018–0.054 px at blur ≤ 1.5, 0.13–0.14 px for
  plain markers at blur 3 (the 2 px-wide annulus is blurred to a blob); the
  coded decoder loses most markers at blur ≥ 1.5. Scatter ratio 1.00–1.18.
- No photometric centres where detection fails: contrast 40 at blur 3 for
  coded and plain rect kinds, some large tilted partial plain-hex views
  (grid labelling), the thin ring at 8 px and contrast 40.

Model misfit (17 px, blur 1.5, contrast 80, 20°; reported σ over Monte-Carlo
scatter, old → new covariance; bias = RMS of the seed-mean error):

| scene effect | coded hex | coded rect | plain kinds (range) |
|---|---|---|---|
| none | 0.99 → 1.06 | 1.00 → 1.08 | 1.01–1.07 |
| anisotropic PSF, ratio 1.8 | 1.42 → 1.80 | 1.72 → 2.09 | 1.02–1.08 |
| Gaussian + 15 % halo | 1.00 → 1.09 | 1.02 → 1.19 | 1.02–1.17 |
| disc PSF | 1.11 → 1.44 | 1.14 → 1.47 | 1.02–1.09 |
| gamma 0.8 | 0.98 → 1.05 | 1.00 → 1.09 | 1.01–1.06 |
| noise correlated over 1 px | 0.34 → 0.99 | 0.35 → 1.02 | 0.99–1.04 (old 0.33–0.38) |
| anisotropic + gamma + correlated noise | 0.72 → 1.19 | 0.84 → 1.30 | 1.00–1.02 |

Bias stays ≤ 0.015 px in every misfit cell; reported σ over error against
the truth is 0.86–1.68 (coded) and 0.95–1.07 (plain). Fitter-level
Monte-Carlo (`monte_carlo` unit tests, 120 seeds, coded and plain): white
noise 0.94–1.05, noise correlated over 1 px 0.95–0.97 (old covariance
0.29–0.32).

### Evidence: plain markers keep the homography's rotation (amendment, 2026-11-17)

Real images (private, not in the repository): a precision plain plate,
58 × 58 rings on a rect 2 mm lattice (outer radius 0.8 mm, inner 0.4 mm,
origin dots), 2 cameras × 28 views, 2048 × 1536, rings 80–120 px across,
fitted blur σ 0.8–6 px; `detect_adaptive`, 6356 reported markers.

| | before | after |
|---|---|---|
| photometric centres | 2068 (32.5 %) | 6356 (100 %) |
| `not_converged` | 4285 | 0 |
| iterations (final fits), median / p90 | 30 (cap) / 30 | 9 / 12 |
| `|C − I|`, median (one image) | 0.088 (rotation 3.5°, up to 60° run to convergence) | 0.002 |
| photometric stage per image (6 images, sequential) | 12.8 s | 5.1 s |
| smooth-fit residual, markers accepted before (1985) | 0.0238 px | 0.0237 px |
| smooth-fit residual, every accepted marker | 0.0238 px (1985) | 0.0163 px (6210) |

Acceptance before fell with ring size (0.47 at 80–90 px across, 0.22 at
110–120 px) and barely with blur or image position. Smooth-fit residual:
per image, a degree-4 polynomial from board to image fitted to the
photometric centres, leave-one-out residuals, pooled RMS over the 43 images
with ≥ 25 markers accepted in both runs. The fits that stopped at the cap
were close but still moving: against the same per-image fit of their
converged neighbours they scattered by 0.034 px RMS, and on one image run to
convergence they had a median 0.13 (max 1.8) of their reported σ still to go.
Reported σ against the leave-one-out residual after the change: 1.13–1.31×
by blur bin (sharpest worst), 1.20 pooled; the residual includes what a
quartic cannot follow, so this bounds the covariance's understatement.

Synthetic, fitter level (`plain_marker_converges_at_every_size`, 30° tilt,
20–111 px across): every fit converges in ≤ 8 iterations, within 4 reported
σ of the truth. Before, a 74 px fit took up to 12 iterations, and 111 px fits
averaged 10–17 (one in four at blur 5 not converged) with a centre RMS of
0.015–0.026 px, against 0.007–0.010 px after. `photometric_scale_matrix`
(pipeline, every kind, outer radius 10–60 px, blur 0.8–5, tilt 0/30°,
2 seeds): unconverged fits 68 → 31, all 31 in one cell whose model misfits (plain rect without
dots, 20 px, blur 3, 30°: relative frame, undetected neighbours blur into the
patch; unchanged, mean fit residual 1.8× the noise) — they still report
`not_converged`. Plain photometric RMS at 30° tilt drops 19–67 % at 33 and
45 px radius (e.g. plain hex 45 px blur 1.5: 0.0104 → 0.0069 px; thin ring
45 px blur 0.8: 0.0188 → 0.0063 px), −22 % to +2 % at 20 px (misfit cell
aside), −34 % to +26 % at 60 px (2–5 markers per cell); at 0° tilt, and at
10 px radius, it moves by ≤ 0.0005 px. Coded markers: bit-identical.
`photometric_target_matrix` (scales 10/20/33/55 px, 3 seeds, 720 × 540)
agrees: plain unconverged fits 103 → 0, plain RMS at 33 px
0.0129/0.0132/0.0123 → 0.0107/0.0119/0.0093 (rect dots / hex dots / thin).
Monte-Carlo scatter ratio (mean over a kind's cells at one scale): unchanged
within 0.02 at 10 and 20 px; at 33 px 1.09 → 1.04, 0.92 → 0.85,
1.15 → 1.01; at 55 px (few markers per cell) 0.86 → 0.82, 0.71 → 0.66,
0.67 → 0.60, where coded hex is at 0.50 unchanged — the under-coverage of
large markers in this harness predates the change. Fitter-level Monte-Carlo
(`monte_carlo`) for plain markers 0.95–0.96 as before. Golden snapshots of
the plain scene with photometric refinement: same markers and statuses,
centres moved by 0.0012 px RMS (max 0.0026 px), homography refit with them.

## Affected Modules

- `crates/ringgrid/src/ring/photometric_center.rs` (new)
- `crates/ringgrid/src/detector/photometric_refine.rs` (new)
- `crates/ringgrid/src/detector/config/{scale,stages,mod}.rs`
- `crates/ringgrid/src/detector/{marker_build,center_correction}.rs`
- `crates/ringgrid/src/pipeline/{result,run}.rs`, `crates/ringgrid/src/api.rs`
- `crates/ringgrid/tests/photometric_target_matrix.rs`,
  `crates/ringgrid/tests/support/blur_scene.rs` (PSF variants, gamma,
  correlated noise, radiance/exposure split)
- `crates/ringgrid-py/python/ringgrid/_api.py` (`PhotometricMetrics`,
  `ResidualAutocorrelation`)
