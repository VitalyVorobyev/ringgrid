# Photometric centre refinement: model and measurements

The evidence behind two decisions:
[ADR-022](../decisions/022-photometric-center-refinement.md) (photometric
centre refinement under defocus blur) and
[ADR-029](../decisions/029-photometric-confidence-error-model.md) (the
confidence of a photometric centre). The ADRs state what was decided; this
document keeps the full model specification, the measurements, the
derivations and the experiments that were tried and rejected. The text was
moved here from the two ADRs unchanged, and each number is as recorded at the
time of its measurement.

## Photometric centre refinement (ADR-022)

### Why edge-based centres fail under blur

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

### The fitted model

The model as ADR-022 specifies it (2026-11-14, amended 2026-11-17). The two
sections it names follow below.

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
  by the PSF's width across that edge, with a second-order curvature
  correction (*Print radii and the curved edge*); code sectors as separable
  radial × angular steps;
- affine paper level and affine contrast; markers whose fit residual exceeds
  1.5× the image median are retried with a cubic paper level, adopted only if it lowers the residual σ below 0.85× (glare halos);
- Levenberg–Marquardt with Huber weights, saturated pixels masked;
- center covariance: the sampling covariance of the regularized estimator,
  `A⁻¹ JᵀW^½ C W^½J A⁻¹` with `A = JᵀWJ + s²P` and `C` the residual noise
  autocovariance over pixel lags (see *Covariance under correlated
  residuals*);
- printed radii estimated per image from free-radii fits of ≤ 16 markers
  (the plain mean of the converged fits within 4 robust σ of their median,
  accepted when its standard error is ≤ 0.008 `r3` and most fits' σ is
  within the measured blur range), on by default: `r0..r2` for coded
  markers, `r0` for plain ones (`r3` is the scale reference); priors and
  gates are fractions of `r3`, not fixed millimetres;
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

### Synthetic accuracy under blur

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

### Every target kind

Amendment of 2026-11-14.

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

### Model misfit and the reported covariance

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

### Model alternatives

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

#### Measurements (amendment, 2026-11-17)

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

### Blur gradient across a marker

**A blur gradient across each marker (two more parameters).** Built and
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

#### Why the residual did not move

calibration-rs measured where the bias
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

### Print radii and the curved edge

ALGO-021(a). The per-image radii were biased low where the blur is
comparable to the ink stroke (thin ring at 17 px, blur 1.5: r0 4.477 mm
against 4.60). Dumping every free-radii fit showed the per-marker fits
unbiased on average (4.601–4.606 mm, all converged with a larger iteration
budget) and the aggregation biased: there blur and stroke width trade off
(corr(r0, σ) ≈ +0.97), a fit's radius uncertainty and its convergence time
grow with the radius it finds, and so the iteration cap (the 63 capped fits
sat at 4.71–4.77 mm), the per-sample precision gate (rejected 4.62–4.66,
kept 4.48) and the inverse-variance weights (−0.65 % against +0.21 % for
the plain mean on the same samples) each favoured thin-stroke fits. The
consensus is now the plain mean after a robust outlier cut, free-radii fits
get four times the centre-fit iteration budget, and a consensus is accepted
on its own standard error rather than on per-sample gates.

Where σ exceeds about 0.3 r0 the per-marker fit itself was biased, by the
model: the first-order curved edge `Φ(z + ε/2)` (`z = d/σ`, `ε = σ/R`). The
outside fraction of a blurred disc is `E_τ[Φ(z + ετ²/2 + O(ε³))]` over the
tangential offset `τ ~ N(0, 1)` (the boundary bends away by `t²/2R`);
expanding and averaging (`E τ² = 1`, `E τ⁴ = 3`) gives
`Φ(z) + φ(z)(ε/2 − 3zε²/8) + O(ε³)`, i.e. `Φ(z(1 − ε²/4) + ε/2)`: the edge
also widens by `1 + ε²/4`. Against the exact profile (Marcum Q) the
largest single-edge error at ε = 0.1 / 0.2 is 7.1e-4 / 3.4e-3 of the
contrast for the first order and 7.6e-5 / 7.7e-4 for the second. Fitted
without noise to the exact blurred annulus (8×8 supersampled, unit test
`noiseless_free_radii_fit_recovers_the_hole_radius`), the free r0 error is:

| σ / r0 | 0.1 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 |
|---|---|---|---|---|---|---|
| first order | +0.04 % | +0.03 % | −0.15 % | −0.31 % | −1.24 % | −3.15 % |
| second order | +0.04 % | +0.04 % | −0.11 % | −0.10 % | −0.61 % | −2.05 % |

(+0.04 % is the renderer's floor.) Beyond σ ≈ 0.35 r0 the hole is blurred
as a whole, its centre never reaches the paper level, and no local
expansion holds: the third order diverges (+13 % at 0.45) and even a
25-node quadrature of the exact profile leaves r0 nearly unidentified there.
The curvature is held at `ε ≤ 1`.

On the photometric target matrix (24 seeds, same scenes), main →
consensus fix and second-order edge:

| Cell (outer radius px, blur px) | r0 bias | σ bias | Centre RMS, px |
|---|---|---|---|
| thin ring 17, 1.5 (6 cells) | −1.0 … −2.7 % → −0.03 … +0.16 % | −1.2 … −3.0 % → −0.1 … +0.2 % | unchanged |
| plain hex 8, 1.5, contrast 80 | −3.0 / −3.8 / −5.3 % → +0.2 / −0.2 / −0.7 % (0/20/30°) | → +0.2 … +0.6 % | unchanged |
| plain hex 8, 1.5, contrast 40 | nominal (no consensus) → 15–18 of 24 exposures | +2.8 … +3.9 % → +0.0 … +0.9 % | unchanged |
| coded hex 17, 3.0, contrast 80, 0° | −1.2 % → −0.5 % | +1.1 % → +0.1 % | 0.0457 → 0.0375 |
| coded hex 8, 1.5, contrast 80, 0° | −1.9 % → −1.0 % | +0.6 % → −0.3 % | 0.0363 → 0.0335 |
| coded rect 8, 1.5, contrast 80, 30° | | | 0.0407 → 0.0412 (+1.7 SE) |
| outer radius 17/33 px, blur ≤ 1.5 | identical | identical | identical |

The consensus fix alone made coded centres worse in the model-limited cells
(up to +16 %, coded hex 17 px blur 3): the old weighting had partly cancelled
the first-order model error. With the second-order edge no coded cell is
worse than before beyond seed noise except the coded rect cell above
(+0.0005 px), and every coded hex cell improves. Wrong labels are unchanged
(0) everywhere. Fixing the radii at their true values was also measured and
rejected: coded hex improves further but coded rect gets worse (+9 … +11 %).

Limits that remain: r0 is not identifiable where the blur reaches the
stroke width (thin ring at blur 3: −0.7 … −1.0 %) or σ > 0.35 r0 (plain hex
8 px blur 1.5 at 20–30° tilt: −0.7 … −1.2 %); a 5.6/2.8 mm plain rect
annulus at 8 px, blur 1.5 keeps a −3 % σ bias even with the true radii (not
diagnosed). At outer radius 8 px and blur 3 (σ = 0.375 R, outside the range
blur is reported in) the number of accepted photometric fits moves (plain hex
+105 over the matrix, plain rect −226, −20 %); centre RMS there improves in
every cell, and rejected fits keep their projective centre. A free-radii fit
costs 4–9 % more per marker.

## Photometric confidence (ADR-029)

### The real-capture residual

The starting point is in the context of ADR-029: on real captures of a
tilted-focus camera, measured with an 8-neighbour homography leave-one-out
(LOO) residual, the photometric centres with the highest confidence looked
like the worst ones.

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

### Synthetic confidence study

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

### Real captures, before and after

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

### Precision harness

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

### Fall-off and scale

**The fall-off and scale.** `1/(1 + σ/s)` with `s = 0.03 px` spans the
errors photometric centres have (0.01 px → 0.75, 0.03 → 0.5, 0.1 → 0.23) and
reaches the default `min_confidence` (0.03) at a 1 px predicted error — where
the edge-fitted confidence drops centres too (ADR-024). Chosen on the
training split (radius 10 / 28 px, coded hex and plain rect, seeds 0–1)
among `1/(1 + (σ/s)²)` and `1/(1 + σ/s)` at `s` = 0.02–0.05 px: the forms
rank errors alike at 0.03 px (held-out AUC > 0.05 px: 0.922 / 0.917), but the
squared one drops 5.3 % of the rtv3d photometric centres (σ ≈ 0.1–0.2 px,
heavily blurred) at the default floor, the linear one 0.24 %.

### Alternatives measured

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

### Effect on reported outputs

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
- On top of #104 (contrast-relative edge gate, size-aware completion), which
  admits more weak top-band rings: rtv3d `adaptive_photometric` 8,945 →
  8,981 decoded IDs (+41 / −5; 38 of the 41 in the top band, confidence
  median 0.038, all within 0.11 marker diameters of where the base run's
  homography puts their board position — a ring one cell off would be about
  1.5 diameters away). Those were photometric centres the old sharpness
  factor zeroed or pushed below 0.03. Photometric precision sweep identical
  to the base, group by group (0 wrong positions; 2 wrong IDs, both on
  scenes with a second board in view).
