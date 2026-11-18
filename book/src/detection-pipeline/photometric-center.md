# Photometric Center Refinement

Edge-based centers degrade under defocus blur. Each outer and inner edge point
is the peak of a radial intensity derivative; when the blur is comparable to
the ring and code-band widths, the blurred code sectors pull every edge point
toward or away from the center depending on whether the neighbouring sector is
paper or ink. The resulting center error depends on the marker's code and on
the blur, so it does not average out across markers. The projective center
inherits it from both ellipses.

`CircleRefinementMethod::Photometric` adds a final stage, on top of
projective-center correction, that fits each marker's pixels directly with a
model of the blurred printed pattern. It is opt-in because it costs a few
milliseconds per marker. It works
for every target kind the detector supports — coded and plain markers, hex and
rect lattices, with or without origin dots: the pattern is derived from the
target definition, so all of them go through the same model.

## Model

For every pixel `x` of a patch around the marker:

```text
I(x) = B(x) - A(x) * P(u(x); sigma)
```

- `u(x)` is the board-local point (millimetres from the marker center) that
  the pixel sees. The map from board to image is the board homography,
  corrected per marker by a free translation `p` and a free 2×2 matrix `C`:
  `y = h0 + C (x - p)`, `u = H_local^-1(y)`. The marker center is imaged
  exactly at `p`, which is the reported center — no ellipse center or conic
  pencil is involved, so the result is unbiased under perspective.
- `P` is the ink coverage of what the target prints around the marker: the
  marker's ink annulus (the two stroked rings and the code band of a coded
  marker, the filled annulus of a plain one), minus the **paper code sectors**
  of a coded marker's decoded codeword, plus the outer edges of the
  neighbouring markers and any origin dots nearby. It is blurred by a Gaussian
  PSF of standard deviation `sigma` with a fitted anisotropy (two more
  parameters, kept near isotropic by a weak prior), since real lenses are often
  astigmatic. Each circular edge is a normal-CDF step of its signed
  image-space distance, blurred by the PSF's width across that edge, with the
  first-order correction for edge curvature (a blurred disc appears smaller by
  `sigma^2 / (2 R)`). Code sectors are the product of a blurred radial band
  and a blurred angular indicator.
- `B` (the paper level) and `A` (paper minus ink) are affine in the image,
  absorbing vignetting and illumination gradients. Markers whose fit leaves an
  excess residual — typically next to a glare halo — are refitted with a cubic
  paper level and keep whichever fit explains the pixels better.

The fit is Levenberg–Marquardt on the pixel residuals with Huber weights.
Saturated pixels (`saturation_level`, default 250) and their 8-neighbours are
excluded. A weak prior keeps `C` near identity, which only matters when blur
erases a coded marker's code band and its in-plane rotation becomes
unobservable. A plain marker looks the same rotated about its center, so its
`C` does not fit that rotation at all: the marker keeps the board homography's
rotation, which the whole board determines. Left free, that rotation would be
held by the prior alone, and on large, tilted plain markers the solver would
creep along it for tens to hundreds of iterations.

## Printed radii

Printed boards do not always match their nominal stroke width, and a code band
modelled at the wrong radius biases the center. With `estimate_print_radii`
(default on), a first pass fits the inner radii freely on up to 16 markers —
the hole and both code-band edges of a coded marker, the inner edge of a plain
annulus (the outer edge sets the scale) — and the precision-weighted consensus
is used for all final fits. The estimate is reported in
`DetectionDiagnostics::photometric_print_radii_mm`.

## Outputs

| Output | Where |
|---|---|
| Refined center | `DetectedMarker::center` |
| Which estimator produced the center | `DetectedMarker::center_source` (`outer_ellipse`, `projective_center`, `photometric`) |
| 2×2 center covariance (px²) | `DetectedMarker::center_cov` (photometric centers only) |
| Fitted blur, contrast, residual, center shift, status | `MarkerDiagnostics::photometric` |
| Homography and RANSAC statistics, refit from the refined centers | `DetectionResult::homography`, `DetectionDiagnostics::ransac` |

## Covariance

The inverse Fisher information of a pixel fit assumes the residuals are
independent from pixel to pixel. Real images break that: demosaicing,
compression and sharpening correlate sensor noise over a few pixels, and a
pattern model that does not quite match the optics leaves structured
residuals. With noise correlated over about a pixel, the independent-pixel
covariance understates the center scatter about threefold.

The reported covariance is therefore a sandwich estimate
`A^-1 J^T C J A^-1` with `C` the autocovariance of the residual noise between
pixel pairs: a white part plus a Gaussian-correlated part whose amplitude and
correlation length are fitted per marker to the residual autocovariance at
lags up to 4 px (accounting for the share of the noise the fit itself
absorbs). The model is positive semidefinite by construction, a correlated part
is used only when it is statistically significant, and with white residuals
the estimate is the inverse Fisher information. The noise statistics are taken
from the part of the residual field that is antisymmetric about the fitted
center: a misfit that is symmetric about the center (a PSF shape or a response
curve the model does not have) cannot move the center and is not mistaken for
noise.

The covariance does not cover a bias common to the whole marker (the model's own
approximation error, an asymmetric misfit), which on synthetic data stays
below about 0.015 px for markers of 17 px outer radius and more.

A fit is rejected — and the marker keeps its previous center and
`center_source` — when it does not converge, moves more than
`max_center_shift_px` (default: half the outer radius), has an implausible blur
or contrast, or leaves a residual more than three times the image's typical fit
residual. The reason is reported in `MarkerDiagnostics::photometric.status`;
a marker that could not be placed on the board at all reports
`no_board_position`. With the method selected, every marker has a status.

## Scope

- Every marker with a board position: a decoded id on a coded target, a
  lattice label on a plain one. Undecoded or unlabelled markers keep their
  center (`no_board_position`).
- A board homography is required; finalize's grid labeling provides it for
  coded and plain targets alike (coded views it cannot label fall back to the
  decode-anchored global filter). When a plain
  board's origin is resolved (absolute frame) the model knows every printed
  marker and origin dot; in the relative frame of an unresolved board it
  knows the labelled markers only, so an undetected neighbour or an origin dot
  next to a marker is not modelled there.
- With a `PixelMapper`, the local board-to-image map is fitted through the
  mapper.
- The stage is the pipeline's last: it runs once per detection call, on the
  finished result — after every pass, scale tier and adaptive candidate — so
  adaptive detection does not pay for it per candidate.
- After refinement the result's homography and its RANSAC statistics are refit
  from the centers now in the result (refined and unrefined markers alike),
  with the same final-homography step finalize ends with: the refit is kept
  when it reprojects the centers better than the homography finalize fitted.

## Accuracy

**Synthetic blur benchmark** (`crates/ringgrid/tests/blur_center_accuracy.rs`,
from [`docs/performance.md`](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/performance.md#blur-benchmark-photometric-centers-2026-11)):
720×540 frames of the default hex board printed with 0.84 mm strokes (a
1.15 mm code band), detected with `detect_adaptive`; ≈33 px markers, tilts
0/20/30°, an illumination gradient, a saturated glare blob and σ = 2
grey-level noise. Center RMS / p95 over decoded markers, px:

| blur σ | contrast | projective-center | photometric |
|---:|---:|---:|---:|
| 0.8 | 40 | 0.485 / 0.559 | 0.013 / 0.023 |
| 1.5 | 40 | 0.449 / 0.693 | 0.022 / 0.038 |
| 0.8 | 80 | 0.614 / 0.924 | 0.008 / 0.014 |
| 1.5 | 80 | 0.715 / 1.095 | 0.013 / 0.022 |
| 2.5 | 80 | 0.600 / 1.118 | 0.028 / 0.052 |
| 3.0 | 80 | 0.736 / 1.387 | 0.040 / 0.073 |

Decoded counts are identical for every method (the photometric stage runs
after detection). At 40 grey levels and blur ≥ 2.5, and at blur 4 for any
contrast, the decoder loses or mis-decodes most markers for every method.

**Every target kind** (`crates/ringgrid/tests/photometric_target_matrix.rs`):
seven kinds — coded hex/rect, plain hex/rect with and without origin dots, a
thin plain ring — at marker outer radius 8/17/33 px, blur 0.8/1.5/3, contrast
40/80 and tilt 0/20/30°. For 17–33 px markers the photometric RMS is
0.012–0.016 px (median over cells) and at most 0.045 px for every kind except
one detection-limited coded-rect cell; the projective center errs by
0.05–1.3 px.

**Real images** (ADR-022, 2026-11-17 amendment): a precision plain plate,
58 × 58 rings on a rect 2 mm lattice (outer radius 0.8 mm, inner 0.4 mm,
origin dots), 2 cameras × 28 views, 2048 × 1536, rings 80–120 px across,
fitted blur σ 0.8–6 px; `detect_adaptive`, 6356 reported markers. Every one of
them gets a photometric center, and the smooth-fit residual over every
accepted marker is 0.0163 px (6210 markers). The smooth-fit residual is, per
image, the leave-one-out residual of the photometric centers against a
degree-4 polynomial from board to image, pooled over the 43 images with ≥ 25
markers accepted in both runs of the amendment's before/after comparison: a
consistency measure, not an error against ground truth.

**Cost.** About 6–15 ms per fit in release, two passes when print radii are
estimated — the reason the method is opt-in rather than the default.

The per-kind tables, Monte-Carlo covariance checks and the model-misfit study
are in
[ADR-022](https://github.com/VitalyVorobyev/ringgrid/blob/main/docs/decisions/022-photometric-center-refinement.md).

## Configuration

```rust
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};

let mut config = DetectConfig::from_target(TargetLayout::default_hex());
config.circle_refinement = CircleRefinementMethod::Photometric;
config.advanced.photometric.saturation_level = 250;
let detector = Detector::with_config(config);
# let _ = detector;
```

See [`PhotometricConfig`](../configuration/sub-configs.md#photometricconfig)
for every field.

**Source:** `crates/ringgrid/src/ring/photometric_center.rs`,
`crates/ringgrid/src/detector/photometric_refine.rs`
