# Photometric Center Refinement

Edge-based centers degrade under defocus blur. Each outer and inner edge point
is the peak of a radial intensity derivative; when the blur is comparable to
the ring and code-band widths, the blurred code sectors pull every edge point
toward or away from the center depending on whether the neighbouring sector is
paper or ink. The resulting center error depends on the marker's code and on
the blur, so it does not average out across markers. The projective center
inherits it from both ellipses.

`CircleRefinementMethod::Photometric` adds a final stage that fits each decoded
marker's pixels directly with a model of the blurred printed pattern.

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
- `P` is the ink coverage of the printed marker: the inner and outer rings,
  the **decoded code sectors** of this marker, and the outer rings of its
  neighbours, blurred by an isotropic Gaussian PSF of standard deviation
  `sigma`. Each circular edge is a normal-CDF step of its signed image-space
  distance, with the first-order correction for edge curvature (a blurred disc
  appears smaller by `sigma^2 / (2 R)`). Code sectors are the product of a
  blurred radial band and a blurred angular indicator.
- `B` (the paper level) and `A` (paper minus ink) are affine in the image,
  absorbing vignetting and illumination gradients. Markers whose fit leaves an
  excess residual — typically next to a glare halo — are refitted with a cubic
  paper level and keep whichever fit explains the pixels better.

The fit is Levenberg–Marquardt on the pixel residuals with Huber weights.
Saturated pixels (`saturation_level`, default 250) and their 8-neighbours are
excluded. A weak prior keeps `C` near identity, which only matters when blur
erases the code band and the marker's in-plane rotation is unobservable.

## Printed radii

Printed boards do not always match their nominal stroke width, and a code band
modelled at the wrong radius biases the center. With `estimate_print_radii`
(default on), a first pass fits the three inner radii freely on up to 16
markers and the precision-weighted consensus is used for all final fits. The
estimate is reported in `DetectionDiagnostics::photometric_print_radii_mm`.

## Outputs

| Output | Where |
|---|---|
| Refined center | `DetectedMarker::center` |
| Which estimator produced the center | `DetectedMarker::center_source` (`outer_ellipse`, `projective_center`, `photometric`) |
| 2×2 center covariance (px²) | `DetectedMarker::center_cov` (photometric centers only) |
| Fitted blur, contrast, residual, status | `MarkerDiagnostics::photometric` |

The covariance is the inverse Fisher information of the fit with the noise
level estimated from the fit's own residuals. It describes the effect of pixel
noise. It does not include model mismatch: a lens PSF that is not Gaussian, a
non-linear sensor response or a glare halo the background cannot follow add
error that the covariance does not see. Use it to weight markers relative to
one another, and check its absolute scale against your own data.

A fit is rejected — and the marker keeps its previous center and
`center_source` — when it does not converge, moves more than
`max_center_shift_px` (default: half the outer radius), has an implausible blur
or contrast, or leaves a residual more than three times the image's typical fit
residual. The reason is reported in `MarkerDiagnostics::photometric.status`.

## Scope

- Coded targets only, and only decoded markers: the model needs the marker's
  code and its board position.
- A board homography is required; it is available whenever the global filter
  runs (the default). With a `PixelMapper`, the local board-to-image map is
  fitted through the mapper.
- The stage runs once per detection call, after scale selection, so adaptive
  detection does not pay for it per candidate. The result's homography is the
  one fitted before refinement.

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
