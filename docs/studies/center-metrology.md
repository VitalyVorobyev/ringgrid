# Paired center metrology under blur

The estimand is the image of the physical marker center in the input pixel frame. Integer coordinates are pixel centers. A signed error is estimated minus reference position. On a real capture with added blur the reference is the **same estimator** on the original image. Native rtv3d centers have no independent image-coordinate truth.

## Reproduce

Run from the RingGrid workspace. The synthetic matrix needs the `study` feature. The real run reads the private IPGS plate, the three-camera ring-grid dataset under `../calibration-rs/privatedata/datasets`, and the rtv3d strips under `data/rtv3d_0`.

```sh
export CENTER_MATRIX_OUT="$PWD/tools/out/center_metrology/synthetic_independent_final.jsonl"
cargo test --release -p ringgrid --features study --test center_metrology_matrix -- --ignored --nocapture
cargo build --release -p ringgrid --features cli,study --example metrology_frame
.venv/bin/python tools/metrology_real.py --jobs 6
.venv/bin/python tools/score_center_metrology.py "$CENTER_MATRIX_OUT" \
  tools/out/center_metrology/real/observations.jsonl \
  --out tools/out/center_metrology/paired.json
```

`CENTER_MATRIX_FULL=1` expands the synthetic sweep. The default real study samples three training and three held-out captures per dataset, retaining all their cameras. Use `--captures-per-split 0` for every eligible capture. The real manifest records the target, capture split, binary hash, per-frame detection counts and confidence failures. Raw results stay under the ignored output directory.

## Interpretation

- **Synthetic truth.** The test renderer evaluates the printed ink pattern in board coordinates, projects the target and blurs the irradiance before the sensor samples it. Paired counterfactuals vary camera sampling phase, display sampling phase, pixel integration, print width, anisotropy, and the sign of a source-position-dependent blur gradient. The zero-gradient and anisotropic controls distinguish a PSF that varies along the ring from one whose width depends only on orientation.
- **Real controls.** IPGS supplies the in-focus precision plate. Its three origin dots check the absolute label assignment. The rtv.t dataset supplies an antialiased displayed target and archived detector correspondences. A centered discrete Gaussian adds a known blur to both sharp controls. The six-camera rtv3d rig supplies native defocus and e-reader display rasterization together. Detector agreement can screen inconsistent IDs but cannot certify absolute identity there.
- **Estimator comparisons.** Per-image radii are the baseline. The fixed-radii fit uses the median of training-capture estimates for the same target with the outer edge as scale reference. The moment comparator measures signed ink after removing the fitted paper and contrast fields. It corrects for the projected sharp annulus and code sectors. Its finite aperture can admit neighboring ink or exclude blurred ink, and it declines clipped or saturated patches.
- **Scores.** Each method has its own output counts. Direct comparisons use common verified observations. The score includes signed displacement, radial mean/RMS/p95, same-pose repeatability per axis, and empirical coverage of the reported 95% center ellipse against synthetic truth. Paired intervals resample complete poses or captures with equal weight. The gradient reversal score is half the signed difference between opposite gradients on the same marker. The planar geometry proxy fits spatially distributed cells per view and projects held-out cells, with common support for each comparison. Distortion, target errors, and changes absorbable by a homography limit this proxy.

## Measurements and decision

The bounded run at base revision `94fd64ce` produced 53,440 synthetic observation rows and 51,045 real rows. The synthetic holdout used six poses with two noise realizations per condition and target variant. The real run used 66 camera views from 18 captures. Each real dataset contributed three held-out captures, so intervals describe this small sample.

**Keep the current photometric estimator and its per-image radii policy.** On the zero-gradient synthetic control, photometric center RMS was 0.0150 px and p95 was 0.0260 px on 720 of 960 expected observations. Same-pose repeatability was 0.0092 px horizontally and 0.0103 px vertically. The moment comparator had 0.0266 px RMS on those same observations. The corresponding median times per synthetic image were 176 ms for the per-image fit, 148 ms for pooled radii, and 198 ms for the moment study including its prerequisite fit. Timings were collected alongside other benchmark runs.

| Synthetic estimator | Signed vertical gradient-reversal bias, px | 95% pose-bootstrap interval, px |
|---|---:|---:|
| Per-image radii | −0.0278 | [−0.0289, −0.0269] |
| Training-pooled radii | −0.0281 | [−0.0292, −0.0271] |
| Corrected ink moment | −0.0288 | [−0.0306, −0.0272] |

The table uses 717 matched opposite-gradient observations per method. The per-image estimator’s reported 95% ellipse covered truth for 91% of verified zero-gradient observations and 37% under either gradient direction. Its repeatability stayed near 0.01 px per axis. This gap measures systematic displacement outside the estimated sampling uncertainty. The existing confidence score also accounts for a gradient-bias term.

Print geometry affected blur and center accuracy differently. With the coded stroke widened by 0.18 mm and zero gradient, nominal geometry gave 0.0290 px center RMS and +0.118 px mean blur error. Per-image radii gave 0.0154 px and +0.017 px. On the altered plain target, estimating radii lowered mean blur error from +0.365 px to +0.008 px while changing center RMS by only 0.0005 px.

**Sharp real controls.** Each table entry compares two estimators on the same markers, with each estimator paired to its own sharp-image measurement. The added Gaussian had σ = 1.5 px.

| Target | Compared estimator | Paired markers | Per-image drift RMS, px | Compared drift RMS, px |
|---|---|---:|---:|---:|
| IPGS | Pooled radii | 859 | 0.0176 | 0.0176 |
| IPGS | Ink moment | 830 | 0.0173 | 0.0222 |
| rtv.t | Pooled radii | 579 | 0.0331 | 0.0383 |
| rtv.t | Ink moment | 139 | 0.0308 | 0.0316 |

All 12 selected IPGS views passed the independent origin-dot check (`tools/metrology_real.py --verify-ipgs-origin`). The rtv.t sharp cohort contained 858 archive anchors. The per-image run agreed on 771, missed 85, failed one fit and disagreed on one ID. The moment estimate was available on a much smaller subset. A single sharp exposure per view supports a blur-drift measurement; real same-pose noise repeatability remains unmeasured.

**Native rtv3d.** On 18 held-out tiles, per-image and pooled radii each supplied accepted centers on 1,322 of 1,357 projective anchors. Their common-center RMS displacement was 0.0031 px. Nominal radii supplied 1,287 centers and differed from per-image by 0.1785 px RMS. The moment comparator supplied 1,296 and differed by 0.1871 px RMS, rising to 0.3076 px in the top band. Across the full output, per-image and pooled each reported 1,447 markers including two center fallbacks and 123 outside the anchor cohort. Display aliasing and physical defocus remain confounded. These data establish the size of the disagreement and give no absolute reference for choosing its direction.

## Sparse calibration follow-on

The gradient reversal justified a four-view synthetic experiment in the adjacent calibration-rs checkout. Its `photometric_planar_study.py` bench example fits three shared pinhole parameters, one pose per view, and a nonlinear Gaussian width (optionally a slope). It solves local brightness and contrast conditionally. Two views used RingGrid absolute labels. Two needed coarse setup geometry for cell correspondence. The 36 spatially held-out markers belong to the fitted views.

| Rendered condition | RingGrid-initialized center fit | Constant-width patch fit | Varying-width patch fit |
|---|---:|---:|---:|
| Alternating blur gradient | 0.1804 px | 0.0550 px | 0.0573 px |
| Constant blur control | 0.1789 px | 0.0560 px | 0.0559 px |

Values are held-out geometric projection RMS. Under the gradient, adding a slope reduced held-out pixel RMS from 3.06 to 2.93 gray levels while geometric RMS increased. The known focal length was 530 px. The three fits recovered 529.84, 528.71 and 528.57 px respectively. The tilted-view Jacobians had full numerical column rank; the frontal control lost three directions. These measurements support the patch-fit prototype for further calibration research. They give no basis for promoting a free blur slope or claiming improved focal accuracy.

The target matrix and the 102-tile rtv3d regression benchmark passed all 54 thresholds. The photometric precision sweep passed its established wrong-output allowances and recall floors. The 0.03/0.05 px synthetic center gates and release entry-point goldens also passed. Separate pre/post comparisons reproduced 24 target artifacts byte for byte and five detection outputs with timing fields excluded.
