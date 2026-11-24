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

## Exposure and saturation (ALGO-022)

Cledat, Rufener and Cucci (Pattern Recognition, 2021) showed that saturation combined with the PSF moves the edges of concentric-circle targets as exposure changes. This section tests three expectations about the photometric stage:

- the per-image print radii absorb an exposure-dependent stroke width;
- a symmetric width change leaves the centre in place;
- masking saturated pixels and their 8-neighbours (grey level ≥ 250) keeps the fit sound.

The study measures the detector at base revision `fc66128f`, the per-image radii consensus and second-order edge of ALGO-021a. The Results, Controls and Answers below describe that detector, before the saturation gate. [Saturation gate](#saturation-gate) gives the gate added in response and its before/after numbers.

### Reproduce

```sh
cargo test --release -p ringgrid --test photometric_target_matrix \
  photometric_exposure_series -- --ignored --nocapture
```

Each cell renders one noise-free radiance. Paper is 150 and ink 70 grey levels at the image centre, with a ±15 % left-to-right illumination gradient. The cells cover the coded hex target and the plain hex target with origin dots, at outer radius 17 px, contrast 80, Gaussian blur 1 and 2 px, and tilt 0° and 30°. The radiance is scaled by gains 0.25, 0.5, 1, 1.5, 2, 2.5 and 3. Gain 0.25 leaves 20 grey levels of contrast. At gain 3 the paper is clipped and only the darkest ink stays below 250. Each scaled radiance is exposed through a camera response with γ = 1, 0.8 or 1.25, plus σ = 2 noise and 8-bit clipping. Six noise seeds are used, and the noise field is the same at every exposure of a seed.

Every exposure is compared with the nominal exposure (gain 1, γ = 1) on the same seed and the same markers. A paired change is the per-seed mean over the markers fitted in both exposures. The ± figures are standard errors over seeds. Radii are converted to pixels at the image centre. The full matrix takes about 80 s with `MATRIX_THREADS=12`, and the output is deterministic. `MATRIX_EXPOSURES`, `MATRIX_GAMMAS`, `MATRIX_BLURS`, `MATRIX_TILTS`, `MATRIX_ILLUM_GRADIENT`, `MATRIX_SEEDS` and `MATRIX_FIXED_RADII` select the controls quoted below.

### Results

The table summarizes all eight cells; each range runs over the cells. Masked is the mean saturated fraction of each attempted fit's patch. Δr is the paired change of the free print radii in pixels. Positive Δr0 and Δr2 with negative Δr1 mean thinner dark strokes; Δr0 is the hole edge. Δσ is the paired change of fitted σ. Δc is the largest paired mean centre change on either axis. The last column compares photometric centre RMS with projective-centre RMS on the same images.

| Exposure | Masked | Radii consensus | Δr (px) | Δσ (px) | Δc (px) | Photometric / projective RMS (px) |
|---|---:|---|---|---:|---|---|
| γ 1, gain 0.25 | 0 | 48/48 | \|Δr\| ≤ 0.022 (≤ 2.4 SE) | −0.008 … +0.002 | ≤ 0.003 (≤ 1.6 SE) | 0.029–0.065 / 0.17–0.64 |
| γ 1, gain 0.5 | 0 | 48/48 | ≤ 0.016 | ≤ 0.003 | ≤ 0.0012 ± 0.0004 | 0.015–0.035 / 0.11–0.46 |
| γ 1, gain 1 (nominal) | 0 | 48/48 | — | — | — | 0.009–0.020 / 0.07–0.39 |
| γ 1, gain 1.5 | 0.02–0.05 | 48/48 | ≤ 0.008 | ≤ 0.004 | ≤ 0.0003 | 0.007–0.016 / 0.06–0.35 |
| γ 1, gain 2 | 0.60–0.74 | coded 24/24, plain 17/24 | coded Δr0 +0.01…+0.10, Δr2 +0.01…+0.12, Δr1 −0.02…+0.01; plain Δr0 +0.26, +0.11 (blur 1), +0.03, −0.30 (blur 2, 3/6 and 2/6 consensus) | −0.065 … +0.025 | coded blur 2: −0.0029 ± 0.0002, +0.0037 ± 0.0003; others ≤ 0.0006 | 0.011–0.029 / 0.05–0.20 |
| γ 1, gain 2.5 | 0.74–0.83 | coded 24/24, plain 0/24 | coded Δr2 +0.07…+0.36 | −0.052 … +0.032 | ≤ 0.0070 ± 0.0007 | 0.020–0.069 / 0.04–0.15 (gated: 0.020–0.061) |
| γ 1, gain 3 | 0.70–0.90 | 10/48 | erratic | −0.15 … +0.25 | ≤ 0.04 ± 0.02 | blur 1: 0.22–0.38 / 0.08–0.17 (gated: 0.060–0.097); blur 2: 0.06–0.09 / 0.09–0.11 (coded blur 2: none detected) |
| γ 0.8, gain 1 | 0 | 48/48 | coded Δr0 +0.06…+0.10, Δr2 +0.08…+0.11, Δr1 −0.02…−0.005; plain Δr0 +0.07 (blur 1), +0.16 (blur 2) | coded −0.004 … −0.013; plain +0.003 … +0.018 | ≤ 0.0007 ± 0.0002 | 0.009–0.021 |
| γ 1.25, gain 1 | 0 | 48/48 | coded Δr0 −0.08…−0.12, Δr2 −0.10…−0.13, Δr1 +0.002…+0.026; plain Δr0 −0.09 (blur 1), −0.19…−0.21 (blur 2) | coded +0.004 … +0.015; plain −0.006 … −0.024 | ≤ 0.0004 | 0.009–0.021 |

No exposure produced a wrong label. The radius changes from γ = 0.8 and γ = 1.25 agree within 0.02 px at every gain from 0.5 to 1.5, and within 0.05 px at gain 0.25.

The second table pools every exposure, response and cell. It omits the 0–30 % bins, where RMS is 0.011 px coded and 0.008 px plain. It bins photometric centres by masked fraction and compares each with the projective centre of the same marker in the same image.

| Masked | Coded: n, RMS, max (px) | Coded: reported σ, 95 % coverage | Coded: projective RMS, share worse | Plain: n, RMS, max (px) | Plain: projective RMS, share worse |
|---|---|---|---|---|---|
| none | 48,927 · 0.029 · 0.26 | 0.029 · 0.92 | 0.294 · 0.01 | 34,864 · 0.024 · 0.17 | 0.222 · 0.01 |
| 0.3–0.5 | 2,131 · 0.020 · 0.10 | 0.025 · 0.92 | 0.228 · 0.01 | 438 · 0.008 · 0.02 | 0.171 · 0.00 |
| 0.5–0.7 | 5,434 · 0.021 · 0.11 | 0.024 · 0.89 | 0.135 · 0.03 | 5,685 · 0.013 · 0.09 | 0.133 · 0.02 |
| 0.7–0.8 | 5,535 · 0.027 · 0.26 | 0.025 · 0.83 | 0.094 · 0.09 | 9,103 · 0.027 · 0.49 | 0.084 · 0.13 |
| ≥ 0.8 | 9,452 · 0.103 · 1.65 | 0.083 · 0.85 | 0.094 · 0.22 | 6,577 · 0.213 · 2.20 | 0.134 · 0.16 |

Before the gate, 64 coded and 252 plain photometric centres in the ≥ 0.8 bin were more than 0.5 px from the truth. All were accepted fits.

### Controls

- **Fixed radii** (`MATRIX_FIXED_RADII=1`, tilt 0). With the nominal radii held, γ moves the fitted σ by −0.054 px (γ 0.8) and +0.075 px (γ 1.25) on plain blur 2. That is −2.6 % and +3.7 %, against +0.018 and −0.021 px with per-image radii. γ also moves coded blur-2 centres by up to 0.0036 ± 0.0001 px, against ≤ 0.0005 px with per-image radii. At gain 2 the coded blur-2 centre change is the same either way: −0.0033 / +0.0036 px fixed, −0.0029 / +0.0037 px estimated.
- **Illumination gradient** (`MATRIX_ILLUM_GRADIENT`, coded blur 2, tilt 0, gain 2, 8 seeds). The paired x change is −0.0003 ± 0.0003 px with no gradient, −0.0029 ± 0.0002 at ±15 % and −0.0050 ± 0.0003 at ±50 %. The centre moves toward the darker, less clipped side. The plain target stays within 0.0004 px at every gradient. With no gradient, a coded y change of +0.0015 ± 0.0003 px remains at gain 2. At gain 2.5 the change is (−0.0046 ± 0.0014, +0.0061 ± 0.0007) px.
- **Code-band dipole.** Regressing each marker's paired change on its image-frame dark-sector dipole gives correlations of −0.17 to +0.11 at gain 2, so the code pattern does not explain the gain-2 shift. At gain 2.5 the correlations are +0.03 to +0.30, and the code pattern explains part of the shift.

### Answers

1. **Do the radii follow exposure?** A linear gain below clipping leaves them unchanged, within 0.022 px over gains 0.25–1.5. Clipping changes them. With the paper clipped at gain 2, the dark strokes mostly come out thinner: the hole edge moves out by 0.01–0.26 px, and the coded outer-ring inner edge by up to 0.12 px. The direction is not universal. The plain target at blur 2 gives +0.03 and −0.30 px from unstable consensus, held in only 2–4 of 6 exposures. Past 75 % masked, plain targets lose the consensus and fall back to nominal radii. Coded radii keep a consensus but drift by up to 0.36 px. A non-linear response changes the radii consistently. γ 0.8 thins the strokes and γ 1.25 thickens them, by 0.06–0.21 px, in proportion to blur and at every gain.
2. **Do the centres move?** Not under gain or response without clipping: the shifts stay ≤ 0.0007 px, with the per-image radii absorbing the response. Once the paper clips, coded centres move beyond noise. The shift is 0.003–0.004 px at gain 2, more than 10 SE, and up to 0.007 px at gain 2.5. Nominal centre RMS there is 0.018 px. The shift scales with the illumination asymmetry across a marker and points toward the less saturated side. Fixing the radii does not change it. Plain centres move by at most 0.0015 ± 0.0004 px up to gain 2.5.
3. **Is σ biased by exposure?** Not by gain below clipping (|Δσ| ≤ 0.008 px). With per-image radii, the response biases σ by up to ±0.024 px (±1.2 %). With the paper clipped at gain 2, σ changes by −0.065 to +0.025 px, mostly downward; the one positive value is within its 0.028 px SE. The worst case is plain blur 1, at −6 %, alongside a +0.26 px hole-edge shift. Under heavy clipping at gain 3, σ errors reach +0.25 / −0.23 px.
4. **Does saturation masking hold up?** It holds up to about 80 % of the patch masked. Over that range accepted centres stay at 0.008–0.027 px RMS, with 0.83–0.92 coverage, and no wrong labels occur. The 70–80 % bin already has outliers of 0.26 px (coded) and 0.49 px (plain). Beyond about 80 % masked the fit degrades, but it is still accepted. Gain 3 at blur 1 is the case where the ink itself nears the clip. There photometric RMS is 0.22–0.38 px against 0.08–0.17 px for the projective centres it replaces. Across all exposures, 316 accepted centres in the ≥ 0.8 bin are more than 0.5 px off, with a maximum of 2.2 px. The covariance grows with the error, reaching 0.08–0.18 px with 0.85–0.93 coverage, so a covariance-weighted consumer down-weights these centres. Still, the stage had no saturation gate, and it replaced a better projective centre in 16–22 % of these fits. Under the precision-over-recall rule this was a fail-open defect. The saturation gate below now closes it: no accepted centre is more than 0.5 px off, and the ≥ 0.8 bin falls to 0.055 / 0.048 px RMS.
5. **Does a non-unit response bias the radii or centres?** Yes for the radii: by 0.06–0.21 px at γ 0.8 and γ 1.25, as in answer 1. That is a fitted stroke-width change the per-image consensus is meant to absorb. Centres are not biased (≤ 0.0007 px) with per-image radii. With nominal radii, coded centres move by up to 0.0036 px.

**Defects found.**

1. **Fixed.** Fail-open at heavy saturation: photometric fits on patches more than 80 % masked were accepted with up to 2.2 px error. See [Saturation gate](#saturation-gate).
2. **Open.** A centre shift of 0.003–0.007 px on coded targets once the paper clips. It grows with the illumination asymmetry across the marker and is not absorbed by the radii. The saturation gate leaves it unchanged: at gain 2 the coded blur-2 shift is still (−0.0029 ± 0.0002, +0.0037 ± 0.0003) px.

### Saturation gate

An accepted photometric fit is now rejected when more than half its patch is masked as saturated **and** its centre σ, `√(tr Σ / 2)`, exceeds 0.08 px. Its status is `InsufficientPixels`, and the marker keeps its previous (projective) centre. Two constants hold the rule: `SATURATED_PATCH_FRACTION` and `SATURATED_MAX_CENTER_SIGMA_PX` in `detector/photometric_refine.rs`. There is no config knob, no new status, and no schema change.

**Choosing the rule.** The exposure series sweeps candidate rules over the 131,117 accepted centres of the ungated detector. That is commit `ba93e5ad`; its `src` is `fc66128f`, and the sweep ships with the test. The table below pools both kinds. "Rejected" counts fits per masked bin: 0.5–0.7, 0.7–0.8 and ≥ 0.8. Lower bins are 0 unless noted. "Kept > 0.1 px" counts all bins, with the patches more than half masked in parentheses. "Good lost" counts rejected fits that were within 0.05 px of the truth.

| Rule | Rejected | Kept > 0.5 px | Kept > 0.1 px | RMS kept (px) | Good lost |
|---|---|---:|---|---:|---:|
| none | — | 316 | 2,690 (2,087) | 0.0602 | 0 |
| masked > 0.7 | 0 · 14,631 · 16,029 | 0 | 607 (4) | 0.0256 | 24,842 |
| masked > 0.8 | 0 · 0 · 16,002 | 0 | 691 (88) | 0.0257 | 10,794 |
| masked > 0.85 | 0 · 0 · 8,971 | 36 | 1,026 (423) | 0.0321 | 4,956 |
| unmasked pixels < 250 | 0 · 211 · 10,216 | 142 | 1,260 (657) | 0.0454 | 6,438 |
| unmasked pixels < 400 | 498 · 6,069 · 15,977 | 0 | 644 (41) | 0.0254 | 16,945 |
| σ > 0.05 (any patch) | 17 · 81 · 3,384, plus 1,402 unmasked | 0 | 665 (347) | 0.0267 | 1,241 |
| σ > 0.1 (any patch) | 3 · 1 · 1,329, plus 25 unmasked | 0 | 1,602 (1,006) | 0.0306 | 73 |
| masked > 0.5 & σ > 0.03 | 422 · 731 · 7,011 | 0 | 652 (49) | 0.0258 | 3,593 |
| masked > 0.5 & σ > 0.05 | 17 · 81 · 3,384 | 0 | 950 (347) | 0.0278 | 730 |
| masked > 0.5 & σ > 0.065 | 3 · 21 · 2,285 | 0 | 1,212 (609) | 0.0291 | 294 |
| **masked > 0.5 & σ > 0.08** | **3 · 4 · 1,721** | **0** | **1,411 (808)** | **0.0298** | **136** |

A masked-fraction cut alone removes the outliers only by also discarding most good saturated fits: 68 % of the ≥ 0.8 bin is within 0.05 px. An absolute pixel count depends on marker scale and does no better. The reported σ separates the bad fits, because the covariance stays roughly honest under saturation (85–93 % coverage). Conditioning on a saturated patch keeps the gate away from noisy fits with no masked pixels. Those are honest: in the darkest exposures photometric RMS is 0.029–0.084 px, against 0.17–0.66 px projective.

The σ limit is where a saturated fit stops improving on the centre it replaces. On patches more than half masked, compared with the projective centre of the same marker:

| Reported σ (px) | n | Photometric RMS (px) | Projective RMS (px) | Photometric worse |
|---|---:|---:|---:|---:|
| < 0.02 | 23,863 | 0.020 | 0.105 | 0.07 |
| 0.02–0.03 | 3,681 | 0.038 | 0.113 | 0.18 |
| 0.03–0.04 | 1,661 | 0.049 | 0.124 | 0.17 |
| 0.04–0.05 | 740 | 0.067 | 0.123 | 0.28 |
| 0.05–0.065 | 503 | 0.082 | 0.115 | 0.29 |
| 0.065–0.08 | 205 | 0.085 | 0.119 | 0.36 |
| 0.08–0.1 | 116 | 0.123 | 0.116 | 0.48 |
| ≥ 0.1 | 162 | 0.259 | 0.133 | 0.68 |

Below 0.08 px the photometric centre is still the better one on average, so rejecting it would worsen the result. From 0.08 px the two are level, and above 0.1 px the photometric centre is clearly worse. The 808 saturated centres with more than 0.1 px error that the gate keeps lie in the bins where the projective alternative is no better.

**After the gate** (same series):
- **Accepted centres:** 129,389, exactly the set the sweep predicted. None is more than 0.5 px off. The rejections per bin are coded 3 · 3 · 767 and plain 0 · 1 · 954.
- **≥ 0.8 masked bin:** RMS 0.055 px coded (from 0.103) and 0.048 px plain (from 0.213). The maximum error is 0.41 / 0.46 px (from 1.65 / 2.20), against 0.092 / 0.135 px for the projective centres of the same markers. The photometric centre is now the worse one in 20 % / 6 % of these fits (from 22 % / 16 %).
- **Gain 3 at blur 1:** photometric RMS 0.060–0.097 px (from 0.22–0.38) against projective 0.084–0.165 px. Coded at tilt 0 is the closest case, 0.086 vs 0.084 px.
- **Other exposures:** radii consensus, wrong labels (none) and every exposure with at most half of each patch masked are unchanged.

**No change elsewhere.** The gate cannot touch a fit with at most half its patch masked. The following runs are byte-identical before and after the gate, excluding timing lines:
- `photometric_target_matrix`: the default cells and the full matrix (378 cells, 6 seeds);
- `blur_center_accuracy`: default tests and `blur_center_benchmark`, whose acceptance scene has a saturated glare blob;
- `blur_gradient`: default tests and `blur_gradient_bias_matrix`;
- `gain_invariance`;
- `precision_audit precision_fast_subset`, with default refinement and with `RINGGRID_PRECISION_REFINE=photometric`.

**Limits.** These are synthetic scenes with a Gaussian PSF. Saturation is a hard clip at 255 with no blooming, smear or charge spill. Noise is additive with fixed σ, with no shot noise and no exposure-dependent noise. The response is a pure power law. The only lighting non-uniformity is a linear left-to-right gradient. The study covers a single marker scale of 17 px. No real exposure series was measured. That remains the real half of ALGO-022.
