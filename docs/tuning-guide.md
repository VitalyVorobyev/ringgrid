# ringgrid Tuning Guide

This guide maps common imaging challenges to the specific configuration knobs that
address them. All settings live in the JSON config file passed via `--config`.

---

## 1. Blurred / Defocused Images

Blur reduces the sharpness of the inner ring transition, causing theta-consistency
gating to reject inner estimates and weakening gradient proposals.

| Setting | Recommended Value | Rationale |
|---|---|---|
| `marker_spec.min_theta_consistency` | `0.18` | Default 0.25 rejects blurry inner peaks; 0.18 accepts them |
| `inner_fit.min_points` | `12` | Fewer but reliable points suffice under blur |
| `inner_fit.ransac.inlier_threshold` | `2.0` | Wider tolerance for spread edge points |
| `completion.enable` | `true` | Recover missed proposals via homography projection |
| `inner_as_outer_recovery.enable` | `true` | Fix markers where inner ring was mistaken for outer |

```json
{
  "marker_spec": { "min_theta_consistency": 0.18 },
  "inner_fit": {
    "min_points": 12,
    "ransac": { "inlier_threshold": 2.0 }
  },
  "completion": { "enable": true },
  "inner_as_outer_recovery": { "enable": true }
}
```

---

## 2. Low Contrast

When marker-to-background contrast is low, code decoding and edge sampling
both suffer from noisy sector samples.

| Setting | Recommended Value | Rationale |
|---|---|---|
| `decode.min_decode_contrast` | `0.01` | Default 0.03 rejects very low-contrast markers |
| `decode.n_radial_rings` | `5` | More rings average out noise |
| `decode.samples_per_sector` | `7` | More angular samples per sector for robustness |

```json
{
  "decode": {
    "min_decode_contrast": 0.01,
    "n_radial_rings": 5,
    "samples_per_sector": 7
  }
}
```

---

## 3. High Noise (Sensor / Quantisation)

Random noise inflates Sampson residuals and reduces RANSAC inlier counts.
Widen the RANSAC tolerance windows.

| Setting | Recommended Value | Rationale |
|---|---|---|
| `outer_fit.ransac.inlier_threshold` | `2.5` | Accept noisier outer edge points as inliers |
| `inner_fit.ransac.inlier_threshold` | `2.5` | Same for inner edge fitting |
| `outer_fit.ransac.min_inliers` | `5` | Fewer inliers required to declare a fit valid |

```json
{
  "outer_fit": {
    "ransac": { "inlier_threshold": 2.5, "min_inliers": 5 }
  },
  "inner_fit": {
    "ransac": { "inlier_threshold": 2.5 }
  }
}
```

---

## 4. Partial Occlusion / Near-Border Markers

Markers near the image edge or partially occluded have large angular gaps in
their edge point sets.

| Setting | Recommended Value | Rationale |
|---|---|---|
| `outer_fit.max_angular_gap_rad` | `2.0` | Allow fits with up to ~115° arc missing |
| `completion.min_arc_coverage` | `0.25` | Attempt completion with only 25% arc visible |
| `completion.enable` | `true` | Necessary to recover occluded-proposal failures |
| `completion.image_margin_px` | `5.0` | Reduce margin to allow near-border completions |

```json
{
  "outer_fit": { "max_angular_gap_rad": 2.0 },
  "completion": {
    "enable": true,
    "min_arc_coverage": 0.25,
    "image_margin_px": 5.0
  }
}
```

---

## 5. Strong Lens Distortion

Radial distortion causes circular rings to appear elliptical with systematic
shape error. Use projective center correction or a calibrated camera model.

| Approach | CLI Flag / Setting | Notes |
|---|---|---|
| Projective center correction | `--circle-refine-method projective-center` | Recommended first step; no calibration file needed |
| Calibrated camera model | `--camera-model <intrinsics.json>` | Enables two-pass distortion-corrected detection |
| Self-undistort | `self_undistort.enable: true` in config | Estimates division-model distortion from detected ellipses |

```json
{
  "self_undistort": { "enable": true }
}
```

Or via CLI:
```bash
ringgrid detect --image img.png --circle-refine-method projective-center
```

---

## 6. Illumination Gradient

Scharr gradient voting is DC-invariant: a smooth illumination gradient does
not affect the ring edge response because only local intensity changes matter.
No specific tuning is required.

If gradient contrast is globally low in dark image regions (e.g. strong vignetting),
the proposal stage may miss markers in those areas. In that case:

| Setting | Recommended Value | Rationale |
|---|---|---|
| `marker_scale.diameter_max_px` | Increase by ~20% | Allows the gradient voting scale to reach larger apparent diameters near dark corners |
| `completion.enable` | `true` | Recover missed proposals from H-projected positions |

```json
{
  "marker_scale": { "diameter_max_px": 72.0 },
  "completion": { "enable": true }
}
```

---

## Quick-Reference Table

| Symptom | Key Settings |
|---|---|
| `inner_fit_reason: estimate_not_ok` | Reduce `marker_spec.min_theta_consistency` |
| Many `inner_fit_status: failed` | Enable `inner_as_outer_recovery`, widen `inner_fit.ransac.inlier_threshold` |
| Missed markers in blurry regions | Enable `completion`, reduce `completion.min_fit_confidence` |
| Wrong IDs | Lower `id_correction.seed_min_decode_confidence`, enable `id_correction` |
| Low overall confidence | Check `decode.min_decode_confidence` and `outer_fit.max_angular_gap_rad` |
| Ellipses look systematically off-center | Use `--circle-refine-method projective-center` |
