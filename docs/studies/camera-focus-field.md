# A shared focus-band direction for rtv3d

Can marker blur elsewhere in a capture predict the blur gradient at a ring well enough to correct its center? This study evaluates a camera-specific band orientation. The detector and its output stay as the baseline. Center corrections are applied only in the study and retain the original IDs.

## Real captures

The input is the adaptive photometric output on all 17 strips in the existing rtv3d benchmark. Nine alternating strips, ordered by numeric strip ID, train the orientation independently for each camera. The other eight supply 48 test tiles across six cameras. Each tile is divided into a 4×4 array of spatial blocks and scored over four interleaved holdouts. Every tested marker's σ and all other measurements in its block are excluded from the fit that predicts its σ and gradient. Only markers with a released blur estimate participate.

The field projects image coordinates onto the learned normal. A capture can have constant blur, a single slope, or two opposite slopes meeting at a focus minimum. A BIC penalty selects among these using only available support. Searching the training angles at 2° spacing gives each training view equal weight after normalizing by its blur variance. The reference local estimator fits a plane to the nine nearest available markers with the detector's spread check. The two additional controls fix the normal to image vertical or estimate it from the available portion of each test tile.

| Camera | Learned normal, degrees from image +x toward +y | Held-out markers | Field σ RMSE, px | Neighbor σ RMSE, px |
|---|---:|---:|---:|---:|
| 0 | 92 | 708 | 0.311 | 0.497 |
| 1 | 88 | 660 | 0.367 | 0.556 |
| 2 | 94 | 653 | 0.347 | 0.561 |
| 3 | 92 | 675 | 0.286 | 0.502 |
| 4 | 90 | 636 | 0.351 | 0.544 |
| 5 | 82 | 598 | 0.316 | 0.505 |

Across 3,930 markers, the field predicted fitted σ with an RMSE of **0.330 px**, compared with **0.528 px** for neighbors and **1.364 px** for a tile mean. It beat the neighbor estimator on 46 of 48 tiles. The vertical control gave **0.370 px**, and learning the angle afresh per tile gave **0.323 px**. The capture-wide profile accounts for most of the gain. The resulting proposed center shifts had camera-wise medians of 0.039–0.049 px and p95 values of 0.095–0.156 px. These scores evaluate predictions of the detector's measured blur, which can include e-reader display rasterization. Absolute rtv3d center error and optical PSF are unknown.

## Geometric test

In the synthetic test the estimator sees only accepted center positions and fitted blur widths. The independently rendered center and gradient are used for scoring. The correction adds 1.24 σ ∇σ in image pixels. The oracle uses the renderer's width and derivative. Whole-pose bootstrap intervals keep noise realizations and markers from the same pose together.

| Held-out condition | Verified / expected | Uncorrected center RMS, px | Neighbor correction, px | Field correction, px | Oracle, px |
|---|---:|---:|---:|---:|---:|
| Constant isotropic | 720 / 960 | 0.01502 | 0.01502 | 0.01502 | 0.01502 |
| Constant anisotropic | 720 / 960 | 0.01905 | 0.01901 | 0.01904 | 0.01905 |
| Positive linear gradient | 719 / 960 | 0.03141 | 0.01819 | 0.01819 | 0.01817 |
| Negative linear gradient | 718 / 960 | 0.03240 | 0.01922 | 0.01916 | 0.01911 |
| Moving two-sided band | 239 / 240 | 0.07303 | 0.05679 | **0.04261** | 0.02839 |

For the moving band, a separate three-pose training set recovered the rendered 82° normal. Six held-out poses varied focus position and slope, with two noise realizations each. The field recovered **68% of the available oracle RMS gain**. Mean signed error along the direction of increasing blur went from −0.0541 px uncorrected to −0.0406 px with neighbors and −0.0231 px with the field. Its p95 center error fell from 0.1475 to 0.0893 px. The paired 95% interval for its RMS improvement over neighbors was **0.0095–0.0185 px**. With all released marker widths available for the correction, the field and the nine-point neighbor plane gave **0.03874** and **0.04842 px** RMS respectively. This full-density comparison includes the query marker's own σ, matching how the production neighbor gradient uses it.

On monotonic linear gradients both estimators already obtain almost all of the oracle gain. The two-sided case is more discriminating because a neighborhood can straddle the minimum. It covers one coded target, one band shape, and six held-out poses. The synthetic results support testing a capture-wide correction against independent real geometry before adopting it. The real-data ablation gives little reason to require prior per-camera training for a first trial.

## Reproduce

Run from the RingGrid root. The directional and moving-band modes are optional variants of the synthetic center-metrology test. For the default held-out matrix, see [the paired study](center-metrology.md). For the real input, run the existing rtv3d benchmark with photometric refinement and supply its detections directory through `--real`.

```sh
CENTER_MATRIX_DIRECTIONAL_TRAIN=1 cargo test --release -p ringgrid --features study \
  --test center_metrology_matrix -- --ignored --nocapture
CENTER_MATRIX_FOCUS_BAND=1 cargo test --release -p ringgrid --features study \
  --test center_metrology_matrix -- --ignored --nocapture
.venv/bin/python tools/camera_focus_field.py \
  --band tools/out/center_metrology/synthetic_focus_band.jsonl
```

The scorer writes per-camera and per-condition metrics to `tools/out/center_metrology/focus_field/results.json`. Its figure overlays held-out marker widths from eight strips per camera. The line in each panel indicates the learned band orientation; its position is arbitrary.
