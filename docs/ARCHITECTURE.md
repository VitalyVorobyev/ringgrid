# ConicMark Architecture

## Overview

ConicMark detects circle/ring-based calibration targets in images from
Scheimpflug cameras where strong, anisotropic defocus blur is present.
Typical conditions: marker outer diameter ~20 px, blur kernel scale ~10 px.

## Pipeline Stages

```
Image → Preprocess → Edges → Conic Fit → Lattice → Refine → Codec → Output
```

### 1. Preprocess (`core::preprocess`)

- **Illumination normalization**: subtract local-mean background to handle
  uneven illumination across the field (common in Scheimpflug setups where
  the focal plane is tilted).
- **Band-pass filtering**: DoG or LoG tuned to the expected ring spatial
  frequency, suppressing both low-frequency shading and high-frequency noise.

### 2. Edge Detection (`core::edges`)

- Gradient computation (Sobel/Scharr) followed by non-maximum suppression.
- Hysteresis thresholding for connected edge segments.
- **Sub-pixel refinement** via parabolic interpolation of gradient magnitude
  perpendicular to the edge direction.
- Edge points are grouped into candidate arcs for ellipse fitting.

### 3. Conic / Ellipse Fitting (`core::conic`)

- **Direct least-squares conic fit** (Fitzgibbon et al., 1999): solves a
  constrained eigenvalue problem enforcing the ellipse condition B²−4AC < 0.
- **RANSAC wrapper** for outlier-robust fitting: samples 6-point minimal
  subsets, scores by inlier count under normalized algebraic distance.
- Point normalization (centroid shift + isotropic scaling) for numerical
  stability.
- Utilities: conic ↔ ellipse conversion, algebraic/Sampson residuals,
  validity checks.

### 4. Lattice / Grid Analysis (`core::lattice`)

- Build a nearest-neighbor graph from detected ellipse centers.
- Extract two dominant grid directions from the neighbor displacement vectors.
- Estimate **vanishing points** for each grid direction.
- Compute the **vanishing line** as the join of the two vanishing points.

#### Why affine rectification for center bias

Under perspective projection, the image of a circle is an ellipse, but the
ellipse center does **not** coincide with the projection of the circle's 3D
center. This systematic error is called *center bias* or *eccentricity error*
and can reach several pixels for markers near the image periphery under
strong perspective.

The key insight is that center bias arises from the **projective** (as opposed
to affine) component of the projection. If we can remove the projective
component — i.e., perform an **affine rectification** — then ellipse centers
become unbiased estimates of the projected circle centers.

The affine rectification homography is:

```
H = [ 1  0  0 ]
    [ 0  1  0 ]
    [ l₁ l₂ l₃]
```

where **l** = (l₁, l₂, l₃) is the vanishing line. This H maps the vanishing
line back to the line at infinity, removing the projective distortion.

After rectification, per-marker refinement proceeds in affine-rectified
coordinates, and the final centers are mapped back to the original image
frame.

### 5. Refinement (`core::refine`)

- **Shared-center dual-ring model**: each marker has two concentric circles
  (inner/outer ring) that project to two ellipses sharing the same center
  (in rectified space).
- Levenberg–Marquardt minimization of robust (Huber/Tukey) geometric
  residuals from edge points to the nearest ring boundary.
- Covariance estimation for uncertainty propagation.

### 6. Codec (`core::codec`)

- Markers encode a unique ID in a pattern of bright/dark sectors between
  the inner and outer rings.
- Intensity sampling along elliptical arcs, adaptive thresholding,
  cyclic-rotation canonicalization, and codebook lookup.

## Crate Structure

```
crates/
  conicmark-core/    # All algorithms, no I/O beyond image loading
  conicmark-cli/     # CLI interface: load image, run pipeline, write JSON
tools/
  gen_synth_dataset.py   # Synthetic dataset generation
  score_run.py           # Scoring / evaluation harness
  viz_debug.py           # Debug visualization
docs/
  ARCHITECTURE.md        # This file
```

## Detection Output Schema

```json
{
  "markers": [
    {
      "center": [x, y],
      "semi_axes": [a, b],
      "angle": theta,
      "id": 42,
      "fit_residual": 0.12
    }
  ],
  "image_size": [width, height]
}
```

## Milestones

- **M0** (current): Workspace setup, conic fitting primitives, CLI skeleton.
- **M1**: Synthetic dataset generator (Python) with ground truth.
- **M2**: Edge detection, full detection pipeline, lattice analysis.
- **M3**: Per-marker refinement (LM), dual-ring model.
- **M4**: Codec (marker ID encoding/decoding).
- **M5**: Performance optimization, benchmarks, documentation.
