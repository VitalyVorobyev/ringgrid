# Why Rings?

Ring markers are a deliberate design choice that addresses fundamental limitations of other fiducial patterns. This chapter explains the three key advantages that motivate the ring geometry.

## Subpixel Edge Detection

The boundary of a circle (or its perspective projection — an ellipse) produces a strong, continuous intensity gradient at every point along the edge. This is fundamentally different from corner-based features:

- **Corners** (checkerboard intersections, square marker corners) are localized features. Their position is estimated from a small neighborhood, and subpixel accuracy depends on the sharpness of the corner response.
- **Ring edges** are extended features. The detector can sample hundreds of edge points along radial rays emanating from the approximate center, then fit an ellipse to all of them simultaneously.

The ringgrid detector uses **gradient-based edge sampling**: for each candidate center, it casts radial rays outward at uniformly spaced angles and locates the intensity transition along each ray using the Scharr gradient magnitude. This yields a dense set of edge points — typically 60–200 per marker — distributed around the full circumference.

These edge points are then passed to the **Fitzgibbon direct least-squares ellipse fitter**, which solves a constrained eigenvalue problem to find the best-fit ellipse in a single algebraic step (no iterative optimization). The resulting ellipse center achieves subpixel accuracy because:

1. The fit uses many points (overdetermined system), averaging out per-point noise
2. Points are distributed around the full ellipse, constraining all five parameters
3. The algebraic constraint guarantees an ellipse (not a hyperbola or degenerate conic)

In synthetic benchmarks with blur σ = 0.8 px, ringgrid achieves **mean center error of 0.054 px** with projective center correction enabled.

## Projective Center Correction

Under perspective projection, a circle in 3D projects to an ellipse in the image. A critical subtlety: **the center of the projected ellipse is not the projection of the circle's center**. This projective bias grows with the viewing angle and distance from the optical axis.

For corner-based markers, this is not an issue — corners project correctly. But for any detector that fits a conic (ellipse) to estimate a circle's center, the projective bias introduces systematic error.

ringgrid solves this problem using **two concentric rings**. When both the outer and inner ellipses are successfully fitted, the detector has two conics that correspond to two concentric circles in 3D. The key mathematical insight is:

> The **conic pencil** spanned by two concentric circle projections contains degenerate conics (pairs of lines) that intersect at the true projected center.

This is the projective center recovery algorithm (detailed in the [Mathematical Foundations](math/projective-center-recovery.md) chapter). It recovers the unbiased projected center **without requiring camera intrinsics** — purely from the geometry of the two fitted ellipses.

The improvement is measurable: on clean synthetic images, projective center correction reduces the mean center error from **0.072 px to 0.054 px** — a 25% improvement in localization accuracy.

## Large Identification Capacity

Each marker carries a unique identity encoded in a 16-sector binary code band between the inner and outer rings. The codebook contains **893 codewords** selected to maximize the minimum Hamming distance between any pair.

This design provides several advantages over other encoding approaches:

| Property | ringgrid | ArUco 4x4 | ArUco 6x6 | Checkerboard |
|---|---:|---:|---:|---:|
| Unique IDs | 893 | 50 | 250 | 0 |
| Rotation invariant | Yes | No (4 orientations) | No | N/A |
| Error tolerance | Hamming distance | Hamming distance | Hamming distance | N/A |
| Encoding mechanism | Angular sectors | Binary grid | Binary grid | None |

Key properties of the coding scheme:

- **Rotation invariance**: The 16-sector code is sampled relative to the marker's geometry, and the decoder tries all 16 cyclic rotations. No marker orientation assumption is needed.
- **Polarity invariance**: The decoder also checks the inverted contrast pattern, handling both dark-on-light and light-on-dark printing.
- **Error tolerance**: The minimum Hamming distance in the codebook provides robustness against individual sector misreads due to blur, noise, or partial occlusion.

## Comparison with Other Calibration Targets

### Checkerboards

Checkerboards are the classic calibration target. They offer excellent corner localization via saddle-point refinement, but have no identity encoding. This means:

- The full board (or a known subset) must be visible for correspondence
- Automatic detection fails with partial occlusion
- Multiple boards in one image cannot be disambiguated

ringgrid markers each carry a unique ID, enabling detection under partial visibility and multi-board setups.

### ArUco / AprilTag

ArUco and AprilTag markers encode identity in a binary grid printed inside a square border. Detection relies on finding the square contour and computing a homography from its four corners. Limitations:

- Corner accuracy is limited by contour detection precision
- The square geometry provides only 4 points per marker for center estimation
- Dense packing is limited by the need for white borders between markers

ringgrid markers provide hundreds of edge points per marker, denser packing on a hex lattice, and rotation-invariant coding.

### Concentric Circles (CCT)

Concentric circle targets (e.g., Huo et al. 2020) share some advantages with ringgrid — subpixel edge fitting and projective center correction. ringgrid adds:

- Binary coding for unique identification (CCTs typically rely on geometric arrangement for correspondence)
- A hex lattice layout for maximum marker density
- A large codebook (893 IDs) enabling scalable target designs

<!-- TODO: Side-by-side comparison figure: checkerboard vs ArUco vs ringgrid -->
