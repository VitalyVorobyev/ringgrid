# H-Guided Refinement

After the global homography filter establishes a valid board-to-image mapping, the detector uses the homography to refine each inlier marker's position.

## Motivation

The initial marker detection starts from gradient-voting proposals, which have limited spatial accuracy. The estimated homography provides a better prior for where each marker should be — the H-projected board position is typically closer to the true marker center than the original proposal.

## Algorithm

For each inlier marker from the global filter:

1. **Project**: use the homography to map the marker's known board position to image coordinates, producing a refined center prior
2. **Refit**: run the full local fit pipeline at the refined position:
   - Edge sampling along radial rays from the new center
   - RANSAC ellipse fitting
   - Inner ellipse estimation
3. **Accept or reject**: the refined fit is accepted only if it produces better quality metrics than the original fit (lower RMS residual or more inliers). Otherwise the original detection is preserved.

## Second Projective Center Pass

After refinement, the projective center correction is reapplied (pass 2 of 3). Since refinement may have produced new ellipse fits with different geometry, the center correction must be recomputed from the updated inner/outer ellipse pair.

## When Refinement Helps

H-guided refinement is most beneficial when:

- The initial proposal was offset from the true center (common at image periphery)
- Perspective distortion is significant (markers appear as elongated ellipses)
- The gradient voting accumulator has ambiguous peaks

In clean, low-perspective images, refinement typically confirms the original fit with minimal change.

**Source**: `detector/refine_h.rs`, `pipeline/finalize.rs`
