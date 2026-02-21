# Completion & Final Refit

The completion stage attempts to detect markers that the initial pipeline missed — typically markers at the image periphery, under heavy blur, or with low contrast. It runs after projective-center correction, structural ID correction, and global homography filtering. It uses the homography to predict where missing markers should be and attempts conservative local fits at those locations.

## Completion Algorithm

For each marker ID in the `BoardLayout` that was **not** detected in the initial pipeline:

1. **Project**: use the homography to map the board position to image coordinates
2. **Boundary check**: skip if the projected position is too close to the image edge (within `image_margin_px`)
3. **Local fit**: run edge sampling and RANSAC ellipse fitting within a limited ROI (`roi_radius_px`) around the projected center
4. **Decode**: attempt code decoding at the fitted position
5. **Gate**: accept the detection only if it passes conservative quality gates

## Conservative Gates

Completion uses stricter acceptance criteria than the initial detection to avoid false positives:

| Parameter | Default | Purpose |
|---|---|---|
| `reproj_gate_px` | 3.0 px | Max distance between fitted center and H-projected position |
| `min_fit_confidence` | 0.45 | Minimum fit quality score |
| `min_arc_coverage` | 0.35 | Minimum fraction of rays with valid edge detections |
| `roi_radius_px` | 24.0 px (derived from scale prior) | Edge sampling extent |
| `image_margin_px` | 10.0 px | Skip attempts near image boundary |
| `max_attempts` | `None` (unlimited) | Optional cap on completion attempts |

The `reproj_gate_px` is the most important gate — it ensures that completed markers are geometrically consistent with the homography. A tight gate (default 3.0 px) prevents false detections from being added.

## Projective Center for Completion Markers

After completion, projective center correction is applied to the newly completed markers only. Previously corrected markers retain their corrections. Each marker is corrected exactly once.

## Final Homography Refit

With the expanded marker set (original + completed), the homography is refit from all corrected centers. This final refit:

1. Uses all available markers for maximum accuracy
2. Accepts the refit only if the mean reprojection error improves
3. Updates `DetectionResult.homography` and `DetectionResult.ransac`

## Disabling Completion

Set `completion.enable = false` in `DetectConfig` or use `--no-complete` in the CLI to skip completion entirely. This is useful when:

- You only want high-confidence initial detections
- Processing speed is more important than recall
- The homography is unreliable (few decoded markers)

Completion also requires a valid homography — if the global filter did not produce one (fewer than 4 decoded markers), completion is automatically skipped.

**Source**: `detector/completion.rs`, `pipeline/finalize.rs`
