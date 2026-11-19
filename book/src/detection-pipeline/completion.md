# Completion & Final Refit

The completion stage attempts to detect markers that the initial pipeline missed — typically markers at the image periphery, under heavy blur, or with low contrast. It runs after grid labeling and origin anchoring (or, in the decode-anchored fallback, after ID correction and the global homography filter). It predicts where missing markers should be and attempts conservative local fits at those locations. On the grid-first path a cell first adopts an already-fitted ring the grid labeler dropped when one lies within the reprojection gate (or half its own outer radius) of the cell's seed, and only then refits; coded completion grows from the labeled patch one lattice ring per round before one sweep over the whole board.

Completion is **coordinate-keyed** and lattice-generic: it targets undetected target *cells*, identified by lattice coordinate, with the cell's ID on coded targets (`CompletionScope`, `complete_with_h`). Coded and plain paths share the same fit/gate machinery.

## Completion Algorithm

For each target cell that was **not** detected in the initial pipeline:

1. **Project**: use the homography to map the cell's board (or frame) position to image coordinates
2. **Expected size**: take the median outer radius of the nearest fitted rings as the cell's expected radius; the refit's edge search reaches 1.4× that radius (the per-ray expected-radius gate), whatever the scale prior
3. **Boundary check**: skip if that reach plus `image_margin_px` crosses the image edge
4. **Local fit**: run edge sampling and RANSAC ellipse fitting around the projected center
5. **Decode**: attempt code decoding at the fitted position (coded targets)
6. **Gate**: accept the detection only if it passes conservative quality gates

## Conservative Gates

Completion uses stricter acceptance criteria than the initial detection to avoid false positives:

| Parameter | Default | Purpose |
|---|---|---|
| `reproj_gate_px` | 3.0 px | Max distance between fitted center and H-projected position |
| `min_fit_confidence` | 0.45 | Minimum fit quality score |
| `min_arc_coverage` | 0.35 | Minimum fraction of rays with valid edge detections |
| `image_margin_px` | 10.0 px | Skip attempts near image boundary |
| `max_attempts` | `None` (unlimited) | Optional cap on completion attempts |

The `reproj_gate_px` is the most important gate — it ensures that completed markers are geometrically consistent with the homography. A tight gate (default 3.0 px) prevents false detections from being added.

## Seed Strategy

The completion stage uses a three-level fallback chain to predict each missing marker's image position:

1. **Lattice-neighbor midpoint** (via `projective-grid`'s `predict_grid_position`): predicts the position from the midpoints of detected opposite-neighbor pairs. This is the most geometrically principled seed and handles local perspective distortion well. It is lattice-generic — hexagonal for hex targets, square for rect.
2. **Local affine**: fits a local affine transform from the 3--4 nearest labeled neighbors and projects the missing position. Requires at least 3 nearby labeled markers.
3. **Global homography**: projects the cell position through the global/frame H matrix. Least accurate under lens distortion at image periphery.

The neighbor-midpoint seed was added via the `projective-grid` crate integration. On the rtv3d validation dataset it provides +5--6 additional decoded markers across strategies.

## Plain-Target Completion and Patch Growth

Plain targets run the same fit and gate machinery, keyed by lattice coordinate (`complete_with_h`). The candidate set depends on whether the [origin was resolved](../targets/origin-fiducials.md):

- **Anchored** (origin resolved): every missing board cell is a candidate, exactly like the coded path over the full board.
- **Unanchored** (relative frame): the labeled patch is **grown** iteratively. Each round attempts the cells inside the current patch bounding box expanded by one lattice ring; predictions improve as the patch fills, so cells the topological labeler dropped are recovered ring by ring, up to a bounded number of rounds.

## Projective Center for Completion Markers

After completion, projective center correction is applied to the newly completed markers only. Previously corrected markers retain their corrections. Each marker is corrected exactly once.

## Final Homography Refit

With the expanded marker set (original + completed), the homography is refit from all corrected centers. This final refit:

1. Uses all available markers for maximum accuracy
2. Accepts the refit only if the mean reprojection error improves
3. Updates `DetectionResult.homography` and `DetectionDiagnostics.ransac`

## Disabling Completion

Set `advanced.completion.enable = false` in `DetectConfig` or use `--no-complete` in the CLI to skip completion entirely. This is useful when:

- You only want high-confidence initial detections
- Processing speed is more important than recall
- The homography is unreliable (few decoded markers)

Completion also requires a valid homography — if neither grid labeling nor the decode-anchored fallback produced one, completion is skipped.

**Source**: `detector/completion.rs`, `pipeline/finalize/` (`grid.rs`, `fallback.rs`)
