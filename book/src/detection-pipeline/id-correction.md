# ID Correction

The `id_correction` stage runs in `pipeline/finalize.rs` after projective-center correction and before optional global homography filtering.

It enforces structural consistency between decoded marker IDs and the board's hex-lattice topology, while recovering safe missing IDs.

## Inputs and Outputs

Inputs:
- marker centers in image pixels (`DetectedMarker.center`)
- decoded IDs (`DetectedMarker.id`)
- board layout topology/coordinates (`BoardLayout`)
- `IdCorrectionConfig`

Outputs:
- corrected IDs in-place (`DetectedMarker.id`)
- unresolved markers either kept with `id=None` or removed (`remove_unverified`)
- diagnostics in `IdCorrectionStats`

Frame semantics:
- local consistency geometry uses image-pixel centers
- board lookups and adjacency use board-space marker coordinates/topology

## Trust Model and Soft Lock

Markers are bootstrapped into trust classes:
- `AnchorStrong`: exact decodes (`best_dist=0`, sufficient margin)
- `AnchorWeak`: valid decoded IDs with enough decode confidence
- recovered markers: assigned later by local/homography recovery

Soft-lock policy (`soft_lock_exact_decode=true`):
- exact decodes are not normally overridden
- they may still be cleared under strict contradiction evidence

## Stage Flow

1. **Bootstrap** trusted anchors from decoded IDs.
2. **Pre-consistency scrub** clears IDs that contradict local hex-neighbor structure.
3. **Local recovery** iteratively votes unresolved markers from trusted neighbors using local-scale gates derived from marker ellipse radii.
4. **Homography fallback** (optional) seeds unresolved markers with a rough, gated board-to-image model built from trusted anchors.
5. **Post-consistency sweep + refill** repeats scrub/recovery to remove late contradictions and fill safe holes.
6. **Cleanup/conflict resolution** clears/removes unresolved markers and resolves duplicate IDs deterministically.

## Local Consistency Rules

For each marker, nearby IDs are evaluated as:
- support edges: neighbors that are 1-hop neighbors on the board
- contradiction edges: nearby IDs that are not board neighbors

Assignments are accepted only when support/contradiction evidence passes configured gates (`consistency_*`, vote gates).

## Homography Fallback

Fallback is conservative:
- requires enough trusted seeds and RANSAC inliers
- uses reprojection gate (`h_reproj_gate_px`)
- cannot override soft-locked exact decodes
- deterministic assignment order (error then ID tie-break)

## Determinism and Diagnostics

`id_correction` is deterministic for a fixed input/config:
- deterministic tie-breaks in voting and assignment
- fixed RANSAC seed for fallback homography

`IdCorrectionStats` reports corrections, recoveries, cleared IDs, unverified reasons, and residual inconsistency count.
