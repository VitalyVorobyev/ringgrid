# ID Correction

The `id_correction` stage runs in the `pipeline/finalize/` module after projective-center correction and before optional global homography filtering.

It enforces structural consistency between decoded marker IDs and the board's hex-lattice topology, while recovering safe missing IDs.

## Applicability

ID correction is a hex-neighbor BFS consensus, so it runs **only for hex coded targets** — its algorithmic domain. Rect coded targets carry decodable IDs but have no hex neighborhood, so they rely on the global RANSAC homography filter plus [geometric verification](overview.md#geometric-verification) instead. Plain targets carry no IDs at all and are labeled by [grid assignment](overview.md#grid-assignment). The gate is `config.advanced.id_correction.enable && target.is_coded() && lattice == Hex`.

## Inputs and Outputs

Inputs:
- marker centers in image pixels (`DetectedMarker.center`)
- decoded IDs (`DetectedMarker.id`)
- target layout topology/coordinates (`TargetLayout`, hex coded)
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

## Voting soundness fixes (0.8)

Several latent voting bugs were fixed in 0.8; each had made a fallback silently misfire on hex boards:

- **Scale-vote pitch.** Scale-vote predictions converted one-hop image distances to millimeters via the axial pitch, but hex board-adjacent centers sit `pitch·√3` apart — predictions fell ~42 % short and outside tolerance, so the scale-vote fallback never fired. It now uses the nearest-neighbor spacing.
- **Rotated scale votes.** Scale votes now follow the locally-estimated board→image rotation (from the same trusted adjacent pairs). The previous axis-aligned prediction assumed an unrotated board and, near the hex 60° symmetry, voted for exactly the wrong lattice site.
- **Affine hypothesis weight.** The affine hypothesis now casts one vote instead of one per neighbor, so a single ill-conditioned affine can no longer satisfy `min_votes` on its own with zero corroboration.
- **Duplicate claims.** Within one batch, duplicate claims on the same ID are resolved by anchor-homography reprojection error (previously both were applied, and a later confidence-based pass could evict a correct assignment for a confident wrong one).
- **Homography fallback floor.** `homography_min_trusted` now acts as a *ceiling*: the effective floor scales with the visible marker count (`max(8, n/3)`), so sparse/partial views are no longer locked out of the geometric fallback.

See the [0.8 changelog](https://github.com/VitalyVorobyev/ringgrid/blob/main/CHANGELOG.md) for the full list.
