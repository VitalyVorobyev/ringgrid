# ADR-001: Inner-as-Outer Recovery — Relaxed-Gate Re-Estimation

- **Status:** proposed
- **Date:** 2026-02-22
- **Author role:** algorithm-engineer
- **Supersedes:** none

## Context

When a marker's outer ellipse fit locks onto the inner ring edge (producing `outer_radius ≈ 0.63 × neighbour_median`), the flagging + recovery system introduced in the prior session detects the anomaly but recovery returns `n_recovered=0` because:

1. The outer estimator's `min_theta_consistency=0.35` gate fails on blurry edges (per-θ peaks scatter widely).
2. The edge sampler's `min_ring_depth=0.05` gate fails when blur reduces intensity contrast.
3. `refine_halfwidth_px=1.0` with coarse stepping misses flat-topped blurry derivative peaks.

The root issue is that production gating parameters are calibrated for sharp-to-moderate blur, but recovery is specifically needed for **soft/blurry markers** — exactly the case where production gates are most restrictive.

The tight `search_halfwidth_px=4.0 px` (already implemented) correctly excludes the inner ring from the search window; the problem is that no valid outer-ring hypothesis survives the theta_consistency and ring_depth gates even within that tight window.

## Decision

Extend `InnerAsOuterRecoveryConfig` with five new serde-default fields that control gate relaxation exclusively within the recovery code path. No production gates are changed. A post-fit size gate (`|r_recovered − r_corrected| / r_corrected ≤ 0.25`) is added to prevent the relaxed estimator from accepting a re-locked inner-ring fit.

**Option A (chosen): Relaxed-gate recovery config** — five new fields in `InnerAsOuterRecoveryConfig`:

| Field | Production default | Recovery default | Purpose |
|-------|-------------------|-----------------|---------|
| `min_theta_consistency` | 0.35 | **0.18** | Blurry peaks: fewer rays agree on radius |
| `min_theta_coverage` | 0.60 | **0.40** | Near-boundary markers: fewer valid rays |
| `min_ring_depth` | 0.05 | **0.02** | Blur smears intensity gradient |
| `refine_halfwidth_px` | 1.0 | **2.5** | Broader per-ray search for flat peaks |
| `size_gate_tolerance` | — | **0.25** | New gate: reject if r_recovered deviates >25% from r_corrected |

**Option B (rejected): Use completion machinery** — `fit_outer_candidate_from_prior_for_completion` already has relaxed `min_arc_coverage`. Rejected because: (a) completion edge config derives `r_max` from `roi_radius_px`, not from `r_corrected`; (b) completion semantics are "spatially missing marker", not "geometry-wrong marker"; (c) changes would require repurposing completion logic in ways that blur its responsibility.

**Option C (rejected): Skip outer estimation; probe directly at r_corrected** — bypass `estimate_outer_from_prior_with_mapper` entirely and collect edge points directly at r_corrected without a consistency gate. More aggressive but removes an important signal quality check; higher false-positive risk in dense targets where edges at r_corrected may belong to adjacent markers.

## Consequences

**Positive:**
- Recovery succeeds on blurry markers (marker 103 in split_02 expected to recover)
- No changes to production detection path — zero regression risk for non-blurry markers
- Config-file tunable: users can set `inner_as_outer_recovery.*` in JSON
- Existing `--no-inner-as-outer-recovery` flag still disables the entire path
- No new public API types; `InnerAsOuterRecoveryConfig` already public

**Negative:**
- Recovery still fails if outer ring is completely undetectable (θ_consistency < 0.18) — acceptable: we do not fabricate an edge where none exists
- Relaxed `min_ring_depth=0.02` can accept very shallow code-band edges — mitigated by decode gate (code band at different radial position would produce a different ID, rejected if it doesn't decode)
- Config has more fields to document

**Neutral:**
- Recovery adds a `config.clone()` allocation per flagged marker (typically ≤3 per image)
- Size gate is a new logical check but `O(1)` arithmetic

## Evidence

Acceptance test (to be run after implementation):

```
split_02: n_recovered=1, recovered outer radius ∈ [18, 28] px
split_00,01,03,04,05: marker counts and IDs identical to pre-change
```

Justification for specific threshold values:

- `min_theta_consistency=0.18`: With 48 rays and typical blur scatter of ±2–3 px, we expect 8–10 rays to agree within `±delta ≈ 1.5 px`. 9/48 = 0.19, just above 0.18. Tracks empirical scatter on blurry synthetic targets.
- `min_ring_depth=0.02`: Outer ring under σ=2 px Gaussian blur reduces peak-to-background from ~0.15 to ~0.03–0.05 normalised. Floor of 0.02 captures the faintest detectable gradient while excluding pure noise (< 0.01).
- `size_gate_tolerance=0.25`: Inner ring is at ~0.63× outer radius (ratio 0.627). The gate at 0.25 creates safe separation: re-locked inner ring gives |14−22.6|/22.6 = 0.38 > 0.25 → rejected.

## Affected Modules

- `crates/ringgrid/src/detector/config.rs` — extend `InnerAsOuterRecoveryConfig`
- `crates/ringgrid/src/pipeline/finalize.rs` — extend recovery config clone; add size gate
- `crates/ringgrid-cli/config_sample.json` — document new fields
