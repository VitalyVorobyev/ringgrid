# Design: Inner-as-Outer Recovery — Relaxed-Gate Re-Estimation

**Date:** 2026-02-22
**Author role:** algorithm-engineer
**Status:** proposed
**Related ADR:** ADR-001

---

## 1. Problem Statement

### Observed failure

In `data/target_3_split_02.png`, marker 103 is detected with its **inner ring edge serving as the outer ellipse**:

- Detected outer radius: **14.17 px** (true inner radius)
- True outer radius (from neighbour median): **22.61 px**
- `neighbor_radius_ratio = 0.627` correctly flags this (threshold = 0.75)

The auto-recovery added in the prior session attempts to re-fit the outer ellipse using `r_corrected = 22.61 px` with a tight 4 px search window `[18.6, 26.6]` px. Recovery returns `n_recovered = 0`.

### Why recovery fails

**Step 1 — outer estimation** calls `estimate_outer_from_prior_with_mapper` which aggregates radial derivatives over 48 rays and applies two quality gates:

| Gate | Threshold | Behaviour on blurry edge |
|------|-----------|--------------------------|
| `min_theta_consistency` | 0.35 | Blurry gradient scatters per-θ peaks → fraction within ±δ of peak drops to ~0.15–0.20 |
| `min_theta_coverage` | 0.60 | Typically passes (34/48 = 0.71 for this marker) |

If theta_consistency is below 0.35, the estimator returns `OuterStatus::Failed` with no hypotheses → recovery aborts.

**Step 2 — edge point collection** (if estimation had succeeded) would then apply:

| Gate | Threshold | Behaviour on blurry edge |
|------|-----------|--------------------------|
| `min_ring_depth` | 0.05 (normalised ≈ 12.7 intensity levels) | Blur smears intensity contrast → signed depth ≈ 0.01–0.03, below gate |
| `refine_halfwidth_px = 1.0` with `r_step ≈ 0.5` | ±1 px around hypothesis | Misses flat-topped blurry gradient peaks |

**Root cause summary:** Normal production gates are calibrated for sharp-to-moderate blur. The blurry outer ring at marker 103 fails theta_consistency and would fail the ring_depth gate. Recovery needs the same geometric prior strength we already have (tight window, known r_corrected) but with relaxed signal-quality gates.

---

## 2. Current Algorithm Location

| Stage | File | Key function |
|-------|------|--------------|
| Outer estimation | `ring/outer_estimate.rs` | `estimate_outer_from_prior_with_mapper` |
| Radial scan + peaks | `ring/radial_estimator.rs` + `ring/radial_profile.rs` | `scan_radial_derivatives`, `theta_consistency` |
| Edge point collection | `detector/outer_fit/sampling.rs` | `collect_outer_edge_points_near_radius`, `pick_best_radius_on_ray` |
| Recovery orchestration | `pipeline/finalize.rs` | `try_recover_inner_as_outer` |
| Recovery config | `detector/config.rs` | `InnerAsOuterRecoveryConfig` |

---

## 3. Baseline (Existing Test Data)

| Image | Total markers | With ID | Anomalous ratio (<0.75) | Recovery success |
|-------|-------------|---------|------------------------|-----------------|
| split_00 | 81 | 81 | 0 | n/a |
| split_01 | 83 | 83 | 0 | n/a |
| split_02 | 70 | 70 | 1 (marker 103) | 0 |
| split_03 | 81 | 81 | 0 | n/a |
| split_04 | 63 | 63 | 0 | n/a |
| split_05 | 59 | 59 | 0 | n/a |

Acceptance criteria:
- split_02: recovery succeeds for marker 103 → `n_recovered = 1`, recovered outer radius ∈ [18, 28] px
- All other images: zero change in marker counts and IDs (no regressions)

---

## 4. Proposed Algorithm

### Name: Relaxed-Gate Recovery Re-Estimation

**Core idea:** When `try_recover_inner_as_outer` invokes `fit_outer_candidate_from_prior`, it already clones the config and sets `outer_estimation.search_halfwidth_px = 4.0 px` (tight window). Extend this clone to also relax the three gating parameters that fail on blurry edges. Add a post-fit size gate to prevent the relaxed config from accepting a mislocated inner-ring re-fit.

### Mathematical specification

**Inputs:**
- Marker center `c = (cx, cy)` in working-frame coordinates
- Corrected radius prior `r_c` = median outer radius of k nearest neighbours (px)
- Tight search window `W = [r_c − h, r_c + h]`, `h = 4.0 px`
- Gray image `I`

**Modified outer estimation — changes to `OuterEstimationConfig` clone:**

| Parameter | Production value | Recovery value | Justification |
|-----------|-----------------|----------------|---------------|
| `search_halfwidth_px` | 10 px (from scale prior) | **4.0 px** | Already implemented; excludes inner ring |
| `min_theta_consistency` | 0.35 | **0.18** | Blurry peak: ~9/48 rays agreeing is sufficient given strong r_c prior |
| `min_theta_coverage` | 0.60 | **0.40** | Near-boundary markers; 19/48 valid rays acceptable |
| `refine_halfwidth_px` | 1.0 px | **2.5 px** | Broader per-ray search to catch flat-topped blurry peaks |

**Modified edge sampling — changes to `EdgeSampleConfig` clone:**

| Parameter | Production value | Recovery value | Justification |
|-----------|-----------------|----------------|---------------|
| `min_ring_depth` | 0.05 | **0.02** | 2.5× relaxation for blur-smeared intensity gradient |

**Decode phase:** unchanged — same 16-sector sampling as normal path.

**Post-fit size gate (new):**

After ellipse fit and decode, verify:

```
|r_recovered − r_c| / r_c ≤ size_gate_tolerance   (default: 0.25)
```

i.e., recovered outer radius must be within 25% of the neighbour-median prior. This prevents a case where the relaxed config finds a completely different ring edge unrelated to the target marker.

**Accept condition (all must hold):**
1. Outer estimation returns `OuterStatus::Ok` with ≥1 hypothesis
2. Ellipse fit succeeds (≥ min_ransac_points inliers)
3. Valid decode (distance to codebook ≤ threshold)
4. Size gate passes

**Reject condition (any one fails → keep original marker):**
- Estimation fails → no edge points → ellipse fit skipped
- Fit or decode fails
- Size gate fails (recovered radius too far from r_c)

### Failure behaviour

- **Not enough gradient at r_c:** Estimation still fails even with relaxed gates → original marker kept. This is correct: if the outer edge is completely undetectable, recovery should not fabricate one.
- **Wrong polarity selected:** Polarity is re-estimated from scratch for the recovery window → unaffected by original mis-detection polarity.
- **Size gate prevents inner-ring re-lock:** Even with relaxed gates, if the estimator finds the inner ring again (at r ≈ 14 px), |14 − 22.6| / 22.6 = 0.38 > 0.25 → rejected.

### Numerical stability

All operations are the same as the existing outer fit path; relaxing thresholds does not change the linear algebra. The only new arithmetic is the size gate ratio, which is `O(1)`.

---

## 5. Config Changes (no new public types)

Extend `InnerAsOuterRecoveryConfig` (already in `detector/config.rs`) with five new fields:

```rust
pub struct InnerAsOuterRecoveryConfig {
    pub enable: bool,            // existing
    pub ratio_threshold: f32,    // existing
    pub k_neighbors: usize,      // existing
    // --- new fields ---
    pub min_theta_consistency: f32,  // default: 0.18
    pub min_theta_coverage: f32,     // default: 0.40
    pub min_ring_depth: f32,         // default: 0.02
    pub refine_halfwidth_px: f32,    // default: 2.5
    pub size_gate_tolerance: f32,    // default: 0.25
}
```

No new enums or public types. JSON serialization via `#[serde(default)]` — existing config files remain valid.

---

## 6. Implementation Plan

1. **Add 5 fields** to `InnerAsOuterRecoveryConfig` with `#[serde(default)]` + `Default` impl.
2. **In `try_recover_inner_as_outer`** (pipeline/finalize.rs), extend the existing config clone:
   ```rust
   let mut recovery_config = config.clone();
   recovery_config.outer_estimation.search_halfwidth_px =
       OuterEstimationConfig::default().search_halfwidth_px;
   // New:
   let rcfg = &config.inner_as_outer_recovery;
   recovery_config.outer_estimation.min_theta_consistency = rcfg.min_theta_consistency;
   recovery_config.outer_estimation.min_theta_coverage   = rcfg.min_theta_coverage;
   recovery_config.outer_estimation.refine_halfwidth_px  = rcfg.refine_halfwidth_px;
   recovery_config.edge_sample.min_ring_depth            = rcfg.min_ring_depth;
   ```
3. **Add size gate** after candidate decode check:
   ```rust
   let recovered_r = candidate.outer.mean_axis() as f32;
   if (recovered_r - r_corrected).abs() / r_corrected > rcfg.size_gate_tolerance {
       tracing::debug!(idx, recovered_r, r_corrected, "recovery: size gate rejected");
       continue;
   }
   ```
4. **Update `config_sample.json`** with the five new fields and their defaults.
5. **Verify** on split_02: expect `n_recovered=1`, recovered radius ∈ [18, 28] px.
6. **Regression check** on all 6 split images: marker counts and IDs must be unchanged.

---

## 7. Affected Modules

- `crates/ringgrid/src/detector/config.rs` — extend `InnerAsOuterRecoveryConfig`
- `crates/ringgrid/src/pipeline/finalize.rs` — extend recovery config clone + add size gate
- `crates/ringgrid-cli/config_sample.json` — document new defaults

No changes to: `ring/outer_estimate.rs`, `ring/radial_estimator.rs`, `detector/outer_fit/sampling.rs`, `lib.rs` (no public API changes).

---

## 8. Test Plan

| Test | Type | Pass condition |
|------|------|----------------|
| Recovery succeeds on split_02 marker 103 | Integration (real image) | n_recovered=1, recovered_r ∈ [18, 28] px |
| Recovered marker passes size gate | Unit | `\|r − r_c\| / r_c ≤ 0.25` |
| Size gate rejects inner-ring re-lock | Unit (synthetic) | r=14 px rejected when r_c=22.6 px |
| No regression: split_00,01,03,04,05 | Integration (real images) | Identical marker counts and IDs |
| No regression: split_02 other 69 markers | Integration | Counts and IDs unchanged |
| Recovery disabled (`enable=false`) | Integration | Behaviour identical to pre-change |
| Config serialisation round-trips | Unit | New fields survive JSON encode/decode |
| Synthetic blurry marker (σ=2px blur) | Synthetic eval | Recovery produces r ∈ [0.9, 1.1] * r_true |

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Recovery finds wrong outer ring in dense target | Low | Size gate (\|r−r_c\|/r_c ≤ 0.25) and decode requirement both must pass |
| Relaxed theta_consistency=0.18 accepts noise peak | Low-medium | Size gate + decode gate provide independent validation |
| min_ring_depth=0.02 misidentifies code-band edge | Medium | Code-band depth is typically 0.04–0.08 (above gate); outer ring depth even blurry ~0.03 |
| Parameter choice doesn't work for this specific marker | Medium | Verify empirically; adjust min_theta_consistency if needed (0.15 is floor) |
