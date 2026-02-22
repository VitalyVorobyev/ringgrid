# Handoff Note — Inner-as-Outer Recovery Improvement

**From:** algorithm-engineer
**To:** Pipeline Architect / Implementer
**Date:** 2026-02-22
**Related:** ADR-001, design doc `inner-as-outer-recovery-relaxed-gates.md`

---

## What this solves

Marker 103 in `data/target_3_split_02.png` is flagged as inner-as-outer (ratio=0.627) but recovery returns `n_recovered=0`. The outer ring at ~22.6 px is blurry and fails the production theta_consistency (0.35) and ring_depth (0.05) gates even with the tight 4 px search window. Recovery needs relaxed signal-quality gates, with a post-fit size gate to prevent re-locking.

---

## Proposed config shape

Extend the **existing** `InnerAsOuterRecoveryConfig` struct — no new public types:

```rust
// crates/ringgrid/src/detector/config.rs

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InnerAsOuterRecoveryConfig {
    pub enable: bool,            // existing — default: true
    pub ratio_threshold: f32,    // existing — default: 0.75
    pub k_neighbors: usize,      // existing — default: 6
    // NEW: relaxed gating for recovery re-estimation
    pub min_theta_consistency: f32,  // default: 0.18
    pub min_theta_coverage: f32,     // default: 0.40
    pub min_ring_depth: f32,         // default: 0.02
    pub refine_halfwidth_px: f32,    // default: 2.5
    pub size_gate_tolerance: f32,    // default: 0.25
}
```

All new fields are `#[serde(default)]` — existing config files remain valid without changes.

---

## Integration in `try_recover_inner_as_outer` (pipeline/finalize.rs)

The existing recovery function already clones the config and sets `outer_estimation.search_halfwidth_px = 4.0`. Extend the same clone:

```rust
let rcfg = &config.inner_as_outer_recovery;
let mut recovery_config = config.clone();
recovery_config.outer_estimation.search_halfwidth_px =
    OuterEstimationConfig::default().search_halfwidth_px;  // existing: 4.0
// NEW — relaxed gates:
recovery_config.outer_estimation.min_theta_consistency = rcfg.min_theta_consistency;
recovery_config.outer_estimation.min_theta_coverage   = rcfg.min_theta_coverage;
recovery_config.outer_estimation.refine_halfwidth_px  = rcfg.refine_halfwidth_px;
recovery_config.edge_sample.min_ring_depth            = rcfg.min_ring_depth;
```

Add size gate **after** the existing `candidate.decode_result.is_none()` check:

```rust
// Existing decode check:
if candidate.decode_result.is_none() {
    tracing::debug!(idx, "inner-as-outer recovery: re-fit produced no decode");
    continue;
}
// NEW — size gate:
let recovered_r = candidate.outer.mean_axis() as f32;
if (recovered_r - r_corrected).abs() / r_corrected > rcfg.size_gate_tolerance {
    tracing::debug!(
        idx, recovered_r, r_corrected,
        "inner-as-outer recovery: size gate rejected (re-locked to wrong ring)"
    );
    continue;
}
```

---

## Toggle / coexistence plan

- **No toggle needed.** The relaxed gates apply only when `inner_as_outer_recovery.enable = true` (default) AND `neighbor_radius_ratio < ratio_threshold`. This path is not reached for normal markers.
- Disable entirely with `enable = false` or `--no-inner-as-outer-recovery`.
- Tighten/loosen per deployment via config JSON.

---

## Migration intention

1. Implement (extend config + update finalize.rs)
2. Validate on 6 real images: split_02 recovers marker 103, no regressions elsewhere
3. Run synthetic blur eval at σ=1.5 px and σ=2.5 px to confirm recovery rates
4. Accept as permanent default (no removal of old path; old path is "recovery disabled")

---

## Public API changes

**None.** `InnerAsOuterRecoveryConfig` is already re-exported from `lib.rs`. New fields are additive with serde defaults. No changes to `DetectedMarker`, `FitMetrics`, `DecodeMetrics`, or `DetectionResult`.

---

## Acceptance criteria (implementer checklist)

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy -- -D warnings` clean
- [ ] split_02: `n_recovered=1`, recovered `outer_radius ∈ [18, 28] px`
- [ ] split_00,01,03,04,05: marker counts and IDs unchanged
- [ ] `config_sample.json` updated with new fields + values
- [ ] Size gate unit test: `|14.17 − 22.61| / 22.61 = 0.37 > 0.25` → rejected
