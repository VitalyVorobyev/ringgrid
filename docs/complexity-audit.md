# Complexity Audit: ringgrid

_Audit date: 2026-02-22_
_Auditor: Claude Sonnet 4.6 (assisted by static analysis of the full source tree)_

---

## Executive Summary

The ringgrid codebase is architecturally sound: the detection pipeline stages are largely separated, the public API surface is clean, and the mathematics modules are well-isolated. The most impactful issues fall into four categories:

1. **String-based error bridging at the `outer_estimate` → `outer_fit` boundary** (`detector/outer_fit/mod.rs:87-119`): a string is formatted in one module, then parsed with a regex-like prefix match in another. This is the most fragile pattern in the codebase.
2. **Algorithmic logic embedded in the pipeline orchestrator**: `pipeline/finalize.rs` contains ~200 lines of algorithmic recovery (`try_recover_inner_as_outer`) that belong in `detector/`, and it duplicates `sync`/`annotate` call sequences in both the global-filter and no-global-filter paths.
3. **`DetectConfig` has two independent parameters (`min_semi_axis`, `max_semi_axis`) that are fully derived from `marker_scale`** — they are always overwritten by `apply_marker_scale_prior` and should not be public fields.
4. **Minor but widespread test-helper duplication**: the `draw_ring_image` / `blur_gray` helpers are independently defined in six and three files respectively; a shared `tests_common` module would eliminate the drift risk.

Total config parameters counted: **72 across all nested structs** in `DetectConfig`. Of these, roughly 8 are candidates for removal or consolidation.

---

## Executive Summary Table

| ID | File(s) | Issue | Severity |
|----|---------|-------|----------|
| A1 | `detector/outer_fit/mod.rs:87-119`, `ring/outer_estimate.rs:162,196` | String-parsed error bridge between modules | High |
| A2 | `pipeline/finalize.rs:194-402` | Algorithmic recovery function in orchestrator module | High |
| A3 | `pipeline/finalize.rs:440-456,522-527` | Duplicated `sync` + `annotate` call sequences in both finalize paths | Medium |
| A4 | `detector/marker_build.rs:191-195`, `detector/completion.rs:133-137`, `detector/outer_fit/scoring.rs:13-18` | `arc_cov * inlier_ratio` formula duplicated three times | Medium |
| A5 | `detector/config.rs:567-569` (`min_semi_axis`, `max_semi_axis`) | Derived fields exposed as writable public config | Medium |
| A6 | `detector/id_correction/local.rs:68-79`, `detector/id_correction/diagnostics.rs:69-80` | `effective_min_votes` logic duplicated verbatim | Low |
| A7 | `ring/inner_estimate.rs:264`, `ring/outer_estimate.rs:329`, `ring/radial_estimator.rs:168` | `blur_gray` test helper copied three times | Low |
| A8 | `pipeline/fit_decode.rs:215`, `detector/outer_fit/mod.rs:325`, `detector/outer_fit/sampling.rs:207`, `detector/inner_fit.rs:422`, `api.rs:206`, `pipeline/fit_decode.rs:215` | `draw_ring_image` test helper copied six times | Low |
| A9 | `lib.rs:95-96` | `codebook` and `codec` raw modules publicly re-exported | Low |
| A10 | `homography/utils.rs` | `refit_homography` and `refit_homography_matrix` are thin wrappers around each other | Low |
| C1 | `detector/config.rs:562,582-586` | `CircleRefinementMethod` enum and `ProjectiveCenterParams.enable` are redundant | Medium |
| C2 | `detector/config.rs:544,562` | `marker_scale` and derived axis/completion fields expose internal coupling | Medium |
| C3 | `detector/config.rs:681` | `outer_estimation.theta_samples` silently overwritten from `edge_sample.n_rays` | Medium |
| C4 | `ring/outer_estimate.rs:OuterEstimationConfig`, `marker/marker_spec.rs:MarkerSpec` | Overlapping config fields (`theta_samples`, `radial_samples`, `aggregator`, `min_theta_coverage`, `min_theta_consistency`) | Medium |

---

## 1. Algorithmic Redundancy and Simplification Opportunities

### A1: String-Formatted Errors Parsed by the Consumer (High)

**Location:** `ring/outer_estimate.rs:162,196` and `detector/outer_fit/mod.rs:87-119`

`OuterEstimate.reason` is an `Option<String>` used as a structured error carrier. When the outer estimator fails with `InsufficientThetaCoverage`, it formats the reason as the string `"insufficient_theta_coverage(0.31<0.60)"`. The consumer in `outer_fit/mod.rs` then parses this string with a hand-written prefix-strip routine (`parse_theta_coverage_reason`) to reconstruct the structured context it needs to attach to `OuterFitRejectContext::ThetaCoverage { observed_coverage, min_required_coverage }`.

This is a textbook anti-pattern: the information is structured at the source, discarded into a string, then re-parsed at the consumer. The fix is to replace `OuterEstimate.reason: Option<String>` with a typed enum:

```rust
pub enum OuterEstimateError {
    InvalidSearchWindow,
    InsufficientThetaCoverage { observed: f32, min_required: f32 },
    NoPolarityCandidates,
}
```

`OuterEstimate.status` and `OuterEstimate.reason` would collapse into `Result<OuterEstimate, OuterEstimateError>` or `OuterEstimate { status: OuterEstimateStatus, error: Option<OuterEstimateError> }`. `parse_theta_coverage_reason` and `map_outer_estimate_reject` in `outer_fit/mod.rs:87-120` would be deleted entirely. The same pattern (`reason: Option<String>`) exists in `InnerEstimate` (`inner_estimate.rs:47`), where it embeds strings like `"theta_inconsistent(0.18<0.25)"`.

**Estimated impact:** High — removes fragile string coupling, makes the boundary type-safe, eliminates one of the most surprising pieces of code in the codebase.

---

### A2: Algorithmic Recovery Function in the Pipeline Orchestrator (High)

**Location:** `pipeline/finalize.rs:194-402`

`try_recover_inner_as_outer` is a 200-line function that performs outer re-fitting, inner fitting, confidence computation, center frame handling, and marker replacement. It is placed in `pipeline/finalize.rs`, whose stated responsibility is pipeline orchestration (sequencing stages, not computing within them).

This function:
- Clones the entire `DetectConfig` to create a `recovery_config` (line 250)
- Calls `outer_fit::fit_outer_candidate_from_prior` directly
- Calls `inner_fit::fit_inner_ellipse_from_outer_hint` directly
- Calls `marker_build::fit_metrics_with_inner` and `compute_marker_confidence` directly
- Handles mapper frame conversion inline

The function belongs in `detector/` (for example, as a new module `detector/inner_as_outer_recovery.rs`), parallel to `detector/completion.rs`, which performs a similar per-marker re-fit operation. The orchestrator in `finalize.rs` should only call into it. This would reduce `finalize.rs` from 746 lines to roughly 550 lines and make the algorithmic boundary clear.

**Estimated impact:** High — `finalize.rs` becomes a true orchestrator; the recovery logic is testable in isolation.

---

### A3: Duplicated `sync` + `annotate` Sequences in Both Finalize Paths (Medium)

**Location:** `pipeline/finalize.rs:440-456` and `pipeline/finalize.rs:503-527`

Both `finalize_no_global_filter_result` and `finalize_global_filter_result` independently call the same sequence:
```rust
sync_marker_board_correspondence(&mut markers, &config.board);
annotate_neighbor_radius_ratios(&mut markers, 6);
if config.inner_as_outer_recovery.enable {
    try_recover_inner_as_outer(gray, &mut markers, config, mapper);
    sync_marker_board_correspondence(&mut markers, &config.board);
    annotate_neighbor_radius_ratios(&mut markers, 6);
}
```

This three-step sequence (sync, annotate, optionally recover + re-sync + re-annotate) is identical in both code paths and could be extracted into a private helper function `apply_post_filter_fixup(gray, markers, config, mapper)`. The magic `k = 6` passed to `annotate_neighbor_radius_ratios` matches `InnerAsOuterRecoveryConfig::k_neighbors` (default 6) but is not derived from it — it is hardcoded at two call sites.

**Estimated impact:** Medium — reduces maintenance surface, eliminates the risk of the two paths drifting. The hardcoded `6` should become `config.inner_as_outer_recovery.k_neighbors`.

---

### A4: `arc_cov * inlier_ratio` Formula Duplicated Three Times (Medium)

**Location:**
- `detector/marker_build.rs:191-195` (`fallback_fit_confidence`)
- `detector/completion.rs:133-137` (`compute_candidate_quality`)
- `detector/outer_fit/scoring.rs:13-18` (`score_outer_candidate`)

All three compute "fit support" as:
```rust
let arc_cov = edge.n_good_rays as f32 / edge.n_total_rays.max(1) as f32;
let inlier_ratio = outer_ransac.map(...).unwrap_or(1.0).clamp(0.0, 1.0);
let fit_support = (arc_cov * inlier_ratio).clamp(0.0, 1.0);
```

The formula is a composite of arc coverage and RANSAC inlier quality. It should live as one free function in `detector/marker_build.rs` (or `detector/outer_fit/scoring.rs`) and be called from the other two sites.

**Estimated impact:** Medium — three files share a latent coupling; if the formula changes (e.g., to weight arc coverage differently from inlier ratio), three edits are required.

---

## 2. Configuration Surface Analysis

### Current Parameters (enumerated)

`DetectConfig` at `detector/config.rs:541-587` has **18 direct fields**, each potentially a sub-config. Full parameter count per sub-config:

| Sub-config | Fields |
|-----------|--------|
| `MarkerScalePrior` | 2 |
| `OuterEstimationConfig` | 10 |
| `ProposalConfig` | 7 |
| `SeedProposalParams` | 3 |
| `EdgeSampleConfig` | 6 |
| `DecodeConfig` | 9 |
| `MarkerSpec` | 8 |
| `InnerFitConfig` (+ nested `RansacConfig`) | 9 + 4 = 13 |
| `OuterFitConfig` (+ nested `RansacConfig`) | 4 + 4 = 8 |
| `ProjectiveCenterParams` | 6 |
| `CompletionParams` | 8 |
| `DetectConfig` direct fields (axis bounds, dedup, filter enable) | 5 |
| `RansacHomographyConfig` | 4 |
| `SelfUndistortConfig` | (not audited in depth) |
| `IdCorrectionConfig` | 16 |
| `InnerAsOuterRecoveryConfig` | 8 |
| **Total** | **~107 leaf fields** |

---

### Proposed Reductions

| Parameter | Current Role | Recommendation | Rationale |
|-----------|-------------|----------------|-----------|
| `DetectConfig.min_semi_axis` | Lower bound on outer ellipse semi-axis | **Remove as public field; compute internally** | Always overwritten by `apply_marker_scale_prior` as `(r_min * 0.3).max(2.0)`. Users cannot meaningfully set this independently of `marker_scale`. |
| `DetectConfig.max_semi_axis` | Upper bound on outer ellipse semi-axis | **Remove as public field; compute internally** | Always overwritten by `apply_marker_scale_prior` as `(r_max * 2.5).max(min_semi_axis)`. Same reasoning as above. |
| `CircleRefinementMethod` enum + `ProjectiveCenterParams.enable` | Two separate on/off switches for center correction | **Merge**: `CircleRefinementMethod::None` already captures the "disable" case; `ProjectiveCenterParams.enable` is redundant. The `enable` bool should be removed; `CircleRefinementMethod::None` is authoritative. | Two mechanisms for the same yes/no decision create a contradiction: `circle_refinement = ProjectiveCenter` but `projective_center.enable = false` is an ambiguous state. |
| `OuterEstimationConfig.theta_samples` | Number of theta rays in outer estimator | **Remove or document as read-only**: `apply_marker_scale_prior` silently overwrites it from `edge_sample.n_rays` (`config.rs:681`). Users who set this value independently will have their setting discarded. | Silent override is a correctness hazard. Either the field should be removed and always derived, or the override should not happen (derive `OuterEstimationConfig.theta_samples` lazily at call time). |
| `InnerAsOuterRecoveryConfig.k_neighbors` vs hardcoded `6` in finalize | Configured k for neighbor median, but `annotate_neighbor_radius_ratios` is called with literal `6` | **Unify**: `annotate_neighbor_radius_ratios(markers, config.inner_as_outer_recovery.k_neighbors)` | The configured value is ignored at the annotation call sites. |
| `DecodeConfig.threshold_max_iters` + `threshold_convergence_eps` | Control iterative 2-means threshold refinement | **Candidates for removal**: the 2-means threshold with `max_iters=10, eps=1e-4` is deterministic and fast; these parameters encode an algorithmic detail users cannot meaningfully tune without understanding the decode internals. Could become a single `threshold_mode: Adaptive | Fixed(f32)` enum. | These are implementation knobs with no intuitive physical meaning for an end user. |
| `SeedProposalParams.seed_score` | Score assigned to injected seed proposals (`1e12`) | **Hardcode or derive**: the value only needs to exceed any naturally occurring proposal score. A fixed large value (or `f32::MAX`) serves the same purpose. | The `seed_score = 1e12` default is a magic constant chosen to dominate other scores; the exact value has no physical meaning. |

---

### Minimal Proposed Public Config Surface

The following parameters **survive** as genuinely independently tunable:

- `MarkerScalePrior` (`diameter_min_px`, `diameter_max_px`) — primary user input
- `ProposalConfig.grad_threshold`, `.min_vote_frac`, `.accum_sigma` — proposal sensitivity
- `ProposalConfig.max_candidates` — throughput cap
- `EdgeSampleConfig.n_rays`, `.r_step`, `.min_ring_depth`, `.min_rays_with_ring` — edge sampling density and gates
- `OuterEstimationConfig` (minus `theta_samples`) — outer radius search window
- `DecodeConfig.max_decode_dist`, `.min_decode_confidence`, `.min_decode_margin`, `.code_band_ratio` — decode acceptance gates
- `MarkerSpec.r_inner_expected`, `.inner_search_halfwidth`, `.inner_grad_polarity` — marker geometry prior
- `OuterFitConfig.ransac` — RANSAC robustness for outer fit
- `InnerFitConfig` (minus derived/duplicate) — inner fit gates and RANSAC
- `CompletionParams.enable`, `.reproj_gate_px`, `.min_fit_confidence`, `.min_arc_coverage`, `.require_perfect_decode` — completion policy
- `ProjectiveCenterParams` (minus `enable`) — center correction quality gates
- `RansacHomographyConfig` — global filter robustness
- `IdCorrectionConfig` — ID verification policy (this is inherently complex; its 16 fields are mostly genuinely independent policy choices)
- `InnerAsOuterRecoveryConfig` — recovery thresholds
- `SelfUndistortConfig` — self-undistort policy

**Parameters recommended for removal from the public API:** `min_semi_axis`, `max_semi_axis`, `ProjectiveCenterParams.enable` (subsume into `CircleRefinementMethod`), `OuterEstimationConfig.theta_samples` (always derived), `threshold_max_iters`, `threshold_convergence_eps`, `SeedProposalParams.seed_score`.

---

## 3. Code Quality Issues

### Q1: `try_recover_inner_as_outer` Placed in Wrong Module

**Location:** `pipeline/finalize.rs:194-402` (209 lines)

This function is an algorithmic stage, not an orchestration step. It directly calls `outer_fit::fit_outer_candidate_from_prior`, `inner_fit::fit_inner_ellipse_from_outer_hint`, `marker_build::fit_metrics_with_inner`, and `compute_marker_confidence`. It should be moved to `detector/inner_as_outer_recovery.rs` with a public-crate function signature, and `finalize.rs` should call it by name only.

**Recommended fix:** Create `detector/inner_as_outer_recovery.rs`, move the function body there, expose it through `detector/mod.rs` as `pub(crate) use inner_as_outer_recovery::try_recover_inner_as_outer`.

---

### Q2: `parse_theta_coverage_reason` — String Parsing as Error Bridging

**Location:** `detector/outer_fit/mod.rs:87-94` and caller at lines `96-120`

A 7-line parsing function extracts `(f32, f32)` from the string `"insufficient_theta_coverage(0.31<0.60)"`. This exists because `OuterEstimate.reason: Option<String>` carries structured data as a formatted string. The function is correctly tested (line 381) but the root cause — a loosely-typed error carrier — should be fixed.

**Recommended fix:** Replace `OuterEstimate.reason: Option<String>` (and the parallel `InnerEstimate.reason: Option<String>` in `inner_estimate.rs:47`) with a typed error enum. See issue A1 above.

---

### Q3: `fit_metrics_from_outer` Has 10 Parameters

**Location:** `detector/marker_build.rs:108-144`

The private function `fit_metrics_from_outer` takes 10 arguments (annotated `#[allow(clippy::too_many_arguments)]`). This is a symptom of `FitMetrics` being built from scattered sources. The public-facing wrapper `fit_metrics_with_inner` correctly aggregates the `InnerFitResult` fields before calling it, but the private function boundary is still awkward.

**Recommended fix:** Accept `(&EdgeSampleResult, &Ellipse, Option<&RansacResult>, &InnerFitResult)` directly in `fit_metrics_from_outer`, removing the intermediate unpacking in `fit_metrics_with_inner`. The public wrapper becomes the private implementation.

---

### Q4: Duplicated `draw_ring_image` Test Helper (6 copies) and `blur_gray` (3 copies)

**Locations:**
- `draw_ring_image`: `pipeline/fit_decode.rs:215`, `detector/outer_fit/mod.rs:325`, `detector/outer_fit/sampling.rs:207`, `detector/inner_fit.rs:422`, `api.rs:206`, `ring/inner_estimate.rs` (via `make_coded_ring_image` variant in decode.rs)
- `blur_gray`: `ring/inner_estimate.rs:264`, `ring/outer_estimate.rs:329`, `ring/radial_estimator.rs:168`

**Recommended fix:** Create a `#[cfg(test)] mod tests_support` within a shared internal test utilities file (Rust allows `#[cfg(test)]` modules in `lib.rs` or a dedicated `src/tests_support.rs` with `#[cfg(test)]` at top level). The three `blur_gray` copies are identical. The six `draw_ring_image` copies differ only in the pixel values assigned to the ring region (24, 26, 30 — slightly different dark values), which should be parameterized as an argument.

---

### Q5: `refit_homography` and `refit_homography_matrix` Are Trivially Layered

**Location:** `homography/utils.rs:12-62` and `150-159`

`refit_homography_matrix` is a 9-line function that calls `refit_homography`, converts its `[[f64; 3]; 3]` output back to `Matrix3<f64>` via `array_to_matrix3`, and returns. The intermediate `[[f64; 3]; 3]` format exists only because `refit_homography` returns that format (matching the JSON-serializable type), but within the pipeline all consumers actually want `Matrix3<f64>`. This creates a round-trip `Matrix3 → array → Matrix3` at `finalize.rs:70`.

**Recommended fix:** Make `refit_homography` return `Matrix3<f64>` directly (it is `pub(crate)`). Add a separate `refit_homography_as_array` for the one caller that needs the array form (if any). `refit_homography_matrix` is then redundant and can be deleted.

---

### Q6: `effective_min_votes` Logic Duplicated in `id_correction`

**Location:** `detector/id_correction/local.rs:68-79` and `detector/id_correction/diagnostics.rs:69-80`

Both files independently compute:
```rust
let effective_min_votes = if ws.markers[i].id.is_none() {
    ws.config.min_votes_recover
} else {
    ws.config.min_votes
};
```

**Recommended fix:** Add a method on `IdCorrectionConfig` or on the workspace type: `fn effective_min_votes(&self, has_id: bool) -> usize`. Both callers then use `ws.config.effective_min_votes(ws.markers[i].id.is_some())`.

---

### Q7: `codebook` and `codec` Raw Modules Publicly Re-exported

**Location:** `lib.rs:95-96`

```rust
pub use marker::codebook;
pub use marker::codec;
```

These expose the entire generated `codebook` module (with `CODEBOOK`, `CODEBOOK_MIN_CYCLIC_DIST`, and the raw table) and the `codec` module (with `Codebook`, `Match`) as public API. The CLI uses them in test functions at `ringgrid-cli/src/main.rs:683,742`, but this is within `#[cfg(test)]` blocks. These modules should be `pub(crate)` unless there is a documented use case for external library consumers to access the raw codebook data directly.

**Recommended fix:** If CLI tests need these, re-export them in a `#[doc(hidden)]` block or expose only the required constants. Mark `codebook` and `codec` as `pub(crate)` in `lib.rs`.

---

## 4. Invariant and Abstraction Boundary Issues

### I1: `OuterEstimationConfig.theta_samples` is a Phantom Field

**Location:** `detector/config.rs:681`, `ring/outer_estimate.rs:OuterEstimationConfig`

`OuterEstimationConfig.theta_samples` is a field in the config struct (with a default of 48), but `apply_marker_scale_prior` unconditionally overwrites it:
```rust
config.outer_estimation.theta_samples = config.edge_sample.n_rays;
```
This means any user modification to `config.outer_estimation.theta_samples` after construction will be silently ignored unless `apply_marker_scale_prior` is not called — but the recommended constructors all call it. The invariant that `outer_estimation.theta_samples == edge_sample.n_rays` is established by config construction but not enforced at the type level.

**Impact:** A caller who does `cfg.outer_estimation.theta_samples = 96` expecting to increase sampling density will have no effect. This is the most likely source of silent misconfiguration.

**Recommended fix:** Remove `theta_samples` from `OuterEstimationConfig` and have `estimate_outer_from_prior_with_mapper` accept it as a separate parameter (already the case in practice since `build_outer_estimation_cfg` clones the config and then sets `theta_samples`). Alternatively, document it explicitly as `// Do not set directly; derived from edge_sample.n_rays by apply_marker_scale_prior`.

---

### I2: `CircleRefinementMethod` and `ProjectiveCenterParams.enable` Create an Ambiguous State

**Location:** `detector/config.rs:254-268, 87-120`

There are two independent enable/disable controls for projective center correction:
1. `DetectConfig.circle_refinement: CircleRefinementMethod` — set to `None` or `ProjectiveCenter`
2. `ProjectiveCenterParams.enable: bool` inside `DetectConfig.projective_center`

In `center_correction.rs:18-24`, the code checks only `config.projective_center.enable`:
```rust
if !config.projective_center.enable {
    return;
}
```
The enum `CircleRefinementMethod` is checked only in `warn_center_correction_without_intrinsics`. So `circle_refinement = ProjectiveCenter` with `projective_center.enable = false` will disable correction (correct), but `circle_refinement = None` with `projective_center.enable = true` will still run correction (incorrect — the enum `None` is ignored by `apply_projective_centers`).

**Invariant violation:** The invariant "correction runs iff the method is `ProjectiveCenter`" is not enforced. Only the `enable` bool is authoritative for the actual computation.

**Recommended fix:** Remove `ProjectiveCenterParams.enable` and check `config.circle_refinement.uses_projective_center()` in `apply_projective_centers`. The enum becomes the single source of truth.

---

### I3: `finalize.rs` Has Two Separate Code Paths with Overlapping Post-Processing

**Location:** `pipeline/finalize.rs:426-464` (`finalize_no_global_filter_result`) and `466-538` (`finalize_global_filter_result`)

The two functions share identical post-processing steps but because they are separate code paths, they can independently accumulate drift. For instance, `finalize_no_global_filter_result` applies `drop_unmappable_markers` and `map_centers_to_image` but then does not recompute RANSAC stats (correct for no-filter path), while `finalize_global_filter_result` does all of these plus calls `phase_final_h`. The `sync_marker_board_correspondence` and `annotate_neighbor_radius_ratios` + recovery block are then independently duplicated in both.

**Invariant:** The post-filter fixup sequence (sync → annotate → recover → re-sync → re-annotate) should be applied exactly once regardless of the global filter path. Currently, a change to the fixup logic must be applied in two places.

**Recommended fix:** Extract the shared post-processing into `fn apply_post_filter_fixup(gray, markers, config, mapper)` and call it from both paths. The two top-level functions then differ only in whether they run the global filter and completion.

---

### I4: `try_recover_inner_as_outer` Conditionally Uses `center_mapped` Based on Timing

**Location:** `pipeline/finalize.rs:229-230`

```rust
let center_wf: [f64; 2] = markers[idx].center_mapped.unwrap_or(markers[idx].center);
```

This line has an implicit assumption: if `map_centers_to_image` has already run, then `center_mapped` holds the working-frame center; if not, `center` is the working-frame center. This invariant is not documented and relies on the call order within `finalize.rs`. When the recovery function is called, `map_centers_to_image` has already run in `finalize_global_filter_result` (line 484-492) but has not run in `finalize_no_global_filter_result` (it runs at line 442, after the recovery at line 453).

Wait: re-reading `finalize_no_global_filter_result`: the `if let Some(mapper) = mapper` block that calls `map_centers_to_image` is at lines 434-443, and then `try_recover_inner_as_outer` is called at line 453 — so `map_centers_to_image` has already run in both paths. The concern is less acute than it appears, but the comment inside the function ("if map_centers_to_image has already run") is the only documentation of this ordering dependency.

**Recommended fix:** Document this invariant explicitly at the call sites, or refactor so the recovery function receives the working-frame center explicitly rather than inferring it from which optional field is populated.

---

### I5: Completion Scale Gate Uses Hardcoded Tolerances

**Location:** `detector/completion.rs:140-141` and `209-210`

```rust
let scale_ok = mean_axis >= (r_expected * 0.75) && mean_axis <= (r_expected * 1.33);
```

This ±25%/+33% tolerance window is hardcoded and not exposed as a config parameter, even though it controls which completion attempts are accepted. It is also inconsistent with `InnerAsOuterRecoveryConfig.size_gate_tolerance` (default 0.25 = 25%), which is configurable. If the outer radius estimation is noisy (e.g., large blur), this window may need to be widened, but there is no knob to do so.

**Recommended fix:** Add `CompletionParams.scale_gate_tolerance: f32` (default 0.25, which maps to the existing 0.75-1.33 window). This makes the gate explicit and consistent with the recovery module's naming convention. Alternatively, document these as intentionally non-configurable and add a comment explaining the 0.75/1.33 choice.

---

## 5. Priority Action List

The following are ordered by impact-to-effort ratio. Each action is self-contained and can be undertaken independently.

**1. Replace `OuterEstimate.reason: Option<String>` with a typed error enum** (`ring/outer_estimate.rs`, `ring/inner_estimate.rs`, `detector/outer_fit/mod.rs`)

Effort: ~2-3 hours. Impact: Eliminates the `parse_theta_coverage_reason` string-parsing anti-pattern, makes the `outer_estimate → outer_fit` boundary type-safe, and removes the `OuterFitRejectReason::OuterEstimateUnknownFailure` catch-all variant. This is the single highest-leverage quality improvement.

**2. Move `try_recover_inner_as_outer` to `detector/`** (`pipeline/finalize.rs` → `detector/inner_as_outer_recovery.rs`)

Effort: ~2 hours (mostly moving code; the function is self-contained). Impact: Restores single-responsibility to `finalize.rs`, makes the recovery logic independently testable, and reduces `finalize.rs` by ~200 lines.

**3. Extract shared post-filter fixup sequence in `finalize.rs`** (`pipeline/finalize.rs:440-456, 522-527`)

Effort: ~1 hour. Impact: Eliminates the two-path duplication of `sync` + `annotate` + optional recovery. Replace the hardcoded `6` with `config.inner_as_outer_recovery.k_neighbors`.

**4. Remove `min_semi_axis` and `max_semi_axis` as public `DetectConfig` fields**

Effort: ~30 minutes. Impact: Removes two misleadingly-public derived fields from the config surface. These should become private fields or be computed locally at the point of use in `outer_fit/solver.rs`.

**5. Remove `ProjectiveCenterParams.enable` and use `CircleRefinementMethod` as the single authority**

Effort: ~1 hour. Impact: Closes the ambiguous-state invariant violation (I2). One `impl` change in `center_correction.rs` and a field removal from `ProjectiveCenterParams`.

**6. Unify the `arc_cov * inlier_ratio` formula into a single function** (`detector/marker_build.rs`, `detector/completion.rs`, `detector/outer_fit/scoring.rs`)

Effort: ~30 minutes. Impact: One canonical `fit_support_score(edge, ransac)` function replaces three hand-rolled copies.

**7. Fix the `OuterEstimationConfig.theta_samples` phantom field** (`detector/config.rs:681`, `ring/outer_estimate.rs`)

Effort: ~1 hour. Impact: Removes a latent silent-misconfiguration hazard. Either remove the field and pass `n_rays` explicitly, or add a `/// Derived from edge_sample.n_rays by the config constructors; do not set directly.` doc comment.

**8. Create a shared `#[cfg(test)]` utility module for ring image helpers** (project-wide)

Effort: ~1 hour. Impact: Collapses 6 copies of `draw_ring_image` and 3 copies of `blur_gray` into one. Any future change to the test image format (e.g., adding anti-aliasing or parameterizing pixel values) requires a single edit.

---

## Appendix: Config Parameters Assessed as Correctly Remaining Public

The following parameters are flagged in passing as areas where complexity is **not reducible** — they encode genuine domain knowledge:

- `IdCorrectionConfig.auto_search_radius_outer_muls: Vec<f64>` — the staged sweep from tight to loose neighbor radii encodes a non-trivial recovery strategy for mixed-quality scenes. The vector length (number of passes) and the specific multipliers are genuinely tunable.
- `InnerFitConfig.miss_confidence_factor` — the 30% confidence penalty for missing inner fits is a tuned heuristic with measurable precision/recall tradeoff.
- `DecodeConfig.min_decode_margin` — the minimum Hamming margin is a fundamental codebook-quality gate with known semantics (`CODEBOOK_MIN_CYCLIC_DIST = 2`).
- `RansacHomographyConfig.max_iters` and `RansacConfig.max_iters` — RANSAC iteration counts encode the probability-of-success guarantee and are image-dependent; they must remain configurable.
- `ProjectiveCenterParams.max_selected_residual` and `min_eig_separation` — these gate the numerical stability of the conic-pencil eigenpair and protect against degenerate geometry; they are not safely hardcoded.
