# ringgrid algorithmic-soundness assessment — 2026-06-25

Scope: correctness, numerical robustness, and missing capabilities of the core
detection algorithms. Three specialized reviewers audited the clusters below;
each verified the math term-by-term and ran the existing unit suites. Findings
cite `file:line` and are graded **critical** (wrong result on valid input),
**major** (fails on plausible edge cases / numerically fragile), or **minor**.

> Verdict in one line: **the core mathematics is correct** — no correctness
> defects in conic fitting, homography, or the ring/decode math on the inputs
> the pipeline actually produces — but there is **one critical robustness bug in
> ID correction** (it drops correct IDs in sparse/partial views), plus a cluster
> of NaN-panic and edge-case-fragility majors. The most intricate algorithms are
> also the least tested, so most findings are currently un-pinned by tests.

---

## What was verified correct (so it is not re-litigated later)

- **Conic fit** (`conic/fit.rs`): Halir–Flusser scatter partition, the reduced
  eigenproblem, back-substitution, and `denormalize_conic` are correct
  term-by-term; Hartley isotropic normalization is applied before the solve.
- **Conic ↔ geometric** (`conic/types.rs`): center/angle/semi-axis recovery and
  the `a≥b` canonicalization round-trip exactly.
- **Homography** (`homography/core.rs`): the Hartley-normalized DLT (delegated to
  `projective-grid`) and the RANSAC wrapper are correct, seeded/deterministic,
  and **refit on the inlier set**; 20/20 conic + 10/10 homography tests pass.
- **Projective center** (`ring/projective_center.rs`): the conic-pencil method
  genuinely removes the centroid's perspective bias for concentric rings
  (verified < 1e-8 on synthetic homographies).
- **Decode** (`marker/decode.rs`, `marker/codec.rs`): the 16-sector winding
  matches the target generator, cyclic matching absorbs board rotation, and all
  893 base codewords decode unambiguously (margin = 2; no self-rotation distance
  < 2). Termination of the ID-correction loop is provably bounded.

---

## Cluster A — Conic & Homography (verdict: sound; 0 critical, 2 major)

| file:line | sev | issue | recommendation |
|---|---|---|---|
| `conic/eigen.rs:13-56,103-139` | major | Eigenvalues via characteristic polynomial + adjugate is the numerically-fragile route (cancellation in tr/det and the cubic discriminant). Mitigated by upstream normalization; no repro found. | Prefer a Schur/QR eigensolver for the non-symmetric `C1⁻¹M`, or add iterative root refinement. |
| `conic/types.rs:293,326` | major | `conic_to_ellipse` (public) rejects on **scale-dependent absolute** epsilons (`1e-15`); a validly-shaped but small-scaled conic is falsely rejected. | Normalize the conic (Frobenius / `A+C`) before the guards, or make thresholds relative. |
| `conic/fit.rs:66-69` | minor | Comment says "Schur decomposition"; impl uses the cubic-formula path. | Fix the comment. |
| `homography/core.rs:135 vs 250` | minor | Doc says non-inlier `errors` are 0; code writes errors for all. | Align doc and code. |
| `homography/core.rs:140-160` | minor | `sample_4_distinct` can silently return non-distinct indices after 200 retries; ellipse RANSAC uses clean Fisher–Yates. | Share one Fisher–Yates partial-shuffle helper. |
| `homography/utils.rs:130` | minor | Inlier count uses `<=` while RANSAC uses `<`. | Pick one convention; centralize. |
| `conic/types.rs:248-259` | minor (perf) | `sampson_distance` recomputes `ellipse_to_conic` every call in the RANSAC hot loop. | Cache the conic per model. |
| `homography/correspondence.rs:191` | minor | p95 index `floor(0.95·len)` is off-by-one vs nearest-rank. | `ceil(0.95·len)−1` or document. |

## Cluster B — Ring estimation (verdict: correct on nominal input; robustness gaps)

| file:line | sev | issue | recommendation |
|---|---|---|---|
| `ring/edge_sample.rs:124-133` | major | `bilinear_sample_u8_checked` returns `Some(NaN)` for NaN coordinates (**confirmed empirically**); the NaN flows into `partial_cmp().unwrap()` sites → panic. Root defensive gap. | Reject non-finite coords at function entry. |
| `ring/radial_profile.rs:20,24,45,51` | major | Shared aggregation/peak core uses `partial_cmp(...).unwrap()` (NaN → panic) and has **0 tests**. | Use `total_cmp`/NaN-skipping reductions; add direct tests. |
| `ring/inner_estimate.rs:142-151` | major | Inner radius chosen by **global** extremum (no proximity-to-expected weighting), unlike the outer estimator's local-peak + consistency gating; a strong code-band edge in-window can win. | Mirror `find_local_peaks` ranked by strength **and** proximity. |
| `ring/outer_estimate.rs:227-243,316` | major | Circular sampling + a `theta_consistency ≥ 0.35` gate is plausibly fragile for oblique/perspective markers — the very case center-correction targets. | Condition the gate on estimated eccentricity; scale the search half-width with the prior. |
| `ring/projective_center.rs:365-370` | major | Eigenvalue-separation is the primary selector; for a **thin ring** (`r_in→r_out`, k→1) the distinct eigenvalue collapses and a wrong center can be returned **silently** (the separation gate lives only in the caller). | Document the k→1 degeneracy; surface `selected_eig_separation` so the gate can fire. |
| `ring/outer_estimate.rs:151-162` | minor | Plateau peak detection (`>=` both sides) marks every flat point a peak → arbitrary tie-break. | Strict `>` on one side or collapse plateaus. |

## Cluster C — ID correction & decode (verdict: decode correct; correction precision-first & asymmetric; 1 critical)

| file:line | sev | issue | recommendation |
|---|---|---|---|
| `id_correction/cleanup.rs:19-26` + `local.rs:54-122` | **critical** | Consistency can only **reject**, never **confirm**. A correctly-decoded non-exact ID in a sparse/partial neighborhood (no affine, no adjacent pair ⇒ 0 votes) is never promoted and is **cleared by cleanup despite being correct**. Drops correct IDs on valid sparse/partial/blurry scenes. | Add a confirm-by-consistency promotion path: trust a decoded marker when `support_edges ≥ 1` and `contradiction_edges == 0`. |
| `marker/decode.rs:363,395-398` | major | Code band is sampled on a **circle** of radius `(a+b)/2`, discarding the fitted ellipse's `a`, `b`, `angle`; under perspective the circle drifts off the elliptical band and corrupts sectors. (Rotation is correctly left to cyclic matching — but eccentricity should not be discarded.) | Sample along the fitted ellipse; let cyclic matching absorb rotation. |
| `id_correction/vote.rs:156-180` | major | Affine path casts one vote **per neighbor for the same shared hypothesis**, inflating `n_votes` and effectively disabling the `min_votes` gate whenever an affine exists. | Cast the affine hypothesis as a single weighted vote; count neighbors, not vote events. |
| `id_correction/vote.rs:104-137` | major | Scale-ratio fallback (<3 neighbors) assumes board axes align with image axes (isotropic scale, **no rotation**) → wrong neighbor prediction for boards rotated ≳30° with sparse neighborhoods. | Estimate local orientation from an adjacent pair, or refuse scale votes when orientation is unconstrained. |
| `id_correction/local.rs:52-136` + `vote.rs:245-268` | major | Batch correction has **no within-batch duplicate-id guard**; two markers can claim the same id in one pass, later resolved by keeping the **higher-confidence** marker — so a higher-confidence *wrong* assignment can evict a lower-confidence *correct* one. | Claim ids within the batch (as `homography.rs` already does), preferring lower reprojection error. |
| `id_correction/index.rs:49-63` | major | `nearest_within` iterates a `HashMap` with strict `<` and **no id tie-break**; on exact equidistance the result is **nondeterministic**, contradicting the module's advertised determinism. | Add the id tie-break (match `nearest_k_ids`). |
| `workspace.rs:80-94` + `consistency.rs` + `config.rs:563-574` | major | A wrong-but-exact decode (≥2-bit error landing on another codeword's rotation; possible since min cyclic distance = 2) is cleared only with `support==0 && contradictions≥2`; in sparse views it survives. The docstring's "no wrong IDs reach the global filter" is overstated. | Soften the docstring; gate soft-locked ids on H-reprojection error when a homography exists. |
| `consistency.rs:10-40,75-140` | major | Consistency is **relative** (reads neighbors' current ids), so a self-consistent **wrong cluster** (60° hex-symmetry rotation, or a lattice translation with no correct anchor) mutually supports itself and is never flagged. | Anchor consistency to absolute strong decodes / the homography. |
| `id_correction/homography.rs:272` + `config.rs:661` | major | `homography_min_trusted = 24` disables the geometric fallback for any scene with <24 trusted markers — exactly the sparse/partial views where local voting is weakest. | Lower the default or scale it to the visible marker count. |
| `marker/codec.rs:182-197` | minor | `margin` second-best includes other rotations of the **same** codeword → understated; for the 63 codewords with self-rotation distance 2 a 1-bit error can spuriously trip `MarginTooLow`. Precision-safe (only false rejects). | Compute margin against the second-best **distinct** id. |
| `id_correction/math.rs:50-104` | minor | `fit_local_affine` uses normal equations with only a `|pivot|<1e-12` check; a near-collinear neighbor triple yields an ill-conditioned but accepted affine. | Add a conditioning/area check before trusting the affine. |

---

## Cross-cutting themes

1. **NaN-safety.** `partial_cmp(...).unwrap()` appears in hot reduction paths
   (`ring/radial_profile.rs`, `ring/outer_estimate.rs`) reachable from the
   confirmed `Some(NaN)` leak in `edge_sample.rs`. A single non-finite guard at
   the sampler plus `total_cmp` in the reducers removes the whole class.
2. **Scale-dependent absolute epsilons** in otherwise scale-invariant math
   (`conic/types.rs`, `eigen.rs`) — make thresholds relative.
3. **Circle-vs-ellipse sampling** under perspective (`decode.rs`, and the outer
   estimator's circular sampling) — the recurring source of accuracy loss on
   tilted boards.
4. **Determinism** is sound except `nearest_within`; the equal-confidence
   tie-breaks also make a few outputs input-order-dependent.
5. **Test coverage** is the inverse of complexity: `radial_profile.rs` (0),
   the ring estimators (1 each), and 13/15 `id_correction` files (0) are exactly
   where these findings concentrate. Every reviewer supplied concrete fixtures.

## Recommended remediation order (each as a separately-validated change)

1. **Safe, behavior-preserving (no detection-result change):** non-finite guard
   in `edge_sample.rs` + `total_cmp` in `radial_profile.rs` (kills the panic
   class); `nearest_within` id tie-break (determinism). Pin with tests.
2. **Critical correctness:** add the confirm-by-consistency promotion path so
   correct non-exact IDs survive in sparse views — validate against the rtv3d
   benchmark (sparse real data) so recall does not regress.
3. **Accuracy under perspective:** sample the code band (and the outer estimator)
   on the fitted ellipse instead of a circle — validate against the distortion
   benchmark.
4. **Robustness majors:** single-weighted affine vote; orientation-aware scale
   fallback; thin-ring degeneracy surfacing; relative-consistency anchoring.
5. **Backfill the supplied unit-test fixtures** for the zero/low-coverage
   modules to lock current behavior before any of the above lands.

## Missing capabilities

- No fit covariance/uncertainty for ellipses or homography (would let ID
  correction and the global filter weight by confidence rather than raw values).
- No orientation estimate in the sparse-neighborhood ID fallback.
- No homography-gated clearing of wrong-but-exact decodes.

**Bottom line:** ship-quality core math with a precision-first ID-correction
stage that trades recall for precision more aggressively than its docstring
claims — fix the one critical promotion gap and the NaN-panic class first, then
the perspective-sampling and robustness majors, each gated on the existing
benchmark suite so the certified accuracy does not regress.
