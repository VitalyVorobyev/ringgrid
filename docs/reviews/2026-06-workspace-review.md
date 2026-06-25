# ringgrid workspace review — 2026-06-25

Scope: a health review of the `ringgrid` workspace focused on practical value,
project design (SOLID), DRY, and the completeness/correctness of tests, comments,
and documentation. Every claim is backed by a `file:line` reference, a metric, or
a command that was run. Findings that were **actioned during this review** are
marked ✅ Resolved.

> Snapshot: 4 crates, ~20.2k LOC in the core crate across 75 source files,
> **210 passing tests** (`cargo test --workspace`), zero `TODO`/`FIXME`,
> **0 `missing_docs`** warnings.

---

## 1. Practical value — strong

ringgrid does something genuinely useful and hard, with no shortcuts:

- A complete, OpenCV-free detection pipeline: proposal → ellipse fit → 16-sector
  decode → dedup → projective-center correction → hex-neighbor ID consensus →
  RANSAC homography → completion of missing markers (`pipeline/`, `detector/`).
- Sub-pixel accuracy verified against synthetic ground truth: with
  projective-center refinement, precision/recall **1.0**, mean center error
  **0.060 px**, homography-vs-GT **0.019 px** (regenerated this review via
  `tools/run_reference_benchmark.sh`; comfortably inside
  `tools/ci/regression_baseline.json`).
- Three usable surfaces from one core: Rust library, `ringgrid` CLI, PyO3 and
  WASM bindings — the WASM demo runs the detector live in a browser.
- Real-world robustness features that go beyond a proof of concept: adaptive
  multi-scale tiers, self-undistort estimation, distortion-aware sampling.

This is a polished, production-intent library, not a prototype.

## 2. Project design & SOLID — strong, with two large files to watch

- **Clean facade.** `lib.rs` is strictly re-exports (`lib.rs:53` onward); type
  definitions live at their construction sites, per the stated conventions.
- **Stable vs. diagnostic split is excellent.** The slim, stable
  `DetectionResult` is cleanly separated from the opt-in `DetectionDiagnostics`
  channel, with a single source of truth for the field partition
  (`pipeline/result.rs` `split_marker_record`). This is exactly the right shape
  for a library that must keep a stable output while still exposing internals.
- **Single responsibility, well-sized functions.** No function exceeds ~250 LOC;
  the pipeline is split into focused stages (`pipeline/{fit_decode,finalize,run}.rs`).
- **Configuration composes rather than mirrors.** Ten focused config structs in
  `detector/config.rs`, bundled via `AdvancedDetectConfig` → `DetectConfig`. No
  `*Params` vs `*Config` duplication anywhere.

Watch items (not defects):
- `detector/config.rs` (1,262 LOC) and `board_layout.rs` (1,186 LOC) are large.
  `board_layout.rs` in particular mixes geometry, JSON I/O, and validation;
  extracting the JSON loader into a submodule would improve cohesion.
- `pipeline/finalize.rs` (922 LOC) carries several distinct phases (filter,
  completion, refit). Functions are well-separated, so this is readability, not
  coupling.

## 3. DRY — good

- Thresholds and gates are config-driven, not copy-pasted algorithms.
- Minor, defensible duplication: `max_angular_gap_rad` is defined on both
  `InnerFitConfig` and `OuterFitConfig`, and each embeds a `RansacConfig` with
  intentionally different seeds. Acceptable; could share a default constant.

## 4. Tests — good coverage with sharp, correctness-critical gaps

210 tests pass. Coverage is strong where it is easy and **thin where the
algorithms are hardest**, which is the inverse of what you want:

Well covered: `board_layout.rs` (21), `conic/fit.rs` (15), `marker/decode.rs`
(14), `marker/codec.rs` (14), `api.rs` (12).

Under-covered, and these are core algorithms:
- `detector/id_correction/` — **13 of 15 files have zero tests** (only
  `vote.rs`=3 and `engine.rs`=4). The BFS consensus, bootstrap, consistency,
  local-affine, and homography-seeding logic — the part most likely to harbor
  subtle ID-assignment bugs — is essentially untested at the unit level.
- `ring/outer_estimate.rs` (498 LOC, **1 test**) and `ring/inner_estimate.rs`
  (413 LOC, **1 test**) — the radius-hypothesis estimators that everything
  downstream depends on.
- `detector/completion.rs` (712 LOC, 4 tests) and `ring/projective_center.rs`
  (541 LOC, 3 tests) carry heavy math relative to their test count.

These modules are exercised indirectly by end-to-end fixture tests, but lack the
focused unit tests that pin down edge-case behavior and catch regressions.

## 5. Documentation & comments — excellent

- **0 `missing_docs` warnings** across the public API (measured with
  `cargo rustc -p ringgrid --lib -- -W missing_docs`). The crate-level docs,
  per-method docs, coordinate-frame semantics, and `no_run` examples are all
  present and high quality.
- ✅ Resolved: added `#![warn(missing_docs)]` to `lib.rs` to **lock** this — it
  passes clean under CI's `-D warnings`, so the excellent coverage can no longer
  silently regress.
- mdBook guide (18 chapters) plus per-crate READMEs and `docs/` notes.
- Algorithm comments cite their methods (Fitzgibbon, RANSAC, projective center).

Gap: examples are `no_run`, so they are compile-checked but not executed; a few
`cargo test --doc` runnable examples would keep them honest over time.

## 6. Dependency health & error handling — the one real problem

- ⚠️→✅ **The workspace did not compile on arrival.** The working tree carried an
  uncommitted, half-finished dependency bump (`radsym` 0.1→0.2,
  `projective-grid` 0.5→0.9, `imageproc` 0.25→0.27, `pyo3`/`numpy` 0.28→0.29)
  with the source not adapted to the new APIs (`radsym` configs became
  `#[non_exhaustive]`; `projective_grid::GridIndex` was renamed `GridCoords`).
  **Resolved this review**: migrated `proposal/mod.rs` (post-default field
  assignment) and `finalize.rs`/`completion.rs` (`GridIndex`→`GridCoords`);
  bindings needed no changes. All crates build, 210 tests pass, accuracy is
  within baseline. **Recommendation:** run `/regression-gate` before merging the
  dependency bump, and consider a scheduled `cargo update` cadence so upgrades
  don't accumulate into a broken multi-crate jump.
- **~154 `unwrap()`/`expect()`/`panic!` in non-test code**, concentrated in
  `board_layout.rs` (31, mostly JSON load/validation). Malformed board JSON will
  panic rather than return an error. A `BoardError` enum with `Result`-returning
  loaders would make the library safe to embed against untrusted input.

---

## Prioritized recommendations

| # | Priority | Recommendation | Evidence |
|---|----------|----------------|----------|
| 1 | High | Run `/regression-gate` to certify the dependency migration before merge | radsym/projective-grid are major bumps to the proposal + grid engines |
| 2 | High | Unit-test `detector/id_correction/` (13/15 files untested) and the `ring/*_estimate` modules | §4 |
| 3 | Medium | Replace JSON-load/validation panics in `board_layout.rs` with a `BoardError` + `Result` | `board_layout.rs` 31 unwrap/expect |
| 4 | Medium | Extract JSON I/O from `board_layout.rs`; optionally split `detector/config.rs` per stage | §2 |
| 5 | Low | Add a few runnable doctests (`cargo test --doc`) | §5 |
| 6 | Low | Share `max_angular_gap_rad` / RANSAC defaults between inner/outer fit configs | §3 |
| ✅ | Done | `#![warn(missing_docs)]` added and passing | §5 |

## Positive signals

- Zero `TODO`/`FIXME` — no lingering debt markers.
- Strictly re-export `lib.rs`; no leaked internals.
- Stable-result vs. opt-in-diagnostics separation is exemplary.
- Fully documented public API, now lint-locked.
- Sub-pixel accuracy proven against ground truth, not asserted.
- Clean four-crate split (core / CLI / Python / WASM) with no cross-leakage.

**Bottom line:** a well-architected, well-documented, genuinely useful library.
The headline risks are operational (an unattended dependency upgrade that had
broken the build) and a test-coverage profile that is thinnest exactly where the
algorithms are most intricate. Both are addressable without redesign.
