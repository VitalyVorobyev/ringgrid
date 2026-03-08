# Architect Report - TASK-009-detect-config-json-cache

- Task ID: `TASK-009-detect-config-json-cache`
- Role: `architect`
- Date: `2026-03-08`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/templates/task-handoff-report.md`
- `docs/handoffs/README.md`
- `.agents/skills/architect/SKILL.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid-py/pyproject.toml`
- `.github/workflows/ci.yml`
- `crates/ringgrid/src/detector/config.rs`

## Summary
- The next unstarted highest-priority backlog item is `INFRA-009`, mapped here to `TASK-009-detect-config-json-cache`.
- `ringgrid-py` currently resolves each `DetectConfig` property getter through a full native `dump_json()` round-trip plus `json.loads`, and convenience setters pay the same read cost before applying a one-field overlay.
- The intended fix is an internal Python-side resolved-config cache with explicit invalidation/refresh rules, while keeping Rust authoritative for derived marker-scale behavior.
- Scope stays inside `ringgrid-py` internals, regression tests, and a small microbenchmark harness; no user-facing API expansion is planned.

## Decisions Made
- Use `TASK-009-detect-config-json-cache` as the required handoff id for backlog item `INFRA-009` because no prior handoff exists for this work.
- Keep the optimization in `crates/ringgrid-py/python/ringgrid/_api.py` rather than introducing a new Rust/PyO3 config API unless the Python-only cache proves insufficient.
- Treat `marker_scale` as a forced refresh/invalidation path because Rust re-derives several public fields from that prior; do not duplicate `apply_marker_scale_prior` logic in Python.
- Require a dedicated, deterministic microbenchmark for named hot getter/setter paths, but keep it out of default CI.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-py/python/ringgrid/_api.py` - add resolved-config cache plumbing, invalidation rules, and cache-aware getters/setters.
- `crates/ringgrid-py/tests/test_api.py` - extend parity/regression coverage for `DetectConfig` cache behavior and detector refresh.
- `crates/ringgrid-py/tools/benchmark_detect_config.py` - add a small microbenchmark script for before/after getter/setter timing.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- `marker_scale` mutates multiple derived public sections (`proposal`, `edge_sample`, `outer_estimation`, `completion`, `projective_center`) via Rust-side logic. If Python tries to patch these locally, parity drift is likely.
- The backlog acceptance says "hot getters/setters" but does not name exact benchmark targets. This plan assumes the 4x gate applies to representative scalar and section accessors that should not require a native full refresh; report `marker_scale` separately if it remains a refresh-triggering special case.
- If any other overlay path performs hidden normalization in Rust, Python cache patching for that field must be replaced with invalidate-and-refresh rather than mirroring logic.

## Next Handoff
- To: `Implementer`
- Requested action: capture a pre-change microbenchmark baseline, implement the cache/invalidation plan in `ringgrid-py`, add parity tests, rerun the benchmark, and document before/after numbers plus any special-case limitations.

---

## Architect Required Sections

### Problem Statement
- `DetectConfig` in `ringgrid-py` currently pays full-config JSON serialization/deserialization cost for routine property access. Every getter rebuilds state from `self._core.dump_json()`, and convenience setters first read a full snapshot before writing a single field back through an overlay.
- This makes notebook-style tuning and repeated config mutation much slower than necessary even though the native config already exists in memory.
- The backlog explicitly requires removing this churn without changing visible behavior: `to_dict()` parity must hold, overlay semantics must remain stable, and hot getter/setter paths must improve by at least 4x versus the current baseline.

### Scope
- In scope:
  - Add an internal resolved-config cache in Python `DetectConfig`.
  - Define cache invalidation/refresh rules so routine getters and simple setters avoid per-access full snapshot dump/load.
  - Preserve current public Python API, config versioning, `Detector` refresh behavior, and visible `to_dict()` output.
  - Add regression tests for cache correctness and a deterministic microbenchmark for before/after measurement.
- Out of scope:
  - Public API additions or breaking changes in `ringgrid-py`.
  - Rust `ringgrid::DetectConfig` schema redesign or new public config wire format.
  - Benchmark enforcement in CI.
  - Broader performance work outside `DetectConfig` property access.

### Constraints
- Rust remains the source of truth for normalization and derived config coupling, especially marker-scale-derived fields.
- Avoid duplicating detector config derivation logic across Python and Rust.
- Preserve installed-package behavior; the change must remain within the package surface already exercised by `maturin develop` and `pytest`.
- Keep diffs reviewable and isolated to `ringgrid-py` internals/tests/benchmark tooling.

### Assumptions
- `INFRA-009` is the correct "next item" because it is the first `todo` item in the Active Sprint with `P0` priority.
- `DetectConfigCore` is mutated only through the Python wrapper methods in `_api.py`; there is no external writer bypassing cache invalidation.
- Returning fresh typed section objects from getters is part of current behavior and must remain unchanged, even after adding caching.
- A local benchmark script committed in `crates/ringgrid-py/tools/` is acceptable evidence for the acceptance target, with before/after numbers recorded in the implementer handoff.

### Affected Areas
- `crates/ringgrid-py/python/ringgrid/_api.py` - `DetectConfig.__init__`, snapshot helpers, `to_dict()`, `_config_json()`, `_apply_overlay()`, and all config property getters/setters.
- `crates/ringgrid-py/tests/test_api.py` - new assertions around cache parity, copy-on-read semantics, derived-field refresh after `marker_scale`, and detector core refresh after config updates.
- `crates/ringgrid-py/tools/benchmark_detect_config.py` - deterministic hot-path timing script for representative getters/setters.
- `.github/workflows/ci.yml` - no planned change; existing Python package validation commands remain the reference gate.

### Plan
1. Refactor `DetectConfig` snapshot handling in `crates/ringgrid-py/python/ringgrid/_api.py`.
   - Add a private resolved snapshot cache and helper methods to read the cached dump, lazily refresh from native when invalidated, and serialize config JSON from the cached resolved state.
   - Update getters to read from the cached resolved mapping instead of calling `dump_json()` every time.
   - Update setters to continue applying overlays through native code, then either patch the cached resolved mapping for simple overlays or invalidate/refresh when native derivation can affect multiple fields.
   - Risk mitigation: if per-overlay patch logic becomes brittle, prefer broader invalidate-and-lazy-refresh over mirroring Rust behavior.
2. Add regression coverage for behavior parity in `crates/ringgrid-py/tests/test_api.py`.
   - Keep existing API-level tests and add focused assertions for repeated getter access, convenience setters, typed section round-trips, `marker_scale`-driven derived updates, and mutation-without-reassignment remaining a no-op.
   - Confirm `Detector` still rebuilds from the latest config version after cached-state changes.
3. Add a deterministic microbenchmark and capture acceptance numbers.
   - Introduce `crates/ringgrid-py/tools/benchmark_detect_config.py` to measure representative scalar and section getters/setters on a warm `DetectConfig`.
   - Require the implementer to capture a baseline on the pre-change tree before editing, then rerun after the implementation and report before/after medians.
   - Use the 4x gate on named hot paths that should now avoid full dump/load churn; report any refresh-triggering special cases separately.

### Acceptance Criteria
- `DetectConfig` property getters no longer rely on `self._core.dump_json()` for every access once the resolved cache is warm.
- Simple property and section setters no longer require a pre-write full snapshot dump/load in Python.
- `cfg.to_dict()` remains behaviorally identical to the resolved public config dump after mixed setter sequences.
- `Detector` refresh after config mutation still uses the updated configuration and versioning behavior remains intact.
- `marker_scale` updates preserve current derived-field behavior exactly, even if that requires cache invalidation/refresh.
- A deterministic microbenchmark demonstrates at least 4x improvement versus the pre-change baseline on the named hot getter/setter scenarios documented in the benchmark script and implementer handoff.

### Test Plan
- Baseline capture before code changes:
  - `python3 -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `python3 crates/ringgrid-py/tools/benchmark_detect_config.py`
- Post-change validation:
  - `python3 crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `python3 -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `python3 -m pytest crates/ringgrid-py/tests -q`
  - `python3 crates/ringgrid-py/tools/benchmark_detect_config.py`
- Expected evidence:
  - parity tests covering `to_dict()`, section setters, convenience setters, `marker_scale` refresh behavior, and detector refresh all pass
  - benchmark output includes before/after timings for the agreed hot getter/setter scenarios and shows the required >=4x improvement on the gated paths

### Out Of Scope
- Adding new `DetectConfigCore` PyO3 methods solely for ergonomics if the Python cache design suffices.
- Reworking `DetectorCore` construction or the Rust detector pipeline.
- Changing README/docs beyond what is necessary to explain benchmark/test execution in code comments or the handoff.
- Expanding this task into broader `ringgrid-py` performance tuning unrelated to config access.

### Handoff To Implementer
- Capture and record the current microbenchmark baseline before editing any code so the acceptance delta is measured against the true pre-change implementation.
- In `crates/ringgrid-py/python/ringgrid/_api.py`, add a private resolved snapshot cache and route getters through it while preserving current copy-on-read behavior for typed section objects.
- Keep `_core.apply_overlay_json(...)` as the write path; after each setter, either patch the cached public dump for simple overlays or invalidate/refresh when native derivation can affect multiple public fields.
- Do not duplicate Rust marker-scale derivation formulas in Python. For `marker_scale`, prefer refresh-from-native after the overlay is applied.
- Add regression tests in `crates/ringgrid-py/tests/test_api.py` for parity and cache invalidation edge cases, especially `marker_scale` and mutation-without-reassignment.
- Add `crates/ringgrid-py/tools/benchmark_detect_config.py`, rerun the before/after benchmark, and document exact numbers plus any non-gated special cases in `02-implementer.md`.
