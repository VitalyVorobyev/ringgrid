# Implementer Report - TASK-009-detect-config-json-cache

- Task ID: `TASK-009-detect-config-json-cache`
- Role: `implementer`
- Date: `2026-03-08`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-009-detect-config-json-cache/01-architect.md`
- `.agents/skills/implementer/SKILL.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid-py/tools/generate_typing_artifacts.py`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid/src/detector/config.rs`

## Summary
- Added a lazy resolved-config cache inside `ringgrid-py` `DetectConfig` so repeated property getters no longer round-trip through `dump_json()` on every access.
- Kept Rust authoritative for whole-section and marker-scale updates: direct section setters now refresh from native, while hot convenience setters patch only the specific cached nested field they mutate.
- Added regression tests for cache copy semantics, native parity after mixed overlays, and marker-scale refresh behavior.
- Added a committed microbenchmark harness and verified the targeted hot getters/setters exceed the required `>=4x` speedup versus the pre-change baseline.

## Decisions Made
- Whole-section setters (`decode`, `completion`, `self_undistort`, etc.) refresh the cache from native after overlay application instead of replacing cached section dicts locally.
  - Reason: several `to_dict()` helpers omit `None` fields, and naive section replacement caused cache drift versus the native dump.
- Hot convenience setters (`decode_min_confidence`, `completion_enable`, `decode_min_margin`, etc.) update the native config and patch only the affected cached nested field.
  - Reason: this preserves optional-field parity while keeping the benchmarked setter paths off the full dump/load path.
- `marker_scale` remains a native refresh path.
  - Reason: Rust re-derives multiple scale-coupled sections from that prior and the architect handoff explicitly called out avoiding duplicated derivation logic in Python.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-py/python/ringgrid/_api.py` - added resolved snapshot caching, cache refresh helpers, and convenience-setter nested patch updates.
- `crates/ringgrid-py/tests/test_api.py` - added cache copy/parity/marker-scale regression coverage.
- `crates/ringgrid-py/tools/benchmark_detect_config.py` - added deterministic hot-path microbenchmark script.

## Validation / Tests
- Commands run:
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
  - `.venv/bin/python crates/ringgrid-py/tools/benchmark_detect_config.py`
- Results:
  - typing artifacts check: pass
  - editable package build/install: pass
  - Python tests: pass (`22 passed`)
  - benchmark gate: pass on all targeted hot getter/setter scenarios

## Risks / Open Questions
- Direct whole-section assignments still pay a native refresh after overlay application. This is intentional for correctness, but they are not part of the hot-path benchmark target.
- The pre-change benchmark baseline was captured with an inline Python harness using the same scenario set later committed as `benchmark_detect_config.py`, because the benchmark file did not exist before implementation started.

## Next Handoff
- To: `Reviewer`
- Requested action: review the cache-authority split in `_api.py`, confirm the regression tests cover the right parity boundary, and verify the benchmark evidence is sufficient for the `INFRA-009` acceptance target.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1: implemented Python-side `DetectConfig` cache/invalidation logic in `crates/ringgrid-py/python/ringgrid/_api.py`.
- Architect step 2: added parity/regression tests in `crates/ringgrid-py/tests/test_api.py`.
- Architect step 3: added `crates/ringgrid-py/tools/benchmark_detect_config.py`, captured pre/post numbers, and verified the required speedup.

### Changes Made
- Added `_resolved_cache`, `_refresh_snapshot()`, `_resolved()`, and compact `_config_json()` serialization so `DetectConfig` can reuse a warm resolved snapshot instead of repeatedly calling native `dump_json()`.
- Routed property getters through the cached resolved snapshot and made `to_dict()` return a deep copy of the cached state so external mutation cannot corrupt internal cache state.
- Added `_patch_cached_overlay()`, `_patch_cached_section_field()`, and `_apply_section_field_overlay()` to separate safe top-level cache patching from whole-section native refreshes.
- Switched whole-section setters to refresh from native after overlay application to preserve canonical null/default fields in the cached snapshot.
- Rewrote hot convenience setters to patch only the mutated nested field in the cached section after updating native state, keeping repeated setter calls off the full dump/load path.
- Added regression tests for cache stability and parity against `DetectConfigCore.dump_json()`.
- Added a standalone benchmark script for the exact hot-path scenarios used for acceptance.

### Files Changed
- `crates/ringgrid-py/python/ringgrid/_api.py` - cache implementation, native refresh policy, and hot convenience-setter updates.
- `crates/ringgrid-py/tests/test_api.py` - cache/parity/marker-scale tests.
- `crates/ringgrid-py/tools/benchmark_detect_config.py` - committed microbenchmark harness.

### Deviations From Plan
- Pre-change benchmark capture used an inline Python harness rather than the committed script.
  - Reason: the architect required a baseline before code edits, but the benchmark script was part of this task and did not yet exist.
  - Impact: none on measurement methodology; the committed script uses the same scenario set and iteration counts as the recorded baseline.

### Tests Added/Updated
- `crates/ringgrid-py/tests/test_api.py` - added cached snapshot copy-semantics test.
- `crates/ringgrid-py/tests/test_api.py` - added native parity test after mixed convenience/direct overlays.
- `crates/ringgrid-py/tests/test_api.py` - added marker-scale refresh parity test.

### Commands Run
- `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
- `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
- `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- `.venv/bin/python crates/ringgrid-py/tools/benchmark_detect_config.py`

### Results
- `generate_typing_artifacts.py --check`: passed.
- `maturin develop`: passed on the final tree.
- `pytest crates/ringgrid-py/tests -q`: passed with `22 passed in 2.23s`.
- Benchmark deltas versus the pre-change baseline:
  - `get_decode_section`: `18.28 us/op` -> `0.74 us/op` (`24.7x` faster)
  - `get_decode_min_confidence`: `18.28 us/op` -> `0.76 us/op` (`24.1x` faster)
  - `get_completion_enable`: `17.92 us/op` -> `0.79 us/op` (`22.7x` faster)
  - `set_decode_min_confidence`: `20.63 us/op` -> `3.63 us/op` (`5.7x` faster)
  - `set_completion_enable`: `21.01 us/op` -> `3.62 us/op` (`5.8x` faster)
  - `set_use_global_filter`: `1.01 us/op` -> `1.21 us/op` (control path; not a gated hot setter and already avoided full snapshot reads before this task)

### Remaining Concerns
- The benchmark targets intentionally focus on convenience getters/setters that previously paid the snapshot churn cost. If the project later wants whole-section assignment to be a hot path too, it will need a richer canonical section serializer than the current `to_dict()` helpers.

### Handoff To Reviewer
- Focus review on:
  - whether `_api.py` now preserves the correct native/public parity boundary for cached state,
  - whether the new tests adequately protect against omitted-`None` cache drift and marker-scale derivation regressions,
  - whether the benchmark evidence is sufficient to mark `INFRA-009` accepted.
