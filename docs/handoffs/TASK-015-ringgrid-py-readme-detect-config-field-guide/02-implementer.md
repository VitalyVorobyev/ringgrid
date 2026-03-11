# Implementer Report - TASK-015-ringgrid-py-readme-detect-config-field-guide

- Task ID: `TASK-015-ringgrid-py-readme-detect-config-field-guide`
- Backlog ID: `DOCS-001`
- Role: `implementer`
- Date: `2026-03-11`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/03-reviewer.md`
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/01-architect.md`
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/02-implementer.md` (prior revision)
- `.agents/skills/implementer/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `crates/ringgrid-py/README.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/python/ringgrid/__init__.pyi`
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid/src/marker/decode.rs`

## Summary
- Kept the original `DOCS-001` deliverable intact: `crates/ringgrid-py/README.md` still contains the full Python `DetectConfig` field guide, and the Python/stub surface still exposes `DecodeConfig.codebook_profile`.
- Addressed the reviewer’s blocking finding by making `DecodeConfig.from_dict(...)` default a missing `codebook_profile` back to `"base"` instead of raising `KeyError`.
- Added a focused regression test that exercises a legacy decode-config mapping without `codebook_profile` and proves it round-trips to the resolved baseline decode config.
- Re-ran the full required validation baseline on the final tree: `fmt`, `clippy`, workspace tests, rustdoc, doctests, `mdbook`, typing-artifact check, `maturin develop`, and Python tests all passed.

## Decisions Made
- Preserved `codebook_profile` as a plain Python `str`.
  - Reason: the architect required the smallest compatibility-preserving parity fix, and the native wire contract remains the string values `"base"` / `"extended"`.
- Defaulted missing `codebook_profile` to `"base"` inside `DecodeConfig.from_dict(...)`.
  - Reason: this restores backward compatibility for legacy Python mappings and matches the native Rust default semantics the reviewer called out.
- Kept the README and typing artifacts unchanged in the rework patch.
  - Reason: the reviewer finding was isolated to constructor compatibility in `_api.py` plus missing regression coverage in `test_api.py`; no additional docs or stub churn was needed.

## Files/Modules Affected
- `crates/ringgrid-py/README.md` - existing field-guide deliverable retained unchanged in the rework.
- `crates/ringgrid-py/python/ringgrid/_api.py` - `DecodeConfig.from_dict(...)` now defaults missing `codebook_profile` to `"base"`.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - existing typed surface from the initial parity fix retained unchanged in the rework.
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - existing template parity update retained unchanged in the rework.
- `crates/ringgrid-py/tests/test_api.py` - added the reviewer-requested regression for legacy decode payloads without `codebook_profile`.

## Validation / Tests
- Required local CI baseline completed on the final tree after the reviewer-requested fix.
- Focused compatibility checks also passed before the full baseline.

## Risks / Open Questions
- `DecodeConfig.codebook_profile` remains intentionally stringly in Python.
  - Impact: this keeps the parity fix small and compatible, but enum-style ergonomics would need a separate API-shaping task.
- The README guard still relies on explicit headings/tokens.
  - Impact: future README reorganizations must update the guard in the same patch, but this rework did not widen that coupling.

## Next Handoff
- To: `Reviewer`
- Requested action: verify that the missing-key compatibility regression is resolved, confirm the new legacy-payload regression test is sufficient, and re-check that the broader README/config parity work still matches the live Python surface.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1 plus reviewer finding:
  - kept `DecodeConfig.codebook_profile` in the Python surface
  - repaired `DecodeConfig.from_dict(...)` so legacy payloads that omit the new field still default to the baseline profile
- Architect step 2:
  - retained the previously implemented README field guide unchanged because the reviewer did not find a docs accuracy problem
- Architect step 3 plus reviewer finding:
  - added a targeted regression test for the missing-key legacy payload path
  - re-ran the full Rust/Python/doc validation baseline on the final tree

### Changes Made
- Backward-compatibility fix:
  - changed `ringgrid.DecodeConfig.from_dict(...)` to use `"base"` when `codebook_profile` is absent
  - preserved the explicit `codebook_profile` round-trip when the field is present
- Tests:
  - added `test_decode_config_from_dict_defaults_missing_profile_to_base`
  - kept the existing parity test that `cfg.decode.to_dict()` matches `cfg.to_dict()["decode"]`
- No new README or stub edits were required in the rework patch.

### Files Changed
- `crates/ringgrid-py/python/ringgrid/_api.py` - backward-compatible default for missing `codebook_profile`.
- `crates/ringgrid-py/tests/test_api.py` - regression coverage for legacy decode payloads.
- `crates/ringgrid-py/README.md` - earlier field-guide expansion retained as part of the task output.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - earlier parity/stub update retained as part of the task output.
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - earlier typing-template update retained as part of the task output.

### Deviations From Plan
- None.

### Tests Added/Updated
- `crates/ringgrid-py/tests/test_api.py`
  - retained `test_decode_config_matches_resolved_dump_surface_and_profile_override`
  - added `test_decode_config_from_dict_defaults_missing_profile_to_base`
  - retained `test_readme_detect_config_field_guide_covers_python_surface`

### Commands Run
- `./.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k 'detect_config or readme or typing'`
- `./.venv/bin/python - <<'PY' ... legacy = dict(cfg.to_dict()["decode"]); legacy.pop("codebook_profile"); parsed = ringgrid.DecodeConfig.from_dict(legacy) ... PY`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo doc --workspace --all-features --no-deps`
- `cargo test --doc --workspace`
- `mdbook build book`
- `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
- `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
- `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q`

### Results
- `pytest crates/ringgrid-py/tests/test_api.py -q -k 'detect_config or readme or typing'` passed: `6 passed, 25 deselected`.
- Manual compatibility repro passed:
  - parsed legacy payload reported `codebook_profile == "base"`
  - parsed legacy payload `to_dict()` matched the resolved baseline `cfg.to_dict()["decode"]`
- `cargo fmt --all --check` passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
- `cargo test --workspace --all-features` passed:
  - `153` Rust unit tests
  - `4` target-generation integration tests
  - `6` CLI tests
  - `5` doctests inside the workspace test run
- `cargo doc --workspace --all-features --no-deps` passed.
- `cargo test --doc --workspace` passed: `5` doctests.
- `mdbook build book` passed.
- `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check` passed: `typing artifacts are up to date`.
- `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release` passed and reinstalled the editable package.
- `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q` passed: `31 passed in 2.63s`.

### Remaining Concerns
- The Python surface is now backward-compatible for the added `codebook_profile` field, but it still exposes that selector as a raw string.
- The README coverage guard is intentionally structural rather than semantic; it catches surface drift, not prose-quality drift.

### Handoff To Reviewer
- Focus first on the reviewer-requested compatibility repair:
  - `crates/ringgrid-py/python/ringgrid/_api.py`
  - `crates/ringgrid-py/tests/test_api.py`
- Verify that `DecodeConfig.from_dict(...)` now accepts the pre-fix decode payload shape with no `codebook_profile` key and defaults it to `"base"`.
- Confirm the full task output still holds:
  - README field guide matches the live Python `DetectConfig` surface
  - typed Python/stub surface still exposes `decode.codebook_profile`
  - focused guard remains meaningful without additional churn
