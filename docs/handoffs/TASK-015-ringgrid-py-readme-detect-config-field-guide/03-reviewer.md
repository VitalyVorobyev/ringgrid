# Reviewer Report - TASK-015-ringgrid-py-readme-detect-config-field-guide

- Task ID: `TASK-015-ringgrid-py-readme-detect-config-field-guide`
- Backlog ID: `DOCS-001`
- Role: `reviewer`
- Date: `2026-03-11`
- Status: `complete`

## Inputs Consulted
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/03-reviewer.md` (previous revision)
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/01-architect.md`
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/02-implementer.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/reviewer/SKILL.md`
- `crates/ringgrid-py/README.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/python/ringgrid/__init__.pyi`
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid/src/marker/decode.rs`
- reviewer-reproduced command outputs listed below

## Summary
- The reviewer-requested compatibility fix is present: `DecodeConfig.from_dict(...)` now defaults a missing `codebook_profile` to `"base"` instead of raising `KeyError`.
- The added regression test covers the missing-key legacy payload path, and the earlier README/config parity work remains intact.
- Reproduced the full required local CI baseline (`fmt`, `clippy`, workspace tests, rustdoc, doctests, `mdbook`, typing-artifact check, `maturin develop`, Python tests); all passed.
- Found no remaining blocking issues. The task now satisfies the architect acceptance criteria and prior reviewer requests.

## Decisions Made
- Treated the missing-key default in `DecodeConfig.from_dict(...)` as the required fix for the prior blocker.
  - Reason: it restores backward compatibility for pre-fix Python decode mappings and matches the native default-to-`base` behavior.
- Accepted the existing README field guide and surface-coverage guard without further changes.
  - Reason: reviewer reruns confirmed the documented Python surface and the live package surface remain aligned after the rework.

## Files/Modules Reviewed
- `crates/ringgrid-py/README.md` - field-guide completeness and live-surface alignment.
- `crates/ringgrid-py/python/ringgrid/_api.py` - runtime Python config surface, especially `DecodeConfig.from_dict(...)`.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - public typing surface for `DecodeConfig.codebook_profile`.
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - stub template alignment.
- `crates/ringgrid-py/tests/test_api.py` - parity coverage, README drift guard, and the new legacy-payload regression.
- `crates/ringgrid/src/marker/decode.rs` - native default semantics for `codebook_profile`.

## Validation / Tests
- Commands run:
  - `./.venv/bin/python - <<'PY' ... legacy = dict(cfg.to_dict()["decode"]); legacy.pop("codebook_profile"); parsed = ringgrid.DecodeConfig.from_dict(legacy) ... PY`
  - `./.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k 'detect_config or readme or typing'`
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- Results:
  - Manual legacy-payload repro passed:
    - parsed config reported `codebook_profile == "base"`
    - parsed config `to_dict()` matched the resolved baseline decode payload
  - `pytest crates/ringgrid-py/tests/test_api.py -q -k 'detect_config or readme or typing'` passed: `6 passed, 25 deselected`.
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
  - `generate_typing_artifacts.py --check` passed: `typing artifacts are up to date`.
  - `maturin develop` passed and reinstalled the editable package.
  - `pytest crates/ringgrid-py/tests -q` passed: `31 passed in 2.55s`.

## Risks / Open Questions
- `DecodeConfig.codebook_profile` remains a plain Python string surface.
  - Impact: this is acceptable for the architected minimal parity fix, but enum ergonomics would be a separate future API task rather than a blocker here.
- The README coverage guard is inventory-oriented rather than prose-semantic.
  - Impact: it catches surface drift well enough for this task, but it is not intended to validate narrative quality.

## Next Handoff
- To: `Human`
- Requested action: accept the task as complete and merge when convenient.

---

## Reviewer Required Sections

### Review Scope
- Re-reviewed the implementation against:
  - the architect requirements for a complete Python-facing `DetectConfig` field guide
  - parity exposure for `decode.codebook_profile`
  - a focused README drift guard
  - the prior reviewer-requested compatibility fix for legacy decode-config payloads
- Focused on correctness of the public Python config surface, backward compatibility, test adequacy, and the full recorded CI baseline.

### Inputs Reviewed
- prior `03-reviewer.md`
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/01-architect.md`
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/02-implementer.md`
- actual changed code, tests, stubs, and README content
- reviewer-reproduced local CI and targeted compatibility outputs

### What Was Checked
- The missing-key path in `DecodeConfig.from_dict(...)` for legacy payloads.
- The new regression test covering that missing-key path.
- Continued parity between:
  - the live resolved `DetectConfig(...).to_dict()` surface
  - the typed Python `DecodeConfig`
  - the committed typing artifacts
  - the README field guide
- The full local CI baseline the implementer recorded.

### Findings
- No blocking findings.
- Previous reviewer finding is resolved.
  - Evidence: [`crates/ringgrid-py/python/ringgrid/_api.py`](/Users/vitalyvorobyev/vision/ringgrid/crates/ringgrid-py/python/ringgrid/_api.py#L1163) now uses `data.get("codebook_profile", "base")`, and reviewer reproduction of the legacy missing-key payload now succeeds.
  - Evidence: [`crates/ringgrid-py/tests/test_api.py`](/Users/vitalyvorobyev/vision/ringgrid/crates/ringgrid-py/tests/test_api.py#L416) adds `test_decode_config_from_dict_defaults_missing_profile_to_base`, which locks the compatibility path in place.

### Test Assessment
- Adequate for approval.
- Coverage now includes:
  - current-surface parity for `DecodeConfig`
  - README field-guide inventory coverage
  - the previously missing legacy payload compatibility path
- Reviewer also reproduced the full required baseline successfully on the current tree.

### Risks
- Residual risk is low and non-blocking.
- The remaining tradeoff is intentional:
  - `codebook_profile` is stringly typed in Python to keep the parity fix minimal and compatibility-preserving

### Required Changes Or Approval Notes
- Approval note:
  - the prior `changes_requested` item is resolved, and no additional implementer changes are required for `DOCS-001`.

### Final Verdict
- `approved`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: merge the approved implementation. No further implementer work is required for `DOCS-001`.
