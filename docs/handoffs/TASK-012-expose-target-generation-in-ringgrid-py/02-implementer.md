# Implementer Report - TASK-012-expose-target-generation-in-ringgrid-py

- Task ID: `TASK-012-expose-target-generation-in-ringgrid-py`
- Backlog ID: `INFRA-011`
- Role: `implementer`
- Date: `2026-03-09`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/03-reviewer.md`
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/01-architect.md`
- previous `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/02-implementer.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/implementer/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/tests-synthetic-fixtures/SKILL.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid-py/python/ringgrid/__init__.py`
- `crates/ringgrid-py/python/ringgrid/__init__.pyi`
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template`
- `crates/ringgrid-py/README.md`
- `README.md`

## Summary
- Kept the original `BoardLayout`-centered installed-package target-generation API and addressed the reviewer’s blocking stale-spec finding.
- `BoardLayout.to_spec_json(...)`, `write_svg(...)`, and `write_png(...)` now rebuild canonical spec JSON from the current public spec fields before emitting output, then refresh the in-memory marker snapshot/cache to match.
- Added a regression test that mutates a board after construction and proves the emitted spec JSON and generated artifacts follow the mutated board rather than the original cached spec.
- Re-ran the full required validation baseline after the rework; all required commands passed.

## Decisions Made
- Chose the mutable-board contract requested by the reviewer instead of making Python `BoardLayout` immutable.
  - Reason: this keeps the additive public API small and avoids a broader Python surface break while fixing the new target-generation methods to follow visible object state.
- Localized the rework to the target-generation path in `BoardLayout`.
  - Reason: the reviewer finding was tied to the newly added spec/SVG/PNG methods, and the fix can be expressed cleanly by rebuilding canonical spec JSON from current spec fields at emission time.
- Refreshed `markers` and `_spec_json` as part of that emission-time normalization.
  - Reason: when geometry fields change, the target-generation outputs and the in-memory board snapshot should converge on one canonical board state rather than only updating the emitted JSON string.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-py/python/ringgrid/_api.py` - reworked `BoardLayout` generation methods to normalize from current spec fields and refresh the cached snapshot before JSON/SVG/PNG output.
- `crates/ringgrid-py/tests/test_api.py` - added reviewer-requested mutation regression coverage for the new target-generation contract.
- `crates/ringgrid-py/src/lib.rs` - unchanged in the rework; still provides the thin Rust-backed target-generation bridge introduced in the original implementation.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - unchanged in the rework; public signatures did not change.
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - unchanged in the rework; public signatures did not change.
- `crates/ringgrid-py/README.md` - unchanged in the rework; installed-package target-generation docs remain accurate.
- `README.md` - unchanged in the rework; top-level Python snippet remains accurate.

## Validation / Tests
- Commands run:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k 'target_generation or mutated_spec_fields'`
- Results:
  - `cargo fmt --all --check` passed.
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
  - `cargo test --workspace --all-features` passed.
  - `cargo doc --workspace --all-features --no-deps` passed.
  - `cargo test --doc --workspace` passed.
  - `mdbook build book` passed.
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check` passed (`typing artifacts are up to date`).
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release` passed.
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q` passed (`28 passed`).
  - Focused regression slice passed (`3 passed`) and confirmed the reviewer-reported mutation case no longer reproduces.

## Risks / Open Questions
- `BoardLayout` remains a mutable public dataclass, so callers can still create transient inconsistencies by editing spec fields and inspecting `markers` before any emission path re-normalizes the object.
- The broader detection/config flow still snapshots board state when the detector/config core is built; this rework is intentionally scoped to the reviewer’s target-generation finding rather than a full redesign of mutable `BoardLayout` semantics across the entire Python API.
- The book target-generation chapter still emphasizes the repo-tool workflow; that documentation drift remains outside this task scope.

## Next Handoff
- To: `Reviewer`
- Requested action: verify that the reviewer-requested mutable-board contract is now explicit in behavior, confirm the new mutation regression closes the stale `_spec_json` finding, and re-check that the rest of the target-generation surface still matches the architect plan.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1 remained intact: the Rust target-generation engine is still the sole implementation for canonical spec JSON and SVG/PNG output.
- Architect step 2 remained intact: the public Python target-generation surface stays `BoardLayout`-centered.
- Reviewer finding was implemented as a bounded rework inside that plan: the new generation methods now derive canonical spec JSON from current public spec fields instead of relying on stale cached `_spec_json`.
- Architect step 3 was extended with reviewer-requested regression coverage for post-construction board mutation.

### Changes Made
- Added `BoardLayout._refresh_from_current_spec_fields()` in `crates/ringgrid-py/python/ringgrid/_api.py`.
  - Builds canonical `ringgrid.target.v3` JSON from `to_spec_dict()`.
  - Rehydrates a fresh Rust-backed board snapshot from that spec JSON.
  - Updates `schema`, `name`, geometry fields, `markers`, and `_spec_json` on the existing Python object.
- Updated `BoardLayout.to_spec_json(...)` to use the refreshed canonical spec instead of the originally cached `_spec_json`.
- Updated `BoardLayout.write_svg(...)` and `BoardLayout.write_png(...)` to emit from that refreshed canonical spec as well.
- Added a regression test that mutates `name` and `pitch_mm` after board construction, then verifies:
  - emitted spec JSON reflects the mutation,
  - the board’s marker snapshot is refreshed,
  - generated SVG/PNG outputs differ from the original fixture as expected for the mutated geometry.

### Files Changed
- `crates/ringgrid-py/python/ringgrid/_api.py` - target-generation methods now normalize from current public spec fields and refresh the board snapshot.
- `crates/ringgrid-py/tests/test_api.py` - added mutation regression coverage for the reviewer-requested contract.
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/02-implementer.md` - updated implementation handoff for the rework cycle.

### Deviations From Plan
- None.
- The rework stays within architect scope and directly addresses the reviewer’s explicit change request.

### Tests Added/Updated
- `crates/ringgrid-py/tests/test_api.py`
  - retained the original fixture-parity and error-contract coverage,
  - added `test_board_layout_target_generation_tracks_mutated_spec_fields(...)` to lock the mutable-board emission contract and prevent stale `_spec_json` regressions.

### Commands Run
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo doc --workspace --all-features --no-deps`
- `cargo test --doc --workspace`
- `mdbook build book`
- `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
- `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
- `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k 'target_generation or mutated_spec_fields'`

### Results
- The reviewer-reported stale `_spec_json` bug no longer reproduces.
- The Python target-generation methods now follow the current public board spec fields instead of the original construction-time snapshot.
- Mutation regression coverage is in place and passing.
- The full required validation baseline passed after the rework.

### Remaining Concerns
- `BoardLayout` normalization currently happens on the target-generation/spec-emission path, not on every arbitrary field assignment. That is enough to satisfy the new emission contract, but it is not a full immutability or always-synchronized-object redesign.
- Because the public API signatures did not change, no stub/template regeneration was required beyond verifying `generate_typing_artifacts.py --check`.

### Handoff To Reviewer
- Focus on `crates/ringgrid-py/python/ringgrid/_api.py` to confirm the mutable-board contract is now implemented in the generation paths without expanding scope unnecessarily.
- Focus on `crates/ringgrid-py/tests/test_api.py` to confirm the new mutation regression covers the stale-spec failure the reviewer reproduced.
- Confirm that the full validation baseline and the focused regression slice are sufficient to clear the prior `changes_requested` verdict.
