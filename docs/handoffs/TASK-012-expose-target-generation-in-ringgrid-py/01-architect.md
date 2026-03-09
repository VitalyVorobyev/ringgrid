# Architect Report - TASK-012-expose-target-generation-in-ringgrid-py

- Task ID: `TASK-012-expose-target-generation-in-ringgrid-py`
- Backlog ID: `INFRA-011`
- Role: `architect`
- Date: `2026-03-09`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/templates/task-handoff-report.md`
- `docs/handoffs/README.md`
- `.agents/skills/architect/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/tests-synthetic-fixtures/SKILL.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/01-architect.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/02-implementer.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/03-reviewer.md`
- `crates/ringgrid/src/lib.rs`
- `crates/ringgrid/src/board_layout.rs`
- `crates/ringgrid/src/target_generation.rs`
- `crates/ringgrid/tests/target_generation.rs`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/python/ringgrid/__init__.py`
- `crates/ringgrid-py/python/ringgrid/__init__.pyi`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid-py/README.md`
- `crates/ringgrid-py/pyproject.toml`
- `README.md`
- `book/src/target-generation.md`

## Summary
- The next unstarted highest-priority backlog item is `INFRA-011`, mapped here to `TASK-012-expose-target-generation-in-ringgrid-py`.
- `INFRA-010` already delivered the Rust target-generation engine (`BoardLayout::new/with_name`, `to_json_string`, `write_json_file`, `write_target_svg`, `write_target_png`), but the installed Python package still exposes only board-load/detect flows and tells users to clone the repo and run `tools/gen_synth.py`.
- The recommended Python API is additive and `BoardLayout`-centered: construct a board from direct geometry args, emit canonical `ringgrid.target.v3` JSON, and write SVG/PNG through the Rust implementation instead of reimplementing geometry/rendering in Python.
- The key contracts to lock down are explicit spec-vs-snapshot semantics, installed-package usability without repo-root tools, and deterministic JSON/SVG/PNG parity tests reusing the committed Rust target-generation fixtures.

## Decisions Made
- Use `TASK-012-expose-target-generation-in-ringgrid-py` as the required handoff id for backlog item `INFRA-011` because no prior handoff exists for this work and `TASK-012` is the next unused task number.
- Keep the public Python surface small and board-centered (`api-shaping`): extend `BoardLayout` rather than introducing a parallel `TargetGenerator` class or a separate module-level generation subsystem.
- Route generation through Rust/PyO3 helpers, not Python-side geometry or file-format logic, so the installed package inherits the exact `INFRA-010` semantics for JSON/SVG/PNG.
- Prefer a Pythonic wrapper over a literal Rust API mirror:
  - direct geometry constructor with optional name,
  - explicit spec-JSON method,
  - file-oriented SVG/PNG methods with keyword-only options.
- Reuse the committed Rust target-generation fixtures and their contract stance (`tests-synthetic-fixtures`): JSON/SVG text can be compared deterministically, while PNG parity should be enforced through dimensions, `pHYs` metadata, and decoded pixels rather than compressed-byte identity.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-py/src/lib.rs` - native helper functions for board construction and JSON/SVG/PNG generation around the existing Rust `BoardLayout` methods.
- `crates/ringgrid-py/python/ringgrid/_api.py` - public `BoardLayout` constructor/method wrappers, path handling, and Python exception mapping.
- `crates/ringgrid-py/python/ringgrid/__init__.py` - public export surface if new symbols are added.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - installed-package type surface for any new `BoardLayout` APIs.
- `crates/ringgrid-py/tools/generate_typing_artifacts.py` and/or `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - typing artifact regeneration path if stub generation needs to learn the new API.
- `crates/ringgrid-py/tests/test_api.py` - installed-package target-generation parity and validation coverage.
- `crates/ringgrid-py/README.md` - installed-package target-generation quickstart and public examples.
- `README.md` - minimal Python target-generation snippet update if the current repo-tools instructions would otherwise stay stale after the API lands.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- `BoardLayout.to_dict()` currently returns the expanded snapshot including `markers`, while canonical `ringgrid.target.v3` JSON is spec-only. Any new Python JSON-generation method must make that distinction explicit to avoid writing the wrong payload.
- The root README and book target-generation chapter currently describe repo-tool workflows. This task should keep docs changes minimal, but if the visible Python quickstart remains repo-tool-only after the package API lands, user-facing drift will persist.
- The current Python `BoardLayout` dataclass is still field-constructible; this task should document and promote the new direct-geometry classmethod rather than attempting a broader redesign of the dataclass constructor.
- File-write failures deserve an explicit Python IO exception contract. If the existing `py_value_error(...)` helper is reused blindly, the public Python API may surface confusing `ValueError`s for filesystem problems.

## Next Handoff
- To: `Implementer`
- Requested action: add the installed-package `ringgrid-py` target-generation surface on top of the existing Rust engine, keep it `BoardLayout`-centered and file-oriented, prove JSON/SVG/PNG parity against the committed Rust fixtures, and update the Python docs so users no longer need repo-root tools just to generate targets.

---

## Architect Required Sections

### Problem Statement
- `INFRA-010` solved the Rust side of target generation, but `INFRA-011` remains open because the installed Python package still cannot generate board JSON or printable SVG/PNG through its public API.
- The current gap is concrete in the code and docs:
  - `crates/ringgrid/src/board_layout.rs` exposes direct-geometry construction plus canonical JSON emit/write,
  - `crates/ringgrid/src/target_generation.rs` exposes SVG/PNG render/write with stable print contracts,
  - `crates/ringgrid-py/src/lib.rs` only exports board-load/snapshot helpers plus detector/config cores,
  - `crates/ringgrid-py/python/ringgrid/_api.py` exposes `BoardLayout.default`, `from_json_file`, and `from_dict`, but no direct geometry constructor or JSON/SVG/PNG generation methods,
  - `crates/ringgrid-py/README.md` explicitly tells installed users to clone the repo and run `tools/gen_synth.py` for target generation.
- That means Python users cannot generate a board from wheel-only installs even though the Rust engine already exists, and the next backlog task (`INFRA-012`) cannot cleanly build on a shared installed-package Python surface.

### Scope
- In scope:
  - Add an additive public Python API in `ringgrid-py` for direct board construction from geometry args, backed by the Rust `BoardLayout` implementation.
  - Expose canonical spec JSON generation/writing from the installed Python package.
  - Expose file-oriented SVG and PNG writing from the installed Python package, reusing the Rust renderer and current print contracts.
  - Add deterministic parity tests for JSON/SVG/PNG against committed Rust fixtures for identical board args.
  - Update the Python package README, and any minimal top-level snippet required to keep the public installed-package story accurate.
- Out of scope:
  - Dedicated repo script or CLI wrapper (`INFRA-012`).
  - Synthetic image / ground-truth generation from Python.
  - DXF export.
  - Changes to the Rust target-generation semantics, fixture geometry, or codebook contents beyond any binding glue required here.
  - Broader README/book restructuring unrelated to this exact installed-package API gap.

### Constraints
- Keep the public diff small (`api-shaping`):
  - extend `BoardLayout`,
  - avoid a redundant `TargetGenerator` class,
  - avoid a second config mirror for target-generation options unless truly necessary.
- Installed-package usability is mandatory:
  - no runtime dependency on repo-root `tools/`,
  - no assumption that the user has cloned the repository.
- Rust generation semantics remain authoritative:
  - Python should not reimplement geometry generation,
  - Python should not reimplement SVG or PNG rendering,
  - Python should call into the existing Rust methods through focused binding helpers.
- Spec-vs-snapshot semantics must stay explicit:
  - canonical file JSON remains `ringgrid.target.v3` spec-only,
  - Python `to_dict()` may continue returning the expanded snapshot with `markers`,
  - any new JSON file/text method must be clearly named as spec JSON rather than generic snapshot JSON.
- Preserve current print/output contracts from `INFRA-010`:
  - SVG uses the Rust page-mm frame and current scale-bar policy,
  - PNG writing preserves DPI via `pHYs` metadata,
  - invalid margins / invalid DPI remain explicit validation failures.
- Keep type stubs and installed-package typing accurate:
  - update `__init__.pyi`,
  - keep `generate_typing_artifacts.py --check` passing.

### Assumptions
- `INFRA-011` is the correct next backlog item because it is the first `todo` item in `docs/backlog.md` under `Up Next`, and the enabling Rust task `INFRA-010` is already done.
- A Python direct-geometry constructor should accept an optional `name`; when omitted, it should reuse the Rust-generated deterministic name behavior rather than duplicating that logic in Python.
- The compact Rust fixture set in `crates/ringgrid/tests/fixtures/target_generation/` is sufficient as the primary cross-language parity baseline; no new large binary fixtures should be added unless the current ones cannot cover the installed-package path.
- A minimal root README correction is acceptable if needed to avoid stale Python target-generation instructions, but broad docs refactoring remains outside this task.

### Affected Areas
- `crates/ringgrid-py/src/lib.rs` - native helper layer for:
  - direct geometry -> validated board/spec JSON,
  - canonical spec JSON writing,
  - SVG writing,
  - PNG writing.
- `crates/ringgrid-py/python/ringgrid/_api.py` - public high-level wrappers on `BoardLayout` and any option/path normalization helpers.
- `crates/ringgrid-py/python/ringgrid/__init__.py` - export updates if public helper types or aliases are added.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - typing updates for the new public methods.
- `crates/ringgrid-py/tests/test_api.py` - parity, error-contract, and installed-package tests.
- `crates/ringgrid-py/README.md` - new quickstart/examples for installed-package target generation.
- `README.md` - only if a minimal Python target-generation snippet update is needed to prevent immediate public drift.
- Future dependent touchpoints that must remain unblocked, but are not in scope for implementation here:
  - `tools/gen_target.py` (future `INFRA-012`)
  - `book/src/target-generation.md` (only touch if a minimal corrective note is truly necessary)

### Plan
1. Add a small native bridge for target generation in `ringgrid-py`.
   - Expose focused PyO3 helpers that:
     - build a validated `BoardLayout` from direct geometry args and optional name,
     - return canonical spec JSON for that board,
     - write spec JSON / SVG / PNG from an existing spec JSON payload.
   - Keep the bridge board-centric and thin over the Rust `BoardLayout` methods.
   - Risk mitigation: do not expose raw Rust target-generation option structs to Python unless wrapper ergonomics force it.
2. Extend the public Python `BoardLayout` surface.
   - Add a documented direct-geometry classmethod, preferably `BoardLayout.from_geometry(..., name: str | None = None)`.
   - Add explicit generation methods on `BoardLayout`:
     - canonical spec JSON text/file output,
     - file-oriented SVG writer,
     - file-oriented PNG writer.
   - Prefer keyword-only scalar options (`margin_mm`, `include_scale_bar`, `dpi`) over separate public Python options dataclasses unless implementation becomes unclear without them.
   - Keep `to_dict()` / `to_spec_dict()` behavior intact and make any new spec-JSON method unambiguous.
3. Add deterministic parity coverage and update installed-package docs.
   - Reuse the committed Rust fixture board and target outputs for parity tests.
   - Add Python tests that operate through the public package API only and write to temp paths.
   - Update `crates/ringgrid-py/README.md` to show target generation directly from `import ringgrid`; update `README.md` only as needed to avoid a stale Python quickstart.

### Acceptance Criteria
- `ringgrid` Python package exposes an additive public API that can:
  - construct a board from direct geometry args,
  - emit canonical `ringgrid.target.v3` JSON from the installed package,
  - write SVG and PNG print targets from the installed package.
- The Python target-generation path routes through the Rust engine; no duplicate Python geometry/rendering implementation is introduced.
- The canonical JSON emitted from Python is spec-only `ringgrid.target.v3` JSON and round-trips through both:
  - Python `BoardLayout.from_json_file` / `from_dict`,
  - Rust `BoardLayout::from_json_file` / `from_json_str`.
- SVG and PNG outputs preserve the `INFRA-010` contracts for identical board args:
  - same geometry/layout semantics,
  - same scale-bar policy,
  - same PNG DPI metadata behavior.
- Deterministic parity tests exist for JSON/SVG/PNG against the committed Rust fixtures or an equivalent shared baseline.
- `crates/ringgrid-py/README.md` no longer tells installed-package users they must clone the repo and use `tools/gen_synth.py` just to generate targets.

### Test Plan
- Python/package-surface tests in `crates/ringgrid-py/tests/test_api.py`:
  - nominal `BoardLayout.from_geometry(...)` case with explicit name,
  - generated-name case when `name` is omitted,
  - canonical spec JSON text/file parity against `crates/ringgrid/tests/fixtures/target_generation/fixture_compact_hex.json`,
  - SVG text/file parity against `crates/ringgrid/tests/fixtures/target_generation/fixture_compact_hex.svg` (newline-normalized if needed),
  - PNG parity via:
    - dimensions,
    - `pHYs` metadata,
    - decoded grayscale pixels against `crates/ringgrid/tests/fixtures/target_generation/fixture_compact_hex.png`,
  - invalid geometry/options raise explicit Python exceptions:
    - `rows=0`,
    - `long_row_cols=1` with `rows>1`,
    - `marker_inner_radius_mm >= marker_outer_radius_mm`,
    - non-finite / non-positive `dpi`,
    - negative `margin_mm`.
- Required validation commands:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo test -p ringgrid --test target_generation`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`

### Out Of Scope
- A dedicated `tools/gen_target.py` wrapper or any new repo-root CLI.
- Synthetic image/ground-truth generation in the Python package.
- DXF export.
- Changes to detection algorithms, `DetectConfig`, or detection result schemas.
- Broad README/book restructuring beyond the minimal corrections needed to keep the installed-package target-generation story accurate.

### Handoff To Implementer
- Reuse the existing Rust `BoardLayout`/target-generation methods end-to-end; do not duplicate board geometry or SVG/PNG drawing logic in Python.
- Keep the Python public diff tight and `BoardLayout`-centered:
  - one direct-geometry constructor path,
  - one explicit spec-JSON method,
  - one SVG writer,
  - one PNG writer.
- Make spec-vs-snapshot semantics explicit in names and docs:
  - `to_dict()` may keep markers,
  - new canonical JSON output must be spec-only.
- Prefer keyword-only scalar writer options unless a separate Python options type is clearly justified by typing or readability.
- Reuse the committed Rust target-generation fixtures instead of adding new large assets; for PNG, compare metadata and decoded pixels rather than compressed-byte identity.
- Update the typed package surface and the Python README in the same change so the installed-package API is immediately discoverable and reviewable.
