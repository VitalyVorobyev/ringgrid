# Architect Report - TASK-015-ringgrid-py-readme-detect-config-field-guide

- Task ID: `TASK-015-ringgrid-py-readme-detect-config-field-guide`
- Backlog ID: `DOCS-001`
- Role: `architect`
- Date: `2026-03-11`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/handoffs/README.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/architect/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/01-architect.md`
- `docs/handoffs/TASK-011-reconcile-codebook-docs-invariants/02-implementer.md`
- `crates/ringgrid-py/README.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/python/ringgrid/__init__.pyi`
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid/src/detector/config.rs`
- `crates/ringgrid/src/marker/decode.rs`
- `book/src/configuration/detect-config.md`
- `book/src/configuration/marker-scale-prior.md`
- `./.venv/bin/python - <<'PY' ... ringgrid.DetectConfig(ringgrid.BoardLayout.default()).to_dict() ... PY`
- `./.venv/bin/python - <<'PY' ... dataclasses.fields(ringgrid.DecodeConfig) ... PY`

## Summary
- `DOCS-001` has no existing workflow directory, so this backlog item is mapped to `TASK-015-ringgrid-py-readme-detect-config-field-guide`.
- `crates/ringgrid-py/README.md` currently mentions `DetectConfig` only in a short usage snippet and does not provide the full field guide the backlog item requires.
- The repo already has useful source material for this guide in Rust-facing docs and code comments, especially `book/src/configuration/detect-config.md`, `book/src/configuration/marker-scale-prior.md`, and `crates/ringgrid/src/detector/config.rs`.
- The Python package surface is large enough that prose alone will drift unless the task includes one focused consistency guard.
- There is one concrete parity gap today: native resolved config dumps include `decode.codebook_profile`, but the checked-in typed Python `DecodeConfig` wrapper and typing artifacts do not expose that field. The README cannot truthfully claim full `decode` coverage unless that small gap is resolved or explicitly scoped out.

## Decisions Made
- Use `TASK-015-ringgrid-py-readme-detect-config-field-guide` because no previous architect handoff exists for backlog item `DOCS-001` and `TASK-015` is the next unused workflow id.
- Keep the task centered on `crates/ringgrid-py/README.md`, not a broad docs reorganization. Broader README reshaping remains part of `DOCS-002`.
- Treat the installed Python `DetectConfig` experience as the primary contract for this task:
  - typed section properties
  - top-level scalars and flags
  - convenience alias properties
  - `to_dict()` wire keys where they are part of the user-visible API
- Document resolved defaults, not just raw struct literals. Fields re-derived from board geometry or marker scale must be labeled as derived, with the derivation rule and the resolved default for `DetectConfig(BoardLayout.default())`.
- Include a minimal Python API/stub parity repair for `decode.codebook_profile` if it is still missing when implementation begins. This is a small compatibility-preserving surface completion, not a new feature.
- Add a lightweight docs consistency check so future config additions cannot silently bypass the README field guide again.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-py/README.md` - main deliverable; needs a full Python-oriented `DetectConfig` field guide.
- `crates/ringgrid-py/python/ringgrid/_api.py` - expected touchpoint if `DecodeConfig` must expose `codebook_profile` to match the native resolved config surface.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - generated/public typing surface must stay aligned with any Python API parity fix.
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - template source for stub regeneration if the typed `DecodeConfig` surface changes.
- `crates/ringgrid-py/tests/test_api.py` - add or extend parity/docs-consistency coverage around the documented config surface.
- `crates/ringgrid/src/detector/config.rs` - source of truth for defaults, derived fields, and tuning comments; expected read-only unless a deeper mismatch is discovered.
- `crates/ringgrid/src/marker/decode.rs` - source of truth for `DecodeConfig`, including `codebook_profile`.
- `book/src/configuration/detect-config.md` - reusable source material for section roles and field descriptions.
- `book/src/configuration/marker-scale-prior.md` - reusable source material for derived-default formulas and scale tuning guidance.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- The current Python surface is internally inconsistent around `decode.codebook_profile`: `cfg.to_dict()` exposes it, but `cfg.decode` and the published stubs do not. If this is left unresolved, the README will either be incomplete or misleading.
- Derived defaults are easy to document incorrectly. Copying raw struct defaults from Rust comments without accounting for `DetectConfig(board)` re-derivation would recreate drift immediately.
- A full per-field guide can become unreadable if it is written as one flat dump. The README needs a user-oriented structure, not just a serialized config pasted into Markdown.
- This task should not accidentally imply Python exposes every Rust-internal detection knob. Only user-reachable Python fields belong in the field guide unless a parity fix is intentionally included.

## Next Handoff
- To: `Implementer`
- Requested action: expand `crates/ringgrid-py/README.md` into a complete, Python-user-facing `DetectConfig` field guide, resolve the existing `decode.codebook_profile` parity gap if it is still present, and lock the documented surface with focused validation so the README stays aligned with the actual Python package.

---

## Architect Required Sections

### Problem Statement
- `DOCS-001` exists because `ringgrid-py` users currently have no authoritative Python-facing field guide for `DetectConfig`.
- The current README shows only basic construction:
  - `cfg = ringgrid.DetectConfig(board)`
  - `detector = ringgrid.Detector(cfg)`
  It does not explain the available sections, defaults, derived values, or when to tune each area.
- That gap is larger than it first appears because `DetectConfig` mixes:
  - nested typed sections
  - top-level scalars/flags
  - convenience aliases for common knobs
  - board/scale-derived values that are re-computed during construction
- The repo already contains enough detail to build a good guide, but it is split across Rust docs, code comments, Python wrappers, stubs, and tests. Without a deliberate synthesis step, Python users must reverse-engineer the package from `cfg.to_dict()` or source.
- There is also a concrete API/docs mismatch today: native resolved config dumps include `decode.codebook_profile`, but the checked-in Python `DecodeConfig` wrapper omits it. That makes a “complete field guide” impossible unless the task fixes or explicitly bounds that gap.

### Scope
- In scope:
  - Add a substantial `DetectConfig` field-guide section to `crates/ringgrid-py/README.md`.
  - Cover every Python-exposed config section, top-level scalar/flag, and convenience alias that Python users can actually set or inspect.
  - Document defaults with the right semantics:
    - literal defaults for fixed fields
    - derived-default rules for board/scale-coupled fields
    - resolved default values for `DetectConfig(BoardLayout.default())`
  - Provide practical tuning guidance for the major user knobs:
    - marker scale
    - decode strictness
    - completion/global filter
    - self-undistort
    - ID correction / inner-as-outer recovery
  - If still needed, make the minimal parity repair so Python `DecodeConfig` exposes `codebook_profile` consistently with the native resolved config dump and generated stubs.
  - Add a focused validation guard that checks the README still covers the exposed config inventory.
- Out of scope:
  - Root `README.md` restructuring or broader docs IA work from `DOCS-002`.
  - Rewriting the mdBook configuration chapters except for optional link reuse.
  - Adding brand-new detector knobs or changing runtime defaults.
  - Exposing Rust-only internal fields that are not already part of the Python user surface.
  - Large Python API redesigns or new config container types.

### Constraints
- Keep the public surface small and compatibility-preserving.
  - If `DecodeConfig.codebook_profile` is added to the typed Python wrapper, it must mirror the already-shipped native field rather than invent a new abstraction.
  - Stub/template updates must match the runtime surface exactly.
- The README must be sourced from current code, not stale local installs.
  - Rebuild the editable package before final validation.
  - Treat the checked-in source plus rebuilt extension as authoritative, not whatever happens to be installed in `.venv`.
- Derived defaults must be documented carefully.
  - Proposal, edge-sample, completion ROI, and projective-center shift gates are not free-floating constants; they depend on `marker_scale` and board geometry.
  - Avoid prose that suggests these are universal fixed numbers across all boards and priors.
- Avoid duplicating long algorithm explanations already covered better in the mdBook. The README should be a practical field guide, with short links to deeper theory where needed.
- Keep all changes within the Python package/docs boundary unless a small parity fix is required to make the documented surface truthful.

### Assumptions
- `DOCS-001` is intended to document the Python package surface users interact with directly, not every internal Rust-only detector field.
- `DetectConfig(BoardLayout.default())` is an acceptable canonical example when showing resolved defaults for derived values.
- A lightweight inventory-level docs consistency check is sufficient; the task does not need a full Markdown parser or docs generation pipeline.
- The `decode.codebook_profile` mismatch is accidental drift rather than an intentional decision to hide the field from Python users.

### Affected Areas
- `crates/ringgrid-py/README.md` - new field-guide structure, tables, tuning notes, and cross-links.
- `crates/ringgrid-py/python/ringgrid/_api.py` - possibly add `DecodeConfig.codebook_profile` and any related `from_dict` / `to_dict` handling.
- `crates/ringgrid-py/python/ringgrid/__init__.pyi` - typing artifact update if Python config surface changes.
- `crates/ringgrid-py/tools/typing_artifacts.pyi.template` - source of generated stubs.
- `crates/ringgrid-py/tests/test_api.py` - targeted surface parity and docs coverage checks.
- `crates/ringgrid/src/detector/config.rs` - authoritative defaults and derivation rules for proposal, edge sampling, completion, and projective-center settings.
- `crates/ringgrid/src/marker/decode.rs` - authoritative `DecodeConfig` field list and defaults.
- `book/src/configuration/detect-config.md` and `book/src/configuration/marker-scale-prior.md` - reusable wording and formulas that should be adapted, not contradicted.

### Plan
1. Reconcile the actual Python config surface before writing the guide.
   - Inventory the resolved `DetectConfig` keys from a rebuilt local package and compare them with the typed Python wrapper/stubs.
   - If `decode.codebook_profile` is still present in the native dump but absent from the typed `DecodeConfig` surface, add that field to `_api.py`, the typing template, and generated stubs.
   - Add or update a focused parity test so `cfg.decode.to_dict()` and `cfg.to_dict()["decode"]` cannot silently diverge on documented fields again.
2. Rewrite `crates/ringgrid-py/README.md` around a practical `DetectConfig` field guide.
   - Add a short orientation section explaining:
     - `DetectConfig(board)` resolves board- and scale-coupled defaults up front
     - `cfg.to_dict()` shows the fully resolved wire view
     - convenience aliases map onto nested sections
   - Provide a top-level inventory of sections, scalars, and aliases.
   - Add concise per-section tables for fields, defaults, and tuning notes.
   - Clearly mark which defaults are derived and reference the derivation rules for `marker_scale`.
   - Cross-link to the mdBook for deeper theory instead of duplicating long algorithm background.
3. Add a drift guard and validate the package/docs together.
   - Add a lightweight test or docs check that verifies the README covers the exposed config inventory and key aliases.
   - Run typing-artifact verification, rebuild the editable package, and run focused Python tests for config parity plus the package test suite.
   - Keep validation scoped to Python/package/docs unless implementation uncovers a real source-of-truth mismatch in Rust comments or docs.

### Acceptance Criteria
- `crates/ringgrid-py/README.md` contains a complete `DetectConfig` field guide for the Python package.
- The guide covers every Python-user-reachable config section and top-level config control, including convenience aliases that the package exposes for common tuning.
- The guide distinguishes:
  - fixed defaults
  - derived defaults
  - resolved example defaults for `DetectConfig(BoardLayout.default())`
- The guide includes practical tuning guidance for the major operational knobs instead of only listing field names.
- The documented `decode` section matches the actual Python/user-visible surface. If `codebook_profile` remains present in native resolved config, it is exposed consistently through the typed Python API/stubs and documented in the README.
- A focused automated check exists to catch future README surface drift for `DetectConfig`.
- Python package validation passes after the change, including typing-artifact verification and relevant `ringgrid-py` tests.

### Test Plan
- Required validation commands:
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k 'detect_config or readme or typing'`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
- Required targeted checks:
  - verify resolved config parity for documented fields, especially `decode`
  - verify the README inventory includes all exposed sections and documented aliases
  - spot-check derived defaults against a rebuilt `DetectConfig(BoardLayout.default()).to_dict()` snapshot
- Optional manual smoke check if the docs test is intentionally lightweight:
  - `./.venv/bin/python - <<'PY' ... print(json.dumps(ringgrid.DetectConfig(ringgrid.BoardLayout.default()).to_dict(), indent=2)) ... PY`

### Out Of Scope
- Reworking the root project README or splitting developer/user docs more broadly.
- General mdBook cleanup outside the minimal reuse needed for accurate Python README content.
- Expanding Python bindings to expose new Rust-only internals beyond the minimum parity repair needed for truthful docs.
- Changing detection behavior, defaults, or tuning heuristics.
- Building a full documentation generation system for config schemas.

### Handoff To Implementer
- Start with the real surface, not the current README. Rebuild the editable package, inspect `DetectConfig(BoardLayout.default()).to_dict()`, and make sure the typed Python classes match what you plan to document.
- Treat `decode.codebook_profile` as an implementation-time gate:
  - if it is still present in the native resolved config and absent in typed Python, fix that first
  - keep the fix minimal and backward-compatible
  - update stubs and targeted parity tests in the same patch
- Keep the README structured for users:
  - short orientation
  - top-level inventory
  - compact per-section tables
  - practical tuning hints
  - links out to mdBook for deep theory
- Do not paste raw JSON dumps into the README as the main content. Translate the surface into stable, readable documentation and explicitly label derived values.
- Add one lightweight drift guard so future config additions force a deliberate README update instead of silently skipping the field guide.
