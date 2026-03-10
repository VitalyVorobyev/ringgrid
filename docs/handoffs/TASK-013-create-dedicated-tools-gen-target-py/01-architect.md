# Architect Report - TASK-013-create-dedicated-tools-gen-target-py

- Task ID: `TASK-013-create-dedicated-tools-gen-target-py`
- Backlog ID: `INFRA-012`
- Role: `architect`
- Date: `2026-03-10`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/templates/task-handoff-report.md`
- `docs/handoffs/README.md`
- `.agents/skills/architect/SKILL.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/01-architect.md`
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/01-architect.md`
- `docs/handoffs/TASK-012-expose-target-generation-in-ringgrid-py/03-reviewer.md`
- `tools/gen_synth.py`
- `tools/gen_board_spec.py`
- `README.md`
- `book/src/fast-start.md`
- `book/src/target-generation.md`
- `crates/ringgrid-py/python/ringgrid/_api.py`
- `crates/ringgrid-py/src/lib.rs`
- `crates/ringgrid-py/tests/test_api.py`
- `crates/ringgrid/tests/target_generation.rs`

## Summary
- `INFRA-012` is now the next unstarted `P1` backlog item and has no existing workflow directory, so this work is mapped to `TASK-013-create-dedicated-tools-gen-target-py`.
- `INFRA-010` and `INFRA-011` already shipped the canonical Rust engine and installed-package Python surface for board JSON/SVG/PNG generation, but repo users still only have `tools/gen_synth.py` and `tools/gen_board_spec.py`.
- The gap is now mostly workflow and discoverability: there is no dedicated repo-root script for direct board geometry -> `board_spec.json` + printable SVG + printable PNG, and current docs still route users through synth-centric flags.
- This task should stay thin: add one dedicated tool that forwards to the existing `ringgrid.BoardLayout` generation methods, add black-box parity tests, and update the user docs that currently point at `gen_synth.py` for print-only generation.

## Decisions Made
- Use `TASK-013-create-dedicated-tools-gen-target-py` as the required workflow id because no prior handoff exists for backlog item `INFRA-012` and `TASK-013` is the next unused task number.
- Keep the implementation as a repo-level Python CLI over the existing `ringgrid-py` target-generation API. Do not add new Rust generation logic, new PyO3 binding methods, or cargo subprocess wrappers unless a concrete blocker appears.
- Make the CLI direct-geometry-first and explicit:
  - required: `--pitch_mm`, `--rows`, `--long_row_cols`, `--marker_outer_radius_mm`, `--marker_inner_radius_mm`
  - optional: `--name`, `--out_dir`, `--basename`, `--dpi`, `--margin_mm`, `--no-scale-bar`
- Keep the output contract opinionated and simple: one invocation writes canonical JSON plus SVG and PNG together, using default artifact names compatible with the current docs (`board_spec.json`, `target_print.svg`, `target_print.png`) unless the user overrides directory or basename.
- Position `gen_target.py` as the dedicated JSON/SVG/PNG generator, while keeping `gen_synth.py` for synthetic image/ground-truth generation and `gen_board_spec.py` as the JSON-only helper.

## Files/Modules Affected (Or Expected)
- `tools/gen_target.py` - new thin CLI entry point for direct geometry -> JSON/SVG/PNG generation.
- `tools/tests/test_gen_target.py` - new subprocess-level parity and error-contract coverage for the script.
- `README.md` - repo-root quickstart/target-generation guidance should mention the dedicated script.
- `book/src/fast-start.md` - fast path docs should reflect the dedicated target-generation tool.
- `book/src/target-generation.md` - full generation reference should distinguish `gen_target.py`, `gen_synth.py`, and `gen_board_spec.py`.
- `crates/ringgrid-py/python/ringgrid/_api.py` - dependency surface only; expected to be consumed, not materially changed.
- `crates/ringgrid-py/src/lib.rs` - dependency surface only; expected to remain unchanged unless the script reveals a binding gap.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- `tools/gen_target.py` will depend on `ringgrid` being importable in the active Python environment. The script and docs must surface a clear setup path instead of failing with a raw import traceback.
- The current root README and book chapters still treat `gen_synth.py` as the main print-target path. Documentation updates need to clarify that `gen_target.py` is for board JSON/SVG/PNG only and does not replace synth dataset generation.
- The dedicated script should not silently reintroduce legacy `board_mm` naming or radius defaults from older tools. Direct geometry args are the contract for this backlog item.

## Next Handoff
- To: `Implementer`
- Requested action: add `tools/gen_target.py` as a thin wrapper over the existing Python target-generation API, cover it with deterministic black-box tests against committed fixtures, and update the repo docs so the dedicated script is discoverable without changing target-generation semantics.

---

## Architect Required Sections

### Problem Statement
- The repository now has two approved target-generation layers:
  - Rust `BoardLayout` JSON/SVG/PNG generation from `INFRA-010`
  - installed-package Python `BoardLayout.from_geometry(...)`, `to_spec_json(...)`, `write_svg(...)`, and `write_png(...)` from `INFRA-011`
- Despite that, repo users still generate printable targets through older tools:
  - `tools/gen_synth.py` mixes print generation with synthetic image/ground-truth generation and carries legacy `board_mm`-centric flags
  - `tools/gen_board_spec.py` only emits JSON and does not cover SVG/PNG
- The missing piece is a dedicated repo script that exposes the new shared generation path with a small, direct CLI for board geometry. Without that script, the docs keep steering users toward synth-specific workflows even when they only want `board_spec.json`, `target_print.svg`, and `target_print.png`.

### Scope
- In scope:
  - Add `tools/gen_target.py` as a direct-geometry CLI for canonical board target generation.
  - Route generation through `ringgrid.BoardLayout.from_geometry(...)`, `to_spec_json(...)`, `write_svg(...)`, and `write_png(...)`.
  - Emit JSON, SVG, and PNG together in one run.
  - Support small, script-appropriate output knobs:
    - output directory
    - output basename for SVG/PNG
    - optional board name
    - PNG DPI
    - print margin
    - scale-bar toggle
  - Add deterministic script-level tests that compare emitted outputs to the committed target-generation fixtures or the same public API outputs for identical inputs.
  - Update repo docs where target-generation guidance currently points print-only users at `gen_synth.py`.
- Out of scope:
  - New Rust target-generation features or changes to Rust output semantics.
  - New Python binding APIs unless a missing script dependency truly blocks implementation.
  - Synthetic image, homography, blur, noise, or ground-truth generation.
  - DXF generation.
  - Removing or rewriting `tools/gen_synth.py` or `tools/gen_board_spec.py`.

### Constraints
- Keep the script thin. It should translate CLI arguments into existing `ringgrid` API calls and not duplicate geometry generation, schema serialization, SVG authoring, or PNG rendering logic.
- Use the backlog’s direct board-geometry contract. Do not add `board_mm` as a required input or derive radii from pitch inside the new script.
- Preserve existing canonical artifact behavior for matching geometry and options:
  - `ringgrid.target.v3` JSON
  - SVG geometry and scale-bar policy
  - PNG pixel content and `pHYs` DPI metadata
- Prefer simple defaults compatible with current user-facing examples:
  - JSON filename: `board_spec.json`
  - SVG/PNG basename: `target_print`
  - scale bar enabled by default
- Keep the Python dependency footprint minimal for the script itself:
  - no `numpy`
  - no `matplotlib`
  - rely on `ringgrid` plus stdlib CLI/path handling
- Validation should stay deterministic and fixture-backed; do not make tests depend on `gen_synth.py` at runtime.

### Assumptions
- `INFRA-012` is the correct next task because it is now the first `todo` item in `Up Next` after `INFRA-011` was closed on `2026-03-10`.
- The script may assume the user has either installed `ringgrid` from PyPI or built the local binding into the active environment with `maturin develop`; this requirement must be documented and surfaced clearly on failure.
- Omitting `--name` should preserve the installed-package API behavior and use the deterministic geometry-derived default name rather than recreating the old `ringgrid_{board_mm}mm_hex` naming policy.
- Black-box parity against the compact committed fixtures in `crates/ringgrid/tests/fixtures/target_generation/` is sufficient for this task; no new large binary fixtures should be added unless the current compact fixture cannot exercise the script contract.

### Affected Areas
- `tools/gen_target.py` - new CLI wrapper and user-facing argument contract.
- `tools/tests/test_gen_target.py` - subprocess tests for nominal generation, directory creation, and invalid-input behavior.
- `README.md` - target-generation quickstart and examples.
- `book/src/fast-start.md` - fast-start board target generation path.
- `book/src/target-generation.md` - detailed target-generation tool selection and flag reference.
- `crates/ringgrid-py/python/ringgrid/_api.py` - consumed public methods that define the script’s behavior contract.
- `crates/ringgrid/tests/fixtures/target_generation/` - reused test baseline; expected unchanged.

### Plan
1. Add the dedicated CLI tool.
   - Create `tools/gen_target.py` with a narrow `argparse` surface around direct board geometry and output options.
   - Construct a board through `ringgrid.BoardLayout.from_geometry(...)`.
   - Create the output directory if needed and write:
     - `board_spec.json`
     - `<basename>.svg`
     - `<basename>.png`
   - Print concise artifact paths and key board metadata on success.
   - Translate import, validation, and filesystem failures into clear stderr output and a nonzero exit.
2. Add deterministic script-level tests.
   - Add `tools/tests/test_gen_target.py`.
   - Invoke the script as a subprocess with the compact fixture geometry (`8.0`, `3`, `4`, `4.8`, `3.2`, `name=fixture_compact_hex`, `dpi=96`, `margin=0`).
   - Compare generated JSON/SVG text and PNG metadata/pixels against the committed fixtures or the same shared public API outputs.
   - Cover invalid geometry/options and parent-directory creation.
3. Update the target-generation docs.
   - Update `README.md`, `book/src/fast-start.md`, and `book/src/target-generation.md` to introduce `gen_target.py`.
   - Keep `gen_synth.py` documented for synth datasets and `gen_board_spec.py` documented as JSON-only, rather than pretending the dedicated tool replaces those workflows.

### Acceptance Criteria
- `tools/gen_target.py` exists and can generate canonical JSON, SVG, and PNG from direct board geometry args in one command.
- The script uses the shared installed-package generation path and does not introduce duplicate geometry/rendering logic.
- For the compact fixture geometry and print options, the script’s outputs match the committed target-generation fixture baseline:
  - JSON matches semantically and textually after newline normalization
  - SVG matches textually after newline normalization
  - PNG matches dimensions, `pHYs` metadata, and decoded grayscale pixels
- The script creates missing parent directories for its outputs.
- Invalid geometry and invalid print options fail with explicit nonzero CLI errors rather than silent fallbacks.
- Repo docs explain when to use:
  - `tools/gen_target.py` for dedicated JSON/SVG/PNG generation
  - `tools/gen_synth.py` for synthetic datasets
  - `tools/gen_board_spec.py` for JSON-only emission

### Test Plan
- Install the local Python binding into the active environment used for tool tests:
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
- Recheck the shared target-generation baseline:
  - `cargo test -p ringgrid --test target_generation`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests/test_api.py -q -k target_generation`
- Run the dedicated tool tests:
  - `.venv/bin/python -m pytest tools/tests/test_gen_target.py -q`
- Rebuild the user docs after updating examples:
  - `mdbook build book`
- Recommended manual smoke check:
  - `.venv/bin/python tools/gen_target.py --pitch_mm 8 --rows 3 --long_row_cols 4 --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 --name fixture_compact_hex --dpi 96 --out_dir /tmp/ringgrid_target_smoke`

### Out Of Scope
- Adding a target-generation command to the Rust CLI.
- Changing `ringgrid.target.v3` schema behavior or default print geometry semantics.
- Replacing `gen_synth.py` synthetic workflows.
- Extending repo tools to emit DXF or additional print formats.
- Broader Python package UX redesign beyond what the script directly consumes.

### Handoff To Implementer
- Keep `tools/gen_target.py` as a thin wrapper only. If you find yourself porting geometry logic from `gen_synth.py`, stop and route through the shipped `ringgrid` API instead.
- Require the full direct geometry argument set from the backlog. Do not add `board_mm` back into the dedicated script or infer radii from pitch.
- Prefer a small, stable CLI contract:
  - direct geometry args
  - `--out_dir`
  - optional `--basename`
  - optional `--name`
  - `--dpi`
  - `--margin_mm`
  - `--no-scale-bar`
- Add black-box tests that execute the script, not just unit tests around helper functions.
- Reuse the committed fixture board and target outputs instead of generating new large assets.
- Update the root README and book pages in the same change so the new script is immediately visible to repo users, but do not overclaim that it handles synth image generation.
