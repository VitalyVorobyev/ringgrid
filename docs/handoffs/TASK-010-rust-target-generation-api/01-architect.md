# Architect Report - TASK-010-rust-target-generation-api

- Task ID: `TASK-010-rust-target-generation-api`
- Role: `architect`
- Date: `2026-03-08`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/templates/task-handoff-report.md`
- `docs/handoffs/README.md`
- `.agents/skills/architect/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/metrology-invariants/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/tests-synthetic-fixtures/SKILL.md`
- `crates/ringgrid/src/lib.rs`
- `crates/ringgrid/src/api.rs`
- `crates/ringgrid/src/board_layout.rs`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid/Cargo.toml`
- `crates/ringgrid/README.md`
- `tools/gen_board_spec.py`
- `tools/gen_synth.py`
- `book/src/target-generation.md`
- `crates/ringgrid-py/src/lib.rs`

## Summary
- The next unstarted highest-priority backlog item is `INFRA-010`, mapped here to `TASK-010-rust-target-generation-api`.
- `ringgrid` currently loads board specs and detects markers in Rust, but target generation remains split across Python tools (`gen_board_spec.py` and `gen_synth.py`), which duplicates geometry/rendering semantics and blocks the next Python/CLI milestones from sharing one engine.
- The recommended API is additive and `BoardLayout`-centered: construct a layout from direct geometry args, emit canonical `ringgrid.target.v3` JSON, and render/write SVG/PNG using the embedded codebook and current pitch-derived print geometry.
- The most important contracts to lock down are explicit frame semantics (board-mm marker frame vs page-mm / raster-pixel output frame) and deterministic parity tests against current Python generation semantics for compact fixtures.

## Decisions Made
- Use `TASK-010-rust-target-generation-api` as the required handoff id for backlog item `INFRA-010` because no prior handoff exists for this work.
- Keep the public surface small and additive by extending `BoardLayout` and adding a focused target-generation module, rather than introducing a parallel board/config mirror.
- Preserve `ringgrid.target.v3` JSON schema and reuse the embedded Rust codebook for print outputs.
- Treat file-oriented API in `ringgrid` as an intentional exception to the usual “CLI owns file I/O” convention because `docs/backlog.md` explicitly locks target generation to a file-oriented Rust library API.
- Define SVG/PNG parity as structural and geometric rather than byte-for-byte string equality.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid/src/board_layout.rs` - public geometry/spec constructor(s), JSON serialization helpers, and validation reuse.
- `crates/ringgrid/src/target_generation.rs` (new) - SVG/PNG rendering helpers, output options, and file writers.
- `crates/ringgrid/src/lib.rs` - additive re-exports only.
- `crates/ringgrid/tests/...` or `#[cfg(test)]` in touched modules - deterministic parity/contract coverage.
- `crates/ringgrid/README.md` - minimal Rust API documentation update if public examples are added or changed.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- `gen_synth.py` print outputs currently take `board_mm`, while the backlog requires the Rust API to use direct runtime board geometry args (`pitch_mm`, `rows`, `long_row_cols`, radii). The implementer must make canvas/page sizing deterministic from `BoardLayout` rather than importing `board_mm` as a hidden dependency.
- `ringgrid.target.v3` JSON requires a `name`, but `board_mm`-based legacy auto-naming is not derivable from the new required arg list. This handoff assumes an optional caller-provided name plus a deterministic geometry-based default is acceptable.
- Python SVG formatting and rasterization details are brittle for exact file equality. Tests should compare stable geometry/pixel outcomes, not incidental formatting.

## Next Handoff
- To: `Implementer`
- Requested action: add the additive Rust target-generation API in `ringgrid`, keep `BoardLayout` as the core contract, implement deterministic JSON/SVG/PNG outputs with explicit frame semantics, and prove parity against current Python generation on compact fixtures.

---

## Architect Required Sections

### Problem Statement
- The Rust crate can currently consume board specs but cannot generate them or emit printable targets. Users must go through Python tooling (`tools/gen_board_spec.py` and `tools/gen_synth.py`) even when their application is otherwise pure Rust.
- That split duplicates target-generation semantics across languages: board spec serialization exists in the Python bridge, while printable SVG/PNG rendering lives only in `gen_synth.py`. This creates drift risk in layout geometry, code ordering, ring-band ratios, scale-bar behavior, and future bindings.
- `INFRA-010` is the enabling milestone for `INFRA-011` and `INFRA-012`: the Rust crate needs a shared, installed-library-safe target-generation engine before Python wrappers or a dedicated script can layer on top.

### Scope
- In scope:
  - Add an additive public Rust API in `ringgrid` to construct a board layout from direct geometry args (`pitch_mm`, `rows`, `long_row_cols`, `marker_outer_radius_mm`, `marker_inner_radius_mm`) plus optional naming/output options.
  - Emit canonical `ringgrid.target.v3` JSON and write it to disk.
  - Render and write printable SVG and PNG from the same layout using the current shipped codebook and current print geometry semantics.
  - Add deterministic tests proving JSON/SVG/PNG parity against current Python generator semantics for compact fixture cases.
  - Add minimal public API docs/rustdoc updates if the new surface would otherwise be undiscoverable or misleading.
- Out of scope:
  - Python bindings or package surface (`INFRA-011`).
  - Dedicated script/CLI wrapper (`INFRA-012`).
  - DXF generation.
  - Synthetic camera/image generation, homography simulation, or any detector-pipeline logic.
  - Changing the `ringgrid.target.v3` schema.

### Constraints
- Public API change must be additive and small-surface (`api-shaping`).
- `lib.rs` remains re-exports only; new public types should live in their defining module.
- `BoardLayout` remains the single source of truth for runtime board geometry; avoid introducing near-identical adapter/config mirrors.
- File I/O in `ringgrid` is permitted for this task because the backlog explicitly requires a file-oriented library API, but the implementation should still keep rendering/serialization logic testable without disk.
- Frame semantics must be explicit (`metrology-invariants`):
  - board marker coordinates remain in normalized board mm (`id=0` anchor at `[0, 0]`);
  - SVG coordinates are page mm with origin at top-left and `+y` downward;
  - PNG pixels follow the existing integer-pixel-center convention.
- Use deterministic fixtures and measurable parity thresholds (`tests-synthetic-fixtures`).
- Avoid heavy new dependencies when existing `image` + string generation are sufficient.

### Assumptions
- `INFRA-010` is the correct “next item” because it is now the first `todo` item in Active Sprint with `P0` priority.
- The new API may accept an optional `name` in addition to the required geometry args, because JSON output requires a name and the old `board_mm`-based auto-name cannot be recovered from the new required arg list.
- Current print semantics should carry over unless explicitly documented otherwise:
  - code-band radii derive from `pitch_mm` using the existing Python ratios (`0.58`, `0.42`);
  - ring stroke derives from `outer_radius * 0.12`;
  - the embedded Rust codebook determines sector fill;
  - scale bar remains included by default.
- Parity with `gen_synth.py` should be judged on geometry, marker/code ordering, output dimensions, and stable pixel/vector content for deterministic fixtures, not literal SVG whitespace or attribute ordering.
- Compact committed fixtures are preferable to Python-at-test-time dependencies for `cargo test`.

### Affected Areas
- `crates/ringgrid/src/board_layout.rs` - likely promote or replace the private spec representation with an additive public construction/serialization contract.
- `crates/ringgrid/src/target_generation.rs` - new module for output options, SVG builder, raster renderer, and thin file writers.
- `crates/ringgrid/src/lib.rs` - additive re-exports for any new public types/options.
- `crates/ringgrid/tests/...` or new module tests - fixture-based JSON/SVG/PNG parity and validation.
- `crates/ringgrid/README.md` - brief usage example for the new public API if needed.
- Future dependent touchpoints, not in scope now but should stay unblocked:
  - `crates/ringgrid-py/src/lib.rs`
  - `tools/gen_target.py` (future)

### Plan
1. Add the public board-spec contract on top of `BoardLayout`.
   - Introduce a small additive public type or constructor path for direct board geometry args plus optional name.
   - Add `BoardLayout` helpers to expose canonical spec serialization and JSON file writing without duplicating schema logic in downstream crates.
   - Keep schema/version semantics explicit and unchanged (`ringgrid.target.v3`).
2. Add SVG/PNG target generation in a dedicated Rust module.
   - Implement pure helpers that render from `BoardLayout` plus explicit output options, then add thin file-oriented wrappers.
   - Reuse the embedded codebook and current Python geometry ratios for code band, ring stroke, and scale bar rules.
   - Document and test the frame conversion from normalized board-mm coordinates to page-mm / raster-pixel output space.
3. Add deterministic parity coverage and minimal API docs.
   - Add compact committed fixtures produced from the current Python toolchain for one or more representable board geometries.
   - Verify JSON semantic equivalence, stable SVG structure/geometry, and PNG pixel/output-dimension parity against those fixtures.
   - Update rustdoc/README only as needed to keep the new public API discoverable and accurate.

### Acceptance Criteria
- `ringgrid` exposes an additive public API that can generate a board from direct geometry args and write JSON, SVG, and PNG files.
- Generated JSON is valid `ringgrid.target.v3` and round-trips through `BoardLayout::from_json_str` / `from_json_file`.
- SVG and PNG outputs use the embedded codebook ordering and the same pitch-derived print geometry semantics currently implemented in `tools/gen_synth.py`.
- Board/layout frame semantics are explicit and unchanged in the runtime API; only rendering/output-space coordinates are transformed.
- Deterministic parity tests cover JSON/SVG/PNG for compact fixture cases and pass locally.
- No DXF API or implementation is added in this milestone.

### Test Plan
- Rust validation:
  - `cargo fmt --all`
  - `cargo test -p ringgrid`
  - `cargo test -p ringgrid --doc` if new public rustdoc examples are added
- Fixture preparation during implementation:
  - use `tools/gen_synth.py` and/or `tools/gen_board_spec.py` only to generate/update reference fixtures during development
  - committed Rust tests must not require Python at runtime
- Required fixture/parity coverage:
  - JSON spec parity for at least one compact board geometry representable by both Python and Rust inputs
  - SVG parity on marker placement/code-band geometry and canvas/dimension contract
  - PNG parity on dimensions and exact or tolerance-free pixel content for a compact deterministic fixture
  - edge-case validation for invalid geometry (`rows=0`, `long_row_cols=1` with `rows>1`, `inner>=outer`, non-finite values)

### Out Of Scope
- Python package wrappers and packaging concerns.
- New command-line UX or repo-level tooling around generation.
- DXF or CAD export.
- Replacing `gen_synth.py` synthetic dataset generation beyond the shared target-print geometry implementation.
- Any change to detection result schemas, decode semantics, or codebook contents.

### Handoff To Implementer
- Use `BoardLayout` as the core public geometry object; do not add a redundant “same data, different name” config mirror unless it carries a genuinely distinct semantic role.
- Keep the public diff additive and explicit:
  - direct-geometry constructor or spec type,
  - JSON emit/write helper(s),
  - SVG/PNG render/write helper(s),
  - small output-option structs only where needed.
- Preserve `ringgrid.target.v3` exactly and document any default-name behavior clearly in code/docs/tests.
- Keep rendering helpers pure and testable; make file-writing wrappers thin.
- Lock frame semantics in tests and doc comments: normalized board-mm input frame, page-mm SVG frame, and raster-pixel PNG frame.
- Build parity tests from compact deterministic fixtures generated by the current Python toolchain, and keep those tests independent of Python at `cargo test` time.
- If exact SVG byte parity proves brittle, compare normalized structure/geometry instead of formatting, but keep PNG and JSON checks strict enough to catch real drift.
