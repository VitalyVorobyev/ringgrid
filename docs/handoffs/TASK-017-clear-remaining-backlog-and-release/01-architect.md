# Architect Report - TASK-017-clear-remaining-backlog-and-release

- Task ID: `TASK-017-clear-remaining-backlog-and-release`
- Backlog ID: `INFRA-008, ALGO-009, ALGO-010, ALGO-011, INFRA-013`
- Role: `architect`
- Date: `2026-03-11`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/workflows/planning.md`
- `crates/ringgrid-cli/src/main.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/detector/outer_fit/mod.rs`
- `crates/ringgrid/src/detector/outer_fit/sampling.rs`
- `crates/ringgrid/src/pipeline/finalize.rs`
- `crates/ringgrid/src/detector/id_correction/math.rs`
- `crates/ringgrid/src/pixelmap/cameramodel.rs`
- `docs/tuning-guide.md`
- `book/src/cli-guide.md`

## Summary
- The remaining backlog items cluster into one reviewable slice: CLI camera-calibration loading, two completion/outer-fit robustness improvements, one post-fit consistency filter, and release/doc bookkeeping.
- The geometry work should reuse existing affine and ellipse metadata instead of adding parallel solvers or new public config knobs.
- The CLI change is additive and should preserve the current `--cam-*` path while adding `--calibration <file.json>` with explicit precedence/conflict handling.
- Release closure depends on shipping code, docs, and a fully rerun validation baseline on the final tree.

## Decisions Made
- Treat the remaining backlog as one bounded implementation/release task rather than five separate PR-sized tasks.
  - Reason: `ALGO-009/010/011` touch the same completion/finalization path, and `INFRA-013` is only meaningful after they land.
- Keep the public API/config surface additive and small.
  - Reason: the backlog items describe behavior changes, not a request for new end-user tuning knobs.
- Accept a calibration JSON file in the same shape as `CameraModel` and also allow the detector-output wrapper shape with a top-level `camera` field.
  - Reason: this reuses existing serde contracts and makes prior detector output or external tooling artifacts directly reusable.

## Files/Modules Affected (Or Expected)
- `crates/ringgrid-cli/src/main.rs` - add calibration-file loading, precedence validation, and CLI tests.
- `crates/ringgrid/src/detector/completion.rs` - local-affine completion seed selection and completion-focused tests.
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` / `crates/ringgrid/src/detector/outer_fit/mod.rs` - outer-ray contamination screening tied to expected radius.
- `crates/ringgrid/src/pipeline/finalize.rs` - inner/outer axis-ratio consistency filter and tests.
- `crates/ringgrid/src/detector/id_correction/math.rs` - shared affine helper visibility/utilities for completion reuse.
- `book/src/cli-guide.md` / `docs/tuning-guide.md` / `CHANGELOG.md` / `docs/backlog.md` - document the new CLI path, robustness changes, and release status.

## Validation / Tests
- Commands run:
  - Not run in architect stage.
- Results:
  - Not run in architect stage.

## Risks / Open Questions
- The backlog note for `INFRA-008` names `RadialTangentialDistortion`, but a usable CLI mapper also requires intrinsics.
  - Impact: implementation must document and validate the full `CameraModel` JSON shape explicitly.
- `ALGO-011` could be implemented either as a logging-only flag or as a hard reject.
  - Impact: choose hard reject for markers with both ellipses and strong ratio outliers; otherwise the backlog item will not materially improve output quality.

## Next Handoff
- To: `Implementer`
- Requested action: implement the combined task with localized changes, add targeted regression tests for each backlog item, update docs/changelog/backlog, run the full local baseline, and prepare reviewer-ready evidence.

---

## Architect Required Sections

### Problem Statement
- The repo is one untagged release behind its actual shipped surface. Four open backlog items remain:
  - `INFRA-008`: CLI support for loading a Brown-Conrady calibration JSON as an external mapper.
  - `ALGO-009`: local-affine completion seeding from nearby decoded neighbors.
  - `ALGO-010`: pre-screen contaminated outer rays before ellipse fitting.
  - `ALGO-011`: inner/outer axis-ratio consistency filter after marker collection.
- The project also needs the release bookkeeping (`INFRA-013`) completed on the final tree: docs updated, validation rerun, and release status made truthful.

### Scope
- In scope:
  - Add `--calibration <file.json>` to the CLI as an additive alternative to `--cam-*`.
  - Reuse existing local-affine math from `id_correction` to seed completion from 3-4 nearest decoded neighbors in board space.
  - Reject outer-ray samples whose recovered radius is too far from the expected outer radius.
  - Reject markers with strong inner/outer mean-axis ratio outliers using a global reference ratio from non-completion markers.
  - Update docs, changelog, backlog state, and release metadata.
- Out of scope:
  - New Rust public APIs beyond additive CLI/config behavior.
  - New user-exposed config knobs for the new gates.
  - Publishing to external registries from this task unless credentials and workflow are already available locally.

### Constraints
- Preserve existing default behavior when `--calibration` is not used.
- Keep geometry/frame semantics explicit:
  - completion seeds and local-affine projection operate in the same frame as current marker centers.
  - output `center` and `center_mapped` semantics must remain unchanged.
- Do not duplicate affine math or create near-identical config mirrors.
- Full local baseline is required before handing off as review-ready.

### Assumptions
- Shipping the remaining backlog into the still-untagged `0.5.0` release is acceptable.
- A calibration JSON file may reasonably follow either:
  - direct `CameraModel` serde shape, or
  - detector output shape with a top-level `camera` object.
- The axis-ratio filter may hard-reject only markers with both ellipses available; markers without inner ellipses remain unchanged.

### Affected Areas
- `crates/ringgrid-cli/src/main.rs` - new CLI flag, JSON parsing, validation, tests.
- `crates/ringgrid/src/detector/completion.rs` - local-affine seed helper, fallback rules, tests.
- `crates/ringgrid/src/detector/id_correction/math.rs` - expose/reuse affine helper.
- `crates/ringgrid/src/detector/outer_fit/sampling.rs` - expected-radius screen for outer rays.
- `crates/ringgrid/src/detector/outer_fit/mod.rs` - pass expected radius into sampling and keep reject behavior coherent.
- `crates/ringgrid/src/pipeline/finalize.rs` - global median ratio computation and outlier rejection.
- `book/src/cli-guide.md` - document `--calibration` and precedence/mutual exclusion rules.
- `docs/tuning-guide.md` - replace stale `--camera-model` wording.
- `CHANGELOG.md` / `docs/backlog.md` - release status and task closure.

### Plan
1. Add CLI calibration-file loading and tests.
2. Implement the three algorithmic robustness changes with targeted unit tests.
3. Update user-facing docs and release bookkeeping, then run the full validation baseline.

### Acceptance Criteria
- `ringgrid detect --calibration file.json ...` loads a valid camera model and feeds `detect_with_mapper`.
- CLI rejects ambiguous camera inputs (`--calibration` mixed with `--cam-*`) and still rejects any camera mode combined with `--self-undistort`.
- Completion uses a local-affine projected seed when at least 3 nearby decoded neighbors exist; otherwise it falls back to the global homography projection.
- Outer-ray sampling drops rays whose chosen radius differs from the expected outer radius by more than 40%.
- Final marker filtering removes strong inner/outer mean-axis ratio outliers using a global median reference from non-completion markers.
- Docs mention the new calibration JSON path and no longer reference the stale `--camera-model` CLI flag.
- Release docs/backlog reflect the implemented work and validation state truthfully.

### Test Plan
- Unit tests in `crates/ringgrid-cli/src/main.rs` for calibration JSON parsing, wrapper support, and conflict handling.
- Unit tests in `crates/ringgrid/src/detector/completion.rs` for local-affine seed selection and H fallback.
- Unit tests in `crates/ringgrid/src/detector/outer_fit/sampling.rs` for expected-radius ray rejection.
- Unit tests in `crates/ringgrid/src/pipeline/finalize.rs` for axis-ratio outlier removal.
- Full local baseline:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `./.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `./.venv/bin/python -m pytest crates/ringgrid-py/tests -q`

### Out Of Scope
- New Python package APIs.
- Broader finalize-pipeline reordering beyond what is needed for the backlog items.
- External publication workflow execution if the environment lacks the required credentials.

### Handoff To Implementer
1. Add a small internal calibration-file parser in the CLI that accepts direct `CameraModel` JSON and wrapped detector-output JSON.
2. Reuse `id_correction` affine math for completion seeding with 3-4 nearest decoded neighbors in board mm.
3. Add the outer-ray expected-radius screen in the outer-fit sampling path.
4. Add a post-collection inner/outer ratio rejector in finalization with tight, deterministic tests.
5. Update docs and release bookkeeping on the final behavior, then run the full baseline and record all results.
