# Architect Report - TASK-016-main-readme-user-first-refactor-split-dev-perf-notes

- Task ID: `TASK-016-main-readme-user-first-refactor-split-dev-perf-notes`
- Backlog ID: `DOCS-002`
- Role: `architect`
- Date: `2026-03-11`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/handoffs/README.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/architect/SKILL.md`
- `docs/handoffs/TASK-015-ringgrid-py-readme-detect-config-field-guide/01-architect.md`
- `README.md`
- `docs/tuning-guide.md`
- `docs/decisions/011-workspace-boundaries.md`
- `book/src/introduction.md`
- `crates/ringgrid/README.md`
- `rg -n '^## ' README.md`

## Summary
- `DOCS-002` has no existing workflow directory, so this backlog item is mapped to `TASK-016-main-readme-user-first-refactor-split-dev-perf-notes`.
- The root `README.md` is currently overloaded: it starts as a product landing page, then continues into API inventory, project layout, detection-mode detail, scoring methodology, benchmark snapshots, asset regeneration, and contributor checks.
- The repo does not currently have dedicated developer or performance landing docs. Deep material exists, but it is scattered across `docs/`, the mdBook, crate READMEs, and benchmark artifacts.
- This task should stay a documentation information-architecture refactor. The goal is to improve entrypoint clarity for workspace users without changing shipped APIs, detector behavior, or benchmark methodology.

## Decisions Made
- Use `TASK-016-main-readme-user-first-refactor-split-dev-perf-notes` because no prior handoff exists for backlog item `DOCS-002` and `TASK-016` is the next unused workflow id.
- Treat the root `README.md` as the GitHub entrypoint for first-time users:
  - what ringgrid is
  - how to install/build the needed entrypoints
  - how to run the common quickstart workflows
  - where to go next for deeper documentation
- Move developer-maintenance material into a dedicated repo doc, expected to be `docs/development.md`, instead of leaving it inline in the root README.
- Move evaluation/benchmark/scoring detail into a dedicated repo doc, expected to be `docs/performance.md`, instead of keeping large synthetic benchmark sections on the front page.
- Prefer short summaries plus links to existing sources (`book/`, crate READMEs, `docs/module_structure.md`, `docs/pipeline_analysis.md`) over duplicating long explanations across multiple Markdown surfaces.
- Keep this task docs-only unless implementation discovers an objective command/path mismatch in existing examples that must be corrected for the README to stay truthful.

## Files/Modules Affected (Or Expected)
- `README.md` - main deliverable; must be restructured into a user-first landing page.
- `docs/development.md` - expected new developer/contributor doc for repo layout, maintenance workflows, and contributor checks.
- `docs/performance.md` - expected new performance/evaluation doc for synthetic scoring notes, benchmark context, and reference commands.
- `docs/module_structure.md` - existing architecture reference likely linked from the new developer doc.
- `docs/pipeline_analysis.md` - existing deep pipeline reference likely linked from the new developer doc.
- `docs/tuning-guide.md` - existing tuning companion doc that may be linked from the user-facing README or developer doc rather than duplicated.
- `book/src/introduction.md` and published mdBook pages - existing user/theory documentation that should be linked, not re-authored, unless a small cross-link fix is needed.
- `crates/ringgrid/README.md` and `crates/ringgrid-py/README.md` - crate-specific usage references that may be linked from the root README.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- The README can become too thin if the refactor removes operational quickstarts along with the deep material. The user-first version still needs a complete install -> generate -> detect path.
- Moving benchmark tables into a new doc can create drift if the new performance page becomes a second source of truth instead of clearly pointing at benchmark artifacts and scripts.
- Relative-link regressions are the most likely failure mode in this task because the value comes from navigation quality rather than code behavior.
- The root README currently mixes user-facing “how to use it” material with contributor-facing “how the repo is organized” material. The implementation needs a clean split, not just section renaming.

## Next Handoff
- To: `Implementer`
- Requested action: refactor the root `README.md` into a user-first entrypoint, create dedicated developer/performance docs for the moved material, and validate that the resulting navigation and commands still match the current repo surface.

---

## Architect Required Sections

### Problem Statement
- `DOCS-002` exists because the root `README.md` is currently trying to serve four different roles at once:
  - product overview for first-time users
  - quickstart for target generation and detection
  - contributor/developer handbook
  - benchmark/performance report
- That mixed role hurts the primary use case of a root README: a user landing on the repo should be able to understand what ringgrid does, install the needed pieces, run a quick workflow, and find the right deeper docs without reading benchmark tables or internal module maps first.
- The current README heading structure makes the problem concrete. After the initial overview and quickstarts, it continues with:
  - `Public API (v1)`
  - `Project Layout`
  - `Examples`
  - `Detection Modes`
  - `Distortion Correction`
  - `Metrics (Synthetic Scoring)`
  - `Performance Snapshots (Synthetic)`
  - `Regenerate Embedded Assets`
  - `Development Checks`
- Much of that material is still useful, but it should not dominate the front page. The repo already has better deep-documentation homes in the mdBook, crate READMEs, and `docs/` references; the missing piece is a clearer top-level information architecture.

### Scope
- In scope:
  - Restructure `README.md` so it is user-first and front-loads install/build prerequisites, quickstart workflows, and documentation navigation.
  - Create a dedicated developer-facing doc under `docs/` for contributor setup, repo layout, maintenance commands, and related internal references.
  - Create a dedicated performance/evaluation doc under `docs/` for scoring definitions, benchmark context, and reference benchmark commands/results that are currently inline in `README.md`.
  - Replace long inline README sections with concise summaries plus links to the new docs, mdBook chapters, crate READMEs, and existing deep references.
  - Keep command examples and links aligned with current repo reality:
    - `tools/gen_target.py` for print-oriented target generation
    - `cargo run -- detect ...` / `ringgrid detect ...` for CLI detection
    - existing Rust/Python package entrypoints that are already shipped
- Out of scope:
  - Changing detector behavior, public APIs, config defaults, or output schemas.
  - Reworking crate-specific README structure beyond minimal link coordination if necessary.
  - Full mdBook information-architecture changes or adding new book chapters.
  - Updating benchmark methodology or generating brand-new benchmark baselines unless needed to verify a moved command/example.
  - Broad cleanup of unrelated docs under `docs/`, `book/`, or historical session reports.

### Constraints
- Follow the backlog’s locked root README policy: user quickstart first, with links to separate developer/performance docs.
- Preserve truthfulness against shipped surfaces. Any commands, filenames, profile counts, or API references retained in the README must match current code and docs.
- Avoid creating new duplicate canonical sources when an existing page already owns the topic:
  - theory/configuration -> mdBook
  - crate-specific API usage -> crate READMEs/rustdoc
  - repo architecture internals -> `docs/module_structure.md`, `docs/pipeline_analysis.md`
- Keep the change reviewable and docs-focused. If implementation discovers stale examples that require code changes, those changes should be minimal and tightly justified.
- New docs should render cleanly on GitHub and use stable relative links from the repo root README.

### Assumptions
- Dedicated developer and performance docs can live in `docs/` as normal repository Markdown; this task does not require them to be mirrored into the mdBook.
- The current benchmark snapshots and scoring notes are still worth preserving, but they belong behind a deliberate “performance/evaluation” link rather than on the README’s main path.
- The root README should still acknowledge the three main user entrypoints:
  - Rust library
  - CLI workflow
  - Python bindings/tools
  but only at a summary level.
- Existing deeper references such as `docs/module_structure.md`, `docs/pipeline_analysis.md`, `docs/tuning-guide.md`, crate READMEs, and the mdBook are acceptable destinations for linked detail.

### Affected Areas
- `README.md` - major content reorganization and navigation rewrite.
- `docs/development.md` - expected new file collecting developer-facing material currently embedded in the README.
- `docs/performance.md` - expected new file collecting performance/evaluation content currently embedded in the README.
- `docs/module_structure.md` - likely linked as the canonical ownership/boundary reference.
- `docs/pipeline_analysis.md` - likely linked as the canonical deep pipeline architecture reference.
- `docs/tuning-guide.md` - likely linked for advanced tuning rather than duplicated.
- `book/src/introduction.md` and published mdBook pages - existing user guide destinations for theory and detailed usage.
- `crates/ringgrid/README.md` and `crates/ringgrid-py/README.md` - existing crate-level usage references for users who need package-specific detail.

### Plan
1. Partition the current README content into three buckets before rewriting.
   - Keep in root README:
     - short overview/value proposition
     - documentation map
     - install/build prerequisites
     - fast-start workflows
     - brief Rust/CLI/Python entrypoint guidance
   - Move to developer doc:
     - project layout
     - examples intended for contributors/library readers
     - asset regeneration steps
     - development checks
     - links to module structure / pipeline architecture / workflow docs
   - Move to performance doc:
     - scoring metric definitions
     - distortion/evaluation framing that is too detailed for the front page
     - synthetic benchmark snapshots
     - benchmark commands and source artifact references
   - Mitigation: create an explicit keep/move/link checklist from the current README headings so no useful section is silently dropped.
2. Create the dedicated docs and relocate deep material with minimal duplication.
   - Add `docs/development.md` as the repo-maintainer/contributor landing page.
   - Add `docs/performance.md` as the benchmark/evaluation landing page.
   - When a moved section is already documented better elsewhere, replace copied prose with a short summary and an explicit link to the existing canonical source.
   - Mitigation: avoid turning the new docs into dump sites. Each should have a clear purpose and outward links to the deeper references already present in the repo.
3. Rewrite the root README around a user-first flow, then validate navigation.
   - Front-load the common “what is it / how do I start / where do I go next” path.
   - Add a clear “More docs” or equivalent navigation section linking:
     - user guide / API reference
     - new developer doc
     - new performance doc
     - crate-specific READMEs where relevant
   - Remove large inline benchmark tables and maintainer-only sections from the README body once their new homes exist.
   - Mitigation: keep short bridge text or link labels for moved topics so returning readers can still find them quickly.

### Acceptance Criteria
- The root `README.md` is visibly user-first:
  - concise overview near the top
  - clear installation/build prerequisites
  - quickstart usage paths before any maintainer-only material
  - explicit links to deeper documentation
- The README no longer contains the current long-form maintainer/performance sections inline, including benchmark snapshot tables and contributor maintenance commands.
- A dedicated developer doc exists under `docs/` and receives the README material needed by contributors/maintainers.
- A dedicated performance doc exists under `docs/` and receives the README material needed for scoring interpretation, benchmark context, and performance snapshots.
- The README links clearly to:
  - the new developer doc
  - the new performance doc
  - the mdBook user guide
  - API/crate references as appropriate
- All retained or moved commands and links match current repo paths and supported workflows.
- The refactor does not introduce new contradictory docs copies; where existing docs already cover a topic well, the README/new docs link to them instead of restating large blocks.

### Test Plan
- Required validation commands:
  - `cargo run -- detect --help`
  - `python3` one-off script to scan `README.md`, `docs/development.md`, and `docs/performance.md` for local Markdown links and fail if any relative target path is missing
- Required manual checks:
  - verify the README render reads naturally top-to-bottom as a landing page, not as a changelog or benchmark report
  - verify moved topics remain discoverable within one click from the README
  - spot-check any edited command examples against the actual repo file/tool names
- Conditional validation:
  - if implementation touches mdBook pages or crate READMEs for link coordination, run the relevant docs build or focused checks for those files
  - if implementation ends up touching code to fix a stale example/path, run the normal Rust/Python validation appropriate to the touched surface rather than treating this as docs-only

### Out Of Scope
- Public API redesign, new bindings, or detector/config changes.
- Reorganizing the entire mdBook or migrating all repo docs into it.
- Refreshing benchmark numbers for new datasets or methodology changes.
- Broad cleanup of historical docs, session notes, or unrelated `docs/decisions/*` files.
- Rewriting `crates/ringgrid/README.md` or `crates/ringgrid-py/README.md` beyond minimal cross-link adjustments.

### Handoff To Implementer
- Start with a concrete content map from the existing `README.md` headings. Mark each section as:
  - keep in root
  - move to `docs/development.md`
  - move to `docs/performance.md`
  - replace with a short link to an existing canonical doc
- Create the destination docs first so the README rewrite can point at real targets immediately.
- Keep the root README tight. It should help a repo visitor do the first useful thing quickly, not explain every internal mode or benchmark result inline.
- Reuse existing deep docs aggressively:
  - module and ownership details -> `docs/module_structure.md`
  - pipeline internals -> `docs/pipeline_analysis.md`
  - theory/configuration details -> mdBook
  - crate/package-specific usage -> crate READMEs
- Validate every new relative link locally and spot-check the key commands you keep in the user path. If you discover a stale path/example, fix it minimally and call it out explicitly in the implementer report.
