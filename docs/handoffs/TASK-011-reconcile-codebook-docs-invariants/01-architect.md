# Architect Report - TASK-011-reconcile-codebook-docs-invariants

- Task ID: `TASK-011-reconcile-codebook-docs-invariants`
- Role: `architect`
- Date: `2026-03-08`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/templates/task-handoff-report.md`
- `docs/handoffs/README.md`
- `.agents/skills/workflow-runner/SKILL.md`
- `.agents/skills/architect/SKILL.md`
- `README.md`
- `book/src/introduction.md`
- `book/src/why-rings.md`
- `book/src/marker-anatomy/coding-scheme.md`
- `book/src/detection-pipeline/decode.md`
- `book/src/cli-guide.md`
- `docs/decisions/006-codebook-invariants.md`
- `docs/decisions/011-workspace-boundaries.md`
- `docs/decisions/013-deterministic-reproducibility.md`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid/src/marker/codec.rs`
- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid-cli/src/main.rs`
- `tools/gen_codebook.py`
- `tools/codebook.json`

## Summary
- The next unstarted highest-priority backlog item is `DOCS-003`, mapped here to `TASK-011-reconcile-codebook-docs-invariants`.
- The shipped codebook artifacts agree on the current facts: `CODEBOOK_N = 893`, `CODEBOOK_BITS = 16`, `CODEBOOK_MIN_CYCLIC_DIST = 2`, and `CODEBOOK_SEED = 1`.
- User-facing docs are inconsistent with those artifacts today: `README.md` still claims minimum cyclic Hamming distance `5`, `book/src/cli-guide.md` shows `3` in the `codebook-info` example, and `docs/decisions/006-codebook-invariants.md` still states `>= 5` and an outdated confidence formula denominator.
- This task should stay doc-only: reconcile `README`, `book`, and `docs/decisions` to the shipped artifacts, and document the provenance/reproducibility path from `tools/gen_codebook.py` and `tools/codebook.json` to the embedded Rust constants and CLI output.

## Decisions Made
- Use `TASK-011-reconcile-codebook-docs-invariants` as the required handoff id for backlog item `DOCS-003` because no prior handoff exists for this work and `TASK-011` is the next unused task number.
- Treat `crates/ringgrid/src/marker/codebook.rs`, `tools/codebook.json`, and `crates/ringgrid-cli/src/main.rs` as the source of truth for shipped codebook facts; narrative docs must conform to those artifacts, not the reverse.
- Keep scope limited to documentation and examples. Do not regenerate the codebook, change decode behavior, or alter acceptance thresholds in code as part of this task.
- Document provenance explicitly:
  - `tools/gen_codebook.py` is the generator,
  - the committed artifacts are produced with `--n 893 --seed 1`,
  - the generator starts from a higher target distance and relaxes until it can reach the requested count,
  - the currently shipped achieved minimum cyclic Hamming distance is `2`.
- Update stale formula wording when it is tied to codebook invariants. The current docs must reflect that confidence scales with `margin / CODEBOOK_MIN_CYCLIC_DIST`, not a hard-coded `/3`.

## Files/Modules Affected (Or Expected)
- `README.md` - correct top-level codebook fact summary and any provenance wording surfaced to new users.
- `book/src/marker-anatomy/coding-scheme.md` - keep the detailed codebook chapter as the canonical long-form explanation of current constants, guarantees, and regeneration steps.
- `book/src/cli-guide.md` - fix the `ringgrid codebook-info` example output so it matches the current CLI implementation.
- `book/src/introduction.md` - adjust any overview wording if it overstates codebook guarantees or omits the shipped invariant needed for consistency.
- `book/src/why-rings.md` - keep comparison/benefit wording accurate and avoid overstating error tolerance beyond what minimum distance `2` supports.
- `book/src/detection-pipeline/decode.md` - ensure decode-confidence wording and unambiguity statements remain consistent with the current codebook minimum distance.
- `docs/decisions/006-codebook-invariants.md` - update the formal invariant record to the shipped constants, current decoding contract wording, and explicit regeneration provenance.
- `docs/decisions/011-workspace-boundaries.md` - narrow or refresh any stale statement about which layer generates codebook artifacts so it matches the current repository reality.
- `docs/decisions/013-deterministic-reproducibility.md` - add or refine the codebook-generation reproducibility expectation if needed so the seed/provenance story is not split across ad hoc prose.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- Minimum cyclic Hamming distance `2` supports weaker guarantees than the stale `>= 5` wording. The implementer must avoid replacing one inaccurate claim with another, especially around single-bit correction versus ambiguity.
- `docs/decisions/011-workspace-boundaries.md` is partially stale for board-generation responsibilities after `INFRA-010`. If touching it would broaden scope too far, keep edits tightly limited to the codebook/provenance sentence and leave wider README/workspace-boundary cleanup to `DOCS-002`.
- The book has multiple overview chapters that mention the codebook qualitatively. The implementer should update only the places that make factual claims or imply stronger guarantees than the shipped artifacts support.

## Next Handoff
- To: `Implementer`
- Requested action: update the targeted docs to match the shipped codebook artifacts exactly, add explicit provenance/reproducibility wording tied to `tools/gen_codebook.py`, and record grep/build evidence showing the stale `5`/`3` claims and outdated formula wording were removed.

---

## Architect Required Sections

### Problem Statement
- `DOCS-003` exists because the repository now ships one set of codebook facts while several docs still describe older values and older decoding semantics.
- The concrete drift is already visible:
  - `README.md` says the 893-codeword codebook has minimum cyclic Hamming distance `5`,
  - `book/src/cli-guide.md` shows `ringgrid codebook-info` output with `min cyclic Hamming: 3`,
  - `docs/decisions/006-codebook-invariants.md` still records `CODEBOOK_MIN_CYCLIC_DIST = 5` and the older confidence expression using `margin / 3`.
- Meanwhile the shipped artifacts and code paths agree on different values:
  - `crates/ringgrid/src/marker/codebook.rs` and `tools/codebook.json` both encode `min_cyclic_dist = 2`, `n = 893`, `seed = 1`,
  - `crates/ringgrid-cli/src/main.rs` prints `CODEBOOK_MIN_CYCLIC_DIST` directly in `codebook-info`,
  - `crates/ringgrid/src/marker/codec.rs` and `crates/ringgrid/src/marker/decode.rs` tie confidence and unambiguity wording to `CODEBOOK_MIN_CYCLIC_DIST`.
- This inconsistency undermines trust in the docs and makes future backlog work on codebook extensions (`ALGO-014`) riskier because the baseline profile is not crisply documented.

### Scope
- In scope:
  - Reconcile codebook facts across `README`, `book`, and `docs/decisions` with the currently shipped artifacts.
  - Correct any stale examples or formulas that directly depend on the codebook minimum cyclic Hamming distance.
  - Document generation provenance and reproducibility expectations for the shipped codebook:
    - generator entrypoint,
    - committed artifact locations,
    - fixed seed,
    - achieved distance versus requested target.
  - Add a bounded consistency sweep so the implementer explicitly proves no stale `5`/`3` values remain in the targeted codebook docs.
- Out of scope:
  - Regenerating or changing the codebook contents.
  - Extending the codebook beyond 893 IDs (`ALGO-014`).
  - Redesigning the root README structure beyond factual corrections needed for this task.
  - Broader docs refactors unrelated to codebook facts or provenance.
  - Any Rust/Python runtime code changes beyond documentation examples if no factual fix is needed in code.

### Constraints
- Shipped artifacts are authoritative:
  - `crates/ringgrid/src/marker/codebook.rs`
  - `tools/codebook.json`
  - `crates/ringgrid-cli/src/main.rs`
- Keep the task reviewable and doc-only. If the implementer discovers an actual code inconsistency, document it and stop rather than silently expanding scope.
- Do not overstate error-tolerance guarantees. Minimum distance `2` does not justify the old `>= 5` prose or any claim of guaranteed single-bit correction.
- Preserve generation ownership:
  - `tools/gen_codebook.py` remains the generator,
  - generated artifacts are committed outputs,
  - docs should describe that provenance clearly without implying manual editing.
- Avoid duplicating a large amount of near-identical prose. Prefer one canonical detailed explanation plus brief consistent references elsewhere.

### Assumptions
- `DOCS-003` is the correct next backlog item because it is the first `todo` item in `docs/backlog.md` under `Active Sprint` with `P0` priority.
- `TASK-011-reconcile-codebook-docs-invariants` is safe to assign because the existing handoff directories stop at `TASK-010-*`.
- The canonical current codebook facts are the committed generated artifacts, not the older decision record text.
- It is acceptable for this task to update a small number of decision docs if that is the cleanest way to capture provenance and remove stale invariant statements.

### Affected Areas
- `README.md` - one-line project description and generation/regeneration references.
- `book/src/marker-anatomy/coding-scheme.md` - detailed codebook invariants and regeneration explanation.
- `book/src/cli-guide.md` - CLI example output for `ringgrid codebook-info`.
- `book/src/introduction.md` - overview claims about identification capacity and guarantees.
- `book/src/why-rings.md` - qualitative error-tolerance explanation tied to codebook minimum distance.
- `book/src/detection-pipeline/decode.md` - decode-confidence and ambiguity semantics.
- `docs/decisions/006-codebook-invariants.md` - formal invariant record and decoding contract.
- `docs/decisions/011-workspace-boundaries.md` - artifact-generation ownership statement if still stale for codebook provenance.
- `docs/decisions/013-deterministic-reproducibility.md` - seeded reproducibility expectations for generated artifacts.

### Plan
1. Perform a targeted consistency sweep across the user-facing docs that make codebook claims.
   - Update `README.md`, the relevant book chapters, and the CLI guide example so every explicit codebook fact matches the shipped artifacts: `n = 893`, `bits = 16`, `min cyclic Hamming = 2`, `seed = 1`.
   - Remove or soften wording that implies stronger codebook guarantees than the current baseline profile actually provides.
   - Risk mitigation: if a chapter mentions the codebook only qualitatively and does not claim a numeric invariant, prefer minimal edits.
2. Refresh the canonical provenance/invariant decision docs.
   - Update `docs/decisions/006-codebook-invariants.md` to record the current constants, the current confidence expression tied to `CODEBOOK_MIN_CYCLIC_DIST`, and the regeneration path through `tools/gen_codebook.py`.
   - Add the reproducibility/provenance details that are currently missing or scattered:
     - seed `1`,
     - generated artifact locations,
     - generator relaxation behavior,
     - "never hand-edit generated artifacts" expectation.
   - If `docs/decisions/011-workspace-boundaries.md` or `docs/decisions/013-deterministic-reproducibility.md` contain the cleanest place for that provenance note, keep the edit tight and explicitly scoped.
3. Validate the docs pass a bounded consistency check.
   - Use targeted `rg` searches to prove the stale `5` and `3` codebook-distance claims are gone from the intended docs.
   - Reproduce current authoritative values from code or CLI output and make sure examples match them.
   - Record whether a book build was run; if tooling is unavailable, note that explicitly rather than guessing.

### Acceptance Criteria
- `README`, the relevant `book/src/...` chapters, and `docs/decisions/...` no longer disagree about the shipped codebook facts.
- Every explicit numeric codebook invariant in the updated docs matches the committed artifacts:
  - `CODEBOOK_N = 893`
  - `CODEBOOK_BITS = 16`
  - `CODEBOOK_MIN_CYCLIC_DIST = 2`
  - `CODEBOOK_SEED = 1`
- The `ringgrid codebook-info` documentation example matches the current CLI output contract.
- The canonical docs explain provenance/reproducibility clearly:
  - `tools/gen_codebook.py` generates the codebook artifacts,
  - the committed outputs live in `tools/codebook.json` and `crates/ringgrid/src/marker/codebook.rs`,
  - the shipped artifacts were generated with seed `1`,
  - the achieved minimum distance is the committed truth, not the initial requested target.
- No updated doc still uses the stale confidence wording `margin / 3` where the text is describing the current shipped decoder.
- Validation evidence includes a targeted grep sweep or equivalent proving the stale numeric claims were removed.

### Test Plan
- Consistency/source-of-truth checks:
  - `rg -n "minimum cyclic Hamming distance of 5|min cyclic Hamming:\\s+3|CODEBOOK_MIN_CYCLIC_DIST = 5|margin / 3" README.md book/src docs/decisions`
  - `sed -n '1,20p' crates/ringgrid/src/marker/codebook.rs`
  - `sed -n '1,20p' tools/codebook.json`
- CLI/example verification:
  - `cargo run --quiet -- codebook-info`
  - if the CLI command is too expensive or unavailable locally, inspect `crates/ringgrid-cli/src/main.rs` and record that substitution explicitly
- Optional docs build:
  - `mdbook build book`
  - if `mdbook` is unavailable, record `not run` rather than failing the task

### Out Of Scope
- Recomputing or replacing the codebook.
- Introducing a new codebook profile or extension mode.
- README information architecture work that belongs to `DOCS-002`.
- Python API documentation work that belongs to `DOCS-001`.
- Any detector behavior or configuration change motivated only by the docs drift discovered here.

### Handoff To Implementer
- Use the committed generated artifacts as the baseline facts and bring the prose into alignment; do not "fix" the codebook to match old docs.
- Prioritize the clearly stale locations first:
  - `README.md`
  - `book/src/cli-guide.md`
  - `docs/decisions/006-codebook-invariants.md`
- Then sweep the book chapters that discuss codebook guarantees and decode confidence so they do not imply stronger guarantees than `CODEBOOK_MIN_CYCLIC_DIST = 2` supports.
- Add concise provenance wording that ties together:
  - `tools/gen_codebook.py`,
  - seed `1`,
  - `tools/codebook.json`,
  - `crates/ringgrid/src/marker/codebook.rs`,
  - and the generated-artifact "do not hand-edit" rule.
- Keep edits tight. If you find a decision doc that is stale beyond this task's narrow scope, either make the smallest safe factual correction or call it out explicitly in `02-implementer.md`.
- Record validation evidence with the exact grep/CLI/source checks you used so the reviewer can reproduce the consistency pass quickly.
